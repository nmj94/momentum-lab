"""Private bundles, WAL snapshots, hostile members and non-destructive restoration."""

import hashlib
import json
import os
import sqlite3
import stat
import struct
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, nullcontext
from pathlib import Path

import pytest

from momentum_lab import (
    BackupError,
    RunBusyError,
    StudyRegistry,
    cli,
    create_backup,
    inspect_backup,
    inspect_run,
    restore_backup,
)
from momentum_lab import backups as bk
from momentum_lab.run_control import RunSession


@pytest.fixture
def research(tmp_path):
    target = tmp_path / "runs" / "example"
    registry = StudyRegistry(tmp_path / "registry.sqlite3")
    context = {
        "ticker": "AAA",
        "start": "2020-01-01",
        "end": "2020-12-31",
        "data_snapshot": "a" * 64,
        "run_id": "other-study",
        "run_path": tmp_path / "other-study",
    }
    claim = registry.claim_test(**context)
    registry.complete_test(claim["access"]["event_id"], {"secret_score": "DO_NOT_PRINT"})
    target.mkdir(parents=True)
    with RunSession(target, "search") as session:
        session.start()
        (target / "run_config.json").write_text(
            json.dumps({"registry_path": str(registry.path), "registry_id": registry.registry_id}), encoding="utf-8"
        )
        (target / "summary.json").write_text('{"secret_score":"DO_NOT_PRINT"}', encoding="utf-8")
        session.complete(["run_config.json", "summary.json"])
    return target, registry, session


def bundle(research, tmp_path):
    path = tmp_path / "example.mlbackup.zip"
    result = create_backup(research[0], path, acknowledge_sensitive=True)
    return path, result


def logical(path):
    with closing(sqlite3.connect(path)) as connection:
        return list(connection.iterdump())


def rewrite(path, *, manifest=None, member=None, extra=None):
    with zipfile.ZipFile(path) as archive:
        entries = [(info.filename, archive.read(info)) for info in archive.infolist()]
    with zipfile.ZipFile(path, "w") as archive:
        for name, raw in entries:
            if name == "manifest.json" and manifest is not None:
                value = json.loads(raw)
                manifest(value)
                raw = json.dumps(value).encode()
            if member is not None:
                name, raw = member(name, raw)
            archive.writestr(bk._info(name, len(raw)), raw)
        if extra is not None:
            archive.writestr(*extra)


def test_whole_registry_outputs_state_and_old_artifacts_survive_inactive_restore(research, tmp_path):
    target, registry, session = research
    (target / "notes").mkdir()
    (target / "notes" / "research.txt").write_text("private notes", encoding="utf-8")
    (target / "summary.old.bak.json").write_text("prior output", encoding="utf-8")
    before = logical(registry.path)
    path, created = bundle(research, tmp_path)
    assert "DO_NOT_PRINT" not in json.dumps(created)
    assert created["files"] == 6 and created["registry_id"] == registry.registry_id
    assert path.stat().st_nlink == 1
    assert inspect_backup(path, expected_sha256=created["archive_sha256"])["integrity"] == "verified"
    output = tmp_path / "recovered"
    restored = restore_backup(path, output, acknowledge_sensitive=True, expected_sha256=created["archive_sha256"])
    assert restored["status"] == "restored_inactive" and restored["active"] is False
    assert "DO_NOT_PRINT" not in json.dumps(restored)
    root = output / "payload"
    assert logical(root / "registry/research.sqlite3") == before == logical(registry.path)
    assert logical(root / "control/state.sqlite3") == logical(session.path)
    for original in target.rglob("*"):
        if original.is_file():
            assert (root / "run" / original.relative_to(target)).read_bytes() == original.read_bytes()
    assert not list(output.rglob("owner.lock"))
    assert inspect_run(root / "run")["status"] == "untracked"
    assert inspect_run(target, verify=True)["integrity"] == "verified"
    assert json.loads((output / "RESTORE.json").read_text())["active"] is False


@pytest.mark.parametrize("acknowledgement", [False, None, 1, "yes"])
def test_sensitive_consent_is_explicit_and_precedes_all_io(tmp_path, acknowledgement):
    with pytest.raises(BackupError, match="acknowledge_sensitive"):
        create_backup(tmp_path / "missing", tmp_path / "backup.zip", acknowledge_sensitive=acknowledgement)
    with pytest.raises(BackupError, match="acknowledge_sensitive"):
        restore_backup(tmp_path / "missing", tmp_path / "restored", acknowledge_sensitive=acknowledgement)
    assert not list(tmp_path.iterdir())


def test_busy_owner_prevents_export_and_does_not_add_attempts(research, tmp_path):
    with RunSession(research[0], "search"), pytest.raises(RunBusyError):
        create_backup(research[0], tmp_path / "backup.zip", acknowledge_sensitive=True)
    assert len(inspect_run(research[0])["history"]) == 1
    assert not (tmp_path / "backup.zip").exists()


def test_concurrent_backup_creators_cannot_overwrite_each_other(research, tmp_path):
    def create():
        try:
            return create_backup(research[0], tmp_path / "backup.zip", acknowledge_sensitive=True)
        except (RunBusyError, BackupError, FileExistsError):
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: create(), range(2)))
    assert sum(result is not None for result in results) == 1
    assert inspect_backup(tmp_path / "backup.zip")["integrity"] == "verified"


@pytest.mark.parametrize("change", ["missing", "identity", "receipt", "state", "missing-lock"])
def test_missing_or_inconsistent_provenance_is_not_silently_repaired(research, tmp_path, change):
    target, registry, session = research
    if change == "missing":
        registry.path.unlink()
    elif change == "identity":
        with sqlite3.connect(registry.path) as connection:
            connection.execute("UPDATE registry_info SET registry_id=?", ("b" * 32,))
    elif change == "receipt":
        (target / "summary.json").write_text("tampered", encoding="utf-8")
    elif change == "state":
        session.path.write_bytes(b"broken")
    else:
        session.lock.path.unlink()
    with pytest.raises((ValueError, OSError, sqlite3.DatabaseError)):
        create_backup(target, tmp_path / "backup.zip", acknowledge_sensitive=True)
    assert not (tmp_path / "backup.zip").exists()
    assert not list(tmp_path.glob(".momentum-backup-*"))


@pytest.mark.parametrize("status", ["failed", "running"])
def test_partial_or_unclean_attempts_are_preserved_not_relabelled_complete(research, tmp_path, status):
    with sqlite3.connect(research[2].path) as connection:
        connection.execute("UPDATE attempts SET status=?,finished_at=NULL,artifacts_json=NULL", (status,))
    path, result = bundle(research, tmp_path)
    assert result["run_status"] == ("interrupted" if status == "running" else "failed")
    output = tmp_path / "restore"
    restore_backup(path, output, acknowledge_sensitive=True)
    assert logical(output / "payload/control/state.sqlite3") == logical(research[2].path)


@pytest.mark.parametrize("database", ["results.sqlite3", "results.old.bak.sqlite3", "registry"])
def test_sqlite_backup_includes_committed_wal_and_never_raw_copies_sidecars(research, tmp_path, database):
    target, registry, _ = research
    source = registry.path if database == "registry" else target / database
    with closing(sqlite3.connect(source)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE wal_proof (value TEXT)")
        writer.execute("INSERT INTO wal_proof VALUES ('committed-only-in-WAL')")
        writer.commit()
        assert Path(str(source) + "-wal").stat().st_size > 0
        path, _ = bundle(research, tmp_path)
        restored = tmp_path / "recovered"
        restore_backup(path, restored, acknowledge_sensitive=True)
        with zipfile.ZipFile(path) as archive:
            assert not any(name.endswith(("-wal", "-shm", "-journal")) for name in archive.namelist())
        name = "registry/research.sqlite3" if database == "registry" else "run/" + database
        with closing(sqlite3.connect(restored / "payload" / name)) as connection:
            assert connection.execute("SELECT value FROM wal_proof").fetchone()[0] == "committed-only-in-WAL"
            assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


def test_database_exclusive_writer_fails_promptly_without_published_archive(research, tmp_path):
    with closing(sqlite3.connect(research[1].path)) as writer:
        writer.execute("BEGIN EXCLUSIVE")
        with pytest.raises((BackupError, ValueError, sqlite3.DatabaseError)):
            create_backup(research[0], tmp_path / "backup.zip", acknowledge_sensitive=True)
    assert not (tmp_path / "backup.zip").exists()


def test_closed_wal_database_can_create_new_sidecars_during_readonly_backup(research, tmp_path, monkeypatch):
    database = research[0] / "results.sqlite3"
    with closing(sqlite3.connect(database)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE proof (value INTEGER)")
        writer.execute("INSERT INTO proof VALUES (42)")
        writer.commit()
    assert not Path(str(database) + "-wal").exists()
    original = bk._snapshot_database

    def with_bookkeeping(source, destination):
        result = original(source, destination)
        if source == database:
            # Reproduce SQLite versions that retain these newly created support
            # files after the last reader closes. They are not payload changes.
            for suffix in ("-wal", "-shm"):
                Path(str(source) + suffix).touch(exist_ok=True)
        return result

    monkeypatch.setattr(bk, "_snapshot_database", with_bookkeeping)
    path, _ = bundle(research, tmp_path)
    with zipfile.ZipFile(path) as archive:
        assert not any(name.endswith(("-wal", "-shm")) for name in archive.namelist())
    restore_backup(path, tmp_path / "recovered", acknowledge_sensitive=True)
    with closing(sqlite3.connect(tmp_path / "recovered/payload/run/results.sqlite3")) as connection:
        assert connection.execute("SELECT value FROM proof").fetchone() == (42,)


def test_orphan_sidecar_and_corrupt_result_database_are_not_omitted(research, tmp_path):
    orphan = research[0] / "results.sqlite3-wal"
    orphan.write_bytes(b"orphan")
    with pytest.raises(BackupError, match="Orphan"):
        create_backup(research[0], tmp_path / "backup.zip", acknowledge_sensitive=True)
    orphan.unlink()
    (research[0] / "results.sqlite3").write_bytes(b"corrupted database")
    with pytest.raises(sqlite3.DatabaseError):
        create_backup(research[0], tmp_path / "backup.zip", acknowledge_sensitive=True)


@pytest.mark.parametrize("link_kind", ["file", "directory", "hardlink", "archive"])
def test_linked_inputs_are_rejected_without_touching_their_targets(research, tmp_path, link_kind):
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "private.txt"
    secret.write_text("UNCHANGED", encoding="utf-8")
    link = research[0] / "link"
    try:
        if link_kind == "hardlink":
            os.link(secret, link)
        elif link_kind == "directory":
            link.symlink_to(outside, target_is_directory=True)
        elif link_kind == "archive":
            path, _ = bundle(research, tmp_path)
            link = tmp_path / "alias.zip"
            link.symlink_to(path)
        else:
            link.symlink_to(secret)
    except OSError as exc:
        pytest.skip(f"Links unavailable: {exc}")
    with pytest.raises(BackupError):
        if link_kind == "archive":
            inspect_backup(link)
        else:
            create_backup(research[0], tmp_path / "backup.zip", acknowledge_sensitive=True)
    assert secret.read_text() == "UNCHANGED"


def test_output_cannot_be_nested_in_live_run_or_control(research):
    for path in (research[0] / "backup.zip", research[2].lock.control / "backup.zip"):
        with pytest.raises(BackupError, match="outside"):
            create_backup(research[0], path, acknowledge_sensitive=True)


def test_existing_archive_and_restore_directories_are_never_overwritten(research, tmp_path):
    path, _ = bundle(research, tmp_path)
    before = path.read_bytes()
    with pytest.raises(BackupError, match="already exists"):
        create_backup(research[0], path, acknowledge_sensitive=True)
    assert path.read_bytes() == before
    for name in ("empty", "occupied"):
        output = tmp_path / name
        output.mkdir()
        if name == "occupied":
            (output / "keep.txt").write_text("KEEP", encoding="utf-8")
        with pytest.raises(BackupError, match="already exists"):
            restore_backup(path, output, acknowledge_sensitive=True)
    assert (tmp_path / "occupied/keep.txt").read_text() == "KEEP"


def test_publication_race_does_not_replace_a_competing_file(research, tmp_path, monkeypatch):
    output = tmp_path / "backup.zip"
    real_link = os.link

    def racing_link(source, destination, **kwargs):
        Path(destination).write_text("OTHER CREATOR", encoding="utf-8")
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(bk.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        create_backup(research[0], output, acknowledge_sensitive=True)
    assert output.read_text() == "OTHER CREATOR"


def test_interrupted_restore_has_no_completion_marker_and_cannot_overwrite_on_retry(research, tmp_path, monkeypatch):
    path, _ = bundle(research, tmp_path)
    output = tmp_path / "restore"
    with monkeypatch.context() as patch:
        patch.setattr(bk.os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError("publication interrupted")))
        with pytest.raises(OSError, match="publication interrupted"):
            restore_backup(path, output, acknowledge_sensitive=True)
    assert (output / "payload/run/run_config.json").exists()
    assert not (output / "RESTORE.json").exists()
    with pytest.raises(BackupError, match="already exists"):
        restore_backup(path, output, acknowledge_sensitive=True)


def test_failed_export_cleans_only_private_staging_and_releases_ownership(research, tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(bk, "_write_member", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
        with pytest.raises(OSError, match="disk full"):
            bundle(research, tmp_path)
    assert not list(tmp_path.glob(".momentum-backup-*"))
    assert not (tmp_path / "example.mlbackup.zip").exists()
    with RunSession(research[0], "search"):
        pass
    bundle(research, tmp_path)


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "C:/drive",
        "run/../x",
        "run/a\\b",
        "run/a:b",
        "run/CON.txt",
        "run/NUL .txt",
        "run/ AUX.txt",
        "run/COM¹.log",
        "run/x\u202etxt",
        "run/x.",
        "run/x ",
        "run//x",
    ],
)
def test_hostile_member_paths_never_escape_restoration(research, tmp_path, name):
    path, _ = bundle(research, tmp_path)
    rewrite(path, member=lambda n, raw: (name if n == "run/summary.json" else n, raw))
    with pytest.raises(BackupError):
        restore_backup(path, tmp_path / "restore", acknowledge_sensitive=True)
    assert not (tmp_path / "restore").exists()
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "duplicate",
        "case-alias",
        "symlink",
        "compressed",
        "wrong-hash",
        "bad-schema",
        "oversized",
        "nan",
        "duplicate-json",
        "central-count",
        "central-length",
        "comment",
    ],
)
def test_invalid_archives_fail_before_publication(research, tmp_path, change):
    path, _ = bundle(research, tmp_path)
    if change == "missing":
        rewrite(path, member=lambda n, raw: ("run/other.json" if n == "run/summary.json" else n, raw))
    elif change in {"extra", "duplicate", "case-alias"}:
        name = {"extra": "run/extra.txt", "duplicate": "run/summary.json", "case-alias": "run/SUMMARY.json"}[change]
        with pytest.warns(UserWarning) if change == "duplicate" else nullcontext():
            rewrite(path, extra=(bk._info(name, 1), b"x"))
    elif change in {"symlink", "compressed"}:
        info = bk._info("run/extra.txt", 1)
        if change == "symlink":
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
        else:
            info.compress_type = zipfile.ZIP_DEFLATED
        rewrite(path, extra=(info, b"x"))
    elif change == "wrong-hash":
        rewrite(path, manifest=lambda value: value["files"][0].update(sha256="0" * 64))
    elif change == "bad-schema":
        rewrite(path, manifest=lambda value: value.update(schema_version=True))
    elif change == "oversized":
        rewrite(path, manifest=lambda value: value["files"][0].update(bytes=bk.MAX_BYTES + 1))
    elif change in {"nan", "duplicate-json"}:
        rewrite(
            path,
            member=lambda n, raw: (
                n,
                b'{"schema_version":NaN}'
                if change == "nan" and n == "manifest.json"
                else b'{"x":1,"x":2}'
                if n == "manifest.json"
                else raw,
            ),
        )
    else:
        raw = bytearray(path.read_bytes())
        if change == "central-count":
            struct.pack_into("<HH", raw, len(raw) - 14, bk.MAX_FILES + 1, bk.MAX_FILES + 1)
        elif change == "central-length":
            struct.pack_into("<L", raw, len(raw) - 10, bk.MAX_DIRECTORY + 1)
        else:
            raw.extend(b"comment")
        path.write_bytes(raw)
    with pytest.raises(BackupError):
        restore_backup(path, tmp_path / "restore", acknowledge_sensitive=True)
    assert not (tmp_path / "restore").exists()


def test_archive_external_hash_and_payload_mutation_are_detected(research, tmp_path):
    path, created = bundle(research, tmp_path)
    rewrite(path, member=lambda n, raw: (n, b"changed" if n == "run/summary.json" else raw))
    with pytest.raises(BackupError, match="SHA-256"):
        inspect_backup(path, expected_sha256=created["archive_sha256"])
    with pytest.raises(BackupError):
        inspect_backup(path)


def test_inspection_does_not_extract_open_archived_sqlite_or_write(research, tmp_path, monkeypatch):
    path, _ = bundle(research, tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(
        bk.sqlite3, "connect", lambda *a, **k: pytest.fail("Inspector must not open archived databases")
    )
    assert inspect_backup(path)["integrity"] == "verified"
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_cli_commands_and_no_score_disclosure(research, tmp_path, monkeypatch, capsys):
    path = tmp_path / "backup.zip"
    monkeypatch.setattr(sys, "argv", ["momentum-lab", "backup", "create", str(research[0]), "--output", str(path)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2 and "acknowledge" in capsys.readouterr().err
    sys.argv.append("--acknowledge-sensitive")
    assert cli.main() == 0
    created = json.loads(capsys.readouterr().out)
    assert created["archive_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(sys, "argv", ["momentum-lab", "backup", "inspect", str(path)])
    assert cli.main() == 0
    assert "DO_NOT_PRINT" not in capsys.readouterr().out
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "momentum-lab",
            "backup",
            "restore",
            str(path),
            "--output",
            str(tmp_path / "restore"),
            "--acknowledge-sensitive",
        ],
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "restored_inactive"


def test_restoring_an_old_snapshot_cannot_roll_back_newer_observation_history(research, tmp_path):
    path, _ = bundle(research, tmp_path)
    registry = research[1]
    registry.record_development(
        ticker="BBB",
        start="2022-01-01",
        end="2022-12-31",
        data_snapshot="b" * 64,
        run_id="after-backup",
        run_path=tmp_path / "newer",
    )
    current = logical(registry.path)
    restore_backup(path, tmp_path / "recovered", acknowledge_sensitive=True)
    snapshot = StudyRegistry(tmp_path / "recovered/payload/registry/research.sqlite3", create=False)
    assert snapshot.registry_id == registry.registry_id
    assert len(snapshot.history()) == 1 and len(registry.history()) == 2
    assert logical(registry.path) == current


@pytest.mark.parametrize("damage", ["new-file", "changed-config", "changed-file"])
def test_source_changes_during_export_prevent_archive_publication(research, tmp_path, monkeypatch, damage):
    original = bk._write_member

    def mutate(archive, source, name, remaining):
        if name == "run/run_config.json":
            if damage == "new-file":
                (research[0] / "unexpected.txt").write_text("new", encoding="utf-8")
            elif damage == "changed-config":
                source.write_bytes(source.read_bytes() + b" ")
            else:
                (research[0] / "summary.json").write_text("changed", encoding="utf-8")
        return original(archive, source, name, remaining)

    monkeypatch.setattr(bk, "_write_member", mutate)
    with pytest.raises(BackupError):
        bundle(research, tmp_path)
    assert not (tmp_path / "example.mlbackup.zip").exists()


def test_database_inside_output_uses_the_same_complete_registry_snapshot(research, tmp_path):
    target, registry, _ = research
    local = bk._snapshot_database(registry.path, target / "registry.sqlite3")
    with RunSession(target, "search", mode="resume") as execution:
        execution.start()
        config = target / "run_config.json"
        values = json.loads(config.read_text())
        config.write_text(json.dumps({**values, "registry_path": str(local)}), encoding="utf-8")
        execution.complete(["run_config.json", "summary.json"])
    path, _ = bundle(research, tmp_path)
    with zipfile.ZipFile(path) as archive:
        assert archive.read("run/registry.sqlite3") == archive.read("registry/research.sqlite3")


@pytest.mark.parametrize("option", ["x", "A" * 64, 1, True])
def test_invalid_external_hash_fails_before_reading_files(tmp_path, option):
    with pytest.raises(BackupError, match="expected_sha256"):
        inspect_backup(tmp_path / "missing", expected_sha256=option)


@pytest.mark.parametrize("limit", ["files", "bytes", "manifest"])
def test_resource_caps_prevent_published_partial_bundles(research, tmp_path, monkeypatch, limit):
    monkeypatch.setattr(bk, {"files": "MAX_FILES", "bytes": "MAX_BYTES", "manifest": "MAX_MANIFEST"}[limit], 1)
    with pytest.raises(BackupError):
        bundle(research, tmp_path)
    assert not (tmp_path / "example.mlbackup.zip").exists()


def test_real_case_alias_cannot_place_a_backup_in_live_storage(research, tmp_path):
    target, _, session = research
    alias = target.with_name(target.name.upper())
    if not alias.exists():
        alias.mkdir()
    if alias.samefile(target):
        with pytest.raises(BackupError, match="outside"):
            create_backup(target, alias / "backup.zip", acknowledge_sensitive=True)
    else:
        assert create_backup(target, alias / "backup.zip", acknowledge_sensitive=True)["integrity"] == "verified"
    assert not (session.lock.control / "backup.zip").exists()


def test_archive_changed_during_read_is_not_published_as_a_completed_restore(research, tmp_path, monkeypatch):
    path, _ = bundle(research, tmp_path)
    original = bk._verify_files

    def mutate(archive, manifest, destination=None):
        original(archive, manifest, destination)
        with path.open("ab") as handle:
            handle.write(b"changed")

    monkeypatch.setattr(bk, "_verify_files", mutate)
    with pytest.raises(BackupError, match="changed"):
        restore_backup(path, tmp_path / "recovered", acknowledge_sensitive=True)
    assert not (tmp_path / "recovered").exists()
