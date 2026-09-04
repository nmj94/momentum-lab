"""Bounded private recovery bundles; restoration never activates a research copy.

The selected output is locked, SQLite databases use the online backup API, and
the entire shared registry is retained. External data/software are NOT bundled.
ZIP members are validated and streamed explicitly, never passed to extractall.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import struct
import tempfile
import time
import unicodedata
import zipfile
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path
from uuid import uuid4

from . import run_control as rc
from ._version import __version__
from .governance import StudyRegistry

BACKUP_SCHEMA = 1
MAX_FILES = 4096
MAX_BYTES = 1024**3
MAX_MANIFEST = 1024**2
MAX_DIRECTORY = 8 * 1024**2
_CHUNK = 1024**2
_SQLITE = b"SQLite format 3\0"
_RESERVED = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\..*)?", re.IGNORECASE)
_REQUIRED = {"run/run_config.json", "control/state.sqlite3", "registry/research.sqlite3"}
_NOTICE = (
    "Private, unencrypted recovery bundle: contains the entire shared registry, including other studies and cached "
    "reveals. External datasets and software environments are not included. Restore is inactive and never replaces "
    "current history, grants reveal permission or establishes previously unseen data."
)


class BackupError(ValueError):
    """A backup cannot be safely created, verified or restored."""


def _acknowledge(value):
    if value is not True:
        raise BackupError("Explicit acknowledge_sensitive=True / --acknowledge-sensitive is required. " + _NOTICE)


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise BackupError("Duplicate JSON field")
            result[key] = value
        return result

    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise BackupError("Non-finite JSON number")
        return result

    try:
        result = json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_float=number, parse_constant=number)
    except (ValueError, RecursionError) as exc:
        raise BackupError("Invalid bounded JSON object") from exc
    if not isinstance(result, dict):
        raise BackupError("Expected a JSON object")
    return result


def _path_key(name):
    if not isinstance(name, str) or not name or len(name.encode("utf-8")) > 512:
        raise BackupError("Invalid portable bundle path")
    parts = name.split("/")
    if len(parts) > 16:
        raise BackupError("Bundle paths exceed the directory-depth limit")
    for part in parts:
        if (
            not part
            or len(part) > 128
            or part in {".", ".."}
            or part != part.strip()
            or not part.isprintable()
            or part.endswith((" ", "."))
            or any(ord(c) < 32 or ord(c) == 127 or c in '\\:<>"|?*' for c in part)
            or _RESERVED.fullmatch(part.split(".", 1)[0].rstrip(" "))
        ):
            raise BackupError("Unsafe or non-portable bundle path")
    return "/".join(unicodedata.normalize("NFC", part).casefold() for part in parts)


def _paths(names):
    files, directories = set(), {}
    for name in names:
        key = _path_key(name)
        if key in files or key in directories:
            raise BackupError("Duplicate or colliding bundle path")
        files.add(key)
        parts, keys = name.split("/"), key.split("/")
        for i in range(1, len(parts)):
            directory, original = "/".join(keys[:i]), "/".join(parts[:i])
            if directory in files or directories.get(directory, original) != original:
                raise BackupError("Conflicting file/directory or native path alias")
            directories[directory] = original


def _stat(path, *, directory=False):
    info = path.lstat()
    reparse = getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if reparse or (not stat.S_ISDIR(info.st_mode) if directory else not stat.S_ISREG(info.st_mode)):
        raise BackupError(
            "Expected an ordinary, non-linked directory" if directory else "Expected a regular backup file"
        )
    if not directory and info.st_nlink != 1:
        raise BackupError("Hard-linked backup inputs are unsupported")
    return info


def _unchanged(path, before, *, identity_only=False):
    after = _stat(path)
    keys = ("st_dev", "st_ino") if identity_only else ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in keys):
        raise BackupError("Input changed during backup/verification")


@contextmanager
def _reader(path):
    before = _stat(path)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise BackupError("Input identity changed while opening")
        yield handle, before
        _unchanged(path, before)


def _stream(source, destination=None, *, limit=MAX_BYTES):
    digest, size = hashlib.sha256(), 0
    while chunk := source.read(min(_CHUNK, limit - size + 1)):
        size += len(chunk)
        if size > limit:
            raise BackupError("Backup byte limit exceeded")
        digest.update(chunk)
        if destination is not None:
            destination.write(chunk)
    return size, digest.hexdigest()


def _walk(root):
    _stat(root, directory=True)
    pending, files, visited = [root], {}, 0
    while pending:
        parent = pending.pop()
        _stat(parent, directory=True)
        for path in sorted(parent.iterdir()):
            visited += 1
            if visited > MAX_FILES:
                raise BackupError("Too many run files/directories")
            name = path.relative_to(root).as_posix()
            _path_key(name)
            if stat.S_ISDIR(path.lstat().st_mode):
                _stat(path, directory=True)
                pending.append(path)
            else:
                _stat(path)
                files[name] = path
    _paths(files)
    return files


def _snapshot_database(source, destination):
    before = _stat(source)
    if before.st_size > MAX_BYTES:
        raise BackupError("Database exceeds the backup limit")
    deadline = time.monotonic() + 60

    def progress(status, remaining, total):
        if status not in {0, 101} or time.monotonic() > deadline:
            raise BackupError("Database busy or backup time limit reached; retry after writers stop")

    # No checkpoint, schema migration, row filtering, or source write transaction.
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=0)) as src:
        src.execute("PRAGMA trusted_schema=OFF")
        src.execute("BEGIN")
        page_size = src.execute("PRAGMA page_size").fetchone()[0]
        if src.execute("PRAGMA page_count").fetchone()[0] * page_size > MAX_BYTES:
            raise BackupError("Database snapshot exceeds the backup limit")
        destination.touch(mode=0o600, exist_ok=False)
        with closing(sqlite3.connect(destination)) as dst:
            dst.execute("PRAGMA synchronous=FULL")
            src.backup(dst, pages=256, progress=progress, sleep=0)
            dst.execute("PRAGMA journal_mode=DELETE")
            dst.execute("PRAGMA trusted_schema=OFF")
            dst.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
            if dst.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise BackupError("Database integrity check failed")
    _unchanged(source, before, identity_only=True)
    return destination


def _new_output(path):
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise BackupError("Output already exists; choose a new path (no overwrite)")
    return path.parent.resolve() / path.name


def _inside(path, root):
    # String prefix checks alone miss case/Unicode aliases on native macOS and
    # Windows filesystems. Existing physical ancestors must agree as well.
    return path.is_relative_to(root) or any(parent.exists() and parent.samefile(root) for parent in path.parents)


def _info(name, size):
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o600) << 16
    info.file_size = size
    info.compress_type = zipfile.ZIP_STORED
    return info


def _write_member(archive, path, name, remaining):
    with _reader(path) as (source, before):
        if before.st_size > remaining:
            raise BackupError("Backup payload exceeds the 1 GiB limit")
        with archive.open(_info(name, before.st_size), "w") as output:
            size, digest = _stream(source, output, limit=remaining)
    return {"path": name, "bytes": size, "sha256": digest}


def _state(lock):
    with closing(rc._connect(lock.control / "state.sqlite3", lock.target)) as connection:
        row = connection.execute("SELECT * FROM attempts ORDER BY sequence DESC LIMIT 1").fetchone()
    if row is None or row["status"] not in {"running", "completed", "failed", "interrupted"}:
        raise BackupError("A recorded operational attempt is required")
    if row["workflow"] not in {"search", "portfolio", "portfolio_study"}:
        raise BackupError("Unknown run workflow")
    return row


def create_backup(run_dir, output, *, acknowledge_sensitive=False):
    """Archive one idle tracked output and its FULL shared registry, without scores in the return value."""
    _acknowledge(acknowledge_sensitive)
    destination = _new_output(output)
    lock = rc._DirectoryLock(run_dir)
    if _inside(destination, lock.target) or _inside(destination, lock.control.parent):
        raise BackupError("Backup output must be outside run outputs and run-control storage")
    try:
        if not lock.acquire(create=False):
            raise rc.RunBusyError("Run is busy; wait before backing it up")
        lock.check()
        attempt = _state(lock)
        files = _walk(lock.target)
        with _reader(lock.target / "run_config.json") as (handle, _):
            config_raw = handle.read(MAX_MANIFEST + 1)
        if len(config_raw) > MAX_MANIFEST:
            raise BackupError("Run configuration exceeds 1 MiB")
        config = _json(config_raw)
        registry_path = config.get("registry_path")
        if not isinstance(registry_path, str) or not Path(registry_path).is_absolute():
            raise BackupError("Original absolute registry_path is required")
        registry_path = Path(registry_path)
        _stat(registry_path)
        registry = StudyRegistry(registry_path, create=False)
        if config.get("registry_id") != registry.registry_id:
            raise BackupError("Run and registry identities differ; never substitute a fresh registry")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".momentum-backup-", dir=destination.parent) as folder:
            staging = Path(folder)
            databases = {}
            for name, path in files.items():
                with _reader(path) as (handle, _):
                    is_database = handle.read(16) == _SQLITE
                if is_database or name.endswith(".sqlite3"):
                    databases[name] = path
            # SQLite may create its bookkeeping files when a read-only WAL
            # database is opened, even if no sidecars existed at inventory time.
            # Only companions of already identified databases are excluded.
            omitted = {name + suffix for name in databases for suffix in ("-wal", "-shm", "-journal")}
            if any(name.endswith(("-wal", "-shm", "-journal")) and name not in omitted for name in files):
                raise BackupError("Orphan database sidecar; preserve and repair provenance before backup")
            snapshots = {}
            for name, path in [
                ("registry/research.sqlite3", registry.path),
                ("control/state.sqlite3", lock.control / "state.sqlite3"),
            ]:
                snapshots[name] = _snapshot_database(path, staging / f"{len(snapshots)}.sqlite3")
            copied_registry = StudyRegistry(snapshots["registry/research.sqlite3"], create=False)
            if copied_registry.registry_id != registry.registry_id:
                raise BackupError("Registry identity changed during snapshot")
            archive_path = staging / "bundle.zip"
            entries, total = [], 0
            archive_path.touch(mode=0o600, exist_ok=False)
            with zipfile.ZipFile(archive_path, "w", allowZip64=False) as archive:
                for name, path in sorted(files.items()):
                    if name in omitted:
                        continue
                    source = path
                    if name in databases:
                        source = (
                            snapshots["registry/research.sqlite3"]
                            if path.resolve() == registry.path
                            else _snapshot_database(path, staging / "run-database.sqlite3")
                        )
                    entry = _write_member(archive, source, "run/" + name, MAX_BYTES - total)
                    entries.append(entry)
                    total += entry["bytes"]
                    if source == staging / "run-database.sqlite3":
                        source.unlink()
                for name, path in snapshots.items():
                    entry = _write_member(archive, path, name, MAX_BYTES - total)
                    entries.append(entry)
                    total += entry["bytes"]
                by_name = {entry["path"]: entry for entry in entries}
                if by_name["run/run_config.json"]["sha256"] != hashlib.sha256(config_raw).hexdigest():
                    raise BackupError("Run configuration changed during backup")
                if attempt["status"] == "completed":
                    for expected in rc._manifest(attempt["artifacts_json"]):
                        actual = by_name.get("run/" + expected["path"])
                        if actual is None or (actual["bytes"], actual["sha256"]) != (
                            expected["bytes"],
                            expected["sha256"],
                        ):
                            raise BackupError("Completion receipt mismatch; preserve original evidence before backup")
                if set(_walk(lock.target)) - omitted != set(files) - omitted:
                    raise BackupError("Run file inventory changed during backup")
                lock.check()
                manifest = {
                    "schema_version": BACKUP_SCHEMA,
                    "bundle_id": uuid4().hex,
                    "created_at": rc._now(),
                    "package_version": __version__,
                    "source_run_dir": str(lock.target),
                    "source_registry_path": str(registry.path),
                    "registry_id": registry.registry_id,
                    "workflow": attempt["workflow"],
                    "attempt_id": attempt["attempt_id"],
                    "run_status": "interrupted" if attempt["status"] == "running" else attempt["status"],
                    "files": entries,
                }
                raw = json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
                if len(raw) > MAX_MANIFEST or len(entries) + 1 > MAX_FILES:
                    raise BackupError("Backup manifest/file count exceeds its limit")
                archive.writestr(_info("manifest.json", len(raw)), raw)
            with archive_path.open("r+b") as handle:
                os.fsync(handle.fileno())
            verified = inspect_backup(archive_path)
            # Atomic exclusive publication on local filesystems supporting hard
            # links. No replace fallback: another creator's destination survives.
            os.link(archive_path, destination)
            archive_path.unlink()
        return {**verified, "backup": str(destination)}
    finally:
        lock.close()


def _directory_header(handle, size):
    # Check bounded central-directory size/count BEFORE ZipFile allocates entries.
    # Schema 1 uses stored members, no ZIP64, comments, prepended data or volumes.
    if not 22 <= size <= MAX_BYTES + MAX_MANIFEST + MAX_DIRECTORY:
        raise BackupError("Invalid backup archive size")
    handle.seek(-22, os.SEEK_END)
    fields = struct.unpack("<4s4H2LH", handle.read(22))
    signature, disk, start_disk, local_count, count, length, offset, comment = fields
    if (
        signature != b"PK\x05\x06"
        or disk
        or start_disk
        or comment
        or local_count != count
        or not 4 <= count <= MAX_FILES
        or length > MAX_DIRECTORY
        or offset + length != size - 22
    ):
        raise BackupError("Unsupported or oversized ZIP directory")
    handle.seek(offset)
    for _ in range(count):
        header = handle.read(46)
        if len(header) != 46 or header[:4] != b"PK\x01\x02":
            raise BackupError("Invalid ZIP directory entry")
        name_size, extra_size, comment_size = struct.unpack_from("<HHH", header, 28)
        if not 1 <= name_size <= 512 or extra_size or comment_size:
            raise BackupError("Unsupported ZIP directory metadata")
        handle.seek(name_size, os.SEEK_CUR)
        if handle.tell() > offset + length:
            raise BackupError("ZIP directory exceeds its bounds")
    if handle.tell() != offset + length:
        raise BackupError("ZIP entry count does not match directory")
    handle.seek(0)


def _manifest(archive):
    members = archive.infolist()
    _paths(info.filename for info in members)
    if min(info.header_offset for info in members) != 0:
        raise BackupError("Prepended ZIP data is unsupported")
    if any(
        info.orig_filename != info.filename
        or info.is_dir()
        or info.flag_bits & 1
        or info.compress_type != zipfile.ZIP_STORED
        or info.compress_size != info.file_size
        or info.file_size > MAX_BYTES
        or stat.S_IFMT(info.external_attr >> 16) != stat.S_IFREG
        for info in members
    ):
        raise BackupError("Unsupported ZIP member: only unencrypted, stored regular files are allowed")
    names = {info.filename for info in members}
    if "manifest.json" not in names or archive.getinfo("manifest.json").file_size > MAX_MANIFEST:
        raise BackupError("Missing or oversized backup manifest")
    manifest = _json(archive.read("manifest.json"))
    fields = {
        "schema_version",
        "bundle_id",
        "created_at",
        "package_version",
        "source_run_dir",
        "source_registry_path",
        "registry_id",
        "workflow",
        "attempt_id",
        "run_status",
        "files",
    }
    if (
        set(manifest) != fields
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != BACKUP_SCHEMA
    ):
        raise BackupError("Unsupported backup manifest schema")
    for key in fields - {"schema_version", "files"}:
        value = manifest[key]
        if not isinstance(value, str) or not value or len(value) > 4096 or any(ord(c) < 32 for c in value):
            raise BackupError("Invalid backup metadata")
    if any(not re.fullmatch(r"[a-f0-9]{32}", manifest[key]) for key in ("bundle_id", "registry_id", "attempt_id")):
        raise BackupError("Invalid backup identity")
    if manifest["workflow"] not in {"search", "portfolio", "portfolio_study"} or manifest["run_status"] not in {
        "completed",
        "failed",
        "interrupted",
    }:
        raise BackupError("Invalid archived run status/workflow")
    entries = manifest["files"]
    if not isinstance(entries, list) or not 3 <= len(entries) < MAX_FILES:
        raise BackupError("Invalid backup file manifest")
    paths, total = [], 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise BackupError("Invalid file receipt")
        name, size, digest = entry["path"], entry["bytes"], entry["sha256"]
        _path_key(name)
        if name not in _REQUIRED and not name.startswith("run/"):
            raise BackupError("Unexpected bundle member scope")
        if (
            type(size) is not int
            or not 0 <= size <= MAX_BYTES
            or not isinstance(digest, str)
            or not re.fullmatch(r"[a-f0-9]{64}", digest)
        ):
            raise BackupError("Invalid file size/digest")
        if name not in names or archive.getinfo(name).file_size != size:
            raise BackupError("ZIP member and receipt disagree")
        paths.append(name)
        total += size
    _paths(paths)
    if not _REQUIRED.issubset(paths) or set(paths) | {"manifest.json"} != names or total > MAX_BYTES:
        raise BackupError("Incomplete, extra or oversized backup payload")
    return manifest


@contextmanager
def _bundle(path, expected_sha256):
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_sha256)
    ):
        raise BackupError("expected_sha256 must be a lowercase SHA-256 digest")
    path = Path(path).absolute()
    try:
        with _reader(path) as (source, before):
            _directory_header(source, before.st_size)
            _, digest = _stream(source, limit=MAX_BYTES + MAX_MANIFEST + MAX_DIRECTORY)
            if expected_sha256 is not None and digest != expected_sha256:
                raise BackupError("Archive SHA-256 differs from the independently recorded digest")
            source.seek(0)
            with zipfile.ZipFile(source) as archive:
                yield archive, _manifest(archive), digest
    except (zipfile.BadZipFile, zipfile.LargeZipFile, UnicodeError, RecursionError, KeyError) as exc:
        raise BackupError("Invalid recovery bundle") from exc


def _verify_files(archive, manifest, destination=None):
    for entry in manifest["files"]:
        with archive.open(entry["path"]) as source:
            if destination is None:
                size, digest = _stream(source, limit=entry["bytes"])
            else:
                path = destination.joinpath(*entry["path"].split("/"))
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "wb") as target:
                    size, digest = _stream(source, target, limit=entry["bytes"])
                    target.flush()
                    os.fsync(target.fileno())
        if (size, digest) != (entry["bytes"], entry["sha256"]):
            raise BackupError("Archived file does not match its receipt")


def _result(manifest, digest):
    return {
        "schema_version": BACKUP_SCHEMA,
        "bundle_id": manifest["bundle_id"],
        "created_at": manifest["created_at"],
        "workflow": manifest["workflow"],
        "run_status": manifest["run_status"],
        "registry_id": manifest["registry_id"],
        "files": len(manifest["files"]),
        "bytes": sum(e["bytes"] for e in manifest["files"]),
        "archive_sha256": digest,
        "integrity": "verified",
        "notice": _NOTICE,
    }


def inspect_backup(path, *, expected_sha256=None):
    """Fully verify a bounded bundle without extracting it, opening its databases or displaying scores."""
    with _bundle(path, expected_sha256) as (archive, manifest, digest):
        _verify_files(archive, manifest)
        result = _result(manifest, digest)
    return result


def restore_backup(path, output, *, acknowledge_sensitive=False, expected_sha256=None):
    """Restore into a NEW, inactive recovery directory, never over current research state."""
    _acknowledge(acknowledge_sensitive)
    destination = _new_output(output)
    with ExitStack() as resources:
        with _bundle(path, expected_sha256) as (archive, manifest, digest):
            # Private staging is fully verified before the destination is reserved.
            destination.parent.mkdir(parents=True, exist_ok=True)
            folder = resources.enter_context(
                tempfile.TemporaryDirectory(prefix=".momentum-restore-", dir=destination.parent)
            )
            staged = Path(folder) / "payload"
            staged.mkdir(mode=0o700)
            _verify_files(archive, manifest, staged)
            with (staged / "manifest.json").open("xb") as handle:
                handle.write(archive.read("manifest.json"))
                handle.flush()
                os.fsync(handle.fileno())
            result = {
                **_result(manifest, digest),
                "status": "restored_inactive",
                "active": False,
                "output": str(destination),
            }
        # _bundle has closed and checked the original archive's identity/content
        # metadata. Publish only the already verified private staging snapshot.
        marker = Path(folder) / "RESTORE.json"
        with marker.open("xb") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        destination.mkdir(mode=0o700, exist_ok=False)
        # An interrupted reservation remains incomplete without RESTORE.json;
        # never delete it automatically or overwrite it on retry.
        staged.rename(destination / "payload")
        os.link(marker, destination / "RESTORE.json")
        marker.unlink()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="momentum-lab backup", description="Private run + registry recovery bundles")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="Back up an idle tracked run and the entire shared registry")
    create.add_argument("run_dir")
    create.add_argument("--output", required=True)
    create.add_argument(
        "--acknowledge-sensitive", action="store_true", help="Confirm unencrypted export of all registry history/caches"
    )
    inspect = commands.add_parser("inspect", help="Verify all hashes without exposing scores or extracting")
    inspect.add_argument("backup")
    inspect.add_argument("--expected-sha256")
    restore = commands.add_parser("restore", help="Extract a verified bundle to a new inactive recovery directory")
    restore.add_argument("backup")
    restore.add_argument("--output", required=True)
    restore.add_argument("--expected-sha256")
    restore.add_argument("--acknowledge-sensitive", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            result = create_backup(args.run_dir, args.output, acknowledge_sensitive=args.acknowledge_sensitive)
        elif args.command == "inspect":
            result = inspect_backup(args.backup, expected_sha256=args.expected_sha256)
        else:
            result = restore_backup(
                args.backup,
                args.output,
                acknowledge_sensitive=args.acknowledge_sensitive,
                expected_sha256=args.expected_sha256,
            )
    except (ValueError, TypeError, OSError, sqlite3.DatabaseError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    return 0
