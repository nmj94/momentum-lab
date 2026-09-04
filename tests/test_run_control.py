"""Deterministic ownership, crash, verification and score-free inspection checks."""

import json
import multiprocessing
import os
import sqlite3
import subprocess
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Timer

import numpy as np
import pandas as pd
import pytest

from momentum_lab import RunBusyError, RunStateError, cli, inspect_run, run_search, search
from momentum_lab import run_control as rc
from momentum_lab.run_control import RunSession


@pytest.fixture
def target(tmp_path):
    return tmp_path / "runs" / "example"


def _files(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _finish(session, *, outcome="completed"):
    session.target.mkdir(parents=True, exist_ok=True)
    (session.target / "summary.json").write_text('{"private_test_score":"NEVER_DISCLOSE"}', encoding="utf-8")
    session.complete(["summary.json"], outcome=outcome)


def _completed(target, *, outcome="completed"):
    with RunSession(target, "search") as session:
        session.start()
        _finish(session, outcome=outcome)
    return session


def test_untracked_and_missing_inspection_never_creates_or_changes_files(target, tmp_path):
    assert inspect_run(target)["status"] == "not_found"
    assert not list(tmp_path.iterdir())
    target.mkdir(parents=True)
    (target / "summary.json").write_text("legacy")
    before = _files(tmp_path)
    result = inspect_run(target, verify=True)
    assert result["status"] == "untracked"
    assert result["integrity"] == "unavailable"
    assert before == _files(tmp_path)


@pytest.mark.parametrize("outcome", ["completed", "no_results"])
def test_completed_receipt_verifies_without_disclosing_scores_or_writing(target, tmp_path, outcome):
    session = _completed(target, outcome=outcome)
    before = _files(tmp_path)
    result = inspect_run(target, verify=True)
    assert result["status"] == "completed"
    assert result["integrity"] == "verified"
    assert result["attempt"]["outcome"] == outcome
    assert result["attempt"]["attempt_id"] == session.attempt_id
    assert result["artifact_count"] == 1
    assert "NEVER_DISCLOSE" not in json.dumps(result)
    assert before == _files(tmp_path)


def test_running_inspection_uses_lock_not_pid_and_never_hashes_live_artifacts(target, monkeypatch):
    with RunSession(target, "search") as session:
        session.start()
        session.stage("research")
        with sqlite3.connect(session.path) as connection:
            connection.execute("UPDATE attempts SET pid=99999999")
        with monkeypatch.context() as patch:
            patch.setattr(rc, "_artifact", lambda *args: pytest.fail("Do not hash a live run"))
            result = inspect_run(target, verify=True)
        assert result["status"] == "running"
        assert result["lock"] == "busy"
        assert result["integrity"] == "unavailable"
        assert result["attempt"]["stage"] == "research"
        assert "delete" in result["recovery"]
        _finish(session)


def test_busy_preflight_and_rejected_preflight_do_not_add_attempts(target):
    _completed(target)
    with RunSession(target, "search"):
        with pytest.raises(RunBusyError), RunSession(target, "search"):
            pytest.fail("Competing invocation entered")
        assert inspect_run(target)["status"] == "busy"
    with pytest.raises(ValueError, match="reject"), RunSession(target, "search"):
        raise ValueError("reject before start")
    assert len(inspect_run(target)["history"]) == 1
    assert inspect_run(target, verify=True)["integrity"] == "verified"


def test_threads_contend_for_one_directory_but_other_directories_remain_independent(target):
    ready, release = Event(), Event()

    def owner():
        with RunSession(target, "search") as session:
            session.start()
            ready.set()
            assert release.wait(10)
            _finish(session)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(owner)
        try:
            assert ready.wait(10)
            with pytest.raises(RunBusyError), RunSession(target, "portfolio"):
                pytest.fail("Cross-workflow lock bypass")
            with RunSession(target.with_name("independent"), "portfolio"):
                pass
        finally:
            release.set()
        future.result(timeout=10)
    assert inspect_run(target)["status"] == "completed"


def test_completed_directory_cannot_change_workflow(target, tmp_path):
    _completed(target)
    before = _files(tmp_path)
    with pytest.raises(RunStateError, match="different workflow"), RunSession(target, "portfolio"):
        pytest.fail("Existing workflow was replaced")
    assert before == _files(tmp_path)


def _symlink(link, destination, *, directory=False):
    try:
        link.symlink_to(destination, target_is_directory=directory)
    except OSError as exc:
        pytest.skip(f"Symbolic links unavailable in this test environment: {exc}")


def test_directory_aliases_share_the_same_lock(target, tmp_path, monkeypatch):
    target.mkdir(parents=True)
    alias = tmp_path / "alias"
    _symlink(alias, target, directory=True)
    monkeypatch.chdir(tmp_path)
    with RunSession(target, "search"):
        for path in (alias, Path("runs") / "example", target / ".." / "example"):
            with pytest.raises(RunBusyError), RunSession(path, "search"):
                pytest.fail("Canonical directory alias bypassed ownership")


def test_case_aliases_follow_native_filesystem_identity_even_before_output_creation(target):
    alternate = target.with_name(target.name.upper())
    with RunSession(target, "search") as session:
        session.start()
        same_native_name = rc._locations(alternate)[1].exists()
        if same_native_name:
            with pytest.raises(RunBusyError), RunSession(alternate, "search"):
                pytest.fail("Case-insensitive alias bypassed ownership")
        else:
            with RunSession(alternate, "search"):
                pass
        _finish(session)
    if same_native_name:
        assert inspect_run(alternate, verify=True)["integrity"] == "verified"


@pytest.mark.parametrize("name", [".momentum-runs", ".MOMENTUM-RUNS", ".momentum-runs."])
def test_reserved_control_namespace_cannot_be_used_as_output(tmp_path, name):
    with pytest.raises(RunStateError), RunSession(tmp_path / name, "search"):
        pytest.fail("Reserved namespace used as a research output")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_failures_preserve_history_and_original_exception_without_score_payload(target, error):
    with pytest.raises(error), RunSession(target, "search") as session:
        session.start()
        raise error("NEVER_DISCLOSE")
    result = inspect_run(target, verify=True)
    assert result["status"] == ("failed" if error is RuntimeError else "interrupted")
    assert result["attempt"]["error_type"] == error.__name__
    assert result["attempt"]["finished_at"] is not None
    assert "NEVER_DISCLOSE" not in json.dumps(result)
    assert "original config" in result["recovery"]
    with RunSession(target, "search", mode="resume") as session:
        session.start()
        _finish(session)
    assert len(inspect_run(target)["history"]) == 2


def test_missing_completion_is_not_mislabelled_success(target):
    with pytest.raises(RunStateError, match="completion receipt"), RunSession(target, "search") as session:
        session.start()
    result = inspect_run(target)
    assert result["status"] == "failed"
    assert result["attempt"]["error_type"] == "MissingCompletion"


@contextmanager
def _child_owner(target):
    code = (
        "import sys; from momentum_lab.run_control import RunSession\n"
        "with RunSession(sys.argv[1], 'search') as session:\n"
        " session.start()\n"
        " print('READY', flush=True)\n"
        " sys.stdin.read()\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", code, str(target)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    timer = Timer(20, process.kill)
    timer.start()
    try:
        ready = process.stdout.readline().strip()
        if ready != "READY":
            process.kill()
            _, errors = process.communicate(timeout=10)
            pytest.fail(f"Child did not acquire ownership: {errors}")
        timer.cancel()
        timer.join(timeout=5)
        yield process
    finally:
        timer.cancel()
        timer.join(timeout=5)
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


def test_process_death_releases_kernel_lock_without_erasing_or_repairing_history(target, tmp_path):
    with _child_owner(target) as process:
        with pytest.raises(RunBusyError), RunSession(target, "search"):
            pytest.fail("Second process entered an owned directory")
        assert inspect_run(target)["status"] == "running"
        process.kill()
        process.wait(timeout=10)
    before = _files(tmp_path)
    interrupted = inspect_run(target, verify=True)
    assert interrupted["status"] == "interrupted"
    assert interrupted["attempt"]["status"] == "running"  # Persisted state is not rewritten by a reader.
    assert interrupted["attempt"]["finished_at"] is None
    assert before == _files(tmp_path)
    with RunSession(target, "search", mode="resume") as session:
        session.start()
        _finish(session)
    history = inspect_run(target)["history"]
    assert len(history) == 2
    assert history[1]["status"] == "interrupted"
    assert history[1]["finished_at"] is None
    assert history[1]["detected_at"] is not None
    assert history[1]["error_type"] == "UncleanExit"


def _fork_wait(ready, release, child_target):
    with RunSession(child_target, "search"):
        ready.set()
        release.wait(15)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork-specific descriptor ownership")
@pytest.mark.parametrize("child_exits_first", [False, True])
def test_forked_worker_neither_unlocks_parent_nor_retains_its_lock(target, child_exits_first):
    context = multiprocessing.get_context("fork")
    ready, release = context.Event(), context.Event()
    process = context.Process(target=_fork_wait, args=(ready, release, target.with_name("child")))
    try:
        with RunSession(target, "search"):
            process.start()
            assert ready.wait(10)
            if child_exits_first:
                release.set()
                process.join(timeout=10)
                assert process.exitcode == 0
            with pytest.raises(RunBusyError), RunSession(target, "search"):
                pytest.fail("Forked child unlocked the parent's descriptor")
        # If the child is still alive, it must not retain the parent's flock.
        with RunSession(target, "search"):
            pass
    finally:
        release.set()
        if process.pid is not None:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)


@pytest.mark.parametrize("change", ["edit", "missing", "symlink"])
def test_verification_detects_changed_missing_and_linked_outputs(target, tmp_path, change):
    _completed(target)
    artifact = target / "summary.json"
    if change == "edit":
        artifact.write_text("changed")
    elif change == "missing":
        artifact.unlink()
    else:
        artifact.unlink()
        external = tmp_path / "external.json"
        external.write_text("outside")
        _symlink(artifact, external)
    result = inspect_run(target, verify=True)
    assert result["status"] == "completed"  # Historical completion is separate from present file integrity.
    assert result["integrity"] == "mismatch"
    assert result["changed_artifacts"] == ["summary.json"]
    assert "restore verified originals" in result["recovery"]


def test_default_status_does_not_read_artifact_contents(target, monkeypatch):
    _completed(target)
    monkeypatch.setattr(rc, "_artifact", lambda *args: pytest.fail("Status must be metadata-only unless --verify"))
    assert inspect_run(target)["integrity"] == "not_checked"


@pytest.mark.parametrize("change", ["schema", "identity", "corrupt", "missing_lock"])
def test_corrupt_control_is_not_reset_or_used_to_force_a_lock(target, tmp_path, change):
    session = _completed(target)
    if change == "corrupt":
        session.path.write_bytes(b"corrupted")
    elif change == "missing_lock":
        session.lock.path.unlink()
        assert inspect_run(target)["status"] == "unknown"
    else:
        with sqlite3.connect(session.path) as connection:
            connection.execute("PRAGMA user_version=99" if change == "schema" else "UPDATE run_info SET target='other'")
    before = _files(tmp_path)
    with pytest.raises((RunStateError, sqlite3.DatabaseError)), RunSession(target, "search"):
        pytest.fail("Untrusted state was reset")
    assert before == _files(tmp_path)


@pytest.mark.parametrize("linked", ["root", "directory", "lock", "journal"])
def test_control_symlinks_fail_without_writing_through_them(target, tmp_path, linked):
    _, control = rc._locations(target)
    outside = tmp_path / "outside"
    outside.mkdir()
    if linked == "root":
        control.parent.parent.mkdir(parents=True)
        _symlink(control.parent, outside, directory=True)
    elif linked == "directory":
        control.parent.mkdir(parents=True)
        _symlink(control, outside, directory=True)
    else:
        control.mkdir(parents=True)
        source = outside / "file"
        source.write_text("untouched")
        _symlink(control / ("owner.lock" if linked == "lock" else "state.sqlite3"), source)
    before = _files(outside)
    with pytest.raises(RunStateError), RunSession(target, "search"):
        pytest.fail("Followed a control symlink")
    assert before == _files(outside)


@pytest.mark.parametrize("name", ["../private.json", "/absolute.json", "sub/file.csv", "x\\file.csv", "x\n.json", "."])
def test_untrusted_manifest_cannot_escape_the_output_directory(target, name):
    session = _completed(target)
    value = json.dumps([{"path": name, "bytes": 1, "sha256": "a" * 64}])
    with sqlite3.connect(session.path) as connection:
        connection.execute("UPDATE attempts SET artifacts_json=?", (value,))
    with pytest.raises(RunStateError, match="manifest"):
        inspect_run(target, verify=True)


@pytest.mark.parametrize("value", ["null", "[]", "{}", "x" * 65537, '[{"path":"x","bytes":true,"sha256":"a"}]'])
def test_malformed_manifests_fail_closed(target, value):
    session = _completed(target)
    with sqlite3.connect(session.path) as connection:
        connection.execute("UPDATE attempts SET artifacts_json=?", (value,))
    with pytest.raises((RunStateError, ValueError)):
        inspect_run(target, verify=True)


@pytest.mark.parametrize("options", [{"verify": 1}, {"limit": True}, {"limit": 0}, {"limit": 101}, {"limit": 1.5}])
def test_bad_inspection_options_fail_without_creating_files(target, tmp_path, options):
    with pytest.raises(RunStateError):
        inspect_run(target, **options)
    assert not list(tmp_path.iterdir())


def test_history_is_bounded_and_new_attempt_does_not_erase_previous_failures(target):
    for _ in range(3):
        with pytest.raises(ValueError), RunSession(target, "search") as session:
            session.start()
            raise ValueError("fail")
    _completed(target)
    result = inspect_run(target, limit=2)
    assert [row["sequence"] for row in result["history"]] == [4, 3]
    assert len(inspect_run(target, limit=100)["history"]) == 4


def test_status_write_failure_does_not_hide_the_original_error(target, monkeypatch):
    with monkeypatch.context() as patch, warnings.catch_warnings(), pytest.raises(RuntimeError, match="original"):
        warnings.simplefilter("error")
        with RunSession(target, "search") as session:
            session.start()
            patch.setattr(rc, "_connect", lambda *a, **k: (_ for _ in ()).throw(OSError("state unavailable")))
            raise RuntimeError("original")
    assert inspect_run(target)["status"] == "interrupted"


def test_reentering_a_session_cannot_release_its_existing_lock(target):
    with RunSession(target, "search") as session:
        session.start()
        with pytest.raises(RunStateError, match="single-use"), session:
            pytest.fail("Re-entered the owner")
        with pytest.raises(RunBusyError), RunSession(target, "search"):
            pytest.fail("Re-entry released the original lock")
        _finish(session)
    with pytest.raises(RunStateError, match="single-use"), session:
        pytest.fail("Reused a finalized session")


def test_failed_state_initialization_never_publishes_an_empty_journal(target, monkeypatch):
    replace = Path.replace

    def fail_state(path, destination):
        if Path(destination).name == "state.sqlite3":
            raise OSError("interrupted initialization")
        return replace(path, destination)

    with RunSession(target, "search") as session, monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_state)
        with pytest.raises(OSError, match="interrupted initialization"):
            session.start()
        assert not session.path.exists()
        assert not list(session.path.parent.glob("*.tmp"))
    _completed(target)
    assert inspect_run(target, verify=True)["integrity"] == "verified"


@pytest.mark.parametrize(
    "operation", ["stage_before_start", "complete_before_start", "start_twice", "invalid_stage", "missing_file"]
)
def test_invalid_lifecycle_transitions_cannot_publish_success(target, operation):
    with RunSession(target, "search") as session:
        if operation in {"stage_before_start", "complete_before_start"}:
            with pytest.raises(RunStateError):
                session.stage("research") if operation == "stage_before_start" else session.complete(["summary.json"])
            assert not session.path.exists()
            return
        session.start()
        with pytest.raises((RunStateError, FileNotFoundError)):
            if operation == "start_twice":
                session.start()
            elif operation == "invalid_stage":
                session.stage("not-a-stage")
            else:
                session.complete(["missing.json"])
        assert inspect_run(target)["status"] == "running"
        _finish(session)


def test_cli_status_and_history_are_read_only_and_return_actionable_exit_codes(target, tmp_path, capsys, monkeypatch):
    _completed(target)
    before = _files(tmp_path)
    monkeypatch.setattr(sys, "argv", ["momentum-lab", "runs", "status", str(target), "--verify"])
    assert cli.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["integrity"] == "verified"
    assert "history" not in result
    assert "NEVER_DISCLOSE" not in json.dumps(result)
    assert before == _files(tmp_path)
    assert rc.main(["history", str(target), "--limit", "1"]) == 0
    assert len(json.loads(capsys.readouterr().out)["history"]) == 1
    (target / "summary.json").write_text("changed")
    assert rc.main(["status", str(target), "--verify"]) == 1
    capsys.readouterr()
    assert rc.main(["status", str(target.with_name("absent"))]) == 1
    capsys.readouterr()
    with pytest.raises(SystemExit) as error:
        rc.main(["status", str(target), "--limit", "0"])
    assert error.value.code == 2


@pytest.fixture
def search_options(target, monkeypatch):
    index = pd.date_range("2020-01-01", periods=120, freq="B")
    time = np.arange(len(index))
    close = 100 * np.exp(0.001 * time + 0.06 * np.sin(time / 5))
    frame = pd.DataFrame({"close": close, "open": close * (1 + 0.03 * np.cos(time / 3)), "volume": 1000.0}, index=index)
    monkeypatch.setattr(search, "prepare_data", lambda *a, **k: ({name: frame[name] for name in frame}, frame))
    monkeypatch.setattr(search, "_quick_sample", lambda *a: [{"lookback": 5, "threshold": 0.001, "long_short": False}])
    return {
        "ticker": "FAKE",
        "strategies": ["tsmom"],
        "result_dir": str(target.parent),
        "run_id": target.name,
        "study_id": "owned-search",
        "robust": False,
        "bootstrap": False,
        "min_validation_bars": 10,
    }


def test_search_lock_covers_loading_and_blocks_rerun_or_resume_before_writes(target, search_options, monkeypatch):
    prepare = search.prepare_data
    ready, release = Event(), Event()

    def wait_for_data(*args, **kwargs):
        ready.set()
        assert release.wait(10)
        return prepare(*args, **kwargs)

    monkeypatch.setattr(search, "prepare_data", wait_for_data)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_search, **search_options)
        try:
            assert ready.wait(10)
            assert inspect_run(target)["attempt"]["stage"] == "loading_data"
            for resume in (False, True):
                with pytest.raises(RunBusyError):
                    run_search(**search_options, resume=resume)
            assert len(inspect_run(target)["history"]) == 1
        finally:
            release.set()
        future.result(timeout=15)
    assert inspect_run(target, verify=True)["integrity"] == "verified"


def test_search_failed_publication_can_resume_with_original_provenance(target, search_options, monkeypatch):
    writer = search._write_frame_atomic

    def fail_top(frame, path):
        if path.name == "top_results.csv":
            raise OSError("publication interrupted")
        return writer(frame, path)

    with monkeypatch.context() as patch:
        patch.setattr(search, "_write_frame_atomic", fail_top)
        with pytest.raises(OSError, match="publication interrupted"):
            run_search(**search_options)
    result = inspect_run(target)
    assert result["status"] == "failed"
    assert result["attempt"]["stage"] == "publishing"
    recovered = run_search(**search_options, resume=True)
    assert recovered["test_access"]["status"] == "sealed"
    assert inspect_run(target, verify=True)["integrity"] == "verified"
    assert len(inspect_run(target)["history"]) == 2


def test_empty_search_is_completed_no_results_not_a_crash(target, search_options, monkeypatch):
    monkeypatch.setattr(search, "_quick_sample", lambda *a: [])
    result = run_search(**search_options)
    assert result["best"] is None
    status = inspect_run(target, verify=True)
    assert status["status"] == "completed"
    assert status["attempt"]["outcome"] == "no_results"
    assert status["integrity"] == "verified"


def test_reveal_and_cached_replay_each_record_an_attempt_without_status_leaking_scores(
    target, search_options, monkeypatch
):
    run_search(**search_options)
    revealed = run_search(**search_options, resume=True, reveal_test=True)
    assert revealed["test_access"]["status"] == "first_recorded_reveal"
    with monkeypatch.context() as patch:
        patch.setattr(search, "_test_payload", lambda *args: pytest.fail("Cached test must not be recomputed"))
        replay = run_search(**search_options, resume=True, reveal_test=True)
    assert replay["test_access"]["status"] == "previously_revealed"
    result = inspect_run(target, verify=True)
    assert [row["mode"] for row in result["history"]] == ["reveal", "reveal", "new"]
    assert "test_metrics" not in json.dumps(result)
    assert result["integrity"] == "verified"
