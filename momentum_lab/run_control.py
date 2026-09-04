"""Local coordinator ownership and score-free, crash-observable run receipts.

OS locks protect cooperating processes using a local filesystem. The separate
SQLite journal records attempts, not research exposure or permission to reveal.
Never delete a lock file to recover a run: its inode must remain stable.
"""

import argparse
import errno
import hashlib
import json
import os
import re
import socket
import sqlite3
import stat
import tempfile
import warnings
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

from ._version import __version__

RUN_STATE_SCHEMA = 1
_OPEN_LOCKS = set()
_FORK_GUARD = Lock()
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
_STAGES = {"preparing", "loading_data", "research", "test_evaluation", "publishing"}
_FIELDS = (
    "sequence,attempt_id,workflow,mode,status,stage,started_at,finished_at,detected_at,"
    "pid,hostname,package_version,outcome,error_type"
)


class RunStateError(ValueError):
    """Run ownership or its local state cannot be safely established."""


class RunBusyError(RunStateError):
    """Another coordinator or verification currently owns this directory."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _locations(run_dir):
    target = Path(run_dir).resolve()
    if target == target.parent or target.name.rstrip(" .").casefold() == ".momentum-runs":
        raise RunStateError("A filesystem root or run-control directory cannot be a run output")
    # Let the SAME filesystem resolve case/Unicode aliases for both the output
    # name and its control directory. Hashing path text would incorrectly split
    # one physical directory into different locks on case-insensitive macOS.
    control = target.parent / ".momentum-runs" / target.name
    return target, control


def _regular(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RunStateError(f"Expected a regular, non-linked run-control file: {path}")
    return info


def _close_inherited_locks():
    # Closing the child's duplicate is essential: an inherited flock must not
    # outlive a killed coordinator in a forked pool worker. Do NOT LOCK_UN here;
    # fork duplicates share the parent's open-file description.
    for fd in tuple(_OPEN_LOCKS):
        try:
            os.close(fd)
        except OSError:
            pass
    _OPEN_LOCKS.clear()
    _FORK_GUARD.release()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_FORK_GUARD.acquire,
        after_in_parent=_FORK_GUARD.release,
        after_in_child=_close_inherited_locks,
    )


def _try_lock(fd):
    try:
        if os.name == "posix":
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        elif os.name == "nt":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            raise RunStateError("Run ownership requires POSIX or Windows file locking")
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
            return False
        raise
    return True


class _DirectoryLock:
    def __init__(self, run_dir):
        self.target, self.control = _locations(run_dir)
        self.path = self.control / "owner.lock"
        self.fd = None
        self.pid = os.getpid()

    def acquire(self, *, create):
        if self.control.parent.is_symlink() or self.control.is_symlink():
            raise RunStateError("Run-control directories must not be symbolic links")
        if create:
            self.control.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.control.mkdir(exist_ok=True, mode=0o700)
        if self.control.parent.is_symlink() or self.control.is_symlink():
            raise RunStateError("Run-control directories must not be symbolic links")
        if self.path.is_symlink():
            raise RunStateError("Run lock must not be a symbolic link")
        if create and not self.path.exists() and (self.control / "state.sqlite3").exists():
            raise RunStateError("Existing run history has no ownership file; stop all owners and use a new run_id")
        flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        # A concurrent fork cannot occur between opening a descriptor and
        # registering/closing it. The child callback can therefore close all
        # inherited ownership descriptors, including probes from other threads.
        with _FORK_GUARD:
            return self._open_and_lock(flags, create)

    def _open_and_lock(self, flags, create):
        fd = os.open(self.path, flags | (os.O_CREAT if create else 0), 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise RunStateError("Run lock must be a regular, non-linked file")
            os.set_inheritable(fd, False)
            if not _try_lock(fd):
                os.close(fd)
                return False
            self.fd = fd
            self.pid = os.getpid()
            _OPEN_LOCKS.add(fd)
            return True
        except BaseException:
            os.close(fd)
            raise

    def check(self):
        if self.fd is None or self.pid != os.getpid():
            raise RunStateError("This process does not own the run directory")
        current, owned = _regular(self.path), os.fstat(self.fd)
        if (current.st_dev, current.st_ino) != (owned.st_dev, owned.st_ino):
            raise RunStateError("Run lock identity changed; do not delete or replace active lock files")

    def close(self):
        with _FORK_GUARD:
            self._close_locked()

    def _close_locked(self):
        if self.fd is not None and self.pid == os.getpid():
            fd = self.fd
            self.fd = None
            _OPEN_LOCKS.discard(fd)
            try:
                if os.name == "posix":
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
                elif os.name == "nt":
                    import msvcrt

                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            finally:
                os.close(fd)
        self.fd = None


def _connect(path, target, *, writable=False):
    _regular(path)
    mode = "rw" if writable else "ro"
    connection = sqlite3.connect(path.as_uri() + f"?mode={mode}", uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version != RUN_STATE_SCHEMA:
            raise RunStateError("Unsupported run-state schema; preserve the journal and use compatible software")
        info = connection.execute("SELECT target FROM run_info WHERE id=1").fetchone()
        matches = info is not None and info["target"] == os.path.normcase(str(target))
        if info is not None and not matches:
            try:
                matches = _locations(info["target"])[1].samefile(path.parent)
            except (OSError, ValueError):
                matches = False
        if not matches:
            raise RunStateError("Run-state directory identity mismatch")
        connection.execute(f"SELECT {_FIELDS},artifacts_json FROM attempts LIMIT 0")
    except BaseException:
        connection.close()
        raise
    return connection


def _initialize(path, target):
    # Publish a fully initialized DB, never an empty/half-created state journal.
    fd, temporary = tempfile.mkstemp(prefix=".state-", suffix=".tmp", dir=path.parent)
    staged = Path(temporary)
    os.close(fd)
    try:
        with closing(sqlite3.connect(staged)) as connection, connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("PRAGMA user_version=1")
            connection.execute("CREATE TABLE run_info (id INTEGER PRIMARY KEY CHECK(id=1), target TEXT NOT NULL)")
            connection.execute("INSERT INTO run_info VALUES (1,?)", (os.path.normcase(str(target)),))
            connection.execute(
                "CREATE TABLE attempts (sequence INTEGER PRIMARY KEY AUTOINCREMENT,"
                "attempt_id TEXT NOT NULL UNIQUE,workflow TEXT NOT NULL,mode TEXT NOT NULL,status TEXT NOT NULL,"
                "stage TEXT NOT NULL,started_at TEXT NOT NULL,finished_at TEXT,detected_at TEXT,"
                "pid INTEGER NOT NULL,hostname TEXT NOT NULL,package_version TEXT NOT NULL,"
                "outcome TEXT,error_type TEXT,artifacts_json TEXT)"
            )
        with staged.open("r+b") as handle:
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise RunStateError("Run-state journal appeared during initialization")
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def _artifact(target, name):
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise RunStateError("Artifact names must be simple relative filenames")
    path = target / name
    before = _regular(path)
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise RunStateError("Artifact identity changed during verification")
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = _regular(path)
    attrs = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, attr) != getattr(after, attr) for attr in attrs):
        raise RunStateError("Artifact changed during verification")
    return {"path": name, "bytes": before.st_size, "sha256": digest.hexdigest()}


class RunSession:
    """Own one output directory through validation, calculation and publication.

    Call start after accepting the invocation, then complete only after exports.
    A preflight rejection without start leaves the previous attempt untouched.
    """

    def __init__(self, run_dir, workflow, *, mode="new"):
        if workflow not in {"search", "portfolio", "portfolio_study"}:
            raise RunStateError("Unknown run workflow")
        if mode not in {"new", "resume", "development", "reveal"}:
            raise RunStateError("Unknown run mode")
        self.lock = _DirectoryLock(run_dir)
        self.target = self.lock.target
        self.path = self.lock.control / "state.sqlite3"
        self.workflow, self.mode = workflow, mode
        self.attempt_id = None
        self.finished = False
        self._used = False

    def __enter__(self):
        if self._used:
            raise RunStateError("RunSession is single-use and cannot be re-entered")
        self._used = True
        try:
            if not self.lock.acquire(create=True):
                raise RunBusyError(f"Run directory is in use: {self.target}; wait for its owner to finish")
            if self.path.exists() or self.path.is_symlink():
                with closing(_connect(self.path, self.target)) as connection:
                    previous = connection.execute(
                        "SELECT workflow FROM attempts ORDER BY sequence DESC LIMIT 1"
                    ).fetchone()
                    if previous is not None and previous["workflow"] != self.workflow:
                        raise RunStateError("Run directory belongs to a different workflow; use a new run_id")
        except BaseException:
            self.lock.close()
            raise
        return self

    @contextmanager
    def _write(self):
        self.lock.check()
        with closing(_connect(self.path, self.target, writable=True)) as connection, connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            yield connection

    def start(self):
        self.lock.check()
        if self.attempt_id is not None:
            raise RunStateError("Run attempt has already started")
        if not self.path.exists():
            _initialize(self.path, self.target)
        attempt_id = uuid4().hex
        now = _now()
        with self._write() as connection:
            # The lock is free, so prior running attempts did not finalize.
            # Detection time is not falsely reported as their actual stop time.
            connection.execute(
                "UPDATE attempts SET status='interrupted',detected_at=?,error_type='UncleanExit' WHERE status='running'",
                (now,),
            )
            connection.execute(
                "INSERT INTO attempts (attempt_id,workflow,mode,status,stage,started_at,pid,hostname,package_version) "
                "VALUES (?,?,?,'running','preparing',?,?,?,?)",
                (attempt_id, self.workflow, self.mode, now, os.getpid(), socket.gethostname(), __version__),
            )
        self.attempt_id = attempt_id

    def stage(self, stage):
        if stage not in _STAGES or self.attempt_id is None or self.finished:
            raise RunStateError("Invalid run stage transition")
        with self._write() as connection:
            changed = connection.execute(
                "UPDATE attempts SET stage=? WHERE attempt_id=? AND status='running'", (stage, self.attempt_id)
            ).rowcount
            if changed != 1:
                raise RunStateError("Run attempt is not active")

    def complete(self, names, *, outcome="completed"):
        if self.attempt_id is None or self.finished or outcome not in {"completed", "no_results"}:
            raise RunStateError("Invalid run completion")
        if not isinstance(names, (list, tuple)) or not 1 <= len(names) <= 64 or len(set(names)) != len(names):
            raise RunStateError("A run receipt requires 1–64 distinct artifact filenames")
        self.lock.check()
        artifacts = [_artifact(self.target, name) for name in names]
        with self._write() as connection:
            changed = connection.execute(
                "UPDATE attempts SET status='completed',stage='publishing',finished_at=?,outcome=?,artifacts_json=? "
                "WHERE attempt_id=? AND status='running'",
                (_now(), outcome, json.dumps(artifacts, allow_nan=False), self.attempt_id),
            ).rowcount
            if changed != 1:
                raise RunStateError("Run attempt is not active")
        self.finished = True

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self.attempt_id is not None and not self.finished and self.lock.pid == os.getpid():
                status = "interrupted" if exc_type and not issubclass(exc_type, Exception) else "failed"
                try:
                    with self._write() as connection:
                        connection.execute(
                            "UPDATE attempts SET status=?,finished_at=?,error_type=? WHERE attempt_id=? AND status='running'",
                            (status, _now(), exc_type.__name__ if exc_type else "MissingCompletion", self.attempt_id),
                        )
                except Exception as state_error:
                    if exc_type is None:
                        raise
                    try:
                        warnings.warn(
                            f"Could not finalize run state ({type(state_error).__name__}); inspect before resuming",
                            RuntimeWarning,
                        )
                    except Warning:
                        pass  # Warning-as-error settings must not replace the original failure.
                if exc_type is None:
                    raise RunStateError("Run exited without a verified completion receipt")
        finally:
            self.lock.close()


def _manifest(value):
    if not isinstance(value, str) or len(value) > 65536:
        raise RunStateError("Invalid run artifact manifest")
    artifacts = json.loads(value)
    if not isinstance(artifacts, list) or not 1 <= len(artifacts) <= 64:
        raise RunStateError("Invalid run artifact manifest")
    names = set()
    for item in artifacts:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "bytes", "sha256"}
            or not isinstance(item["path"], str)
            or not _NAME.fullmatch(item["path"])
            or item["path"] in names
            or type(item["bytes"]) is not int
            or item["bytes"] < 0
            or not isinstance(item["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        ):
            raise RunStateError("Invalid run artifact manifest")
        names.add(item["path"])
    return artifacts


def _recovery(result):
    state = result["status"]
    if result["integrity"] == "mismatch":
        return "Artifacts differ from the completion receipt. Preserve history and restore verified originals; do not silently recompute a revealed test."
    if state in {"running", "busy"}:
        return "Wait for the owner to finish. Never delete or replace a lock file while a process may hold it."
    if state in {"unknown", "untracked", "not_found"}:
        return "Completion and ownership are not established. Inspect original artifacts/history; do not infer successful or previously unseen research."
    if state in {"failed", "interrupted"}:
        if result["attempt"]["workflow"] == "search":
            return "Resume only with the original config, data, environment and registry. If provenance is missing, restore the complete backup set or use a new run_id; retain exposure history."
        return "Use a new run_id with the original recipe and registry. Portfolio studies still require explicit reveal/reuse consent; a completed reveal may be replayed from its existing cache."
    return "Completion is recorded; use --verify to check published files. This does not certify investment results or unseen data."


def inspect_run(run_dir, *, verify=False, limit=20):
    """Read score-free state/history; optionally verify completed file snapshots.

    Never create, repair, unlock, reveal or resume anything. Lock possession,
    rather than PIDs or timestamps, determines whether an unfinished run is live.
    """
    if not isinstance(verify, bool) or type(limit) is not int or not 1 <= limit <= 100:
        raise RunStateError("verify must be boolean and limit an integer in [1, 100]")
    lock = _DirectoryLock(run_dir)
    path = lock.control / "state.sqlite3"
    lock_state = "available"
    try:
        try:
            if not lock.acquire(create=False):
                lock_state = "busy"
        except FileNotFoundError:
            lock_state = "missing"
        except PermissionError:
            lock_state = "unknown"
        result = {
            "schema_version": RUN_STATE_SCHEMA,
            "run_dir": str(lock.target),
            "lock": lock_state,
            "status": "untracked",
            "attempt": None,
            "history": [],
            "integrity": "unavailable" if verify else "not_checked",
        }
        if not path.exists():
            if path.is_symlink():
                raise RunStateError("Run-state journal must not be a symbolic link")
            if lock_state == "busy":
                result["status"] = "busy"
            elif not lock.target.exists():
                result["status"] = "not_found"
            result["recovery"] = _recovery(result)
            return result
        with closing(_connect(path, lock.target)) as connection:
            rows = connection.execute(
                f"SELECT {_FIELDS},artifacts_json FROM attempts ORDER BY sequence DESC LIMIT ?", (limit,)
            ).fetchall()
        for row in rows:
            public = dict(row)
            public.pop("artifacts_json")
            if public["status"] not in {"running", "completed", "failed", "interrupted"}:
                raise RunStateError("Invalid recorded run status")
            result["history"].append(public)
        if not rows:
            result["recovery"] = _recovery(result)
            return result
        result["attempt"] = result["history"][0]
        status = rows[0]["status"]
        result["status"] = (
            "running"
            if lock_state == "busy" and status == "running"
            else "busy"
            if lock_state == "busy"
            else "unknown"
            if lock_state in {"unknown", "missing"}
            else "interrupted"
            if status == "running"
            else status
        )
        if verify and result["status"] == "completed":
            artifacts = _manifest(rows[0]["artifacts_json"])
            failures = []
            for expected in artifacts:
                try:
                    actual = _artifact(lock.target, expected["path"])
                except (OSError, RunStateError):
                    failures.append(expected["path"])
                else:
                    if actual != expected:
                        failures.append(expected["path"])
            result["integrity"] = "mismatch" if failures else "verified"
            result["changed_artifacts"] = failures
            result["artifact_count"] = len(artifacts)
        elif verify:
            result["integrity"] = "unavailable"
        result["recovery"] = _recovery(result)
        return result
    finally:
        lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="momentum-lab runs", description="Read-only, score-free run status and history"
    )
    parser.add_argument("command", choices=["status", "history"])
    parser.add_argument("run_dir", help="Exact output directory, e.g. experiments/gld-dev")
    parser.add_argument(
        "--verify", action="store_true", help="Check the latest completed artifact hashes; never recalculate"
    )
    parser.add_argument("--limit", type=int, default=20, help="Maximum attempts to display (1–100)")
    args = parser.parse_args(argv)
    try:
        result = inspect_run(args.run_dir, verify=args.verify, limit=args.limit)
    except (RunStateError, OSError, sqlite3.DatabaseError, ValueError) as exc:
        parser.error(str(exc))
    if args.command == "status":
        result.pop("history")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 1 if result["status"] == "not_found" or result["integrity"] == "mismatch" else 0
