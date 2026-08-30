"""Local research protocols and durable, cross-run observation records.

This is an accident-prevention/audit boundary, not encrypted data custody or a
tamper-proof service. A first *recorded* reveal never establishes virgin data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
from platformdirs import user_data_dir

REGISTRY_SCHEMA_VERSION = 1
AUDIT_WARNING = (
    "Local registry evidence only; history outside this registry is unknown. "
    "First recorded reveal is not proof of untouched out-of-sample data. "
    "Canonical ticker and overlapping inclusive daily dates are checked across data versions. "
    "This is not encryption, tamper-proof custody, or a correction for selection bias."
)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_TICKER = re.compile(r"[A-Za-z0-9._^=-]+\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class RegistryError(ValueError):
    """The audit contract cannot safely be satisfied."""


class TestReuseError(RegistryError):
    """Known prior observations require explicit reuse acknowledgement."""

    __test__ = False


def registry_path(path=None):
    """Resolve independently of result directories; evaluate the override at call time."""
    value = path if path is not None else os.environ.get("MOMENTUM_LAB_REGISTRY_PATH")
    if value is None:
        value = Path(user_data_dir("momentum-lab")) / "research-registry.sqlite3"
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise RegistryError("registry_path must be a non-empty filesystem path")
    return Path(value).expanduser().resolve()


def validate_study_options(study_id, reveal_test=False, allow_test_reuse=False, test_reuse_reason=None):
    if study_id is not None and (not isinstance(study_id, str) or not _ID.fullmatch(study_id)):
        raise RegistryError(
            "study_id must be 1-128 ASCII letters, digits, dots, underscores or hyphens; start alphanumeric"
        )
    if not isinstance(reveal_test, bool) or not isinstance(allow_test_reuse, bool):
        raise RegistryError("reveal_test and allow_test_reuse must be boolean")
    if reveal_test and study_id is None:
        raise RegistryError("reveal_test requires a registered study_id")
    if allow_test_reuse and not reveal_test:
        raise RegistryError("allow_test_reuse requires reveal_test")
    if allow_test_reuse:
        if not isinstance(test_reuse_reason, str) or not 1 <= len(test_reuse_reason.strip()) <= 2000:
            raise RegistryError("test_reuse_reason must contain 1-2000 characters when acknowledging reuse")
    elif test_reuse_reason is not None:
        raise RegistryError("test_reuse_reason requires allow_test_reuse")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ticker(value):
    if not isinstance(value, str) or not _TICKER.fullmatch(value.strip()):
        raise RegistryError("invalid ticker for observation registry")
    return value.strip().upper()


def _dates(start, end):
    try:
        for value in (start, end):
            if not isinstance(value, (str, date, datetime, pd.Timestamp)):
                raise TypeError("expected daily date labels")
            if isinstance(value, str) and not re.match(r"^\d{4}-\d{2}-\d{2}(?:$|[ T])", value):
                raise ValueError("expected ISO date labels, not relative dates")
        # Daily session labels, as in prepared Yahoo data, not intraday UTC instants.
        left, right = pd.Timestamp(start), pd.Timestamp(end)
        if pd.isna(left) or pd.isna(right):
            raise ValueError("missing date")
        left, right = left.date().isoformat(), right.date().isoformat()
    except (ValueError, TypeError, OverflowError) as exc:
        raise RegistryError("observation boundaries must be valid dates") from exc
    if left > right:
        raise RegistryError("observation start must not follow end")
    return left, right


def _snapshot(value):
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise RegistryError("data_snapshot must be a SHA-256 hex digest")
    return value


_SCHEMA = (
    "CREATE TABLE registry_info (id INTEGER PRIMARY KEY CHECK(id = 1), registry_id TEXT NOT NULL, created_at TEXT NOT NULL)",
    """CREATE TABLE studies (
        study_id TEXT PRIMARY KEY, ticker TEXT NOT NULL, test_start TEXT NOT NULL, test_end TEXT NOT NULL,
        protocol_json TEXT NOT NULL, protocol_sha256 TEXT NOT NULL, registered_at TEXT NOT NULL,
        selection_json TEXT, selection_sha256 TEXT, selected_at TEXT
    )""",
    """CREATE TABLE observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE, kind TEXT NOT NULL,
        ticker TEXT NOT NULL, start_date TEXT NOT NULL, end_date TEXT NOT NULL, data_snapshot TEXT NOT NULL,
        study_id TEXT REFERENCES studies(study_id), run_id TEXT NOT NULL, run_path TEXT NOT NULL,
        recorded_at TEXT NOT NULL, status TEXT NOT NULL, assessment TEXT NOT NULL,
        prior_overlap_count INTEGER NOT NULL, reason TEXT, dedupe_key TEXT UNIQUE,
        result_json TEXT, result_sha256 TEXT, completed_at TEXT, error TEXT,
        source_event_id TEXT REFERENCES observations(event_id)
    )""",
    "CREATE INDEX observation_overlap ON observations(ticker, start_date, end_date)",
    "CREATE INDEX study_observation ON observations(study_id, kind, status)",
)


class StudyRegistry:
    """SQLite-backed protocol locks and observation reservations; no deletion API.

    Reservations commit BEFORE computation and count even after interruption.
    Reader methods never return cached test scores. Registry identity is also
    pinned in run manifests, so a missing/replaced registry cannot be resumed.
    """

    def __init__(self, path=None, *, create=True):
        self.path = registry_path(path)
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        elif not self.path.is_file():
            raise RegistryError(f"registry not found: {self.path}; restore it rather than resetting access history")
        mode = "rwc" if create else "ro"
        try:
            with self._connect(mode) as connection:
                if create:
                    connection.execute("BEGIN IMMEDIATE")
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                if version == 0 and not tables and create:
                    for sql in _SCHEMA:
                        connection.execute(sql)
                    connection.execute("INSERT INTO registry_info VALUES (1, ?, ?)", (uuid4().hex, _now()))
                    connection.execute(f"PRAGMA user_version = {REGISTRY_SCHEMA_VERSION}")
                elif version != REGISTRY_SCHEMA_VERSION:
                    raise RegistryError(f"unsupported registry schema {version}; no automatic reset or migration")
                row = connection.execute("SELECT registry_id, created_at FROM registry_info WHERE id=1").fetchone()
                if row is None or not re.fullmatch(r"[0-9a-f]{32}", row["registry_id"]):
                    raise RegistryError("invalid registry identity; restore a verified registry backup")
                self.registry_id, self.created_at = row["registry_id"], row["created_at"]
        except sqlite3.DatabaseError as exc:
            raise RegistryError(f"cannot open observation registry {self.path}: {exc}") from exc

    @contextmanager
    def _connect(self, mode="ro"):
        connection = sqlite3.connect(f"{self.path.as_uri()}?mode={mode}", uri=True, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            if hasattr(self, "registry_id"):
                identity = connection.execute("SELECT registry_id FROM registry_info WHERE id=1").fetchone()
                if identity is None or identity[0] != self.registry_id:
                    raise RegistryError("registry identity changed; refusing to reset access history")
            if mode != "ro":
                connection.execute("PRAGMA synchronous = FULL")
            with connection:
                yield connection
        finally:
            connection.close()

    @contextmanager
    def _write(self):
        with self._connect("rw") as connection:
            connection.execute("BEGIN IMMEDIATE")
            # Re-check identity inside each mutation transaction, not only at construction.
            identity = connection.execute("SELECT registry_id FROM registry_info WHERE id=1").fetchone()
            if identity is None or identity[0] != self.registry_id:
                raise RegistryError("registry identity changed; refusing to reset access history")
            yield connection

    @staticmethod
    def _study(connection, study_id):
        row = connection.execute("SELECT * FROM studies WHERE study_id=?", (study_id,)).fetchone()
        if row is None:
            raise RegistryError(f"study {study_id!r} is not registered; run a sealed search first")
        if _hash(row["protocol_json"]) != row["protocol_sha256"]:
            raise RegistryError("registered protocol integrity check failed")
        if row["selection_json"] is not None and _hash(row["selection_json"]) != row["selection_sha256"]:
            raise RegistryError("frozen selection integrity check failed")
        return row

    @staticmethod
    def _overlaps(connection, ticker, start, end):
        # Never add data_snapshot, run_id, study_id or source version to this predicate.
        predicate = "ticker=? AND start_date<=? AND end_date>=?"
        args = (ticker, end, start)
        count = connection.execute(f"SELECT COUNT(*) FROM observations WHERE {predicate}", args).fetchone()[0]
        rows = connection.execute(
            f"SELECT event_id, kind, study_id, run_id, start_date, end_date, status, recorded_at "
            f"FROM observations WHERE {predicate} ORDER BY id DESC LIMIT 20",
            args,
        ).fetchall()
        return count, [dict(row) for row in rows]

    def _base(self, study_id=None):
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "mode": "registered" if study_id else "legacy",
            "study_id": study_id,
            "registry_id": self.registry_id,
            "registry_path": str(self.path),
            "registry_created_at": self.created_at,
            "history_outside_registry": "unknown",
            "warning": AUDIT_WARNING,
            "test_results_visible": False,
        }

    def unregistered_status(self):
        return {**self._base(), "status": "history_unknown"}

    def register(self, study_id, protocol):
        """Freeze data, strategy space and evaluation rules before candidate work."""
        validate_study_options(study_id)
        if study_id is None:
            raise RegistryError("registration requires study_id")
        protocol = json.loads(_canonical(protocol))
        protocol["ticker"] = _ticker(protocol["ticker"])
        _snapshot(protocol["data_snapshot"])
        periods = {name: _dates(*protocol["periods"][name]) for name in ("train", "val", "test")}
        if not periods["train"][1] < periods["val"][0] or not periods["val"][1] < periods["test"][0]:
            raise RegistryError("registered train, validation and test dates must be disjoint and ordered")
        text = _canonical(protocol)
        with self._write() as connection:
            exists = connection.execute("SELECT 1 FROM studies WHERE study_id=?", (study_id,)).fetchone()
            if exists:
                row = self._study(connection, study_id)
                if row["protocol_json"] != text:
                    previous = json.loads(row["protocol_json"])
                    changed = sorted(
                        key for key in set(previous) | set(protocol) if previous.get(key) != protocol.get(key)
                    )
                    raise RegistryError(
                        f"study protocol mismatch on {', '.join(changed)}; use a new study_id, not a reset"
                    )
            else:
                connection.execute(
                    "INSERT INTO studies (study_id,ticker,test_start,test_end,protocol_json,protocol_sha256,registered_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (study_id, protocol["ticker"], *periods["test"], text, _hash(text), _now()),
                )
        return self.status(study_id)

    def require_reveal_ready(self, study_id):
        """Require a completed sealed selection from an earlier invocation."""
        with self._connect() as connection:
            row = self._study(connection, study_id)
            if row["selection_json"] is None:
                raise RegistryError("study has no frozen selection; finish a sealed search before revealing")

    def bind_selection(self, study_id, selection):
        text = _canonical(selection)
        with self._write() as connection:
            row = self._study(connection, study_id)
            if row["selection_json"] is not None and row["selection_json"] != text:
                raise RegistryError("frozen study selection changed; revealing a different winner is not permitted")
            if row["selection_json"] is None:
                connection.execute(
                    "UPDATE studies SET selection_json=?,selection_sha256=?,selected_at=? WHERE study_id=?",
                    (text, _hash(text), _now(), study_id),
                )

    @staticmethod
    def _context(ticker, start, end, data_snapshot, run_id, run_path, study_id=None):
        left, right = _dates(start, end)
        if not isinstance(run_id, str) or not run_id:
            raise RegistryError("observation run_id must be a non-empty string")
        validate_study_options(study_id)
        return {
            "ticker": _ticker(ticker),
            "start_date": left,
            "end_date": right,
            "data_snapshot": _snapshot(data_snapshot),
            "run_id": run_id,
            "run_path": str(Path(run_path).resolve()),
            "study_id": study_id,
        }

    @staticmethod
    def _insert(
        connection,
        context,
        *,
        kind,
        status,
        assessment,
        prior_count=0,
        reason=None,
        dedupe_key=None,
        source_event_id=None,
    ):
        values = {
            **context,
            "event_id": uuid4().hex,
            "kind": kind,
            "status": status,
            "assessment": assessment,
            "prior_overlap_count": prior_count,
            "reason": reason,
            "dedupe_key": dedupe_key,
            "recorded_at": _now(),
            "source_event_id": source_event_id,
        }
        names = list(values)
        connection.execute(
            f"INSERT INTO observations ({','.join(names)}) VALUES ({','.join('?' for _ in names)})",
            tuple(values[key] for key in names),
        )
        return values

    def record_development(self, **kwargs):
        """Conservatively reserve training/validation observations before search work."""
        context = self._context(**kwargs)
        key = "development:" + _hash(_canonical(context))
        with self._write() as connection:
            if context["study_id"]:
                self._study(connection, context["study_id"])
            if not connection.execute("SELECT 1 FROM observations WHERE dedupe_key=?", (key,)).fetchone():
                self._insert(
                    connection,
                    context,
                    kind="development",
                    status="recorded",
                    assessment="development_observed",
                    dedupe_key=key,
                )

    def claim_test(self, *, allow_reuse=False, reason=None, **kwargs):
        """Atomically return a cached reveal or commit a possible exposure first."""
        context = self._context(**kwargs)
        study_id = context["study_id"]
        validate_study_options(study_id, bool(study_id), allow_reuse, reason)
        with self._write() as connection:
            if study_id:
                study = self._study(connection, study_id)
                protocol = json.loads(study["protocol_json"])
                if (
                    study["selection_json"] is None
                    or context["ticker"] != study["ticker"]
                    or context["data_snapshot"] != protocol["data_snapshot"]
                    or (context["start_date"], context["end_date"]) != (study["test_start"], study["test_end"])
                ):
                    raise RegistryError("test claim does not match the registered data and frozen selection")
                cached = connection.execute(
                    "SELECT * FROM observations WHERE study_id=? AND kind='registered_reveal' "
                    "AND status='completed' ORDER BY id LIMIT 1",
                    (study_id,),
                ).fetchone()
                if cached:
                    if cached["result_json"] is None or _hash(cached["result_json"]) != cached["result_sha256"]:
                        raise RegistryError("cached reveal integrity check failed; will not silently re-evaluate")
                    count, _ = self._overlaps(connection, context["ticker"], context["start_date"], context["end_date"])
                    replay = self._insert(
                        connection,
                        context,
                        kind="reveal_replay",
                        status="recorded",
                        assessment="previously_revealed",
                        prior_count=count,
                        source_event_id=cached["event_id"],
                        reason="Explicit replay of cached test results; no new evaluation",
                    )
                    return {
                        "cached": True,
                        "payload": json.loads(cached["result_json"]),
                        "access": {
                            **self._base(study_id),
                            "status": "previously_revealed",
                            "event_id": cached["event_id"],
                            "replay_event_id": replay["event_id"],
                            "recorded_at": cached["recorded_at"],
                            "original_assessment": cached["assessment"],
                            "prior_overlap_count": count,
                            "cached": True,
                            "test_results_visible": True,
                        },
                    }
            count, overlaps = self._overlaps(connection, context["ticker"], context["start_date"], context["end_date"])
            if study_id and count and not allow_reuse:
                raise TestReuseError(
                    f"test dates overlap {count} recorded observation(s), including possible interrupted reveals; "
                    "use allow_test_reuse / --allow-test-reuse with a test_reuse_reason / --test-reuse-reason "
                    "only to acknowledge reuse, never to claim fresh evidence"
                )
            assessment = "repeated_use" if count else "first_recorded_reveal" if study_id else "history_unknown"
            event = self._insert(
                connection,
                context,
                kind="registered_reveal" if study_id else "legacy_auto",
                status="reserved",
                assessment=assessment,
                prior_count=count,
                reason=reason,
            )
            access = {
                **self._base(study_id),
                "status": assessment,
                "event_id": event["event_id"],
                "recorded_at": event["recorded_at"],
                "prior_overlap_count": count,
                "prior_overlaps": overlaps,
                "reuse_reason": reason,
                "cached": False,
                "test_results_visible": False,
            }
        return {"cached": False, "payload": None, "access": access}

    def complete_test(self, event_id, payload):
        text = _canonical(payload)
        with self._write() as connection:
            changed = connection.execute(
                "UPDATE observations SET status='completed',result_json=?,result_sha256=?,completed_at=? "
                "WHERE event_id=? AND status='reserved' AND kind IN ('registered_reveal','legacy_auto')",
                (text, _hash(text), _now(), event_id),
            ).rowcount
            if changed != 1:
                raise RegistryError("test reservation is missing or already finalized")

    def fail_test(self, event_id, error):
        with self._write() as connection:
            connection.execute(
                "UPDATE observations SET status='failed',error=?,completed_at=? WHERE event_id=? AND status='reserved'",
                (str(error)[:2000], _now(), event_id),
            )

    def status(self, study_id):
        """Read protocol and exposure metadata, never test scores or cached payloads."""
        validate_study_options(study_id)
        with self._connect() as connection:
            row = self._study(connection, study_id)
            count, overlaps = self._overlaps(connection, row["ticker"], row["test_start"], row["test_end"])
            revealed = connection.execute(
                "SELECT event_id FROM observations WHERE study_id=? AND kind='registered_reveal' "
                "AND status='completed' ORDER BY id LIMIT 1",
                (study_id,),
            ).fetchone()
            return {
                **self._base(study_id),
                "status": "previously_revealed" if revealed else "known_prior_exposure" if count else "sealed",
                "ticker": row["ticker"],
                "test_start": row["test_start"],
                "test_end": row["test_end"],
                "protocol_sha256": row["protocol_sha256"],
                "registered_at": row["registered_at"],
                "selection_sha256": row["selection_sha256"],
                "selected_at": row["selected_at"],
                "event_id": revealed[0] if revealed else None,
                "prior_overlap_count": count,
                "prior_overlaps": overlaps,
            }

    def list_studies(self):
        with self._connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT study_id,ticker,test_start,test_end,registered_at,protocol_sha256,selection_sha256 "
                    "FROM studies ORDER BY registered_at,study_id"
                )
            ]

    def history(self, ticker=None, *, limit=100):
        """Bounded access metadata, including legacy observations and cached replays."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise RegistryError("history limit must be an integer in [1, 1000]")
        where, args = ("WHERE ticker=?", [_ticker(ticker)]) if ticker is not None else ("", [])
        with self._connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT event_id,source_event_id,kind,ticker,start_date,end_date,data_snapshot,study_id,run_id,"
                    "recorded_at,status,assessment,prior_overlap_count,reason,completed_at,error "
                    f"FROM observations {where} ORDER BY id DESC LIMIT ?",
                    (*args, limit),
                )
            ]

    def import_legacy(self, run_dir):
        """Record existing test evidence; never rewrite or certify historical artifacts."""
        run_dir = Path(run_dir)
        try:
            config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            if not isinstance(config, dict) or not isinstance(summary, dict):
                raise TypeError("expected JSON objects")
            if summary["run_id"] != config["run_id"]:
                raise ValueError("run IDs do not match")
            best = summary.get("best")
            if not isinstance(best, dict) or not isinstance(best.get("test_metrics"), dict) or not best["test_metrics"]:
                raise ValueError("no visible legacy test metrics to import")
            context = self._context(
                config["ticker"], *config["periods"]["test"], config["data_snapshot"], config["run_id"], run_dir
            )
            dev_start, dev_end = _dates(config["periods"]["train"][0], config["periods"]["val"][1])
            if dev_end >= context["start_date"]:
                raise ValueError("legacy development and test periods overlap")
            key = "legacy-import:" + _hash(_canonical({"config": config, "summary": summary}))
        except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
            raise RegistryError(f"cannot import legacy research evidence: {exc}") from exc
        with self._write() as connection:
            old = connection.execute("SELECT event_id FROM observations WHERE dedupe_key=?", (key,)).fetchone()
            if old:
                return {"event_id": old[0], "status": "history_unknown", "already_imported": True}
            self._insert(
                connection,
                {**context, "start_date": dev_start, "end_date": dev_end},
                kind="development",
                status="recorded",
                assessment="history_unknown",
                dedupe_key=key + ":dev",
            )
            count, _ = self._overlaps(connection, context["ticker"], context["start_date"], context["end_date"])
            event = self._insert(
                connection,
                context,
                kind="legacy_import",
                status="recorded",
                assessment="history_unknown",
                prior_count=count,
                reason="Imported historical artifacts; original access history is unknown",
                dedupe_key=key,
            )
            return {"event_id": event["event_id"], "status": "history_unknown", "already_imported": False}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="momentum-lab study", description="Inspect research protocols and access history."
    )
    parser.add_argument(
        "--registry", dest="registry_path", help="Shared registry path; defaults to the user data directory"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List registered studies without disclosing test scores")
    status = commands.add_parser("status", help="Inspect a study without disclosing test scores")
    status.add_argument("study_id")
    history = commands.add_parser("history", help="Inspect observation metadata, never cached test scores")
    history.add_argument("--ticker", help="Filter by canonical ticker")
    history.add_argument("--limit", type=int, default=100, help="Maximum recent events (1-1000)")
    legacy = commands.add_parser("import-legacy", help="Record existing run artifacts; their files remain unchanged")
    legacy.add_argument("run_dir")
    args = parser.parse_args(argv)
    try:
        registry = StudyRegistry(args.registry_path, create=args.command == "import-legacy")
        if args.command == "list":
            result = {"registry_id": registry.registry_id, "studies": registry.list_studies(), "warning": AUDIT_WARNING}
        elif args.command == "status":
            result = registry.status(args.study_id)
        elif args.command == "history":
            result = {
                "registry_id": registry.registry_id,
                "observations": registry.history(args.ticker, limit=args.limit),
                "warning": AUDIT_WARNING,
            }
        else:
            result = registry.import_legacy(args.run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (RegistryError, OSError, sqlite3.DatabaseError) as exc:
        parser.error(str(exc))
    return 0
