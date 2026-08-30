"""Atomic portfolio exposure groups in the existing shared observation registry.

The base schema/identity and single-asset history are preserved. Portfolio
studies have their own namespace; all asset events share the existing overlap
index, including inactive candidates, failed reveals and explicit replays.
"""

import json
import sqlite3
from uuid import uuid4

from .governance import (
    RegistryError,
    StudyRegistry,
    TestReuseError,
    _canonical,
    _hash,
    _now,
    _snapshot,
    validate_study_options,
)
from .portfolio import MAX_PORTFOLIO_ASSETS, PortfolioError, _symbols
from .universe import daily_date

PORTFOLIO_REGISTRY_SCHEMA = 1
MAX_CACHED_SUMMARY_BYTES = 4 * 1024 * 1024
_TABLES = {"portfolio_registry_info", "portfolio_studies", "portfolio_access_batches", "portfolio_batch_events"}
_SCHEMA = (
    "CREATE TABLE portfolio_registry_info (id INTEGER PRIMARY KEY CHECK(id=1), schema_version INTEGER NOT NULL)",
    """CREATE TABLE portfolio_studies (
        study_id TEXT PRIMARY KEY, protocol_json TEXT NOT NULL, protocol_sha256 TEXT NOT NULL,
        registered_at TEXT NOT NULL, selection_json TEXT, selection_sha256 TEXT, selected_at TEXT,
        development_json TEXT, development_sha256 TEXT,
        cache_batch_id TEXT REFERENCES portfolio_access_batches(batch_id)
    )""",
    """CREATE TABLE portfolio_access_batches (
        batch_id TEXT PRIMARY KEY, study_id TEXT NOT NULL REFERENCES portfolio_studies(study_id),
        kind TEXT NOT NULL, status TEXT NOT NULL, assessment TEXT NOT NULL, run_id TEXT NOT NULL,
        run_path TEXT NOT NULL, recorded_at TEXT NOT NULL, completed_at TEXT, reason TEXT,
        result_json TEXT, result_sha256 TEXT, error TEXT,
        source_batch_id TEXT REFERENCES portfolio_access_batches(batch_id)
    )""",
    """CREATE TABLE portfolio_batch_events (
        batch_id TEXT NOT NULL REFERENCES portfolio_access_batches(batch_id),
        event_id TEXT NOT NULL UNIQUE REFERENCES observations(event_id), PRIMARY KEY(batch_id,event_id)
    )""",
    "CREATE INDEX portfolio_reveal_cache ON portfolio_access_batches(study_id,kind,status)",
)


def _encode(value, name):
    if not isinstance(value, dict):
        raise RegistryError(f"{name} must be a JSON object")
    try:
        text = _canonical(value)
        if len(text.encode("utf-8")) > MAX_CACHED_SUMMARY_BYTES:
            raise RegistryError(f"{name} exceeds the 4 MiB cache limit")
    except (TypeError, ValueError, RecursionError) as exc:
        raise RegistryError(f"{name} must be bounded finite JSON: {exc}") from exc
    return text


def _protocol(value):
    value = json.loads(_encode(value, "Portfolio protocol"))
    if value.get("kind") != "fixed_rule_portfolio_v1":
        raise RegistryError("Unsupported portfolio protocol kind")
    assets = value.get("assets")
    if not isinstance(assets, dict) or not 2 <= len(assets) <= MAX_PORTFOLIO_ASSETS:
        raise RegistryError("Portfolio protocol requires 2-64 assets with snapshot hashes")
    try:
        tickers = _symbols(assets)
        value["assets"] = dict(sorted(zip(tickers, (_snapshot(item) for item in assets.values()))))
        periods = value.get("periods")
        if not isinstance(periods, dict) or set(periods) != {"development", "test"}:
            raise RegistryError("Portfolio periods must be development and test")
        for name, bounds in periods.items():
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise RegistryError("Portfolio periods require pairs of daily dates")
            dates = [daily_date(item, name) for item in bounds]
            if dates[0] > dates[1]:
                raise RegistryError("Portfolio period start must not follow end")
        if periods["development"][1] >= periods["test"][0]:
            raise RegistryError("Portfolio development and test must be disjoint and ordered")
    except PortfolioError as exc:
        raise RegistryError(str(exc)) from exc
    return value


class PortfolioStudyRegistry(StudyRegistry):
    """Fixed-rule portfolio protocols, all-asset claims and summary-only caches."""

    def __init__(self, path=None, *, create=True):
        super().__init__(path, create=create)
        try:
            with self._write() if create else self._connect() as connection:
                present = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                installed = present & _TABLES
                if not installed and create:
                    for sql in _SCHEMA:
                        connection.execute(sql)
                    connection.execute("INSERT INTO portfolio_registry_info VALUES (1,?)", (PORTFOLIO_REGISTRY_SCHEMA,))
                elif installed != _TABLES:
                    raise RegistryError(
                        "Portfolio registry extension missing/incomplete; restore it, do not reset history"
                    )
                self._extension(connection)
        except sqlite3.DatabaseError as exc:
            raise RegistryError(f"Cannot open portfolio registry: {exc}") from exc

    @staticmethod
    def _extension(connection):
        row = connection.execute("SELECT schema_version FROM portfolio_registry_info WHERE id=1").fetchone()
        if row is None or row[0] != PORTFOLIO_REGISTRY_SCHEMA:
            raise RegistryError("Unsupported portfolio registry schema; no automatic reset")

    @classmethod
    def _portfolio(cls, connection, study_id):
        cls._extension(connection)
        row = connection.execute("SELECT * FROM portfolio_studies WHERE study_id=?", (study_id,)).fetchone()
        if row is None:
            raise RegistryError(
                f"Portfolio study {study_id!r} is not registered; finish a sealed development run first"
            )
        for prefix in ("protocol", "selection", "development"):
            text, digest = row[f"{prefix}_json"], row[f"{prefix}_sha256"]
            if (text is None) != (digest is None) or (text is not None and _hash(text) != digest):
                raise RegistryError(f"Portfolio {prefix} integrity check failed")
        if (
            row["protocol_json"] is None
            or len({row[name] is None for name in ("selection_json", "development_json", "selected_at")}) != 1
        ):
            raise RegistryError("Incomplete frozen portfolio development record")
        _protocol(json.loads(row["protocol_json"]))
        return row

    def register(self, study_id, protocol):
        validate_study_options(study_id)
        if study_id is None:
            raise RegistryError("Portfolio registration requires study_id")
        text = _canonical(_protocol(protocol))
        with self._write() as connection:
            self._extension(connection)
            exists = connection.execute("SELECT 1 FROM portfolio_studies WHERE study_id=?", (study_id,)).fetchone()
            if exists:
                if self._portfolio(connection, study_id)["protocol_json"] != text:
                    raise RegistryError("Portfolio study protocol mismatch; use a new study_id, never reset history")
            else:
                connection.execute(
                    "INSERT INTO portfolio_studies(study_id,protocol_json,protocol_sha256,registered_at) VALUES (?,?,?,?)",
                    (study_id, text, _hash(text), _now()),
                )
        return self.status(study_id)

    def require_protocol(self, study_id, protocol):
        text = _canonical(_protocol(protocol))
        with self._connect() as connection:
            if self._portfolio(connection, study_id)["protocol_json"] != text:
                raise RegistryError(
                    "Portfolio study protocol mismatch; data, membership, recipe and software are frozen"
                )

    def require_reveal_ready(self, study_id):
        with self._connect() as connection:
            if self._portfolio(connection, study_id)["selection_json"] is None:
                raise RegistryError("Portfolio study has no frozen development; finish a sealed run before revealing")

    def complete_development(self, study_id, selection, payload):
        selected, development = _encode(selection, "Selection"), _encode(payload, "Development summary")
        with self._write() as connection:
            row = self._portfolio(connection, study_id)
            if row["selection_json"] is not None:
                if row["selection_json"] != selected:
                    raise RegistryError("Frozen portfolio selection changed")
                return  # Keep the first completed development payload and timestamp.
            connection.execute(
                "UPDATE portfolio_studies SET selection_json=?,selection_sha256=?,selected_at=?,"
                "development_json=?,development_sha256=? WHERE study_id=?",
                (selected, _hash(selected), _now(), development, _hash(development), study_id),
            )

    def development_payload(self, study_id):
        with self._connect() as connection:
            row = self._portfolio(connection, study_id)
            if row["development_json"] is None:
                raise RegistryError("Portfolio study has no frozen development")
            return json.loads(row["development_json"])

    def _contexts(self, row, period, run_id, run_path):
        protocol = json.loads(row["protocol_json"])
        return {
            ticker: self._context(ticker, *protocol["periods"][period], snapshot, run_id, run_path)
            for ticker, snapshot in protocol["assets"].items()
        }

    @staticmethod
    def _cached_batch(connection, row):
        """First COMPLETED result is pinned; a late older claim cannot replace it."""
        if row["cache_batch_id"] is None:
            if connection.execute(
                "SELECT 1 FROM portfolio_access_batches WHERE study_id=? AND kind='reveal' AND status='completed' LIMIT 1",
                (row["study_id"],),
            ).fetchone():
                raise RegistryError("Portfolio cache pointer is missing; no silent re-evaluation")
            return None
        batch = connection.execute(
            "SELECT * FROM portfolio_access_batches WHERE batch_id=?", (row["cache_batch_id"],)
        ).fetchone()
        if (
            batch is None
            or batch["study_id"] != row["study_id"]
            or batch["kind"] != "reveal"
            or batch["status"] != "completed"
        ):
            raise RegistryError("Portfolio cache pointer integrity check failed")
        return batch

    def record_portfolio_development(self, study_id, run_id, run_path):
        with self._write() as connection:
            row = self._portfolio(connection, study_id)
            for context in self._contexts(row, "development", run_id, run_path).values():
                key = "portfolio-development:" + _hash(_canonical({**context, "portfolio_study_id": study_id}))
                if not connection.execute("SELECT 1 FROM observations WHERE dedupe_key=?", (key,)).fetchone():
                    self._insert(
                        connection,
                        context,
                        kind="portfolio_development",
                        status="recorded",
                        assessment="development_observed",
                        dedupe_key=key,
                    )

    @staticmethod
    def _overlap_groups(connection, contexts):
        counts, overlaps = {}, {}
        for ticker, context in contexts.items():
            counts[ticker], overlaps[ticker] = StudyRegistry._overlaps(
                connection, ticker, context["start_date"], context["end_date"]
            )
        return counts, overlaps

    def status(self, study_id):
        validate_study_options(study_id)
        with self._connect() as connection:
            row = self._portfolio(connection, study_id)
            protocol = json.loads(row["protocol_json"])
            counts, overlaps = self._overlap_groups(connection, self._contexts(row, "test", "status", self.path.parent))
            revealed = self._cached_batch(connection, row)
            return {
                **self._base(study_id),
                "portfolio_registry_schema": PORTFOLIO_REGISTRY_SCHEMA,
                "status": "previously_revealed"
                if revealed
                else "known_prior_exposure"
                if sum(counts.values())
                else "sealed",
                "assets": sorted(protocol["assets"]),
                "periods": protocol["periods"],
                "protocol_sha256": row["protocol_sha256"],
                "registered_at": row["registered_at"],
                "selection_sha256": row["selection_sha256"],
                "selected_at": row["selected_at"],
                "batch_id": revealed["batch_id"] if revealed else None,
                "prior_overlap_count": sum(counts.values()),
                "prior_overlap_counts": counts,
                "prior_overlaps": overlaps,
            }

    def list_studies(self):
        with self._connect() as connection:
            self._extension(connection)
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT study_id,registered_at,protocol_sha256,selection_sha256,selected_at FROM portfolio_studies ORDER BY registered_at,study_id"
                )
            ]

    def _batch_events(self, connection, batch, expected_status):
        row = self._portfolio(connection, batch["study_id"])
        expected = self._contexts(row, "test", batch["run_id"], batch["run_path"])
        events = connection.execute(
            "SELECT o.* FROM observations o JOIN portfolio_batch_events p ON o.event_id=p.event_id WHERE p.batch_id=?",
            (batch["batch_id"],),
        ).fetchall()
        if len(events) != len(expected) or {event["ticker"] for event in events} != set(expected):
            raise RegistryError("Incomplete or inconsistent portfolio exposure group")
        for event in events:
            context = expected[event["ticker"]]
            if (
                any(event[key] != value for key, value in context.items())
                or event["kind"] != "portfolio_reveal"
                or event["status"] != expected_status
                or event["assessment"] != batch["assessment"]
                or event["reason"] != batch["reason"]
            ):
                raise RegistryError("Portfolio exposure group integrity check failed")
        return {event["ticker"]: event for event in events}

    def _insert_batch(self, connection, study_id, contexts, *, kind, assessment, counts, reason, source=None):
        batch_id, now = uuid4().hex, _now()
        first = next(iter(contexts.values()))
        status = "reserved" if kind == "reveal" else "recorded"
        connection.execute(
            "INSERT INTO portfolio_access_batches(batch_id,study_id,kind,status,assessment,run_id,run_path,recorded_at,reason,source_batch_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                batch_id,
                study_id,
                kind,
                status,
                assessment,
                first["run_id"],
                first["run_path"],
                now,
                reason,
                source["batch_id"] if source else None,
            ),
        )
        event_ids = {}
        for ticker, context in contexts.items():
            event = self._insert(
                connection,
                context,
                kind=f"portfolio_{kind}",
                status=status,
                assessment=assessment,
                prior_count=counts[ticker],
                reason=reason,
                source_event_id=source["events"][ticker]["event_id"] if source else None,
            )
            event_ids[ticker] = event["event_id"]
            connection.execute("INSERT INTO portfolio_batch_events VALUES (?,?)", (batch_id, event["event_id"]))
        return {"batch_id": batch_id, "event_ids": event_ids, "recorded_at": now}

    def claim_test(self, study_id, run_id, run_path, *, allow_reuse=False, reason=None):
        """Reserve every asset in one transaction BEFORE any test computation."""
        validate_study_options(study_id, True, allow_reuse, reason)
        with self._write() as connection:
            row = self._portfolio(connection, study_id)
            if row["selection_json"] is None:
                raise RegistryError("Portfolio study has no frozen development")
            contexts = self._contexts(row, "test", run_id, run_path)
            counts, overlaps = self._overlap_groups(connection, contexts)
            cached = self._cached_batch(connection, row)
            access = {
                **self._base(study_id),
                "prior_overlap_count": sum(counts.values()),
                "prior_overlap_counts": counts,
                "prior_overlaps": overlaps,
                "reuse_reason": reason,
            }
            if cached:
                if cached["result_json"] is None or _hash(cached["result_json"]) != cached["result_sha256"]:
                    raise RegistryError("Cached portfolio reveal integrity check failed; no silent re-evaluation")
                payload = json.loads(cached["result_json"])
                _encode(payload, "Cached portfolio summary")
                events = self._batch_events(connection, cached, "completed")
                replay = self._insert_batch(
                    connection,
                    study_id,
                    contexts,
                    kind="replay",
                    assessment="previously_revealed",
                    counts=counts,
                    reason="Explicit replay of cached test results; no new evaluation",
                    source={"batch_id": cached["batch_id"], "events": events},
                )
                return {
                    "cached": True,
                    "payload": payload,
                    "access": {
                        **access,
                        **replay,
                        "source_batch_id": cached["batch_id"],
                        "status": "previously_revealed",
                        "original_assessment": cached["assessment"],
                        "cached": True,
                        "test_results_visible": True,
                    },
                }
            if sum(counts.values()) and not allow_reuse:
                raise TestReuseError(
                    "Portfolio test dates overlap recorded observations, including possible interrupted reveals; "
                    "use --allow-test-reuse with --test-reuse-reason only to acknowledge reuse, never fresh evidence"
                )
            assessment = "repeated_use" if sum(counts.values()) else "first_recorded_reveal"
            group = self._insert_batch(
                connection, study_id, contexts, kind="reveal", assessment=assessment, counts=counts, reason=reason
            )
            return {
                "cached": False,
                "payload": None,
                "access": {
                    **access,
                    **group,
                    "status": assessment,
                    "cached": False,
                    "test_results_visible": False,
                },
            }

    def complete_test(self, batch_id, payload):
        text = _encode(payload, "Portfolio test summary")
        with self._write() as connection:
            batch = connection.execute(
                "SELECT * FROM portfolio_access_batches WHERE batch_id=?", (batch_id,)
            ).fetchone()
            if batch is None or batch["kind"] != "reveal" or batch["status"] != "reserved":
                raise RegistryError("Portfolio test reservation is missing or already finalized")
            self._batch_events(connection, batch, "reserved")
            self._cached_batch(connection, self._portfolio(connection, batch["study_id"]))
            now = _now()
            connection.execute(
                "UPDATE observations SET status='completed',completed_at=? WHERE event_id IN (SELECT event_id FROM portfolio_batch_events WHERE batch_id=?)",
                (now, batch_id),
            )
            connection.execute(
                "UPDATE portfolio_access_batches SET status='completed',completed_at=?,result_json=?,result_sha256=? WHERE batch_id=?",
                (now, text, _hash(text), batch_id),
            )
            connection.execute(
                "UPDATE portfolio_studies SET cache_batch_id=? WHERE study_id=? AND cache_batch_id IS NULL",
                (batch_id, batch["study_id"]),
            )

    def fail_test(self, batch_id, error):
        with self._write() as connection:
            now, message = _now(), str(error)[:2000]
            connection.execute(
                "UPDATE observations SET status='failed',error=?,completed_at=? WHERE status='reserved' AND event_id IN (SELECT event_id FROM portfolio_batch_events WHERE batch_id=?)",
                (message, now, batch_id),
            )
            connection.execute(
                "UPDATE portfolio_access_batches SET status='failed',error=?,completed_at=? WHERE batch_id=? AND status='reserved'",
                (message, now, batch_id),
            )
