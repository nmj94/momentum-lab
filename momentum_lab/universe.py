"""Strict, declared as-of membership histories; not verified point-in-time data.

All candidate assets still require complete positive synchronized daily prices.
A removal is a portfolio instruction, not a fabricated delisting settlement.
"""

import hashlib
import json
import re
from bisect import bisect_left
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .portfolio import MAX_PORTFOLIO_ASSETS, MAX_PORTFOLIO_CELLS, PortfolioError, _symbols

UNIVERSE_SCHEMA = 1
MAX_UNIVERSE_BYTES = 2 * 1024 * 1024
MAX_UNIVERSE_EVENTS = 10_000
UNIVERSE_NOTE = (
    "User-declared membership history, not independently verified point-in-time data. "
    "known_on means available before that session's close; retroactive changes are unsupported. "
    "The candidate superset can still introduce selection/survivorship bias. "
    "Removal is not delisting settlement: complete positive prices, including delayed exit fills, remain required."
)
_FIELDS = {
    "schema_version",
    "universe_id",
    "source",
    "license",
    "coverage_start",
    "coverage_end",
    "initial_known_on",
    "initial_members",
    "events",
}


def daily_date(value, name="date"):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise PortfolioError(f"{name} must be an ISO YYYY-MM-DD daily date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise PortfolioError(f"{name} must be a valid daily date") from exc


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PortfolioError(f"Duplicate membership field: {key}")
        result[key] = value
    return result


def _fields(value, expected, name):
    if not isinstance(value, dict) or set(value) != expected:
        raise PortfolioError(f"{name} requires exactly these fields: {', '.join(sorted(expected))}")


def _read_manifest(path):
    try:
        with Path(path).open("rb") as source:
            raw = source.read(MAX_UNIVERSE_BYTES + 1)
        if len(raw) > MAX_UNIVERSE_BYTES:
            raise PortfolioError("Membership manifest exceeds the 2 MiB limit")
        manifest = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique)
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise PortfolioError(f"Cannot read membership manifest: {exc}") from exc
    _fields(manifest, _FIELDS, "Membership manifest")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != UNIVERSE_SCHEMA:
        raise PortfolioError("Unsupported membership schema_version")
    for name in ("universe_id", "source", "license"):
        value = manifest[name]
        if (
            not isinstance(value, str)
            or not 1 <= len(value) <= 2048
            or value != value.strip()
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
        ):
            raise PortfolioError(f"{name} must be non-empty, trimmed, control-free text (at most 2048 characters)")
    left = daily_date(manifest["coverage_start"], "coverage_start")
    right = daily_date(manifest["coverage_end"], "coverage_end")
    known = daily_date(manifest["initial_known_on"], "initial_known_on")
    if not known <= left <= right:
        raise PortfolioError("Require initial_known_on <= coverage_start <= coverage_end")
    initial = manifest["initial_members"]
    if not isinstance(initial, list) or len(initial) > MAX_PORTFOLIO_ASSETS:
        raise PortfolioError("initial_members must be a list of at most 64 unique tickers")
    manifest["initial_members"] = sorted(_symbols(initial))
    events = manifest["events"]
    if not isinstance(events, list) or len(events) > MAX_UNIVERSE_EVENTS:
        raise PortfolioError("events must be a list of at most 10000 membership changes")
    normalized, seen = [], set()
    for event in events:
        _fields(event, {"ticker", "known_on", "effective_on", "action"}, "Membership event")
        ticker = _symbols([event["ticker"]])[0]
        announcement = daily_date(event["known_on"], "known_on")
        effective = daily_date(event["effective_on"], "effective_on")
        if not left < effective <= right:
            raise PortfolioError("Event effective_on must be after coverage_start and within coverage_end")
        if announcement > effective:
            raise PortfolioError("Retroactive membership changes are unsupported: known_on must be <= effective_on")
        if event["action"] not in ("add", "remove"):
            raise PortfolioError("Membership action must be add or remove")
        if (ticker, effective) in seen:
            raise PortfolioError("Conflicting/duplicate membership events for an asset on one effective date")
        seen.add((ticker, effective))
        normalized.append({**event, "ticker": ticker})
    normalized.sort(key=lambda event: (event["effective_on"], event["ticker"]))
    active = set(manifest["initial_members"])
    for event in normalized:
        ticker = event["ticker"]
        adding = event["action"] == "add"
        if adding == (ticker in active):
            raise PortfolioError(f"Redundant membership {event['action']} for {ticker}")
        if adding:
            active.add(ticker)
        else:
            active.remove(ticker)
    manifest["events"] = normalized
    return manifest, hashlib.sha256(raw).hexdigest()


def load_membership(path, index, assets):
    """Return a boolean eligibility frame and path-independent provenance.

    Membership changes on the first observed session on/after effective_on.
    Events before a sliced price range establish its opening membership state.
    """
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.empty
        or index.hasnans
        or index.tz is not None
        or not index.is_unique
        or not index.is_monotonic_increasing
        or not index.equals(index.normalize())
    ):
        raise PortfolioError("Membership requires sorted unique timezone-free daily session dates")
    symbols = _symbols(list(assets))
    if not 1 <= len(symbols) <= MAX_PORTFOLIO_ASSETS or len(index) * len(symbols) > MAX_PORTFOLIO_CELLS:
        raise PortfolioError("Membership exceeds the asset/session work limit")
    manifest, raw_sha = _read_manifest(path)
    dates = list(index.date)
    if daily_date(manifest["coverage_start"]) > dates[0] or daily_date(manifest["coverage_end"]) < dates[-1]:
        raise PortfolioError("Membership coverage must include every evaluated session")
    referenced = set(manifest["initial_members"]) | {event["ticker"] for event in manifest["events"]}
    if not referenced.issubset(symbols):
        raise PortfolioError("Every named membership asset requires a supplied offline dataset")
    matrix = np.zeros((len(index), len(symbols)), dtype=bool)
    grouped = {ticker: [] for ticker in symbols}
    for event in manifest["events"]:
        grouped[event["ticker"]].append(event)
    for column, ticker in enumerate(symbols):
        state = ticker in manifest["initial_members"]
        start = 0
        for event in grouped[ticker]:
            stop = bisect_left(dates, daily_date(event["effective_on"]))
            matrix[start:stop, column] = state
            start, state = stop, event["action"] == "add"
        matrix[start:, column] = state
    frame = pd.DataFrame(matrix, index=index.copy(), columns=symbols)
    frame.index.name = "date"
    try:
        canonical = json.dumps(
            manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except UnicodeError as exc:
        raise PortfolioError("Membership text must contain valid UTF-8 Unicode characters") from exc
    provenance = {
        **{
            key: manifest[key]
            for key in (
                "schema_version",
                "universe_id",
                "source",
                "license",
                "coverage_start",
                "coverage_end",
                "initial_known_on",
            )
        },
        "manifest_sha256": raw_sha,
        "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        "event_count": len(manifest["events"]),
        "note": UNIVERSE_NOTE,
    }
    return frame, provenance
