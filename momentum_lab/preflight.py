"""Read-only offline data preflight: metadata and dates, never prices or scores.

This is not investment evaluation, point-in-time certification or a registry
reservation. A report describes only the bytes read during this invocation;
research must still revalidate its inputs and obtain normal reveal consent.
"""

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import pandas as pd

from ._version import __version__
from .datasets import DatasetError, _read_bytes, _unique_object, load_dataset
from .portfolio import MAX_PORTFOLIO_CELLS, PortfolioError
from .universe import daily_date, load_membership

PREFLIGHT_SCHEMA_VERSION = 1
MAX_SESSION_BYTES = 2 * 1024 * 1024
MAX_SESSIONS = 100_000
SAMPLE_DATES = 10
NOTICE = (
    "Read-only structural checks, not strategy evaluation, a research reservation, permission to reveal, "
    "or certification of investment readiness. No prices, returns, volume values or strategy scores are exported. "
    "Dates and hashes may still be sensitive. Sources, licenses, calendars and adjustment bases are user declarations; "
    "matching dates do not verify synchronized closes, security identity, corporate actions or point-in-time availability. "
    "Input files can change after this report; research must validate them again."
)
_SESSION_FIELDS = {"schema_version", "calendar_id", "source", "license", "coverage_start", "coverage_end", "sessions"}
_CONVENTIONS = ("currency", "calendar", "annualization", "price_adjustment")


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _bounds(start, end):
    try:
        left = daily_date(start, "start") if start is not None else None
        right = daily_date(end, "end") if end is not None else None
    except PortfolioError as exc:
        raise DatasetError("Preflight bounds must be valid ISO YYYY-MM-DD dates") from exc
    if left is not None and right is not None and left > right:
        raise DatasetError("end must be on or after start")
    return left, right


def _session_calendar(path):
    if path is None:
        return None
    try:
        if Path(path).is_symlink():
            raise DatasetError("Session calendar must not be a symlink")
        raw = _read_bytes(path, MAX_SESSION_BYTES)
        obj = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object)
        if not isinstance(obj, dict) or set(obj) != _SESSION_FIELDS:
            raise DatasetError("Invalid session-calendar fields")
        if type(obj["schema_version"]) is not int or obj["schema_version"] != 1:
            raise DatasetError("Unsupported session-calendar schema")
        for key in ("calendar_id", "source", "license"):
            text = obj[key]
            if (
                not isinstance(text, str)
                or not 1 <= len(text) <= 2048
                or text != text.strip()
                or any(ord(char) < 32 or ord(char) == 127 for char in text)
            ):
                raise DatasetError("Session-calendar declarations must be non-empty, trimmed text")
            text.encode("utf-8")
        left = daily_date(obj["coverage_start"])
        right = daily_date(obj["coverage_end"])
        if left > right or not isinstance(obj["sessions"], list) or not 1 <= len(obj["sessions"]) <= MAX_SESSIONS:
            raise DatasetError("Invalid session-calendar coverage or session count")
        dates = [daily_date(value) for value in obj["sessions"]]
        if dates != sorted(set(dates)) or dates[0] < left or dates[-1] > right:
            raise DatasetError("Sessions must be sorted, unique and within declared coverage")
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError) as exc:
        # Do not echo file values: even a malformed calendar could contain prices.
        raise DatasetError("Invalid session calendar; check schema, dates, size and regular-file constraints") from exc
    return {
        "dates": dates,
        "left": left,
        "right": right,
        "provenance": {
            "schema_version": 1,
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "contract_sha256": _digest(obj),
            "coverage_start": str(left),
            "coverage_end": str(right),
            "session_count": len(dates),
            "verification": "user_declared_not_independently_verified",
        },
    }


def _issue(issues, code, message, *, asset=None, severity="error", dates=None):
    item = {"code": code, "severity": severity, "message": message}
    if asset is not None:
        item["asset"] = asset
    if dates is not None:
        item.update(count=len(dates), sample_dates=[str(value) for value in dates[:SAMPLE_DATES]])
    issues.append(item)


def _date_checks(index, source, left, right, calendar, issues, asset):
    if calendar is not None:
        if left < calendar["left"] or right > calendar["right"]:
            _issue(
                issues,
                "calendar_coverage",
                "Session calendar does not cover the entire requested interval.",
                asset=asset,
            )
            return
        expected = [day for day in calendar["dates"] if left <= day <= right]
        if source["calendar"] == "continuous" and len(expected) != max(0, (right - left).days + 1):
            _issue(
                issues,
                "calendar_declaration_conflict",
                "A continuous dataset requires every calendar day; the supplied session calendar disagrees.",
                asset=asset,
            )
    elif source["calendar"] == "continuous":
        if (right - left).days + 1 > MAX_SESSIONS:
            _issue(issues, "calendar_work_limit", "Continuous calendar exceeds the session work limit.", asset=asset)
            return
        expected = list(pd.date_range(left, right, freq="D").date)
    else:
        _issue(
            issues,
            "calendar_unverified",
            "Exchange-session completeness is unknown without an explicit session calendar; weekdays are not an exchange calendar.",
            asset=asset,
            severity="warning",
        )
        return
    actual = set(index.date)
    expected_set = set(expected)
    missing, unexpected = sorted(expected_set - actual), sorted(actual - expected_set)
    if missing:
        _issue(
            issues,
            "missing_sessions",
            "Declared sessions have no observations; no rows were filled.",
            asset=asset,
            dates=missing,
        )
    if unexpected:
        _issue(
            issues,
            "unexpected_sessions",
            "Observations fall outside the declared session list.",
            asset=asset,
            dates=unexpected,
        )


def _scan(entries, left, right, calendar, *, portfolio=False, lookback=None, universe=None):
    issues, assets, indices = [], {}, {}
    cells = 0
    for label, path in entries:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                frame, source = load_dataset(path, ticker=label)
        except (DatasetError, OSError, ValueError, TypeError):
            _issue(
                issues,
                "invalid_dataset",
                "Cannot verify snapshot. Use data inspect privately to diagnose its manifest, hash or rows.",
                asset=label,
            )
            assets[label or "dataset"] = {"integrity": "invalid"}
            continue
        asset = source["ticker"]
        source_left, source_right = frame.index[0].date(), frame.index[-1].date()
        selected_left, selected_right = left or source_left, right or source_right
        # Compare using date values, avoiding silently clipping requested bounds.
        mask = (frame.index.date >= selected_left) & (frame.index.date <= selected_right)
        index = frame.index[mask]
        metadata = {
            "integrity": "verified",
            "contract_sha256": source["contract_sha256"],
            "csv_sha256": source["sha256"],
            "rows": source["rows"],
            "first_date": source["first_date"],
            "last_date": source["last_date"],
            "selected_rows": len(index),
            "selected_first": str(index[0].date()) if len(index) else None,
            "selected_last": str(index[-1].date()) if len(index) else None,
            "has_volume": source["has_volume"],
            **{key: source[key] for key in _CONVENTIONS},
        }
        assets[asset] = metadata
        if selected_left < source_left:
            _issue(
                issues,
                "late_start",
                "Snapshot begins after the requested start; listing history is not verified.",
                asset=asset,
                severity="warning",
            )
        if selected_right > source_right:
            _issue(
                issues,
                "truncated_end",
                "Snapshot ends before the requested end; automatic extension is disabled.",
                asset=asset,
            )
        if index.empty:
            _issue(issues, "empty_range", "No observations fall within the requested interval.", asset=asset)
        cells += len(index)
        if cells > MAX_PORTFOLIO_CELLS:
            raise DatasetError("Preflight exceeds the asset-session work limit")
        indices[asset] = index
        _date_checks(index, source, selected_left, selected_right, calendar, issues, asset)
        if source["price_adjustment"] != "split_and_dividend_adjusted":
            _issue(
                issues,
                "adjustment_risk",
                "Declared OHLC may omit distributions or contain corporate-action jumps; no reconstruction is performed.",
                asset=asset,
                severity="warning",
            )

    if portfolio and indices:
        reference = min(indices)
        reference_index = indices[reference]
        for asset in sorted(indices):
            if asset != reference:
                for key in _CONVENTIONS:
                    if assets[asset][key] != assets[reference][key]:
                        _issue(
                            issues,
                            "convention_mismatch",
                            f"{key} differs from {reference}; no conversion or inference is performed.",
                            asset=asset,
                        )
                if not indices[asset].equals(reference_index):
                    differing = sorted(set(indices[asset].date) ^ set(reference_index.date))
                    _issue(
                        issues,
                        "session_mismatch",
                        f"Session dates differ from {reference}; no intersection or filling is performed.",
                        asset=asset,
                        dates=differing,
                    )
            if len(indices[asset]) < lookback + 2:
                _issue(
                    issues,
                    "insufficient_history",
                    "Need lookback + 2 observations for a signal and delayed execution.",
                    asset=asset,
                )
        seen_hashes = {}
        for asset in sorted(indices):
            digest = assets[asset]["csv_sha256"]
            if digest in seen_hashes:
                _issue(
                    issues,
                    "identical_price_files",
                    f"CSV bytes match {seen_hashes[digest]}; verify that these are distinct securities, not duplicated input files.",
                    asset=asset,
                    severity="warning",
                )
            else:
                seen_hashes[digest] = asset
        if universe is not None:
            if (
                len(indices) != len(entries)
                or any(not idx.equals(reference_index) for idx in indices.values())
                or reference_index.empty
            ):
                _issue(
                    issues,
                    "membership_not_checked",
                    "Membership requires valid, non-empty, exactly aligned asset sessions.",
                )
            else:
                try:
                    _, membership = load_membership(universe, reference_index, sorted(indices))
                except (PortfolioError, OSError, ValueError, TypeError):
                    _issue(
                        issues,
                        "invalid_membership",
                        "Membership dates, assets or declarations do not satisfy the portfolio contract.",
                    )
                else:
                    return (
                        assets,
                        issues,
                        {key: membership[key] for key in ("manifest_sha256", "canonical_sha256", "event_count")},
                    )
    elif portfolio and universe is not None:
        _issue(issues, "membership_not_checked", "Membership cannot be checked without valid asset sessions.")
    return assets, issues, None


def _report(assets, issues, membership, calendar, start, end, workflow, recipe=None):
    errors = sum(issue["severity"] == "error" for issue in issues)
    warning_count = len(issues) - errors
    result = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "package_version": __version__,
        "workflow": workflow,
        "status": "error" if errors else "warning" if warning_count else "passed",
        "error_count": errors,
        "warning_count": warning_count,
        "scope": {"start": start, "end": end, "sample_date_limit": SAMPLE_DATES},
        "assets": assets,
        "issues": issues,
        "membership": membership,
        "session_calendar": calendar["provenance"] if calendar else None,
        "checked_recipe_sha256": _digest(recipe) if recipe is not None else None,
        "notice": NOTICE,
    }
    result["report_sha256"] = _digest(result)
    return result


def preflight_dataset(manifest_path, *, start=None, end=None, sessions=None):
    """Verify a snapshot and diagnose date coverage without scores or writes.

    ``sessions`` is an optional user-declared schema-1 session-calendar JSON.
    Only ISO daily strings are accepted as range bounds. Invalid snapshots are
    reported as errors; invalid invocation options raise DatasetError.
    """
    left, right = _bounds(start, end)
    calendar = _session_calendar(sessions)
    assets, issues, membership = _scan([(None, manifest_path)], left, right, calendar)
    return _report(assets, issues, membership, calendar, start, end, "dataset")


def preflight_portfolio(config, *, sessions=None):
    """Check a fixed-rule PortfolioConfig without evaluator or registry access.

    This checks data compatibility and structural recipe validity, not strategy
    eligibility, execution feasibility, test access, or available output paths.
    Registered study-specific fields intentionally require the study workflow.
    """
    from .portfolio_research import _validate_config, load_portfolio_config

    recipe = load_portfolio_config(config)
    _validate_config(recipe)
    left, right = _bounds(recipe.start, recipe.end)
    calendar = _session_calendar(sessions)
    entries = sorted((symbol.upper(), path) for symbol, path in recipe.datasets.items())
    assets, issues, membership = _scan(
        entries, left, right, calendar, portfolio=True, lookback=recipe.lookback, universe=recipe.universe
    )
    checked_recipe = {
        key: value
        for key, value in recipe.to_dict().items()
        if key not in {"datasets", "universe", "result_dir", "run_id", "registry_path"}
    }
    return _report(assets, issues, membership, calendar, recipe.start, recipe.end, "portfolio", checked_recipe)


def write_preflight_report(report, output_dir):
    """Save metadata-only JSON/Markdown in a NEW directory; JSON commits last.

    An interrupted write may leave a partial directory; it is not overwritten
    on retry. A report digest binds contents, not authenticity or later inputs.
    """
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    if report.get("schema_version") != PREFLIGHT_SCHEMA_VERSION or _digest(body) != report.get("report_sha256"):
        raise DatasetError("Invalid preflight report digest or schema")
    lines = [
        "# Momentum Lab data preflight",
        "",
        report["notice"],
        "",
        f"Status: {report['status']}",
        "",
        f"Errors: {report['error_count']}; warnings: {report['warning_count']}",
        "",
        f"Report SHA-256: {report['report_sha256']}",
        "",
    ]
    for issue in report["issues"]:
        lines.append(f"- {issue['severity']}: {issue.get('asset', 'inputs')} / {issue['code']}: {issue['message']}")
        if "count" in issue:
            lines.append(f"  Count: {issue['count']}; first {SAMPLE_DATES} dates: {', '.join(issue['sample_dates'])}")
    if not report["issues"]:
        lines.append("No structural issues found in this invocation. This is not a certification of data quality.")
    output = Path(output_dir)
    try:
        output.mkdir(parents=True, exist_ok=False)
        with (output / "report.md").open("x", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
        with (output / "report.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    except OSError as exc:
        raise DatasetError(f"Cannot publish preflight in a new directory: {output}") from exc
    return output / "report.json"


def main(argv=None, *, portfolio=False):
    parser = argparse.ArgumentParser(
        prog="momentum-lab portfolio preflight" if portfolio else "momentum-lab data check", description=NOTICE
    )
    if portfolio:
        parser.add_argument("--config", required=True, help="Fixed-rule portfolio recipe; no run or registration")
    else:
        parser.add_argument("manifest")
        parser.add_argument("--start", help="Inclusive ISO session date")
        parser.add_argument("--end", help="Inclusive ISO session date")
    parser.add_argument("--sessions", help="User-declared session-calendar JSON; no calendar is downloaded")
    parser.add_argument("--output", help="Optional NEW report directory; never overwrites input files or prior reports")
    args = parser.parse_args(argv)
    try:
        if args.output and Path(args.output).exists():
            raise DatasetError("Preflight output directory already exists")
        report = (
            preflight_portfolio(args.config, sessions=args.sessions)
            if portfolio
            else preflight_dataset(args.manifest, start=args.start, end=args.end, sessions=args.sessions)
        )
        if args.output:
            write_preflight_report(report, args.output)
    except (DatasetError, PortfolioError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return {"passed": 0, "warning": 1, "error": 2}[report["status"]]
