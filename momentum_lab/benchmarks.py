"""Offline, frozen accounting regressions; not investment-performance evidence.

The bundled synthetic suite is deliberately public and never participates in
search selection. Its complete ledgers, not just rounded scores, are compared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import tracemalloc
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from statistics import median
from uuid import uuid4

import numpy as np
import pandas as pd

from ._version import __version__
from .backtest import backtest, evaluate
from .data import _validate_ohlcv
from .indicators import IndicatorDAG
from .search import ENGINE_SCHEMA_VERSION, _environment_manifest, _source_fingerprint, _write_text_atomic
from .strategies import get_strategy

SNAPSHOT_SCHEMA_VERSION = 1
RTOL = 1e-9
ATOL = 1e-11
MEASUREMENT = "median wall seconds and maximum tracemalloc peak bytes; includes tracing overhead; not RSS"
DISCLAIMER = (
    "Synthetic software-regression fixtures, not historical market observations or out-of-sample investment evidence. "
    "A pass means compatibility with a reviewed reference, not profitability or proof of correctness."
)
LEDGER_FIELDS = (
    "targets",
    "returns",
    "equity",
    "positions",
    "trades",
    "requested_trades",
    "transaction_costs",
    "participation",
    "capacity_constrained",
    "borrow_blocked",
)
BOOLEAN_FIELDS = {"capacity_constrained", "borrow_blocked"}
METRIC_FIELDS = (
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "cagr",
    "total_return",
    "volatility",
    "win_rate",
    "profit_factor",
    "skew",
    "kurtosis",
    "final_equity",
    "turnover",
    "requested_turnover",
    "transaction_cost_drag",
    "trade_count",
    "capacity_constrained_bars",
    "borrow_blocked_bars",
)


def _json_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assets():
    return files("momentum_lab").joinpath("benchmark_data")


def load_benchmark_suite() -> dict:
    """Read the packaged, versioned contract. No provider or cache is accessed."""
    suite = json.loads(_assets().joinpath("suite_v1.json").read_text(encoding="utf-8"))
    if suite.get("schema_version") != 1 or suite.get("data_kind") != "synthetic":
        raise ValueError("Unsupported benchmark suite contract")
    return suite


def _load_frame(dataset):
    raw = _assets().joinpath(dataset["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != dataset["sha256"]:
        raise ValueError(f"Frozen dataset hash mismatch: {dataset['id']}")
    frame = pd.DataFrame(json.loads(raw), columns=["date", "open", "high", "low", "close", "volume"])
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    frame = frame.set_index("date").astype(float)
    if len(frame) != dataset["bars"]:
        raise ValueError(f"Frozen dataset bar count mismatch: {dataset['id']}")
    _validate_ohlcv(frame, dataset["id"])
    if not np.isfinite(frame["volume"]).all() or (frame["volume"] < 0).any():
        raise ValueError(f"Invalid frozen volume: {dataset['id']}")
    return frame


def _execute_case(suite, dataset, strategy, frame):
    data = {column: frame[column] for column in frame}
    data["annualization"] = dataset["annualization"]
    data["_indicator_dag"] = IndicatorDAG(data, max_entries=suite["indicator_cache_size"])
    name = strategy["name"]
    if name in {"cash", "buy_and_hold"}:
        targets = pd.Series(0.0 if name == "cash" else 1.0, index=frame.index)
    else:
        targets = get_strategy(name).run(data, **strategy["params"])
    kwargs = {**suite["backtest"], **dataset["backtest"], "annualization": dataset["annualization"]}
    availability = pd.Series(True, index=frame.index)
    for start, stop in dataset["borrow_unavailable_bars"]:
        availability.iloc[start:stop] = False
    bt = backtest(targets, frame["close"], volume=frame["volume"], borrow_available=availability, **kwargs)
    metrics = evaluate(bt["returns"], risk_free_rate=suite["risk_free_rate"], annualization=dataset["annualization"])
    metrics.update(
        final_equity=bt["equity"].iloc[-1],
        turnover=bt["trades"].sum(),
        requested_turnover=bt["requested_trades"].sum(),
        transaction_cost_drag=bt["transaction_costs"].sum(),
        trade_count=(bt["trades"] > 1e-12).sum(),
        capacity_constrained_bars=bt["capacity_constrained"].sum(),
        borrow_blocked_bars=bt["borrow_blocked"].sum(),
    )
    bt["targets"] = targets
    return {
        "metrics": {name: float(metrics[name]) for name in METRIC_FIELDS},
        "ledger": {name: bt[name].tolist() for name in LEDGER_FIELDS},
    }


def run_benchmarks(*, repeats: int = 1, measure_resources: bool = True) -> dict:
    """Run all fixed cases without downloading data, searching, or selecting.

    Resource profiling is serial and requires an inactive process-global
    tracemalloc tracer. Disable it when embedding in an already-profiled app.
    Repeats use fresh caches and must produce identical numerical results.
    """
    if type(repeats) is not int or not 1 <= repeats <= 100:
        raise ValueError("repeats must be an integer between 1 and 100")
    if type(measure_resources) is not bool:
        raise ValueError("measure_resources must be boolean")
    if measure_resources and tracemalloc.is_tracing():
        raise ValueError("Disable measure_resources when tracemalloc is already active")
    suite = load_benchmark_suite()
    results, performance = {}, {}
    for dataset in suite["datasets"]:
        frame = _load_frame(dataset)
        for strategy in suite["strategies"]:
            case_id = f"{dataset['id']}/{strategy['id']}"
            seconds, peaks = [], []
            for _ in range(repeats):
                if measure_resources:
                    tracemalloc.start()
                started = time.perf_counter()
                try:
                    result = _execute_case(suite, dataset, strategy, frame)
                    seconds.append(time.perf_counter() - started)
                    if measure_resources:
                        peaks.append(tracemalloc.get_traced_memory()[1])
                finally:
                    if measure_resources:
                        tracemalloc.stop()
                if case_id in results and results[case_id] != result:
                    raise ValueError(f"Non-deterministic repeated benchmark: {case_id}")
                results[case_id] = result
            if measure_resources:
                performance[case_id] = {"seconds": median(seconds), "peak_traced_bytes": max(peaks)}
    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "suite": suite,
        "suite_sha256": _json_hash(suite),
        "provenance": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "package_version": __version__,
            "engine_schema_version": ENGINE_SCHEMA_VERSION,
            "source_sha256": _source_fingerprint(),
            "environment": _environment_manifest(),
        },
        "measurement": MEASUREMENT,
        "repeats": repeats,
        "results": results,
        "performance": performance,
    }
    _validate_snapshot(snapshot)
    return snapshot


def load_benchmark_reference() -> dict:
    """Load reviewed references shipped inside both source and wheel installs."""
    reference = json.loads(_assets().joinpath("reference_v2.json").read_text(encoding="utf-8"))
    results = {}
    for dataset in reference["suite"]["datasets"]:
        fragment = json.loads(_assets().joinpath(f"expected_{dataset['id']}_v2.json").read_text(encoding="utf-8"))
        if set(results) & set(fragment):
            raise ValueError("Duplicate cases in packaged benchmark reference")
        results.update(fragment)
    reference["results"] = results
    _validate_snapshot(reference)
    return reference


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _validate_snapshot(snapshot):
    """Reject corrupt, incomplete and non-finite snapshots before comparison."""
    if type(snapshot["schema_version"]) is not int or snapshot["schema_version"] != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("Unsupported benchmark snapshot schema")
    suite = snapshot["suite"]
    if not isinstance(suite, dict) or snapshot["suite_sha256"] != _json_hash(suite):
        raise ValueError("Benchmark contract hash mismatch")
    provenance = snapshot["provenance"]
    if (
        not isinstance(provenance, dict)
        or not all(isinstance(provenance.get(name), str) for name in ("package_version", "source_sha256"))
        or not isinstance(provenance.get("environment"), dict)
    ):
        raise ValueError("Missing benchmark provenance")
    datasets = suite["datasets"]
    strategies = suite["strategies"]
    if not datasets or not strategies:
        raise ValueError("Benchmark contract cannot be empty")
    if len({row["id"] for row in datasets}) != len(datasets) or len({row["id"] for row in strategies}) != len(
        strategies
    ):
        raise ValueError("Duplicate benchmark dataset or strategy IDs")
    expected = {f"{d['id']}/{s['id']}": d["bars"] for d in datasets for s in strategies}
    if not isinstance(snapshot["results"], dict) or set(snapshot["results"]) != set(expected):
        raise ValueError("Missing or unexpected benchmark cases")
    for case_id, result in snapshot["results"].items():
        bars = expected[case_id]
        if type(bars) is not int or bars < 2:
            raise ValueError(f"Invalid bar count: {case_id}")
        if set(result) != {"metrics", "ledger"} or set(result["metrics"]) != set(METRIC_FIELDS):
            raise ValueError(f"Missing or unexpected metrics: {case_id}")
        if not all(_finite_number(value) for value in result["metrics"].values()):
            raise ValueError(f"Non-finite or invalid metric: {case_id}")
        if set(result["ledger"]) != set(LEDGER_FIELDS):
            raise ValueError(f"Missing or unexpected ledger fields: {case_id}")
        for name, values in result["ledger"].items():
            if not isinstance(values, list) or len(values) != bars:
                raise ValueError(f"Ledger length mismatch: {case_id}/{name}")
            valid = (
                all(type(value) is bool for value in values)
                if name in BOOLEAN_FIELDS
                else all(_finite_number(value) for value in values)
            )
            if not valid:
                raise ValueError(f"Non-finite or invalid ledger: {case_id}/{name}")
    performance = snapshot["performance"]
    if not isinstance(performance, dict) or (performance and set(performance) != set(expected)):
        raise ValueError("Incomplete resource measurements")
    if snapshot["measurement"] != MEASUREMENT or type(snapshot["repeats"]) is not int or snapshot["repeats"] < 1:
        raise ValueError("Invalid resource measurement contract")
    for values in performance.values():
        if set(values) != {"seconds", "peak_traced_bytes"} or not all(
            _finite_number(value) and value > 0 for value in values.values()
        ):
            raise ValueError("Invalid resource measurements")


def compare_benchmarks(current: dict, reference: dict, *, max_slowdown=None, max_memory_growth=None) -> dict:
    """Compare compatible inputs symmetrically; higher returns also count as change.

    Numerical tolerances are fixed and do not adapt to the observed results.
    Runtime/memory limits are optional ratios and fail closed without matching
    measurements. Machine-dependent resource changes do not gate by default.
    """
    limits = {"seconds": max_slowdown, "peak_traced_bytes": max_memory_growth}
    for limit in limits.values():
        if limit is not None and (not _finite_number(limit) or limit <= 0):
            raise ValueError("Resource ratio limits must be positive and finite")
    comparison = {
        "status": "passed",
        "rtol": RTOL,
        "atol": ATOL,
        "resource_limits": limits,
        "issues": [],
        "cases": {},
    }
    try:
        _validate_snapshot(current)
        _validate_snapshot(reference)
        comparison["reference"] = reference["provenance"]
        comparison["environment_changed"] = (
            current["provenance"]["environment"] != reference["provenance"]["environment"]
        )
        if current["suite_sha256"] != reference["suite_sha256"]:
            raise ValueError("Different frozen data, parameters or execution assumptions; snapshots are incomparable")
        if any(value is not None for value in limits.values()) and (
            not current["performance"] or not reference["performance"]
        ):
            raise ValueError("Resource gates require measurements in both snapshots; the bundled reference has none")
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
        comparison.update(status="incompatible", issues=[str(exc)])
        return comparison

    for case_id, result in current["results"].items():
        expected = reference["results"][case_id]
        differences = []
        metric_deltas = {}
        for name, value in result["metrics"].items():
            old = expected["metrics"][name]
            metric_deltas[name] = value - old
            if not math.isclose(value, old, rel_tol=RTOL, abs_tol=ATOL):
                differences.append({"field": f"metrics.{name}", "expected": old, "actual": value})
        for name, values in result["ledger"].items():
            actual = np.asarray(values)
            old = np.asarray(expected["ledger"][name])
            equal = actual == old if name in BOOLEAN_FIELDS else np.isclose(actual, old, rtol=RTOL, atol=ATOL)
            if not equal.all():
                first = int(np.flatnonzero(~equal)[0])
                differences.append(
                    {
                        "field": f"ledger.{name}",
                        "changed_bars": int((~equal).sum()),
                        "first_bar": first,
                        "expected": old[first].item(),
                        "actual": actual[first].item(),
                        "max_abs_delta": float(np.max(np.abs(actual.astype(float) - old.astype(float)))),
                    }
                )
        ratios = {}
        if current["performance"] and reference["performance"]:
            for name, limit in limits.items():
                ratio = current["performance"][case_id][name] / reference["performance"][case_id][name]
                ratios[name] = ratio
                if limit is not None and ratio > limit:
                    differences.append({"field": f"performance.{name}", "ratio": ratio, "limit": limit})
        comparison["cases"][case_id] = {
            "status": "changed" if differences else "passed",
            "differences": differences,
            "metric_deltas": metric_deltas,
            "resource_ratios": ratios,
        }
    if any(case["status"] != "passed" for case in comparison["cases"].values()):
        comparison["status"] = "changed"
    return comparison


def _markdown(value):
    return (
        str(value).replace("&", "&amp;").replace("<", "&lt;").replace("|", "\\|").replace("`", "\\`").replace("\n", " ")
    )


def render_benchmark_report(snapshot: dict, comparison: dict) -> str:
    """Render a portable report; do not rank or select strategies by returns."""
    reference = comparison.get("reference", {})
    lines = [
        "# Momentum Lab frozen regression report",
        "",
        f"> {DISCLAIMER}",
        "",
        f"Status: **{comparison['status']}**",
        "",
        f"Suite: `{_markdown(snapshot['suite']['suite_id'])}`",
        f"Contract SHA-256: `{snapshot['suite_sha256']}`",
        f"Package: `{_markdown(snapshot['provenance']['package_version'])}`",
        f"Source SHA-256: `{snapshot['provenance']['source_sha256']}`",
        "",
        f"Reference package: `{_markdown(reference.get('package_version', 'unknown'))}`",
        f"Reference source SHA-256: `{_markdown(reference.get('source_sha256', 'unknown'))}`",
        "",
        f"Environment changed: {comparison.get('environment_changed', 'unknown')}.",
        "",
        f"Tolerance: rtol={RTOL:g}, atol={ATOL:g}. Any numerical change, including improved returns, is reviewed.",
        "",
        "## Fixed scenarios",
        "",
        "| Dataset | Synthetic asset style | Regime | Bars/year | Data SHA-256 |",
        "|---|---|---|---:|---|",
    ]
    for dataset in snapshot["suite"]["datasets"]:
        lines.append(
            f"| {_markdown(dataset['id'])} | {_markdown(dataset['asset_style'])} | {_markdown(dataset['regime'])} "
            f"| {dataset['annualization']} | `{dataset['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Results (fixed order, not a performance ranking)",
            "",
            "| Case | Total return | Sharpe | Max drawdown | Turnover | Time (s) | Traced peak (KiB) | Comparison |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for case_id, result in snapshot["results"].items():
        metrics = result["metrics"]
        resources = snapshot["performance"].get(case_id)
        elapsed = f"{resources['seconds']:.4f}" if resources else "unmeasured"
        memory = f"{resources['peak_traced_bytes'] / 1024:.1f}" if resources else "unmeasured"
        status = comparison["cases"].get(case_id, {}).get("status", comparison["status"])
        lines.append(
            f"| {_markdown(case_id)} | {metrics['total_return']:.4%} | {metrics['sharpe']:.4f} "
            f"| {metrics['max_drawdown']:.4%} | {metrics['turnover']:.4f} | {elapsed} | {memory} | {status} |"
        )
    lines.extend(["", "## Comparison details", ""])
    lines.extend(f"- {_markdown(issue)}" for issue in comparison["issues"])
    for case_id, case in comparison["cases"].items():
        for difference in case["differences"]:
            lines.append(f"- {_markdown(case_id)}: {_markdown(json.dumps(difference, sort_keys=True))}")
    if comparison["status"] == "passed":
        lines.append("All metrics and complete ledgers match the reference within the fixed tolerances.")
    lines.extend(
        [
            "",
            "## Assumptions and limitations",
            "",
            "- Next-close execution, costs, leverage and annualization are fixed by the embedded suite contract.",
            "- No training, tuning, final-test access, network downloads, or historical-data redistribution.",
            "- Runtime and allocation observations are environment-dependent; they do not gate unless requested.",
            f"- Measurement: {MEASUREMENT}. Repeats: {snapshot['repeats']}.",
            "- Full parameters, raw ledgers and environment versions are in snapshot.json; deltas are in comparison.json.",
            "- Reference updates require a reviewed explanation, never automatic acceptance of better returns.",
            "",
        ]
    )
    return "\n".join(lines)


def write_benchmark_report(snapshot: dict, comparison: dict, output_dir: str | Path) -> Path:
    """Create a new report directory; existing artifacts are never overwritten."""
    payloads = {
        "snapshot.json": json.dumps(snapshot, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n",
        "comparison.json": json.dumps(comparison, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        "report.md": render_benchmark_report(snapshot, comparison),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for filename, payload in payloads.items():
        _write_text_atomic(output / filename, payload)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(prog="momentum-lab benchmark", description=DISCLAIMER)
    parser.add_argument("--output", type=Path, help="New report directory (existing directories are rejected)")
    parser.add_argument(
        "--compare", type=Path, help="Reference snapshot.json; defaults to the reviewed bundled reference"
    )
    parser.add_argument("--repeat", type=int, default=1, help="Fresh-cache repetitions per case, 1-100 (default: 1)")
    parser.add_argument("--max-slowdown", type=float, help="Optional per-case elapsed-time ratio limit")
    parser.add_argument("--max-memory-growth", type=float, help="Optional per-case traced-peak ratio limit")
    args = parser.parse_args(argv)
    output = args.output or Path("experiments") / "benchmarks" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    )
    try:
        if output.exists():
            raise ValueError(f"Output directory already exists: {output}")
        reference = json.loads(args.compare.read_text(encoding="utf-8")) if args.compare else load_benchmark_reference()
        snapshot = run_benchmarks(repeats=args.repeat)
        comparison = compare_benchmarks(
            snapshot, reference, max_slowdown=args.max_slowdown, max_memory_growth=args.max_memory_growth
        )
        write_benchmark_report(snapshot, comparison, output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Frozen benchmark: {comparison['status']} ({len(snapshot['results'])} cases)")
    print(f"Report: {output / 'report.md'}")
    return {"passed": 0, "changed": 1, "incompatible": 2}[comparison["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
