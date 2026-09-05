"""Frozen regressions plus independent oracles for the new benchmark harness."""

import copy
import json
import subprocess
import sys
import tracemalloc

import numpy as np
import pandas as pd
import pytest

from momentum_lab import benchmarks as bench
from momentum_lab import cli


@pytest.fixture(scope="module")
def snapshot():
    return bench.run_benchmarks(measure_resources=False)


@pytest.fixture(scope="module")
def reference():
    return bench.load_benchmark_reference()


def test_packaged_reference_matches_complete_frozen_ledgers(snapshot, reference):
    comparison = bench.compare_benchmarks(snapshot, reference)
    assert comparison["status"] == "passed"
    assert len(comparison["cases"]) == 16
    assert all(not case["differences"] for case in comparison["cases"].values())
    assert snapshot["suite"]["data_kind"] == "synthetic"
    assert snapshot["suite_sha256"] == "2ccdb94ac7e9f55f3c079d77491f95ed70009b3125a0aa4b3a1250adf8eea8cf"


def test_benchmarks_are_offline_and_deterministic(snapshot, monkeypatch):
    import momentum_lab.data as data_module

    monkeypatch.setattr(data_module.yf, "download", lambda *a, **kw: pytest.fail("Benchmark attempted a download"))
    repeat = bench.run_benchmarks(repeats=2, measure_resources=False)
    assert repeat["results"] == snapshot["results"]
    assert repeat["suite_sha256"] == snapshot["suite_sha256"]
    assert repeat["performance"] == {}


@pytest.mark.parametrize("dataset_index", range(4))
def test_cash_matches_independent_calendar_accrual(dataset_index, snapshot):
    dataset = snapshot["suite"]["datasets"][dataset_index]
    frame = bench._load_frame(dataset)
    days = frame.index.to_series().diff().dt.total_seconds().fillna(0).to_numpy() / 86400
    expected = np.cumprod(1 + 0.02 * days / 365.25)
    result = snapshot["results"][f"{dataset['id']}/cash"]
    np.testing.assert_allclose(result["ledger"]["equity"], expected, rtol=1e-12, atol=1e-12)
    assert not any(result["ledger"]["trades"])


def test_buy_and_hold_matches_independent_entry_and_price_ratio(snapshot):
    dataset = snapshot["suite"]["datasets"][0]
    frame = bench._load_frame(dataset)
    # No position until the next close: first earn one calendar day's cash,
    # then pay 1 bp commission + 0.5 bp slippage + half of the 2 bp spread.
    entry = (1 + 0.02 / 365.25) / (1 + 2.5 / 10000)
    expected = np.r_[1.0, entry * frame["close"].iloc[1:].to_numpy() / frame["close"].iloc[1]]
    actual = snapshot["results"]["equity_trend/buy_and_hold"]["ledger"]["equity"]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_stress_fixtures_actually_exercise_capacity_borrow_and_insolvency(snapshot):
    results = snapshot["results"]
    assert results["illiquid_stress/buy_and_hold"]["metrics"]["capacity_constrained_bars"] > 50
    assert results["commodity_range/tsmom"]["metrics"]["borrow_blocked_bars"] > 0
    ledger = results["crypto_jumps/tsmom"]["ledger"]
    first_zero = ledger["equity"].index(0.0)
    assert first_zero < len(ledger["equity"]) - 1
    assert all(value == 0 for value in ledger["equity"][first_zero:])
    assert all(value == 0 for value in ledger["trades"][first_zero + 1 :])
    illiquid = results["illiquid_stress/tsmom"]["ledger"]
    assert all(value == 0 for value in illiquid["trades"][35:45])
    assert max(illiquid["participation"]) <= 0.02 + 1e-12


@pytest.mark.parametrize("dataset_index", range(4))
@pytest.mark.parametrize("strategy_index", range(4))
def test_frozen_signals_and_execution_are_prefix_causal_and_cache_neutral(dataset_index, strategy_index, snapshot):
    suite = snapshot["suite"]
    dataset = suite["datasets"][dataset_index]
    strategy = suite["strategies"][strategy_index]
    frame = bench._load_frame(dataset)
    full = snapshot["results"][f"{dataset['id']}/{strategy['id']}"]
    prefix = bench._execute_case(suite, dataset, strategy, frame.iloc[:80])
    for name in bench.LEDGER_FIELDS:
        np.testing.assert_allclose(prefix["ledger"][name], full["ledger"][name][:80], atol=1e-12, rtol=1e-12)
    uncached = bench._execute_case({**suite, "indicator_cache_size": 0}, dataset, strategy, frame)
    assert uncached == full


def test_changed_data_hash_is_rejected():
    dataset = copy.deepcopy(bench.load_benchmark_suite()["datasets"][0])
    dataset["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        bench._load_frame(dataset)


def test_changed_data_bar_count_is_rejected():
    dataset = copy.deepcopy(bench.load_benchmark_suite()["datasets"][0])
    dataset["bars"] += 1
    with pytest.raises(ValueError, match="bar count"):
        bench._load_frame(dataset)


@pytest.mark.parametrize("delta", [-0.05, 0.05])
def test_both_better_and_worse_metrics_fail_regression(delta, snapshot):
    changed = copy.deepcopy(snapshot)
    changed["results"]["equity_trend/ma_cross"]["metrics"]["total_return"] += delta
    comparison = bench.compare_benchmarks(changed, snapshot)
    assert comparison["status"] == "changed"
    case = comparison["cases"]["equity_trend/ma_cross"]
    assert case["metric_deltas"]["total_return"] == pytest.approx(delta)
    assert case["differences"][0]["field"] == "metrics.total_return"


def test_interior_ledger_changes_fail_even_with_identical_summary(snapshot):
    changed = copy.deepcopy(snapshot)
    changed["results"]["equity_trend/cash"]["ledger"]["equity"][20] += 0.001
    comparison = bench.compare_benchmarks(changed, snapshot)
    assert comparison["status"] == "changed"
    detail = comparison["cases"]["equity_trend/cash"]["differences"][0]
    assert detail["field"] == "ledger.equity"
    assert detail["first_bar"] == 20
    assert detail["changed_bars"] == 1
    assert detail["max_abs_delta"] == pytest.approx(0.001)


def test_fixed_tolerance_accepts_only_small_numeric_noise(snapshot):
    changed = copy.deepcopy(snapshot)
    changed["results"]["equity_trend/cash"]["ledger"]["returns"][0] = 1e-13
    changed["results"]["equity_trend/cash"]["metrics"]["sharpe"] += 1e-13
    assert bench.compare_benchmarks(changed, snapshot)["status"] == "passed"


def test_boolean_ledger_change_fails(snapshot):
    changed = copy.deepcopy(snapshot)
    changed["results"]["equity_trend/cash"]["ledger"]["capacity_constrained"][10] = True
    comparison = bench.compare_benchmarks(changed, snapshot)
    assert comparison["status"] == "changed"


@pytest.mark.parametrize(
    "corruption",
    [
        "schema",
        "contract_hash",
        "missing_case",
        "extra_case",
        "missing_metric",
        "missing_ledger",
        "short_ledger",
        "nan_metric",
        "inf_ledger",
        "boolean_metric",
        "numeric_flag",
        "partial_performance",
        "invalid_measurement",
        "empty_contract",
        "duplicate_dataset",
    ],
)
def test_corrupt_snapshots_cannot_silently_pass(corruption, snapshot):
    changed = copy.deepcopy(snapshot)
    case = changed["results"]["equity_trend/cash"]
    if corruption == "schema":
        changed["schema_version"] = 99
    elif corruption == "contract_hash":
        changed["suite_sha256"] = "bogus"
    elif corruption == "missing_case":
        del changed["results"]["equity_trend/cash"]
    elif corruption == "extra_case":
        changed["results"]["extra/cash"] = case
    elif corruption == "missing_metric":
        del case["metrics"]["sharpe"]
    elif corruption == "missing_ledger":
        del case["ledger"]["trades"]
    elif corruption == "short_ledger":
        case["ledger"]["returns"].pop()
    elif corruption == "nan_metric":
        case["metrics"]["sharpe"] = float("nan")
    elif corruption == "inf_ledger":
        case["ledger"]["returns"][10] = float("inf")
    elif corruption == "boolean_metric":
        case["metrics"]["turnover"] = True
    elif corruption == "numeric_flag":
        case["ledger"]["borrow_blocked"][0] = 0
    elif corruption == "partial_performance":
        changed["performance"] = {"equity_trend/cash": {"seconds": 1, "peak_traced_bytes": 2}}
    elif corruption == "invalid_measurement":
        changed["measurement"] = "RSS"
    elif corruption == "empty_contract":
        changed["suite"]["datasets"] = []
        changed["suite_sha256"] = bench._json_hash(changed["suite"])
    elif corruption == "duplicate_dataset":
        changed["suite"]["datasets"].append(changed["suite"]["datasets"][0])
        changed["suite_sha256"] = bench._json_hash(changed["suite"])
    # Even two identically broken files must not produce a false green.
    assert bench.compare_benchmarks(changed, snapshot)["status"] == "incompatible"
    assert bench.compare_benchmarks(changed, changed)["status"] == "incompatible"


def test_changed_assumptions_are_incomparable_but_new_source_versions_are_not(snapshot):
    changed = copy.deepcopy(snapshot)
    changed["provenance"]["package_version"] = "99.0.0"
    changed["provenance"]["source_sha256"] = "new-source"
    changed["provenance"]["environment"]["python"] = "different-python"
    comparison = bench.compare_benchmarks(changed, snapshot)
    assert comparison["status"] == "passed"
    assert comparison["environment_changed"] is True
    assert comparison["reference"] == snapshot["provenance"]
    changed["suite"]["backtest"]["cost_bps"] = 7
    changed["suite_sha256"] = bench._json_hash(changed["suite"])
    assert bench.compare_benchmarks(changed, snapshot)["status"] == "incompatible"


def test_resource_changes_are_observational_unless_limits_are_requested(snapshot):
    old, current = copy.deepcopy(snapshot), copy.deepcopy(snapshot)
    old["performance"] = {case_id: {"seconds": 1.0, "peak_traced_bytes": 1000} for case_id in old["results"]}
    current["performance"] = {case_id: {"seconds": 2.0, "peak_traced_bytes": 3000} for case_id in current["results"]}
    assert bench.compare_benchmarks(current, old)["status"] == "passed"
    comparison = bench.compare_benchmarks(current, old, max_slowdown=1.5, max_memory_growth=2)
    assert comparison["status"] == "changed"
    assert comparison["cases"]["equity_trend/cash"]["resource_ratios"] == {"seconds": 2, "peak_traced_bytes": 3}
    assert len(comparison["cases"]["equity_trend/cash"]["differences"]) == 2
    assert bench.compare_benchmarks(snapshot, old, max_slowdown=2)["status"] == "incompatible"
    old["performance"]["equity_trend/cash"]["seconds"] = 0
    assert bench.compare_benchmarks(current, old)["status"] == "incompatible"


@pytest.mark.parametrize("limit", [0, -1, True, float("inf"), float("nan")])
def test_resource_limit_validation(limit, snapshot):
    with pytest.raises(ValueError, match="positive and finite"):
        bench.compare_benchmarks(snapshot, snapshot, max_slowdown=limit)


@pytest.mark.parametrize("repeats", [0, -1, 101, 1.5, True])
def test_repeat_validation(repeats):
    with pytest.raises(ValueError, match="repeats"):
        bench.run_benchmarks(repeats=repeats)


def test_profiling_is_positive_and_preserves_external_tracer(snapshot):
    measured = bench.run_benchmarks(repeats=2)
    assert len(measured["performance"]) == 16
    assert measured["results"] == snapshot["results"]
    assert all(row["seconds"] > 0 and row["peak_traced_bytes"] > 0 for row in measured["performance"].values())
    assert not tracemalloc.is_tracing()
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="already active"):
            bench.run_benchmarks()
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()


def test_missing_provenance_is_rejected(snapshot):
    changed = copy.deepcopy(snapshot)
    changed["provenance"] = {}
    assert bench.compare_benchmarks(changed, snapshot)["status"] == "incompatible"


def test_resource_switch_requires_a_boolean():
    with pytest.raises(ValueError, match="measure_resources must be boolean"):
        bench.run_benchmarks(measure_resources="false")


def test_profiling_is_stopped_after_execution_error(monkeypatch):
    def broken(*args):
        raise RuntimeError("execution failed")

    monkeypatch.setattr(bench, "_execute_case", broken)
    with pytest.raises(RuntimeError, match="execution failed"):
        bench.run_benchmarks()
    assert not tracemalloc.is_tracing()


def test_nondeterministic_repetitions_are_rejected(monkeypatch):
    original = bench._execute_case
    calls = 0

    def changing(*args):
        nonlocal calls
        calls += 1
        result = original(*args)
        result["metrics"]["total_return"] += calls
        return result

    monkeypatch.setattr(bench, "_execute_case", changing)
    with pytest.raises(ValueError, match="Non-deterministic"):
        bench.run_benchmarks(repeats=2, measure_resources=False)


def test_reports_are_json_safe_explicit_and_never_overwrite(tmp_path, snapshot):
    comparison = bench.compare_benchmarks(snapshot, snapshot)
    output = bench.write_benchmark_report(snapshot, comparison, tmp_path / "run")
    assert json.loads((output / "snapshot.json").read_text()) == snapshot
    assert json.loads((output / "comparison.json").read_text())["status"] == "passed"
    report = (output / "report.md").read_text()
    assert "not historical market observations" in report
    assert "not RSS" in report
    assert "Contract SHA-256" in report
    assert "unmeasured" in report
    with pytest.raises(FileExistsError):
        bench.write_benchmark_report(snapshot, comparison, output)
    assert json.loads((output / "snapshot.json").read_text()) == snapshot


def test_report_contains_differences_and_escapes_markdown(snapshot):
    changed = copy.deepcopy(snapshot)
    changed["results"]["equity_trend/cash"]["metrics"]["sharpe"] = 999
    comparison = bench.compare_benchmarks(changed, snapshot)
    assert "metrics.sharpe" in bench.render_benchmark_report(changed, comparison)
    comparison.update(status="incompatible", issues=["<script>x</script>|`test`"])
    report = bench.render_benchmark_report(changed, comparison)
    assert "<script>" not in report
    assert "&lt;script>" in report


@pytest.mark.parametrize("status,code", [("passed", 0), ("changed", 1), ("incompatible", 2)])
def test_cli_exit_codes_and_reports(tmp_path, monkeypatch, snapshot, reference, status, code):
    current = copy.deepcopy(snapshot)
    if status == "changed":
        current["results"]["equity_trend/cash"]["metrics"]["sharpe"] += 1
    elif status == "incompatible":
        current["suite"]["risk_free_rate"] = 0.1
        current["suite_sha256"] = bench._json_hash(current["suite"])
    monkeypatch.setattr(bench, "run_benchmarks", lambda **kwargs: current)
    target = tmp_path / status
    assert bench.main(["--output", str(target)]) == code
    assert json.loads((target / "comparison.json").read_text())["status"] == status


def test_cli_reads_custom_baseline_and_refuses_existing_output(tmp_path, monkeypatch, snapshot):
    monkeypatch.setattr(bench, "run_benchmarks", lambda **kwargs: snapshot)
    first = tmp_path / "first"
    assert bench.main(["--output", str(first)]) == 0
    assert bench.main(["--output", str(tmp_path / "second"), "--compare", str(first / "snapshot.json")]) == 0
    with pytest.raises(SystemExit) as raised:
        bench.main(["--output", str(first)])
    assert raised.value.code == 2
    with pytest.raises(SystemExit) as raised:
        bench.main(["--compare", str(tmp_path / "missing.json")])
    assert raised.value.code == 2


def test_existing_cli_dispatches_without_changing_ticker_interface(monkeypatch):
    calls = []
    monkeypatch.setattr(bench, "main", lambda argv: calls.append(argv) or 1)
    monkeypatch.setattr(sys, "argv", ["momentum-lab", "benchmark", "--repeat", "2"])
    assert cli.main() == 1
    assert calls == [["--repeat", "2"]]


def test_python_module_propagates_benchmark_failure_exit_status(tmp_path, snapshot):
    bad = copy.deepcopy(snapshot)
    bad["results"]["equity_trend/cash"]["metrics"]["sharpe"] += 1
    reference_dir = bench.write_benchmark_report(bad, bench.compare_benchmarks(bad, snapshot), tmp_path / "reference")
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "momentum_lab",
            "benchmark",
            "--compare",
            str(reference_dir / "snapshot.json"),
            "--output",
            str(tmp_path / "actual"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert run.returncode == 1, run.stdout + run.stderr
    assert "changed (16 cases)" in run.stdout


def test_frozen_dates_have_expected_calendar():
    suite = bench.load_benchmark_suite()
    weekdays = bench._load_frame(suite["datasets"][0])
    daily = bench._load_frame(suite["datasets"][2])
    assert not (weekdays.index.dayofweek >= 5).any()
    assert (daily.index.dayofweek >= 5).any()
    assert all(daily.index.to_series().diff().dropna() == pd.Timedelta(days=1))
