"""Statistical oracles, fail-closed diagnostics and search/report integration."""

import hashlib
import json
from inspect import signature
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from momentum_lab import cli, paired_block_bootstrap, search, uncertainty
from momentum_lab.config import SearchConfig
from momentum_lab.reporting import render_html_report, render_markdown_report


@pytest.fixture(scope="module")
def frozen():
    raw = (Path(__file__).parent / "fixtures" / "block_bootstrap_v1.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "2c7c89aaa20378f5d47dfbce66fb46ce9fc198f88ee356b140246fecee216bb2"
    return json.loads(raw)


def _series(frozen):
    values = np.asarray(frozen["returns"])
    index = pd.date_range("2020-01-02", periods=len(values), freq="B")
    return pd.Series(values[:, 0], index=index), pd.Series(values[:, 1], index=index)


def test_frozen_correlated_data_matches_independent_scalar_oracle(frozen):
    current = paired_block_bootstrap(*_series(frozen), **frozen["options"])
    assert current["status"] == "ok"
    assert current["n_observations"] == 512
    for key, expected in frozen["expected"].items():
        assert current["statistics"][key]["estimate"] == pytest.approx(expected["estimate"], abs=1e-12)
        np.testing.assert_allclose(current["statistics"][key]["ci"], expected["ci"], atol=1e-12, rtol=1e-11)
        assert current["statistics"][key]["valid_resamples"] == 1000
    assert current["blocks_per_resample"] == 52  # Last block is truncated, not dropped.
    json.dumps(current, allow_nan=False)


def test_reproducible_across_batches_and_preserves_global_rng(frozen, monkeypatch):
    state = np.random.get_state()
    first = paired_block_bootstrap(*_series(frozen), **frozen["options"])
    monkeypatch.setattr(uncertainty, "_BATCH_CELLS", 1100)
    second = paired_block_bootstrap(*_series(frozen), **frozen["options"])
    assert first == second
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


def test_block_method_reflects_positive_dependence_in_this_frozen_fixture(frozen):
    strategy, benchmark = _series(frozen)
    options = {**frozen["options"], "n_resamples": 2000}
    block = paired_block_bootstrap(strategy, benchmark, **options)
    iid = paired_block_bootstrap(strategy, benchmark, **{**options, "block_length": 1})
    block_ci = block["statistics"]["strategy_annualized_mean"]["ci"]
    iid_ci = iid["statistics"]["strategy_annualized_mean"]["ci"]
    assert block_ci[1] - block_ci[0] > 1.5 * (iid_ci[1] - iid_ci[0])
    assert "does not preserve serial dependence" in iid["warning"]


def test_pairing_preserves_cross_series_dependence(frozen):
    strategy, benchmark = _series(frozen)
    paired = paired_block_bootstrap(strategy, benchmark, **frozen["options"])
    # Deliberately destroy the fixture's cross-series dependence while retaining
    # its labels, only as a diagnostic oracle, not as an accepted market dataset.
    permuted = pd.Series(np.random.default_rng(77).permutation(benchmark.to_numpy()), index=benchmark.index)
    unpaired = paired_block_bootstrap(strategy, permuted, **frozen["options"])
    paired_ci = paired["statistics"]["annualized_mean_excess"]["ci"]
    unpaired_ci = unpaired["statistics"]["annualized_mean_excess"]["ci"]
    assert paired_ci[1] - paired_ci[0] < (unpaired_ci[1] - unpaired_ci[0]) / 2


def test_raw_arithmetic_mean_sharpe_annualization_and_risk_free_rate(frozen):
    strategy, benchmark = _series(frozen)
    result = paired_block_bootstrap(strategy, benchmark, annualization=365, risk_free_rate=0.05, n_resamples=200)
    metrics = result["statistics"]
    assert metrics["strategy_annualized_mean"]["estimate"] == pytest.approx(strategy.mean() * 365)
    assert metrics["strategy_sharpe"]["estimate"] == pytest.approx(
        (strategy.mean() * 365 - 0.05) / (strategy.std() * np.sqrt(365))
    )
    assert metrics["annualized_mean_excess"]["estimate"] == pytest.approx((strategy - benchmark).mean() * 365)
    assert "not CAGR" in result["warning"]


def test_without_benchmark_marks_relative_statistics_unavailable(frozen):
    strategy, _ = _series(frozen)
    result = paired_block_bootstrap(strategy, n_resamples=200)
    assert result["status"] == "ok"
    assert result["paired"] is False
    assert result["statistics"]["annualized_mean_excess"]["status"] == "not_provided"
    assert result["statistics"]["benchmark_sharpe"]["ci"] == [None, None]


@pytest.mark.parametrize(
    "n,block_length,required", [(0, 10, 60), (1, 10, 60), (59, 10, 60), (60, 20, 100), (60, 100, 500)]
)
def test_insufficient_samples_never_get_a_fabricated_interval(n, block_length, required):
    values = pd.Series(np.linspace(-0.02, 0.02, n), dtype=float)
    result = paired_block_bootstrap(values, values, block_length=block_length)
    assert result["status"] == "insufficient_data"
    assert result["required_observations"] == required
    assert result["completed_resamples"] == 0
    assert all(value["ci"] == [None, None] for value in result["statistics"].values())
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("constant", [0.0, 0.001])
def test_zero_variance_means_and_sharpes_are_explicitly_unavailable(constant):
    values = pd.Series(constant, index=pd.RangeIndex(100))
    result = paired_block_bootstrap(values, values, n_resamples=200)
    assert result["status"] == "unavailable"
    for value in result["statistics"].values():
        assert value["status"] == "zero_variance"
        assert value["ci"] == [None, None]
    assert result["statistics"]["strategy_sharpe"]["estimate"] is None
    assert result["statistics"]["annualized_mean_excess"]["estimate"] == 0.0


def test_undefined_sharpe_draws_are_not_silently_discarded():
    values = pd.Series([0.02] + [0.0] * 99)
    result = paired_block_bootstrap(values, n_resamples=200, block_length=1)
    statistic = result["statistics"]["strategy_sharpe"]
    assert statistic["status"] == "degenerate_resamples"
    assert 0 < statistic["valid_resamples"] < 200
    assert statistic["ci"] == [None, None]
    assert result["statistics"]["strategy_annualized_mean"]["status"] == "ok"


def test_constant_block_sums_are_marked_degenerate():
    values = pd.Series(np.tile([-0.01, 0.02], 50))
    result = paired_block_bootstrap(values, n_resamples=200, block_length=2)
    assert result["statistics"]["strategy_annualized_mean"]["status"] == "degenerate_resamples"
    assert result["statistics"]["strategy_sharpe"]["ci"] == [None, None]


def test_resource_limit_is_checked_before_allocating_resamples(frozen, monkeypatch):
    monkeypatch.setattr(uncertainty, "MAX_RESAMPLE_CELLS", 100)
    monkeypatch.setattr(uncertainty.np.random, "PCG64", lambda *args: pytest.fail("Must not allocate RNG work"))
    result = paired_block_bootstrap(*_series(frozen), **frozen["options"])
    assert result["status"] == "resource_limit"
    assert result["completed_resamples"] == 0
    assert all(value["ci"] == [None, None] for value in result["statistics"].values())


def test_numpy_integer_overflow_cannot_bypass_workload_limit(monkeypatch):
    monkeypatch.setattr(uncertainty.np.random, "PCG64", lambda *args: pytest.fail("Workload must fail closed"))
    values = pd.Series(np.tile([-0.01, 0.02], 75_000))
    result = paired_block_bootstrap(values, n_resamples=np.int32(20_000), block_length=np.int32(10))
    assert result["status"] == "resource_limit"
    json.dumps(result, allow_nan=False)


def test_numpy_scalar_options_are_normalized_for_json(frozen):
    result = paired_block_bootstrap(
        *_series(frozen),
        n_resamples=np.int32(200),
        block_length=np.int64(10),
        seed=np.uint32(17),
        min_observations=np.int64(60),
        confidence_level=np.float32(0.95),
        annualization=np.float64(252),
    )
    assert result["status"] == "ok"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "options",
    [
        {"n_resamples": 199},
        {"n_resamples": 20001},
        {"n_resamples": True},
        {"n_resamples": 200.5},
        {"block_length": 0},
        {"block_length": -1},
        {"block_length": True},
        {"block_length": 1_000_001},
        {"seed": -1},
        {"seed": 2**32},
        {"seed": False},
        {"seed": "42"},
        {"confidence_level": 0},
        {"confidence_level": 1},
        {"confidence_level": float("nan")},
        {"confidence_level": True},
        {"confidence_level": 0.999},
        {"min_observations": 1},
        {"min_observations": True},
        {"annualization": 0},
        {"annualization": True},
        {"annualization": float("inf")},
        {"risk_free_rate": float("nan")},
        {"risk_free_rate": True},
    ],
)
def test_invalid_bootstrap_options_are_rejected(options):
    with pytest.raises(ValueError):
        paired_block_bootstrap(pd.Series(np.linspace(-0.01, 0.02, 100)), **options)


@pytest.mark.parametrize(
    "values",
    [
        pd.Series([0.1, np.nan]),
        pd.Series([0.1, np.inf]),
        pd.Series([True, False]),
        pd.Series([0.1 + 1j, 0.2]),
        pd.Series(["0.1", "0.2"]),
        pd.Series([1, None], dtype="Float64"),
    ],
)
def test_invalid_returns_are_never_silently_dropped_or_coerced(values):
    with pytest.raises(ValueError):
        paired_block_bootstrap(values)


def test_return_indexes_must_match_exactly(frozen):
    strategy, benchmark = _series(frozen)
    with pytest.raises(ValueError, match="match exactly"):
        paired_block_bootstrap(strategy, benchmark.iloc[1:])
    with pytest.raises(ValueError, match="sorted"):
        paired_block_bootstrap(strategy.iloc[::-1])
    with pytest.raises(ValueError, match="unique"):
        paired_block_bootstrap(pd.Series([0.01, 0.02], index=[1, 1]))
    with pytest.raises(ValueError, match="index"):
        paired_block_bootstrap(pd.Series([0.01, 0.02], index=[1, np.nan]))
    with pytest.raises(TypeError, match="MultiIndex"):
        paired_block_bootstrap(pd.Series([0.01, 0.02], index=pd.MultiIndex.from_tuples([(1, 1), (2, 2)])))
    with pytest.raises(TypeError, match="Series"):
        paired_block_bootstrap([0.01, 0.02])


def test_numeric_overflow_cannot_leak_nan_into_json():
    result = paired_block_bootstrap(pd.Series([1e308, -1e308] * 50), n_resamples=200)
    assert result["status"] == "unavailable"
    json.dumps(result, allow_nan=False)


def test_fingerprint_binds_returns_and_time_index(frozen):
    strategy, benchmark = _series(frozen)
    first = paired_block_bootstrap(strategy, benchmark, n_resamples=200)
    altered = strategy.copy()
    altered.iloc[3] += 0.001
    changed = paired_block_bootstrap(altered, benchmark, n_resamples=200)
    shifted = paired_block_bootstrap(
        strategy.set_axis(strategy.index + pd.Timedelta(days=1)),
        benchmark.set_axis(benchmark.index + pd.Timedelta(days=1)),
        n_resamples=200,
    )
    assert len({first["returns_sha256"], changed["returns_sha256"], shifted["returns_sha256"]}) == 3


def _market(monkeypatch, frozen, *, test_change=False):
    strategy, _ = _series(frozen)
    prices = 100 * (1 + strategy).cumprod()
    if test_change:
        start = int(len(prices) * 0.8)
        prices.iloc[start:] *= np.linspace(1, 1.5, len(prices) - start)
    frame = pd.DataFrame({"close": prices, "volume": 100000.0})
    data = {column: frame[column] for column in frame}
    data["annualization"] = 252
    monkeypatch.setattr(search, "prepare_data", lambda *args, **kwargs: (data, frame))
    params = [
        {"lookback": value, "threshold": 0.002, "long_short": True, "skip_recent": 1, "signal_smooth": 0}
        for value in (12, 24)
    ]
    monkeypatch.setattr(search, "_quick_sample", lambda *args, **kwargs: params)
    return frame


def _run(tmp_path, run_id, **kwargs):
    return search.run_search(
        ticker="FAKE",
        strategies=["tsmom"],
        result_dir=tmp_path,
        run_id=run_id,
        quick=True,
        robust=False,
        bootstrap_resamples=200,
        keep_all_results=True,
        **kwargs,
    )


def test_post_selection_diagnostics_do_not_change_ranking_or_leak_to_candidates(tmp_path, monkeypatch, frozen):
    frame = _market(monkeypatch, frozen)
    disabled = _run(tmp_path, "disabled", bootstrap=False)
    selected = False
    original_select, original_bootstrap, original_backtest = (
        search._select_from_store,
        search.paired_block_bootstrap,
        search.backtest,
    )
    observations, full_backtests = [], []

    def selection(*args, **kwargs):
        nonlocal selected
        result = original_select(*args, **kwargs)
        selected = True
        return result

    def diagnostic(strategy, benchmark, **kwargs):
        assert selected
        assert strategy.index.equals(benchmark.index)
        observations.append(strategy.index)
        return original_bootstrap(strategy, benchmark, **kwargs)

    def accounting(positions, prices, **kwargs):
        if len(prices) == len(frame):
            full_backtests.append(len(prices))
        return original_backtest(positions, prices, **kwargs)

    monkeypatch.setattr(search, "_select_from_store", selection)
    monkeypatch.setattr(search, "paired_block_bootstrap", diagnostic)
    monkeypatch.setattr(search, "backtest", accounting)
    enabled = _run(tmp_path, "enabled")
    assert enabled["best"] is not None
    assert enabled["top_results"] == disabled["top_results"]
    assert enabled["all_results"] == disabled["all_results"]
    assert enabled["selection_diagnostics"] == disabled["selection_diagnostics"]
    assert enabled["best"]["test_metrics"] == disabled["best"]["test_metrics"]
    assert disabled["bootstrap_diagnostics"]["status"] == "disabled"
    assert enabled["bootstrap_diagnostics"]["used_for_selection"] is False
    assert len(full_backtests) == 2  # One selected strategy and one benchmark; no bootstrap replay.
    periods = search._split_periods(frame.index)
    assert len(observations) == 2
    assert observations[0].equals(frame.loc[slice(*periods["val"])].index)
    assert observations[1].equals(frame.loc[slice(*periods["test"])].index)
    for filename in ("all_results.csv", "top_results.csv"):
        assert "bootstrap" not in (tmp_path / "enabled" / filename).read_text()
    assert all(
        "bootstrap" not in json.dumps(row)
        for row in search._iter_result_store(tmp_path / "enabled" / "results.sqlite3")
    )
    summary = json.loads((tmp_path / "enabled" / "summary.json").read_text())
    assert summary["bootstrap_diagnostics"] == enabled["bootstrap_diagnostics"]
    for filename in ("report.md", "report.html"):
        content = (tmp_path / "enabled" / filename).read_text()
        assert "Paired block-bootstrap uncertainty" in content
        assert "Annualized mean excess" in content
        assert "not selection-adjusted" in content


def test_test_price_changes_cannot_change_validation_intervals(tmp_path, monkeypatch, frozen):
    _market(monkeypatch, frozen)
    first = _run(tmp_path, "original", generate_report=False)
    _market(monkeypatch, frozen, test_change=True)
    changed = _run(tmp_path, "changed-test", generate_report=False)
    assert first["top_results"] == changed["top_results"]
    assert (
        first["bootstrap_diagnostics"]["periods"]["validation"]
        == changed["bootstrap_diagnostics"]["periods"]["validation"]
    )
    assert (
        first["bootstrap_diagnostics"]["periods"]["test"]["returns_sha256"]
        != changed["bootstrap_diagnostics"]["periods"]["test"]["returns_sha256"]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("bootstrap", False),
        ("bootstrap_resamples", 400),
        ("bootstrap_block_length", 12),
        ("bootstrap_confidence", 0.9),
        ("bootstrap_seed", 9),
        ("bootstrap_min_observations", 70),
    ],
)
def test_bootstrap_configuration_is_resume_locked(tmp_path, monkeypatch, frozen, field, value):
    _market(monkeypatch, frozen)
    _run(tmp_path, "locked", generate_report=False)
    with pytest.raises(ValueError, match=field):
        search.run_search(
            ticker="FAKE",
            strategies=["tsmom"],
            result_dir=tmp_path,
            run_id="locked",
            robust=False,
            resume=True,
            **{"bootstrap_resamples": 200, field: value},
        )


def test_config_wiring_preserves_old_positional_options_and_resume_determinism(tmp_path, monkeypatch, frozen):
    _market(monkeypatch, frozen)
    config = SearchConfig(
        ticker="FAKE",
        strategies=["tsmom"],
        result_dir=str(tmp_path),
        run_id="config",
        robust=False,
        bootstrap_resamples=400,
        bootstrap_block_length=12,
        bootstrap_seed=123,
        bootstrap_confidence=0.9,
        bootstrap_min_observations=70,
    )
    assert SearchConfig.from_mapping(json.loads(json.dumps(config.to_dict()))) == config
    for obj in (SearchConfig, search.run_search):
        names = list(signature(obj).parameters)
        assert names.index("bootstrap") > names.index("indicator_cache_size")
    first = search.run_search(config=config)
    resumed = search.run_search(config=config, resume=True)
    assert first["bootstrap_diagnostics"] == resumed["bootstrap_diagnostics"]
    metadata = json.loads((tmp_path / "config" / "run_config.json").read_text())
    for name in (
        "bootstrap",
        "bootstrap_resamples",
        "bootstrap_block_length",
        "bootstrap_seed",
        "bootstrap_confidence",
        "bootstrap_min_observations",
    ):
        assert metadata[name] == getattr(config, name)


@pytest.mark.parametrize("kwargs", [{"bootstrap": 1}, {"bootstrap_resamples": 0}, {"bootstrap_seed": -1}])
def test_search_rejects_invalid_bootstrap_before_accessing_data(monkeypatch, kwargs):
    monkeypatch.setattr(search, "prepare_data", lambda *args, **kwargs: pytest.fail("No data access expected"))
    with pytest.raises((ValueError, TypeError)):
        search.run_search(**kwargs)


def test_no_selection_does_not_run_bootstrap(tmp_path, monkeypatch, frozen):
    _market(monkeypatch, frozen)
    monkeypatch.setattr(search, "paired_block_bootstrap", lambda *args, **kwargs: pytest.fail("No selected strategy"))
    result = _run(tmp_path, "no-selection", min_validation_trades=1_000_000)
    assert result["best"] is None
    assert result["bootstrap_diagnostics"]["status"] == "no_selection"


def test_cli_wires_bootstrap_flags(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "run_search", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        "sys.argv",
        [
            "momentum-lab",
            "SPY",
            "--no-bootstrap",
            "--bootstrap-resamples",
            "1000",
            "--bootstrap-block-length",
            "20",
            "--bootstrap-confidence",
            "0.9",
            "--bootstrap-seed",
            "17",
            "--bootstrap-min-observations",
            "100",
        ],
    )
    cli.main()
    assert captured["bootstrap"] is False
    assert captured["bootstrap_resamples"] == 1000
    assert captured["bootstrap_block_length"] == 20
    assert captured["bootstrap_confidence"] == 0.9
    assert captured["bootstrap_seed"] == 17
    assert captured["bootstrap_min_observations"] == 100


def test_reports_explain_unavailable_intervals_and_escape_reasons():
    unavailable = paired_block_bootstrap(pd.Series([0.0] * 20))
    unavailable["statistics"]["strategy_sharpe"]["reason"] = "<script>alert(1)</script>|example"
    summary = {"bootstrap_diagnostics": {"status": "unavailable", "periods": {"validation": unavailable}}}
    markdown = render_markdown_report(summary, {})
    html = render_html_report(summary, {})
    for content in (markdown, html):
        assert "Not estimated" in content
        assert "insufficient_data" in content
        assert "<script>" not in content
    assert "\\|example" in markdown
    assert "analytic" in markdown


def test_changing_seed_affects_intervals_not_point_estimates(frozen):
    first = paired_block_bootstrap(*_series(frozen), **frozen["options"])
    second = paired_block_bootstrap(*_series(frozen), **{**frozen["options"], "seed": 18})
    assert first["returns_sha256"] == second["returns_sha256"]
    assert (
        first["statistics"]["strategy_annualized_mean"]["estimate"]
        == second["statistics"]["strategy_annualized_mean"]["estimate"]
    )
    assert (
        first["statistics"]["strategy_annualized_mean"]["ci"] != second["statistics"]["strategy_annualized_mean"]["ci"]
    )
