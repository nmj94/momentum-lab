"""Causal cross-sectional rules and independently checked shared-cash ledgers."""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from momentum_lab import PortfolioError, backtest_portfolio, cross_sectional_momentum, portfolio_metrics
from momentum_lab import portfolio as pf


def panel(n=8):
    return pd.DataFrame(
        {"BBB": np.full(n, 100.0), "AAA": 100.0 + 10 * np.arange(n)},
        index=pd.date_range("2024-01-02", periods=n, freq="B", name="date"),
    )


def orders(prices, *instructions):
    result = pd.DataFrame(np.nan, index=prices.index, columns=sorted(prices.columns))
    for row, weights in instructions:
        result.iloc[row] = weights
    return result


def test_scores_use_lookback_minus_skipped_recent_sessions():
    prices = panel()
    result = cross_sectional_momentum(prices, lookback=3, skip_recent=1, rebalance="daily")
    expected = prices["AAA"].shift(1) / prices["AAA"].shift(3) - 1
    pd.testing.assert_series_equal(result["scores"]["AAA"], expected)
    assert result["targets"].iloc[:3].isna().all().all()
    assert result["scores"]["AAA"].iloc[3] == pytest.approx(0.2)
    assert result["targets"].iloc[3].tolist() == [1, 0]


def test_exact_ties_break_by_canonical_ticker_not_column_order():
    prices = panel()
    prices["BBB"] = prices["AAA"]
    prices.columns = ["bbb", "aaa"]
    first = cross_sectional_momentum(prices, lookback=2, rebalance="daily")
    second = cross_sectional_momentum(prices.iloc[:, ::-1], lookback=2, rebalance="daily")
    for key in first:
        if isinstance(first[key], pd.DataFrame):
            pd.testing.assert_frame_equal(first[key], second[key])
        else:
            pd.testing.assert_series_equal(first[key], second[key])
    assert first["targets"].iloc[2].tolist() == [1, 0]


@pytest.mark.parametrize("cap, expected", [(1.0, 0.5), (0.3, 0.3)])
def test_abs_filter_vacant_top_k_slots_remain_cash(cap, expected):
    result = cross_sectional_momentum(panel(), lookback=1, top_k=2, max_weight=cap, rebalance="daily")
    assert result["targets"].iloc[1].tolist() == [expected, 0]
    # A zero score does not pass the strict >0 absolute-momentum filter.
    unfiltered = cross_sectional_momentum(panel(), lookback=1, top_k=2, absolute_threshold=None)
    assert unfiltered["targets"].iloc[1].tolist() == [0.5, 0.5]


def test_negative_absolute_momentum_can_hold_all_cash():
    prices = panel().iloc[::-1].set_axis(panel().index)
    result = cross_sectional_momentum(prices, lookback=2, rebalance="daily")
    assert (result["targets"].iloc[2:] == 0).all().all()
    allowed = cross_sectional_momentum(prices, lookback=2, absolute_threshold=None)
    assert allowed["targets"].iloc[2].tolist() == [0, 1]


@pytest.mark.parametrize(
    "frequency, rows", [("daily", [1, 2, 3, 4, 5, 6, 7]), ("weekly", [1, 2, 7]), ("monthly", [1, 5])]
)
def test_period_signals_use_first_observed_session_not_future_month_end(frequency, rows):
    prices = panel().set_axis(
        pd.DatetimeIndex(
            [
                "2024-01-25",
                "2024-01-26",
                "2024-01-29",
                "2024-01-30",
                "2024-01-31",
                "2024-02-01",
                "2024-02-02",
                "2024-02-05",
            ],
            name="date",
        )
    )
    result = cross_sectional_momentum(prices, lookback=1, rebalance=frequency)
    assert np.flatnonzero(result["rebalance"]).tolist() == rows
    assert result["targets"].loc[~result["rebalance"]].isna().all().all()


@pytest.mark.parametrize("frequency", ["daily", "weekly", "monthly"])
def test_future_price_mutation_or_prefix_truncation_does_not_change_past(frequency):
    prices = panel(70)
    config = {"lookback": 5, "skip_recent": 2, "top_k": 2, "rebalance": frequency}
    full = cross_sectional_momentum(prices, **config)
    changed = prices.copy()
    changed.iloc[40:] *= 0.01
    future = cross_sectional_momentum(changed, **config)
    prefix = cross_sectional_momentum(prices.iloc[:40], **config)
    for key in ("scores", "targets"):
        pd.testing.assert_frame_equal(full[key].iloc[:40], future[key].iloc[:40])
        pd.testing.assert_frame_equal(full[key].iloc[:40], prefix[key])
    book = backtest_portfolio(full["targets"], prices)
    truncated = backtest_portfolio(prefix["targets"], prices.iloc[:40])
    pd.testing.assert_frame_equal(book["ledger"].iloc[:40], truncated["ledger"])


@pytest.mark.parametrize(
    "options",
    [
        {"lookback": 0},
        {"lookback": 1.1},
        {"lookback": True},
        {"lookback": 8},
        {"skip_recent": -1},
        {"skip_recent": 2},
        {"skip_recent": False},
        {"top_k": 0},
        {"top_k": 3},
        {"top_k": True},
        {"rebalance": "quarterly"},
        {"absolute_threshold": float("nan")},
        {"absolute_threshold": "0"},
        {"max_weight": 0},
        {"max_weight": 1.01},
        {"max_weight": True},
        {"max_weight": 10**400},
    ],
)
def test_invalid_signal_options(options):
    with pytest.raises(PortfolioError):
        cross_sectional_momentum(panel(), **{"lookback": 2, **options})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda df: df.iloc[0:0],
        lambda df: df.iloc[::-1],
        lambda df: df.set_axis(pd.RangeIndex(len(df))),
        lambda df: df.set_axis(pd.DatetimeIndex([df.index[0]] * len(df))),
        lambda df: df.set_axis(df.index.tz_localize("UTC")),
        lambda df: df.set_axis(df.index + pd.Timedelta(hours=1)),
        lambda df: df.set_axis(pd.DatetimeIndex([pd.NaT, *df.index[1:]])),
        lambda df: df.astype(str),
        lambda df: df.astype(bool),
        lambda df: df.astype(complex),
        lambda df: df * 0,
        lambda df: -df,
        lambda df: df * np.nan,
        lambda df: df * np.inf,
        lambda df: df.set_axis(["aaa", "AAA"], axis=1),
        lambda df: df.set_axis(["=SUM(A1)", "BBB"], axis=1),
        lambda df: df.set_axis([0, 1], axis=1),
    ],
)
def test_invalid_price_contract(mutate):
    with pytest.raises(PortfolioError):
        cross_sectional_momentum(mutate(panel()), lookback=2)


def test_cell_and_asset_limits(monkeypatch):
    monkeypatch.setattr(pf, "MAX_PORTFOLIO_CELLS", 4)
    with pytest.raises(PortfolioError, match="cell"):
        cross_sectional_momentum(panel(), lookback=2)
    monkeypatch.setattr(pf, "MAX_PORTFOLIO_CELLS", 1_000_000)
    monkeypatch.setattr(pf, "MAX_PORTFOLIO_ASSETS", 1)
    with pytest.raises(PortfolioError, match="assets"):
        cross_sectional_momentum(panel(), lookback=2)


def test_score_overflow_is_not_ranked():
    prices = panel(3)
    prices["AAA"] = [1e-200, 1e200, 1e200]
    with pytest.raises(PortfolioError, match="non-finite scores"):
        cross_sectional_momentum(prices, lookback=1)


def test_post_fee_full_investment_is_funded_from_cash_not_borrowing():
    prices = panel(3)
    targets = orders(prices, (0, [1, 0]))
    book = backtest_portfolio(targets, prices, initial_capital=100, cost_bps=100)
    assert book["ledger"]["nav"].iloc[1] == pytest.approx(100 / 1.01)
    assert book["ledger"]["transaction_cost"].iloc[1] == pytest.approx(100 - 100 / 1.01)
    assert book["holdings"]["AAA"].iloc[1] == pytest.approx((100 / 1.01) / 110)
    assert book["ledger"]["cash"].min() >= 0
    assert book["ledger"]["nav"].iloc[2] == pytest.approx((100 / 1.01) * 120 / 110)


def test_rotation_charges_both_legs_and_cash_reconciles_every_day():
    prices = panel(6)
    targets = orders(prices, (0, [1, 0]), (1, [0, 1]), (3, [0.25, 0.25]), (4, [0, 0]))
    book = backtest_portfolio(targets, prices, initial_capital=1000, cost_bps=10, slippage_bps=5, spread_bps=20)
    ledger = book["ledger"]
    np.testing.assert_allclose(ledger["nav"], book["asset_values"].sum(axis=1) + ledger["cash"])
    np.testing.assert_allclose(book["weights"].sum(axis=1) + ledger["cash_weight"], 1, atol=1e-14)
    np.testing.assert_allclose(ledger["transaction_cost"], book["trades"].abs().sum(axis=1) * 0.0025, atol=1e-12)
    np.testing.assert_allclose(
        ledger["nav"].diff().iloc[1:],
        (ledger["asset_pnl"] + ledger["cash_interest"] - ledger["transaction_cost"]).iloc[1:],
        atol=1e-12,
    )
    assert ledger["turnover"].iloc[2] > 1.9
    assert ledger["cash_weight"].iloc[-1] == pytest.approx(1)
    assert ledger["cash"].min() >= 0


def test_hold_rows_keep_units_not_weights_and_target_caps_can_drift():
    prices = panel(4)
    prices["AAA"] = [100, 100, 200, 400]
    book = backtest_portfolio(orders(prices, (0, [0.5, 0.25])), prices, initial_capital=1000, cost_bps=0)
    assert book["ledger"]["nav"].tolist() == [1000, 1000, 1500, 2500]
    assert book["holdings"]["AAA"].tolist() == [0, 5, 5, 5]
    assert book["weights"]["AAA"].iloc[-1] == pytest.approx(0.8)
    assert book["ledger"]["rebalance_executed"].sum() == 1


@pytest.mark.parametrize("lag", [1, 2, 9])
def test_execution_delay_and_last_unfilled_signal(lag):
    prices = panel(4)
    book = backtest_portfolio(orders(prices, (0, [1, 0]), (3, [0, 1])), prices, cost_bps=0, execution_lag=lag)
    assert book["holdings"]["BBB"].eq(0).all()
    assert book["holdings"]["AAA"].iloc[:lag].eq(0).all()
    assert np.flatnonzero(book["ledger"]["rebalance_executed"]).tolist() == ([lag] if lag < 4 else [])


@pytest.mark.parametrize("cash_rate, nav", [(0.1, [100, 110, 121]), (-0.1, [100, 90, 81]), (0, [100, 100, 100])])
def test_effective_annual_cash_compounds_actual_elapsed_days(cash_rate, nav):
    prices = panel(3).set_axis(pd.DatetimeIndex(["2020-01-01", "2020-12-31", "2021-12-31"], name="date"))
    book = backtest_portfolio(orders(prices), prices, initial_capital=100, cash_rate=cash_rate)
    np.testing.assert_allclose(book["ledger"]["nav"], nav)
    assert (book["trades"] == 0).all().all()


def test_cash_only_metrics_keep_positive_return_even_when_sharpe_is_undefined():
    prices = panel(3).set_axis(pd.DatetimeIndex(["2020-01-01", "2020-12-31", "2021-12-31"], name="date"))
    metrics = portfolio_metrics(
        backtest_portfolio(orders(prices), prices, initial_capital=100, cash_rate=0.1), annualization=1
    )
    assert metrics["total_return"] == pytest.approx(0.21)
    assert metrics["cagr"] == pytest.approx(0.1)
    assert metrics["sharpe"] is None
    assert metrics["return_intervals"] == 2
    assert metrics["max_drawdown"] == 0
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize("n", [1, 2])
def test_short_ledger_statistics_are_explicitly_undefined(n):
    prices = panel(n)
    metrics = portfolio_metrics(backtest_portfolio(orders(prices), prices))
    assert metrics["volatility"] is None and metrics["sharpe"] is None
    assert metrics["return_intervals"] == n - 1
    assert metrics["cagr"] == (None if n == 1 else 0)


@pytest.mark.parametrize(
    "options", [{"annualization": 0}, {"annualization": 367}, {"risk_free_rate": -1}, {"risk_free_rate": True}]
)
def test_invalid_metric_options(options):
    prices = panel()
    with pytest.raises(PortfolioError):
        portfolio_metrics(backtest_portfolio(orders(prices), prices), **options)


@pytest.mark.parametrize(
    "options",
    [
        {"initial_capital": 0},
        {"initial_capital": np.inf},
        {"cost_bps": -1},
        {"cost_bps": 10000},
        {"spread_bps": 20000},
        {"slippage_bps": True},
        {"cash_rate": -1},
        {"cash_rate": np.nan},
        {"execution_lag": 0},
        {"execution_lag": 1.5},
        {"execution_lag": True},
    ],
)
def test_invalid_execution_options(options):
    prices = panel()
    with pytest.raises(PortfolioError):
        backtest_portfolio(orders(prices), prices, **options)


@pytest.mark.parametrize("weights", [[0.5, np.nan], [-0.1, 0.5], [0.5, np.inf], [0.6, 0.6]])
def test_invalid_target_rows_fail_closed(weights):
    prices = panel()
    with pytest.raises(PortfolioError, match="long-only"):
        backtest_portfolio(orders(prices, (0, weights)), prices)


@pytest.mark.parametrize("dtype", [str, bool, complex])
def test_targets_reject_non_real_numeric_dtypes(dtype):
    prices = panel()
    targets = orders(prices, (0, [0.5, 0.5])).fillna(0).astype(dtype)
    with pytest.raises(PortfolioError, match="numeric"):
        backtest_portfolio(targets, prices)


def test_only_machine_roundoff_weight_overshoot_is_normalized_without_input_mutation():
    prices = panel()
    targets = orders(prices, (0, [0.5 + 1e-13, 0.5]))
    before, original = prices.copy(), targets.copy()
    book = backtest_portfolio(targets, prices)
    assert book["weights"].iloc[1].sum() <= 1.0 + 1e-15
    pd.testing.assert_frame_equal(targets, original)
    pd.testing.assert_frame_equal(prices, before)


def test_targets_require_exact_dates_and_assets_but_accept_column_permutations():
    prices = panel()
    targets = orders(prices, (0, [0.5, 0.25]))
    with pytest.raises(PortfolioError, match="dates"):
        backtest_portfolio(targets.iloc[1:], prices)
    with pytest.raises(PortfolioError, match="assets"):
        backtest_portfolio(targets.rename(columns={"AAA": "CCC"}), prices)
    with pytest.raises(PortfolioError, match="Duplicate"):
        backtest_portfolio(targets.set_axis(["AAA", "aaa"], axis=1), prices)
    result = backtest_portfolio(targets, prices)
    permuted = backtest_portfolio(targets.iloc[:, ::-1], prices.iloc[:, ::-1])
    for key in result:
        pd.testing.assert_frame_equal(result[key], permuted[key])


def test_extreme_scale_overflow_fails_instead_of_exporting_infinite_nav():
    prices = panel(3).set_axis(pd.DatetimeIndex(["2020-01-01", "2021-12-31", "2023-12-31"], name="date"))
    with pytest.raises(PortfolioError, match="NAV"):
        backtest_portfolio(orders(prices), prices, cash_rate=1e200)


def test_frozen_reference_is_independent_exact_fraction_oracle():
    root = Path(__file__).resolve().parents[1]
    frozen = root / "momentum_lab" / "benchmark_data" / "portfolio_reference_v1.json"
    oracle = runpy.run_path(str(root / "scripts" / "portfolio_reference_oracle.py"))
    assert json.loads(frozen.read_text()) == oracle["reference"]()
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == pf.PORTFOLIO_REFERENCE_SHA256
    assert pf.check_portfolio_reference() == {"status": "passed", "cases": 6, "schema_version": 1}


def test_frozen_reference_detects_tamper_and_changed_accounting(monkeypatch):
    original = pf.PORTFOLIO_REFERENCE_SHA256
    monkeypatch.setattr(pf, "PORTFOLIO_REFERENCE_SHA256", "0" * 64)
    with pytest.raises(PortfolioError, match="SHA-256"):
        pf.check_portfolio_reference()
    monkeypatch.setattr(pf, "PORTFOLIO_REFERENCE_SHA256", original)
    production = pf.backtest_portfolio

    def altered(*args, **kwargs):
        result = production(*args, **kwargs)
        result["ledger"].iloc[1, 0] += 1
        return result

    monkeypatch.setattr(pf, "backtest_portfolio", altered)
    with pytest.raises(PortfolioError, match="ledger changed"):
        pf.check_portfolio_reference()
