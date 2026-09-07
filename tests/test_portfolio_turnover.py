"""Independent turnover ledgers, causality, funding and execution diagnostics."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from momentum_lab import PortfolioError, backtest_portfolio, cli, portfolio_metrics
from momentum_lab import portfolio_turnover_benchmarks as frozen
from momentum_lab.portfolio_study import _test_books


def panel(n=12):
    return pd.DataFrame(
        {"AAA": np.linspace(10, 18, n), "BBB": np.linspace(20, 15, n)},
        index=pd.date_range("2024-01-02", periods=n, freq="B", name="date"),
    )


def orders(prices, rows):
    result = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for row, weights in rows.items():
        result.iloc[row] = weights
    return result


@pytest.mark.parametrize("bad", [True, False, np.bool_(True), -0.01, 2.01, np.nan, np.inf, -np.inf, "0.2", 1j])
def test_invalid_budget_fails_before_any_price_access(bad):
    with pytest.raises(PortfolioError, match="max_turnover"):
        backtest_portfolio(None, None, max_turnover=bad)


def test_hand_calculated_partial_entry_hold_and_partial_exit():
    prices = panel(6)
    prices["AAA"] = [10, 10, 20, 20, 10, 15]
    prices["BBB"] = 20
    targets = orders(prices, {0: [1, 0], 2: [0, 0], 5: [0, 1]})
    book = backtest_portfolio(targets, prices, initial_capital=1000, cost_bps=0, max_turnover=0.25)
    np.testing.assert_allclose(book["ledger"]["nav"], [1000, 1000, 1250, 1250, 1156.25, 1203.125])
    np.testing.assert_allclose(book["ledger"]["cash"], [1000, 750, 750, 1062.5, 1062.5, 1062.5])
    np.testing.assert_allclose(book["holdings"]["AAA"], [0, 25, 25, 9.375, 9.375, 9.375])
    np.testing.assert_allclose(book["requested_trades"]["AAA"], [0, 1000, 0, -500, 0, 0])
    np.testing.assert_allclose(book["trades"]["AAA"], [0, 250, 0, -312.5, 0, 0])
    np.testing.assert_allclose(book["unfilled_trades"]["AAA"], [0, 750, 0, -187.5, 0, 0])
    assert book["execution"]["turnover_limited"].tolist() == [False, True, False, True, False, False]
    assert book["execution"]["fill_fraction"].tolist() == [0, 0.25, 0, 0.625, 0, 0]
    assert book["holdings"]["BBB"].eq(0).all()  # Last signal never executes outside the data.
    metrics = portfolio_metrics(book)
    assert metrics["limited_rebalances"] == 2
    assert metrics["requested_notional"] == 1500 and metrics["unfilled_notional"] == 937.5


@pytest.mark.parametrize("cost", [0, 12, 9000])
@pytest.mark.parametrize("cap", [0.0, 0.05, 0.4, 2.0])
def test_random_rotations_reconcile_cash_fees_all_legs_and_budget(cost, cap):
    rng = np.random.Generator(np.random.PCG64(31))
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.025, (80, 3)), axis=0)),
        index=pd.date_range("2024-01-02", periods=80, name="date"),
        columns=["A", "B", "C"],
    )
    weights = rng.dirichlet(np.ones(4), len(prices))[:, :3]
    weights[:20] = [1, 0, 0]  # Build a funded holding, then rotate.
    weights[20:30] = [0, 1, 0]
    targets = pd.DataFrame(weights, index=prices.index, columns=prices.columns)
    targets.iloc[::7] = np.nan
    before_prices, before_targets = prices.copy(), targets.copy()
    book = backtest_portfolio(targets, prices, cost_bps=cost, max_turnover=cap, cash_rate=0.03)
    ledger, trades, audit = book["ledger"], book["trades"], book["execution"]
    assert (ledger["turnover"] <= cap + 1e-12).all()
    assert ledger["cash"].ge(0).all() and book["holdings"].ge(0).all().all()
    np.testing.assert_allclose(ledger["nav"], ledger["cash"] + book["asset_values"].sum(axis=1))
    np.testing.assert_allclose(ledger["nav"], ledger["pre_trade_nav"] - ledger["transaction_cost"])
    np.testing.assert_allclose(ledger["transaction_cost"], trades.abs().sum(axis=1) * cost / 10000, atol=1e-10)
    np.testing.assert_allclose(trades, book["requested_trades"].mul(audit["fill_fraction"], axis=0), atol=1e-10)
    np.testing.assert_allclose(book["requested_trades"], trades + book["unfilled_trades"], atol=1e-10)
    np.testing.assert_allclose(
        audit["requested_notional"], audit["filled_notional"] + audit["unfilled_notional"], atol=1e-10
    )
    np.testing.assert_allclose(
        ledger["cash"].diff().iloc[1:],
        (ledger["cash_interest"] - trades.sum(axis=1) - ledger["transaction_cost"]).iloc[1:],
        atol=1e-8,
    )
    np.testing.assert_allclose(book["asset_values"], book["holdings"] * prices)
    pd.testing.assert_frame_equal(prices, before_prices)
    pd.testing.assert_frame_equal(targets, before_targets)


def test_zero_budget_preserves_cash_interest_and_is_not_a_filled_rebalance():
    prices = panel(3).set_axis(pd.DatetimeIndex(["2020-01-01", "2020-12-31", "2021-12-31"], name="date"))
    book = backtest_portfolio(
        orders(prices, {0: [1, 0], 1: [0, 1]}), prices, initial_capital=100, cash_rate=0.1, max_turnover=0
    )
    np.testing.assert_allclose(book["ledger"]["cash"], [100, 110, 121])
    assert book["holdings"].eq(0).all().all()
    assert book["execution"]["filled_notional"].eq(0).all()
    assert book["ledger"]["rebalance_executed"].sum() == 2  # Processed instructions, not filled-trade count.
    assert book["execution"]["turnover_limited"].sum() == 2


def test_budget_uses_nav_after_marking_and_cash_interest():
    prices = panel(3).set_axis(pd.DatetimeIndex(["2020-01-01", "2020-12-31", "2021-12-31"], name="date"))
    book = backtest_portfolio(
        orders(prices, {0: [1, 0]}), prices, initial_capital=1000, cash_rate=0.1, max_turnover=0.2, cost_bps=100
    )
    assert book["ledger"]["pre_trade_nav"].iloc[1] == pytest.approx(1100)
    assert book["trades"].iloc[1].sum() == pytest.approx(220)
    assert book["ledger"]["transaction_cost"].iloc[1] == pytest.approx(2.2)


@pytest.mark.parametrize("lag", [1, 2, 4])
def test_causal_prefix_column_order_and_future_changes_preserve_entire_book(lag):
    prices = panel()
    targets = orders(prices, {0: [1, 0], 2: [0, 1], 4: [0.5, 0.25], 7: [0, 0]})
    options = {"max_turnover": 0.15, "cost_bps": 150, "execution_lag": lag}
    book = backtest_portfolio(targets, prices, **options)
    prefix = backtest_portfolio(targets.iloc[:8], prices.iloc[:8], **options)
    future_prices, future_targets = prices.copy(), targets.copy()
    future_prices.iloc[8:] *= 20
    future_targets.iloc[8:] = [0, 1]
    mutated = backtest_portfolio(future_targets, future_prices, **options)
    permuted = backtest_portfolio(targets.iloc[:, ::-1], prices.iloc[:, ::-1], **options)
    for name, frame in book.items():
        pd.testing.assert_frame_equal(frame.iloc[:8], prefix[name])
        pd.testing.assert_frame_equal(frame.iloc[:8], mutated[name].iloc[:8])
        pd.testing.assert_frame_equal(frame, permuted[name])


def test_nonbinding_budget_is_identical_to_default_and_explicit_none():
    prices = panel()
    targets = orders(prices, {0: [1, 0], 1: [0, 1], 3: [0, 0], 4: [0.2, 0.3]})
    books = [
        backtest_portfolio(targets, prices, **options) for options in ({}, {"max_turnover": None}, {"max_turnover": 2})
    ]
    for name, expected in books[0].items():
        for other in books[1:]:
            pd.testing.assert_frame_equal(expected, other[name])


def test_satisfied_zero_target_is_processed_without_being_limited():
    prices = panel()
    book = backtest_portfolio(orders(prices, {0: [0, 0]}), prices, max_turnover=0)
    assert book["execution"]["fill_fraction"].iloc[1] == 1
    assert not book["execution"]["turnover_limited"].any()
    assert not book["execution"]["instruction_processed"].iloc[0]


def test_test_anchor_clears_all_flow_diagnostics_and_keeps_carried_holdings():
    prices = panel()
    targets = orders(prices, {0: [1, 0], 1: [0, 1], 2: [0.5, 0.5], 4: [0, 0]})
    book = backtest_portfolio(targets, prices, max_turnover=0.2)
    original = {key: frame.copy(deep=True) for key, frame in book.items()}
    anchor = prices.index[2]
    result = _test_books({"result": book, "benchmark": book, "plan": {}, "eligibility": None}, anchor)
    for account in ("result", "benchmark"):
        test = result[account]
        assert (test["execution"].iloc[0] == 0).all()
        assert test["requested_trades"].iloc[0].eq(0).all() and test["unfilled_trades"].iloc[0].eq(0).all()
        pd.testing.assert_series_equal(test["holdings"].iloc[0], book["holdings"].loc[anchor])
        pd.testing.assert_frame_equal(test["execution"].iloc[1:], book["execution"].loc[anchor:].iloc[1:])
        metrics = portfolio_metrics(test)
        assert metrics["unfilled_notional"] == pytest.approx(book["execution"]["unfilled_notional"].iloc[3:].sum())
    for name, frame in original.items():
        pd.testing.assert_frame_equal(frame, book[name])


def test_independent_exact_oracle_and_all_frozen_cells():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/portfolio_turnover_reference_oracle.py")],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    raw = (root / "momentum_lab/benchmark_data/portfolio_turnover_reference_v1.json").read_bytes()
    assert json.loads(result.stdout) == json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == frozen.PORTFOLIO_TURNOVER_REFERENCE_SHA256
    assert frozen.check_portfolio_turnover_reference() == {"status": "passed", "cases": 8, "schema_version": 1}


@pytest.mark.parametrize("mutation", ["hash", "cash", "fill", "columns"])
def test_frozen_gate_detects_tamper_or_changed_execution(monkeypatch, mutation):
    if mutation == "hash":
        monkeypatch.setattr(frozen, "PORTFOLIO_TURNOVER_REFERENCE_SHA256", "0" * 64)
    else:
        original = frozen.backtest_portfolio

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            if mutation == "columns":
                result["execution"] = result["execution"].iloc[:, ::-1]
            elif mutation == "cash":
                result["ledger"].iloc[1, 2] += 1
            else:
                result["unfilled_trades"].iloc[1, 0] += 1
            return result

        monkeypatch.setattr(frozen, "backtest_portfolio", changed)
    with pytest.raises(PortfolioError, match="Frozen portfolio turnover"):
        frozen.check_portfolio_turnover_reference()


def test_turnover_benchmark_cli_and_failure_exit(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["momentum-lab", "portfolio", "benchmark", "--turnover"])
    assert cli.main() == 0
    assert "passed (8 cases)" in capsys.readouterr().out
    monkeypatch.setattr(frozen, "PORTFOLIO_TURNOVER_REFERENCE_SHA256", "0" * 64)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
