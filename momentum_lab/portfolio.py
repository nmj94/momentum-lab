"""Causal cross-sectional signals and a self-financing long-only portfolio book.

This is a separate accounting model (schema 1), not an average of independent
single-asset backtests. NaN target rows mean HOLD SHARES, not daily rebalancing.
"""

import hashlib
import json
import math
import re
from importlib import resources
from numbers import Real

import numpy as np
import pandas as pd

PORTFOLIO_ENGINE_SCHEMA = 1
MAX_PORTFOLIO_ASSETS = 64
MAX_PORTFOLIO_CELLS = 1_000_000
REBALANCE_FREQUENCIES = ("daily", "weekly", "monthly")
_SYMBOL = re.compile(r"[A-Za-z0-9^][A-Za-z0-9._^=-]{0,63}")


class PortfolioError(ValueError):
    """An unsupported or invalid portfolio-research input."""


def _number(value, name, *, minimum=None, exclusive_minimum=None, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise PortfolioError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise PortfolioError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise PortfolioError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise PortfolioError(f"{name} must be at least {minimum}")
    if exclusive_minimum is not None and value <= exclusive_minimum:
        raise PortfolioError(f"{name} must be greater than {exclusive_minimum}")
    if maximum is not None and value > maximum:
        raise PortfolioError(f"{name} cannot exceed {maximum}")
    return float(value)


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise PortfolioError(f"{name} must be an integer of at least {minimum}")
    return int(value)


def _symbols(columns):
    if not all(isinstance(name, str) and _SYMBOL.fullmatch(name) for name in columns):
        raise PortfolioError("Asset names must be canonical tickers beginning with a letter, digit or '^'")
    normalized = [name.upper() for name in columns]
    if len(set(normalized)) != len(normalized):
        raise PortfolioError("Duplicate portfolio assets (including case aliases)")
    return normalized


def _prices(prices):
    if not isinstance(prices, pd.DataFrame) or prices.empty or not 1 <= len(prices.columns) <= MAX_PORTFOLIO_ASSETS:
        raise PortfolioError(f"prices must be a non-empty DataFrame with 1-{MAX_PORTFOLIO_ASSETS} assets")
    if prices.size > MAX_PORTFOLIO_CELLS:
        raise PortfolioError(f"Portfolio exceeds the {MAX_PORTFOLIO_CELLS}-cell work limit")
    index = prices.index
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.hasnans
        or index.tz is not None
        or not index.is_monotonic_increasing
        or not index.is_unique
        or not index.equals(index.normalize())
    ):
        raise PortfolioError("Prices require sorted unique timezone-free daily session dates")
    if any(
        not pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_complex_dtype(dtype)
        for dtype in prices.dtypes
    ):
        raise PortfolioError("Prices must contain real numeric, non-boolean values")
    result = prices.copy()
    result.columns = _symbols(result.columns)
    result = result.sort_index(axis=1).astype(float)
    if not np.isfinite(result.to_numpy()).all() or (result.to_numpy() <= 0).any():
        raise PortfolioError("Portfolio prices must be finite and positive; missing sessions are never filled")
    result.index.name = "date"
    return result


def validate_momentum_options(lookback, skip_recent, top_k, rebalance, absolute_threshold, max_weight):
    _integer(lookback, "lookback")
    _integer(skip_recent, "skip_recent", 0)
    _integer(top_k, "top_k")
    if skip_recent >= lookback:
        raise PortfolioError("skip_recent must be smaller than lookback")
    if rebalance not in REBALANCE_FREQUENCIES:
        raise PortfolioError(f"rebalance must be one of {REBALANCE_FREQUENCIES}")
    if absolute_threshold is not None:
        _number(absolute_threshold, "absolute_threshold")
    _number(max_weight, "max_weight", exclusive_minimum=0, maximum=1)


def cross_sectional_momentum(
    prices,
    *,
    lookback=126,
    skip_recent=0,
    top_k=1,
    rebalance="monthly",
    absolute_threshold=0.0,
    max_weight=1.0,
    eligibility=None,
):
    """Rank assets using ``P[t-skip_recent] / P[t-lookback] - 1``.

    Signals use the current/past close only and must execute on a later bar.
    Monthly/weekly signals occur on the first observed session of a new period,
    plus the first fully warmed-up session. Ties break by uppercase ticker.
    Each selected asset gets min(1/top_k, max_weight); empty slots stay in cash.
    The cap is a REBALANCE TARGET cap; actual weights drift between fills.
    Optional boolean membership changes force an additional delayed rebalance.
    """
    validate_momentum_options(lookback, skip_recent, top_k, rebalance, absolute_threshold, max_weight)
    prices = _prices(prices)
    if top_k > len(prices.columns):
        raise PortfolioError("top_k cannot exceed the number of assets")
    if len(prices) <= lookback:
        raise PortfolioError("Not enough observations for the requested lookback")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        scores = prices.shift(skip_recent) / prices.shift(lookback) - 1.0
    if not np.isfinite(scores.iloc[lookback:].to_numpy()).all():
        raise PortfolioError("Momentum calculation produced non-finite scores")
    if eligibility is not None:
        if not isinstance(eligibility, pd.DataFrame) or not eligibility.index.equals(prices.index):
            raise PortfolioError("Eligibility dates must match price dates exactly")
        eligibility = eligibility.copy()
        eligibility.columns = _symbols(eligibility.columns)
        if set(eligibility.columns) != set(prices.columns):
            raise PortfolioError("Eligibility assets must match price assets exactly")
        if any(not pd.api.types.is_bool_dtype(dtype) for dtype in eligibility.dtypes) or eligibility.isna().any().any():
            raise PortfolioError("Eligibility must contain complete boolean values")
        eligibility = eligibility.reindex(columns=prices.columns).astype(bool)
        scores = scores.where(eligibility)
    if rebalance == "daily":
        flags = np.ones(len(prices), dtype=bool)
    else:
        periods = prices.index.to_period("M" if rebalance == "monthly" else "W-SUN")
        flags = np.r_[True, periods[1:] != periods[:-1]]
    if eligibility is not None:
        flags |= eligibility.ne(eligibility.shift()).any(axis=1).to_numpy()
    flags[:lookback] = False
    flags[lookback] = True
    targets = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    weight = min(1.0 / top_k, float(max_weight))
    for row in np.flatnonzero(flags):
        values = scores.iloc[row].dropna()
        eligible = values if absolute_threshold is None else values[values > absolute_threshold]
        # Stable sorting preserves the pre-sorted ticker order for exact ties.
        selected = eligible.sort_values(ascending=False, kind="stable").index[:top_k]
        targets.iloc[row] = 0.0
        targets.loc[prices.index[row], selected] = weight
    return {
        "scores": scores,
        "targets": targets,
        "rebalance": pd.Series(flags, index=prices.index, name="signal_scheduled"),
    }


def validate_execution_options(initial_capital, cost_bps, slippage_bps, spread_bps, cash_rate, execution_lag):
    _number(initial_capital, "initial_capital", exclusive_minimum=0)
    for name, value in (("cost_bps", cost_bps), ("slippage_bps", slippage_bps), ("spread_bps", spread_bps)):
        _number(value, name, minimum=0)
    if cost_bps + slippage_bps + spread_bps / 2 >= 10000:
        raise PortfolioError("Combined one-way costs must be below 10000 bps")
    _number(cash_rate, "cash_rate", exclusive_minimum=-1)
    _integer(execution_lag, "execution_lag")


def _post_cost_nav(nav, current_values, weights, rate):
    """Solve N + rate * sum(abs(weights*N-current_values)) = pre-trade NAV.

    With long-only weights summing to <=1 and rate<1 the root is unique.
    Keep the lower feasible bound so fees are paid from cash, never a loan.
    """
    if rate == 0:
        return nav
    lower, upper = 0.0, nav
    for _ in range(64):
        midpoint = lower + (upper - lower) * 0.5
        if midpoint == lower or midpoint == upper:
            break
        required = midpoint + rate * np.abs(weights * midpoint - current_values).sum()
        if required <= nav:
            lower = midpoint
        else:
            upper = midpoint
    return lower


def backtest_portfolio(
    targets,
    prices,
    *,
    initial_capital=1_000_000.0,
    cost_bps=1.0,
    slippage_bps=0.0,
    spread_bps=0.0,
    cash_rate=0.0,
    execution_lag=1,
):
    """One shared cash/holdings account, with delayed close fills and paid fees.

    Targets must match the price index/assets exactly. A fully NaN row means
    no instruction; partial missing rows are invalid. Finite rows are long-only
    post-cost NAV fractions summing to at most one (1e-12 rounding tolerance).
    Unfilled last-bar instructions are not executed beyond the available data.
    Cash accrues an effective annual rate over actual elapsed days / 365.
    No FX, leverage, shorts, taxes, capacity, minimum fees or asynchronous fills.
    """
    validate_execution_options(initial_capital, cost_bps, slippage_bps, spread_bps, cash_rate, execution_lag)
    prices = _prices(prices)
    if not isinstance(targets, pd.DataFrame) or not targets.index.equals(prices.index):
        raise PortfolioError("Target dates must match price dates exactly; no reindexing/filling")
    targets = targets.copy()
    targets.columns = _symbols(targets.columns)
    if set(targets.columns) != set(prices.columns):
        raise PortfolioError("Target assets must match price assets exactly")
    if any(
        not pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_complex_dtype(dtype)
        for dtype in targets.dtypes
    ):
        raise PortfolioError("Target weights must be real numeric, non-boolean values")
    try:
        targets = targets.reindex(columns=prices.columns).astype(float)
    except (TypeError, ValueError) as exc:
        raise PortfolioError("Target weights must be numeric") from exc
    values = targets.to_numpy(copy=True)
    active = ~np.isnan(values).all(axis=1)
    valid = values[active]
    if not np.isfinite(valid).all() or (valid < 0).any() or (valid.sum(axis=1) > 1.0 + 1e-12).any():
        raise PortfolioError("Targets must be complete long-only rows with total weight <=1, or wholly NaN HOLD rows")
    # Normalize numerical overshoots only; never clip economically invalid
    # weights or renormalize unfilled selection slots to 100% exposure.
    if len(valid):
        values[active] = valid / np.maximum(1.0, valid.sum(axis=1))[:, None]
    scheduled = pd.DataFrame(values, index=prices.index, columns=prices.columns).shift(execution_lag)
    quotes = prices.to_numpy()
    orders = scheduled.to_numpy()
    units = np.zeros(len(prices.columns), dtype=float)
    cash = float(initial_capital)
    previous_nav = cash
    rate = (float(cost_bps) + float(slippage_bps) + float(spread_bps) / 2.0) / 10000.0
    holdings, weight_rows, value_rows, trade_rows, ledger = [], [], [], [], []
    elapsed_days = prices.index.to_series().diff().dt.days.fillna(0).to_numpy(dtype=float)
    for row, quote in enumerate(quotes):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            interest = cash * np.expm1(np.log1p(cash_rate) * elapsed_days[row] / 365.0)
            current_values = units * quote
            asset_pnl = float(np.dot(units, quote - quotes[row - 1])) if row else 0.0
            cash += float(interest)
            pre_nav = cash + float(current_values.sum())
        if not np.isfinite(pre_nav) or pre_nav <= 0:
            raise PortfolioError("Portfolio produced non-finite/non-positive NAV; review input scales and rates")
        traded = np.zeros(len(quote), dtype=float)
        fees = 0.0
        executed = bool(np.isfinite(orders[row]).all())
        if executed:
            desired = orders[row]
            post_nav = _post_cost_nav(pre_nav, current_values, desired, rate)
            new_values = desired * post_nav
            traded = new_values - current_values
            fees = float(np.abs(traded).sum() * rate)
            cash = pre_nav - fees - float(new_values.sum())
            if cash < -1e-12 * pre_nav:
                raise PortfolioError("Self-financing check failed: a rebalance would borrow cash")
            cash = max(0.0, cash)  # At most machine-roundoff from summing filled legs.
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                units = new_values / quote
            if not np.isfinite(units).all():
                raise PortfolioError("Holdings exceed numerical range")
            current_values = new_values
        nav = cash + float(current_values.sum())
        if not np.isfinite(nav) or nav <= 0:
            raise PortfolioError("Post-cost NAV must remain finite and positive")
        weights = current_values / nav
        ledger.append(
            {
                "nav": nav,
                "equity": nav / initial_capital,
                "cash": cash,
                "cash_weight": cash / nav,
                "asset_pnl": asset_pnl,
                "cash_interest": float(interest),
                "pre_trade_nav": pre_nav,
                "transaction_cost": fees,
                "traded_notional": float(np.abs(traded).sum()),
                "turnover": float(np.abs(traded).sum()) / pre_nav,
                "gross_return": pre_nav / previous_nav - 1.0,
                "return": nav / previous_nav - 1.0,
                "rebalance_executed": executed,
            }
        )
        if not all(math.isfinite(value) for value in ledger[-1].values()):
            raise PortfolioError("Portfolio ledger exceeds numerical range")
        holdings.append(units.copy())
        weight_rows.append(weights)
        value_rows.append(current_values.copy())
        trade_rows.append(traded)
        previous_nav = nav
    return {
        "ledger": pd.DataFrame(ledger, index=prices.index),
        "weights": pd.DataFrame(weight_rows, index=prices.index, columns=prices.columns),
        "holdings": pd.DataFrame(holdings, index=prices.index, columns=prices.columns),
        "asset_values": pd.DataFrame(value_rows, index=prices.index, columns=prices.columns),
        "trades": pd.DataFrame(trade_rows, index=prices.index, columns=prices.columns),
        "executed_targets": scheduled,
    }


def portfolio_metrics(result, *, annualization=252.0, risk_free_rate=0.0):
    """Descriptive full-history metrics; no selection/OOS significance claims.

    The initial structural zero return is excluded. Warm-up cash intervals are
    included for both strategies. Undefined Sharpe/volatility is explicit null.
    """
    annualization = _number(annualization, "annualization", exclusive_minimum=0, maximum=366)
    risk_free_rate = _number(risk_free_rate, "risk_free_rate", exclusive_minimum=-1)
    ledger = result["ledger"]
    returns = ledger["return"].iloc[1:]
    equity = ledger["equity"]
    intervals = len(returns)
    std = float(returns.std(ddof=1)) if intervals >= 2 else None
    volatility = std * math.sqrt(annualization) if std is not None else None
    sharpe = (
        (float(returns.mean()) * annualization - risk_free_rate) / volatility
        if volatility and volatility > 1e-12
        else None
    )
    try:
        cagr = math.expm1(math.log(float(equity.iloc[-1])) * annualization / intervals) if intervals else None
    except OverflowError:
        cagr = None
    metrics = {
        "return_intervals": intervals,
        "final_nav": float(ledger["nav"].iloc[-1]),
        "total_return": float(equity.iloc[-1]) - 1.0,
        "cagr": cagr,
        "sharpe": sharpe,
        "volatility": volatility,
        "max_drawdown": float((equity / equity.cummax().clip(lower=1.0) - 1.0).min()),
        "transaction_costs": float(ledger["transaction_cost"].sum()),
        "traded_notional": float(ledger["traded_notional"].sum()),
        "turnover": float(ledger["turnover"].sum()),
        "rebalances": int(ledger["rebalance_executed"].sum()),
        "average_cash_weight": float(ledger["cash_weight"].mean()),
    }
    # Extreme annualization/variance can overflow despite a finite daily book.
    # Such derived statistics are undefined, never non-standard JSON NaN/Infinity.
    return {
        key: None if isinstance(value, float) and not math.isfinite(value) else value for key, value in metrics.items()
    }


def check_portfolio_reference():
    """Compare every ledger/holding/weight/trade cell with a frozen scalar oracle."""
    payload = resources.files("momentum_lab").joinpath("benchmark_data/portfolio_reference_v1.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != PORTFOLIO_REFERENCE_SHA256:
        raise PortfolioError("Frozen portfolio reference SHA-256 mismatch")
    suite = json.loads(payload)
    for case in suite["cases"]:
        index = pd.DatetimeIndex(case["dates"], name="date")
        prices = pd.DataFrame(case["prices"], index=index)
        targets = pd.DataFrame(case["targets"], index=index, columns=prices.columns, dtype=float)
        result = backtest_portfolio(targets, prices, **case["config"])
        for name, expected in case["expected"].items():
            actual = result[name].to_numpy(dtype=float)
            reference = np.asarray(expected, dtype=float)
            if (
                actual.shape != reference.shape
                or not np.isfinite(actual).all()
                or not np.allclose(actual, reference, rtol=1e-11, atol=1e-11)
            ):
                raise PortfolioError(f"Frozen portfolio ledger changed: {case['id']} / {name}")
        if list(result["ledger"].columns) != suite["ledger_columns"]:
            raise PortfolioError("Frozen portfolio ledger columns changed")
    return {"status": "passed", "cases": len(suite["cases"]), "schema_version": PORTFOLIO_ENGINE_SCHEMA}


# Filled from the reviewed independent Fraction-based oracle, never rebaselined
# automatically from production engine output.
PORTFOLIO_REFERENCE_SHA256 = "5732501ee84163d09edb6714c25d814685bbc6cf30b3408af255747c4f1aa6db"
