"""backtest.py - Vectorized backtest engine and evaluation metrics.

All functions accept pandas Series indexed by date. The backtest engine
uses previous-day positions to compute current-day returns (no look-ahead bias).
"""

from numbers import Real

import numpy as np
import pandas as pd

RISK_FREE_RATE: float = 0.0


def _real_series(value, name, *, allow_missing=False, positive=False):
    if not isinstance(value, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if (
        not pd.api.types.is_numeric_dtype(value.dtype)
        or pd.api.types.is_bool_dtype(value.dtype)
        or pd.api.types.is_complex_dtype(value.dtype)
    ):
        raise ValueError(f"{name} must contain real numeric values")
    values = value.to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(values) | (np.isnan(values) if allow_missing else False)
    if not valid.all() or (positive and (values <= 0).any()):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return value.astype(float)


def _aligned_numeric_input(value, index, name, *, minimum=None, exclusive_minimum=None):
    """Return a scalar or dated schedule aligned without looking forward."""
    if isinstance(value, pd.Series):
        value = _real_series(value, name)
        if not value.index.is_monotonic_increasing or not value.index.is_unique:
            raise ValueError(f"{name} schedule index must be sorted and unique")
        try:
            aligned = value.astype(float).reindex(index, method="ffill")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} schedule index is incompatible with prices") from exc
        if aligned.isna().any():
            raise ValueError(f"{name} schedule must cover the first price bar")
    else:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be numeric")
        try:
            scalar = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric or a pandas Series") from exc
        aligned = pd.Series(scalar, index=index, dtype=float)

    values = aligned.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    if minimum is not None and (values < minimum).any():
        raise ValueError(f"{name} cannot be less than {minimum}")
    if exclusive_minimum is not None and (values <= exclusive_minimum).any():
        raise ValueError(f"{name} must be greater than {exclusive_minimum}")
    return aligned


def _period_accrual_input(value, index, name, annualization, *, minimum=None, exclusive_minimum=None):
    """Integrate an annual-rate scalar or effective-dated schedule per bar."""
    aligned = _aligned_numeric_input(value, index, name, minimum=minimum, exclusive_minimum=exclusive_minimum)
    if not isinstance(index, pd.DatetimeIndex):
        result = aligned / float(annualization)
        if len(result):
            result.iloc[0] = 0.0
        return result
    result = pd.Series(0.0, index=index, dtype=float)
    if not isinstance(value, pd.Series):
        elapsed = index.to_series().diff().dt.total_seconds().div(365.25 * 24 * 60 * 60).fillna(0.0)
        return aligned * elapsed
    schedule = value.astype(float)
    if not isinstance(schedule.index, pd.DatetimeIndex) or schedule.index.tz != index.tz:
        raise ValueError(f"{name} schedule index is incompatible with prices")
    year_seconds = 365.25 * 24 * 60 * 60
    for i in range(1, len(index)):
        left, right = index[i - 1], index[i]
        changes = schedule.loc[(schedule.index > left) & (schedule.index < right)]
        cursor = left
        rate = float(schedule.loc[:left].iloc[-1])
        accrued = 0.0
        for effective_at, next_rate in changes.items():
            accrued += rate * (effective_at - cursor).total_seconds() / year_seconds
            cursor, rate = effective_at, float(next_rate)
        accrued += rate * (right - cursor).total_seconds() / year_seconds
        result.iloc[i] = accrued
    return result


def _execution_cost_dollars(
    trade_dollars,
    dollar_volume,
    cost_bps,
    slippage_bps,
    spread_bps,
    impact_bps,
    impact_exponent,
    impact_reference_participation,
    min_fee,
):
    """Return total execution cost for an absolute currency notional."""
    if trade_dollars <= 0.0:
        return 0.0
    commission = max(trade_dollars * cost_bps / 10000.0, min_fee)
    impact = 0.0
    if impact_bps > 0.0:
        participation = trade_dollars / dollar_volume if dollar_volume > 0.0 else 0.0
        impact = impact_bps * (participation / impact_reference_participation) ** impact_exponent
    return commission + trade_dollars * (slippage_bps + spread_bps / 2.0 + impact) / 10000.0


def _target_trade_dollars(asset_value, pre_trade_nav, desired, cost_function):
    """Solve the signed notional that reaches a post-cost target weight."""
    if desired == 0.0:
        return -asset_value

    def residual(trade):
        return asset_value + trade - desired * (pre_trade_nav - cost_function(abs(trade)))

    initial = residual(0.0)
    if abs(initial) <= 1e-14 * max(pre_trade_nav, 1.0):
        return 0.0
    direction = -1.0 if initial > 0.0 else 1.0
    low, high = 0.0, max(abs(desired * pre_trade_nav - asset_value), pre_trade_nav * 1e-12)
    for _ in range(64):
        if residual(direction * high) * initial <= 0.0:
            break
        high *= 2.0
        if not np.isfinite(high) or high > pre_trade_nav * 1e6:
            raise ValueError("transaction-cost model cannot reach the requested target")
    else:
        raise ValueError("transaction-cost model cannot reach the requested target")
    for _ in range(80):
        middle = (low + high) / 2.0
        if residual(direction * middle) * initial > 0.0:
            low = middle
        else:
            high = middle
    candidate = direction * high
    post_cost_nav = pre_trade_nav - cost_function(abs(candidate))
    if post_cost_nav <= 0.0:
        return 0.0
    achieved_error = abs((asset_value + candidate) / post_cost_nav - desired)
    no_trade_error = abs(asset_value / pre_trade_nav - desired)
    return candidate if achieved_error < no_trade_error else 0.0


def _aligned_bool_input(value, index, name):
    """Align a boolean availability schedule without backfilling future data."""
    if isinstance(value, pd.Series):
        if not value.index.is_monotonic_increasing or not value.index.is_unique:
            raise ValueError(f"{name} schedule index must be sorted and unique")
        try:
            aligned = value.reindex(index, method="ffill")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} schedule index is incompatible with prices") from exc
        if aligned.isna().any():
            raise ValueError(f"{name} schedule must cover the first price bar")
        invalid = ~aligned.isin([True, False, 0, 1])
        if invalid.any():
            raise ValueError(f"{name} must contain only boolean values")
        return aligned.astype(bool)
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be boolean or a pandas Series")
    return pd.Series(bool(value), index=index, dtype=bool)


def backtest(
    positions: pd.Series,
    prices: pd.Series,
    cost_bps: float = 1.0,
    vol_target: float | None = None,
    vol_lookback: int = 21,
    max_leverage: float = 2.0,
    annualization: float = 252,
    financing_rate: float | pd.Series = 0.0,
    borrow_bps: float | pd.Series = 0.0,
    slippage_bps: float = 0.0,
    cash_rate: float | pd.Series = 0.0,
    short_rebate_rate: float | pd.Series = 0.0,
    execution_lag: int = 0,
    financing_spread: float | pd.Series = 0.0,
    borrow_available: bool | pd.Series = True,
    spread_bps: float | pd.Series = 0.0,
    impact_bps: float = 0.0,
    impact_exponent: float = 0.5,
    impact_reference_participation: float = 0.01,
    max_participation: float | None = None,
    volume: pd.Series | None = None,
    initial_capital: float = 1_000_000.0,
    min_fee: float = 0.0,
) -> dict:
    """Run a vectorized backtest.

    Args:
        positions: pd.Series of daily target positions (+1 long, 0 flat, -1 short).
        prices: pd.Series of daily close prices.
        cost_bps: Transaction cost in basis points per unit traded.
        vol_target: If not None, scale positions to target annualized volatility.
        vol_lookback: Lookback for volatility calculation.
        max_leverage: Final absolute exposure cap, applied to every strategy.
        annualization: Number of return periods per year (252 for trading days,
            365 for continuously traded daily assets).
        financing_rate: Annual base financing rate applied to absolute exposure
            above 1x. A dated Series is integrated from each effective timestamp
            without applying later values retroactively.
        borrow_bps: Annualized short borrow fee in basis points. May be a dated
            Series for time-varying borrow conditions.
        slippage_bps: Additional transaction slippage in basis points.
        cash_rate: Annual return earned by uninvested cash. May be a Series.
        short_rebate_rate: Annual short-collateral rebate. May be a Series.
        execution_lag: Whole bars between observing a target and executing it.
            ``0`` models a same-close/MOC fill; ``1`` models a next-close fill.
            Search runs default to ``1`` so close-derived signals are not
            assumed to trade at the close that created them.
        financing_spread: Annual spread added to ``financing_rate`` for exposure
            above 1x. May be a dated Series.
        borrow_available: Boolean or dated boolean Series. A false value blocks
            new short targets and requests a cover of an existing short.
        spread_bps: Quoted full bid/ask spread in basis points. Each unit traded
            pays half the spread. May be a dated Series.
        impact_bps: One-way market-impact cost at
            ``impact_reference_participation``. Zero disables impact.
        impact_exponent: Positive exponent applied to participation; total
            impact cost is nonlinear in order size when this is positive.
        impact_reference_participation: Participation rate at which
            ``impact_bps`` is quoted.
        max_participation: Optional maximum fraction of bar dollar volume that
            may be traded. Capacity-constrained orders are filled gradually.
        volume: Bar share/unit volume aligned to ``prices``. Required when
            impact or participation constraints are enabled.
        initial_capital: Starting NAV in currency units, used for capacity and
            minimum-fee calculations.
        min_fee: Minimum currency fee charged on each non-zero rebalance.

    Returns:
        dict containing returns/equity, actual and requested turnover, filled
        positions, transaction costs, participation, and constraint flags.
    """
    if isinstance(annualization, (bool, np.bool_)) or not np.isfinite(annualization) or annualization <= 0:
        raise ValueError("annualization must be positive and finite")
    if isinstance(max_leverage, (bool, np.bool_)) or not np.isfinite(max_leverage) or max_leverage <= 0:
        raise ValueError("max_leverage must be positive and finite")
    scalar_nonnegative = {
        "cost_bps": cost_bps,
        "slippage_bps": slippage_bps,
        "impact_bps": impact_bps,
        "min_fee": min_fee,
    }
    for name, value in scalar_nonnegative.items():
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if isinstance(initial_capital, (bool, np.bool_)) or not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be finite and positive")
    if isinstance(impact_exponent, (bool, np.bool_)) or not np.isfinite(impact_exponent) or impact_exponent <= 0:
        raise ValueError("impact_exponent must be finite and positive")
    if (
        isinstance(impact_reference_participation, (bool, np.bool_))
        or not np.isfinite(impact_reference_participation)
        or not 0 < impact_reference_participation <= 1
    ):
        raise ValueError("impact_reference_participation must be in (0, 1]")
    if max_participation is not None and (
        isinstance(max_participation, (bool, np.bool_))
        or not np.isfinite(max_participation)
        or not 0 < max_participation <= 1
    ):
        raise ValueError("max_participation must be in (0, 1]")
    if isinstance(execution_lag, bool) or not isinstance(execution_lag, (int, np.integer)) or execution_lag < 0:
        raise ValueError("execution_lag must be a non-negative integer")
    if vol_target is not None:
        if (
            isinstance(vol_target, (bool, np.bool_))
            or not isinstance(vol_target, Real)
            or not np.isfinite(vol_target)
            or vol_target < 0
        ):
            raise ValueError("vol_target must be finite and non-negative")
        if (
            isinstance(vol_lookback, (bool, np.bool_))
            or not isinstance(vol_lookback, (int, np.integer))
            or vol_lookback < 2
        ):
            raise ValueError("vol_lookback must be an integer of at least 2")

    prices = _real_series(prices, "prices", positive=True)
    positions = _real_series(positions, "positions", allow_missing=True)
    if prices.empty:
        empty = prices.astype(float).copy()
        return {
            "returns": empty,
            "equity": empty.copy(),
            "trades": empty.copy(),
            "requested_trades": empty.copy(),
            "positions": empty.copy(),
            "transaction_costs": empty.copy(),
            "participation": empty.copy(),
            "capacity_constrained": empty.astype(bool),
            "borrow_blocked": empty.astype(bool),
        }
    if (
        isinstance(prices.index, pd.MultiIndex)
        or prices.index.hasnans
        or (not prices.index.is_monotonic_increasing or not prices.index.is_unique)
    ):
        raise ValueError("prices index must be sorted and unique")

    positions = positions.reindex(prices.index).ffill().fillna(0)
    if not np.isfinite(positions.to_numpy(dtype=float)).all():
        raise ValueError("positions must be finite")
    returns = prices.pct_change().fillna(0)
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("price returns exceed numerical range")

    if vol_target is not None:
        realized_vol = returns.rolling(vol_lookback).std() * np.sqrt(annualization)
        # Stay flat until the rolling volatility estimate is available.  Leaving
        # the warm-up as NaN contaminates trades, returns, and cumulative equity.
        scaling = (vol_target / (realized_vol + 1e-10)).fillna(0.0)
        positions = positions * scaling

    # Strategy-level sizing can be composed (for example internal vol scaling
    # followed by ``position_size``).  Enforce the portfolio risk limit after
    # every transformation so a nominal 2x cap can never become 4x.
    positions = positions.clip(-max_leverage, max_leverage)

    # ``positions`` contains targets observed at each bar.  Delay the target
    # before constructing the execution ledger; a one-bar lag means a signal
    # formed from close[t] is first filled at close[t+1].
    executed_targets = positions.shift(int(execution_lag)).fillna(0.0)

    cash_accruals = _period_accrual_input(cash_rate, prices.index, "cash_rate", annualization, exclusive_minimum=-1.0)
    financing_accruals = _period_accrual_input(
        financing_rate, prices.index, "financing_rate", annualization, minimum=0.0
    )
    financing_spread_accruals = _period_accrual_input(
        financing_spread, prices.index, "financing_spread", annualization, minimum=0.0
    )
    borrow_accruals = (
        _period_accrual_input(borrow_bps, prices.index, "borrow_bps", annualization, minimum=0.0) / 10000.0
    )
    short_rebate_accruals = _period_accrual_input(
        short_rebate_rate, prices.index, "short_rebate_rate", annualization, exclusive_minimum=-1.0
    )
    spreads = _aligned_numeric_input(spread_bps, prices.index, "spread_bps", minimum=0.0)
    borrow_schedule = _aligned_bool_input(borrow_available, prices.index, "borrow_available")

    liquidity_enabled = impact_bps > 0 or max_participation is not None
    if liquidity_enabled:
        if volume is None:
            raise ValueError("volume is required when impact or max_participation is enabled")
        if not isinstance(volume, pd.Series):
            raise ValueError("volume must be a pandas Series")
        volumes = volume.reindex(prices.index).fillna(0.0).astype(float)
        if not np.isfinite(volumes.to_numpy()).all() or (volumes < 0).any():
            raise ValueError("volume must be finite and non-negative")
    else:
        volumes = pd.Series(0.0, index=prices.index, dtype=float)

    strategy_returns = pd.Series(0.0, index=prices.index, dtype=float)
    trades = pd.Series(0.0, index=prices.index, dtype=float)
    requested_trades = pd.Series(0.0, index=prices.index, dtype=float)
    filled_positions = pd.Series(0.0, index=prices.index, dtype=float)
    transaction_costs = pd.Series(0.0, index=prices.index, dtype=float)
    participation = pd.Series(0.0, index=prices.index, dtype=float)
    capacity_constrained = pd.Series(False, index=prices.index, dtype=bool)
    borrow_blocked = pd.Series(False, index=prices.index, dtype=bool)
    nav = float(initial_capital)
    asset_value = 0.0
    insolvent = False

    # A small stateful ledger is intentional.  Target positions are portfolio
    # weights, so market movement makes the held weight drift between fills.
    # Charging only ``target.diff()`` misses the turnover needed to rebalance a
    # constant fractional target (for example 50% invested).
    for i, idx in enumerate(prices.index):
        if insolvent:
            strategy_returns.iloc[i] = 0.0
            trades.iloc[i] = 0.0
            requested_trades.iloc[i] = 0.0
            filled_positions.iloc[i] = 0.0
            continue

        asset_return = float(returns.iloc[i])
        held = asset_value / nav
        long_exposure = max(held, 0.0)
        short_exposure = max(-held, 0.0)
        base_cash = max(1.0 - long_exposure, 0.0)
        borrowed_exposure = max(abs(held) - 1.0, 0.0)

        cash_pnl = base_cash * float(cash_accruals.iloc[i])
        rebate_pnl = short_exposure * float(short_rebate_accruals.iloc[i])
        financing_cost = borrowed_exposure * float(financing_accruals.iloc[i] + financing_spread_accruals.iloc[i])
        borrow_cost = short_exposure * float(borrow_accruals.iloc[i])
        pre_trade_return = held * asset_return + cash_pnl + rebate_pnl - financing_cost - borrow_cost
        if not np.isfinite(pre_trade_return):
            raise ValueError("backtest return exceeds numerical range; review input scales and rates")

        # Normalize the marked-to-market asset leg by pre-trade NAV to obtain
        # its drifted portfolio weight.  If NAV is already gone, liquidate and
        # clamp the path at zero instead of allowing negative equity to revive.
        pre_trade_factor = 1.0 + pre_trade_return
        desired = float(executed_targets.iloc[i])
        if pre_trade_factor <= 0.0:
            turnover = 0.0
            period_return = -1.0
            insolvent = True
        else:
            if desired < 0.0 and not bool(borrow_schedule.iloc[i]):
                desired = 0.0
                borrow_blocked.iloc[i] = True

            pre_trade_nav = nav * pre_trade_factor
            if not np.isfinite(pre_trade_nav) or pre_trade_nav <= 0:
                raise ValueError("backtest NAV exceeds numerical range")

            pre_trade_asset = asset_value * (1.0 + asset_return)
            dollar_volume = float(prices.iloc[i]) * float(volumes.iloc[i])
            current_spread = float(spreads.iloc[i])

            def cost_function(amount, bar_volume=dollar_volume, spread=current_spread):
                return _execution_cost_dollars(
                    amount,
                    bar_volume,
                    cost_bps,
                    slippage_bps,
                    spread,
                    impact_bps,
                    impact_exponent,
                    impact_reference_participation,
                    min_fee,
                )

            requested_trade_dollars = _target_trade_dollars(pre_trade_asset, pre_trade_nav, desired, cost_function)
            requested_dollars = abs(requested_trade_dollars)
            requested_turnover = requested_dollars / pre_trade_nav
            requested_trades.iloc[i] = requested_turnover
            traded_dollars = requested_dollars
            if liquidity_enabled and requested_dollars > 0.0:
                participation_limit = max_participation if max_participation is not None else 1.0
                max_trade_dollars = max(dollar_volume * participation_limit, 0.0)
                traded_dollars = min(requested_dollars, max_trade_dollars)
                participation.iloc[i] = traded_dollars / dollar_volume if dollar_volume > 0.0 else 0.0
                capacity_constrained.iloc[i] = traded_dollars + 1e-12 < requested_dollars
            trade_dollars = float(np.sign(requested_trade_dollars)) * traded_dollars
            turnover = traded_dollars / pre_trade_nav
            execution_cost_dollars = cost_function(traded_dollars)
            asset_value = pre_trade_asset + trade_dollars
            transaction_cost = execution_cost_dollars / nav
            if not np.isfinite(transaction_cost):
                raise ValueError("backtest transaction costs exceed numerical range")
            transaction_costs.iloc[i] = transaction_cost
            period_return = pre_trade_return - transaction_cost
            if period_return <= -1.0:
                period_return = -1.0
                insolvent = True

        trades.iloc[i] = turnover
        strategy_returns.iloc[i] = period_return
        nav *= 1.0 + period_return
        if not np.isfinite(nav) or (nav <= 0 and not insolvent):
            raise ValueError("backtest NAV exceeds numerical range")
        if insolvent:
            asset_value = 0.0
        else:
            filled_positions.iloc[i] = asset_value / nav

    equity = (1.0 + strategy_returns).cumprod().clip(lower=0.0)
    if not np.isfinite(equity.to_numpy()).all():
        raise ValueError("backtest equity exceeds numerical range")

    return {
        "returns": strategy_returns,
        "equity": equity,
        "trades": trades,
        "requested_trades": requested_trades,
        "positions": filled_positions,
        "transaction_costs": transaction_costs,
        "participation": participation,
        "capacity_constrained": capacity_constrained,
        "borrow_blocked": borrow_blocked,
    }


def evaluate(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    annualization: float = 252,
) -> dict:
    """Compute comprehensive evaluation metrics.

    Require finite real returns; missing observations are not silently removed.
    Bankruptcy is absorbing. Constant and one-observation series retain their
    actual performance; undefined Sharpe/Sortino and insufficient-sample moments
    keep the legacy numeric zero sentinel. Search excludes undefined Sharpe.

    Args:
        returns: pd.Series of daily strategy returns.
        risk_free_rate: Annualized risk-free rate as a decimal.
        annualization: Number of return periods per year.

    Returns:
        dict of metrics: sharpe, sortino, calmar, max_drawdown, cagr,
        total_return, volatility, win_rate, profit_factor, skew, kurtosis.
    """
    for name, value in (("annualization", annualization), ("risk_free_rate", risk_free_rate)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{name} must be finite and numeric")
    if annualization <= 0:
        raise ValueError("annualization must be positive")
    returns = _real_series(returns, "returns").copy()
    if returns.empty:
        return {
            k: 0.0
            for k in [
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
            ]
        }

    # The same absorbing-bankruptcy convention as backtest(). In particular,
    # two losses below -100% must never multiply into a positive equity curve.
    losses_at_or_below_nav = np.flatnonzero(returns.to_numpy() <= -1.0)
    if len(losses_at_or_below_nav):
        first_loss = losses_at_or_below_nav[0]
        returns.iloc[first_loss] = -1.0
        returns.iloc[first_loss + 1 :] = 0.0

    ann = annualization
    mean_ret = returns.mean()
    std_ret = returns.std() if len(returns) >= 2 else 0.0
    minimum_acceptable_return = risk_free_rate / ann
    downside = np.minimum(returns.to_numpy(dtype=float) - minimum_acceptable_return, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))))

    # Keep legacy zero sentinels for undefined ratios, but never erase actual
    # gain/loss, drawdown or CAGR merely because variance is zero or n == 1.
    sharpe = (mean_ret * ann - risk_free_rate) / (std_ret * np.sqrt(ann) + 1e-10) if std_ret >= 1e-10 else 0.0
    sortino = (
        (mean_ret * ann - risk_free_rate) / (downside_deviation * np.sqrt(ann)) if downside_deviation >= 1e-10 else 0.0
    )

    equity = (1 + returns).cumprod()
    # Anchor the peak at initial capital 1.0 so first-bar entry costs count as
    # drawdown instead of becoming an artificially lower starting peak.
    equity_peak = equity.cummax().clip(lower=1.0)
    drawdown = equity / equity_peak.where(equity_peak != 0) - 1
    max_dd = drawdown.min()
    if pd.isna(max_dd):
        max_dd = -1.0

    total_return = equity.iloc[-1] - 1
    years = len(returns) / ann
    # Leveraged or short strategies can lose more than 100%, making the
    # compounded return undefined once equity reaches zero or below.  Keep
    # the report numeric and conservatively treat that path as a total loss.
    if years > 0 and (equity > 0).all():
        cagr = (1 + total_return) ** (1 / years) - 1
    else:
        cagr = -1.0
    calmar = cagr / (abs(max_dd) + 1e-10)

    win_rate = (returns > 0).mean()
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = gains / (losses + 1e-10)

    metrics = {
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "calmar": round(calmar, 4),
        "max_drawdown": round(max_dd, 4),
        "cagr": round(cagr, 4),
        "total_return": round(total_return, 4),
        "volatility": round(std_ret * np.sqrt(ann), 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "skew": round(returns.skew(), 4) if len(returns) >= 3 and std_ret >= 1e-10 else 0.0,
        "kurtosis": round(returns.kurtosis(), 4) if len(returns) >= 4 and std_ret >= 1e-10 else 0.0,
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("evaluation metrics exceed numerical range")
    return metrics


def evaluate_strategy(
    positions: pd.Series, prices: pd.Series, risk_free_rate: float = RISK_FREE_RATE, **bt_kwargs
) -> dict:
    """Backtest + evaluate in one step.

    ``risk_free_rate`` is consumed by ``evaluate``; everything else is
    forwarded to ``backtest``.
    """
    result = backtest(positions, prices, **bt_kwargs)
    result["metrics"] = evaluate(
        result["returns"],
        risk_free_rate=risk_free_rate,
        annualization=bt_kwargs.get("annualization", 252),
    )
    return result


def get_buy_and_hold(prices: pd.Series) -> pd.Series:
    """Buy and hold benchmark positions (always +1)."""
    return pd.Series(1.0, index=prices.index)
