"""backtest.py - Vectorized backtest engine and evaluation metrics.

All functions accept pandas Series indexed by date. The backtest engine
uses previous-day positions to compute current-day returns (no look-ahead bias).
"""

import numpy as np
import pandas as pd

RISK_FREE_RATE: float = 0.0


def _aligned_numeric_input(value, index, name, *, minimum=None, exclusive_minimum=None):
    """Return a scalar or dated schedule aligned without looking forward."""
    if isinstance(value, pd.Series):
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
            above 1x. A dated Series is forward-filled without look-ahead.
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
    if not np.isfinite(prices.to_numpy(dtype=float)).all() or (prices <= 0).any():
        raise ValueError("prices must be finite and positive")
    if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError("prices index must be sorted and unique")

    positions = positions.reindex(prices.index).ffill().fillna(0)
    if not np.isfinite(positions.to_numpy(dtype=float)).all():
        raise ValueError("positions must be finite")
    returns = prices.pct_change().fillna(0)

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

    cash_rates = _aligned_numeric_input(cash_rate, prices.index, "cash_rate", exclusive_minimum=-1.0)
    financing_rates = _aligned_numeric_input(financing_rate, prices.index, "financing_rate", minimum=0.0)
    financing_spreads = _aligned_numeric_input(financing_spread, prices.index, "financing_spread", minimum=0.0)
    borrow_rates = _aligned_numeric_input(borrow_bps, prices.index, "borrow_bps", minimum=0.0) / 10000.0
    short_rebate_rates = _aligned_numeric_input(
        short_rebate_rate, prices.index, "short_rebate_rate", exclusive_minimum=-1.0
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

    if isinstance(prices.index, pd.DatetimeIndex):
        elapsed_years = prices.index.to_series().diff().dt.total_seconds().div(365.25 * 24 * 60 * 60).fillna(0.0)
        if (elapsed_years < 0).any():
            raise ValueError("prices index must be sorted and unique")
    else:
        elapsed_years = pd.Series(1.0 / annualization, index=prices.index)
        elapsed_years.iloc[0] = 0.0

    def _period_rate(rate, fraction):
        # Financing and borrow inputs are quoted as simple annual rates; use
        # actual elapsed calendar time for dated data and bars/year otherwise.
        return rate * fraction

    strategy_returns = pd.Series(0.0, index=prices.index, dtype=float)
    trades = pd.Series(0.0, index=prices.index, dtype=float)
    requested_trades = pd.Series(0.0, index=prices.index, dtype=float)
    filled_positions = pd.Series(0.0, index=prices.index, dtype=float)
    transaction_costs = pd.Series(0.0, index=prices.index, dtype=float)
    participation = pd.Series(0.0, index=prices.index, dtype=float)
    capacity_constrained = pd.Series(False, index=prices.index, dtype=bool)
    borrow_blocked = pd.Series(False, index=prices.index, dtype=bool)
    previous_target = 0.0
    nav = float(initial_capital)
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
        fraction = float(elapsed_years.iloc[i])
        held = previous_target
        long_exposure = max(held, 0.0)
        short_exposure = max(-held, 0.0)
        base_cash = max(1.0 - long_exposure, 0.0)
        borrowed_exposure = max(abs(held) - 1.0, 0.0)

        cash_pnl = base_cash * _period_rate(float(cash_rates.iloc[i]), fraction)
        rebate_pnl = short_exposure * _period_rate(float(short_rebate_rates.iloc[i]), fraction)
        financing_cost = borrowed_exposure * _period_rate(
            float(financing_rates.iloc[i] + financing_spreads.iloc[i]), fraction
        )
        borrow_cost = short_exposure * _period_rate(float(borrow_rates.iloc[i]), fraction)
        pre_trade_return = held * asset_return + cash_pnl + rebate_pnl - financing_cost - borrow_cost

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
            drifted_weight = held * (1.0 + asset_return) / pre_trade_factor
            if desired < 0.0 and not bool(borrow_schedule.iloc[i]):
                desired = 0.0
                borrow_blocked.iloc[i] = True

            requested_turnover = abs(desired - drifted_weight)
            requested_trades.iloc[i] = requested_turnover
            turnover = requested_turnover
            pre_trade_nav = nav * pre_trade_factor

            if liquidity_enabled and requested_turnover > 0.0:
                dollar_volume = float(prices.iloc[i]) * float(volumes.iloc[i])
                participation_limit = max_participation if max_participation is not None else 1.0
                max_trade_dollars = max(dollar_volume * participation_limit, 0.0)
                requested_dollars = requested_turnover * pre_trade_nav
                traded_dollars = min(requested_dollars, max_trade_dollars)
                turnover = traded_dollars / pre_trade_nav
                participation.iloc[i] = traded_dollars / dollar_volume if dollar_volume > 0.0 else 0.0
                capacity_constrained.iloc[i] = traded_dollars + 1e-12 < requested_dollars

            direction = float(np.sign(desired - drifted_weight))
            filled_weight = drifted_weight + direction * turnover
            filled_positions.iloc[i] = filled_weight

            commission_dollars = turnover * pre_trade_nav * cost_bps / 10000.0
            if turnover > 0.0:
                commission_dollars = max(commission_dollars, min_fee)
            impact_cost_bps = 0.0
            if turnover > 0.0 and impact_bps > 0.0:
                relative_participation = participation.iloc[i] / impact_reference_participation
                impact_cost_bps = impact_bps * relative_participation**impact_exponent
            execution_cost_bps = slippage_bps + float(spreads.iloc[i]) / 2.0 + impact_cost_bps
            execution_cost_dollars = turnover * pre_trade_nav * execution_cost_bps / 10000.0
            transaction_cost = (commission_dollars + execution_cost_dollars) / nav
            transaction_costs.iloc[i] = transaction_cost
            period_return = pre_trade_return - transaction_cost
            if period_return <= -1.0:
                period_return = -1.0
                insolvent = True

        trades.iloc[i] = turnover
        strategy_returns.iloc[i] = period_return
        nav *= 1.0 + period_return
        previous_target = 0.0 if insolvent else filled_positions.iloc[i]

    equity = (1.0 + strategy_returns).cumprod().clip(lower=0.0)

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

    Args:
        returns: pd.Series of daily strategy returns.
        risk_free_rate: Annualized risk-free rate as a decimal.
        annualization: Number of return periods per year.

    Returns:
        dict of metrics: sharpe, sortino, calmar, max_drawdown, cagr,
        total_return, volatility, win_rate, profit_factor, skew, kurtosis.
    """
    if annualization <= 0:
        raise ValueError("annualization must be positive")
    returns = returns.dropna()
    if len(returns) < 2 or returns.std() < 1e-10:
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

    ann = annualization
    mean_ret = returns.mean()
    std_ret = returns.std()
    downside_std = returns[returns < 0].std()
    if pd.isna(downside_std):
        downside_std = 0.0

    sharpe = (mean_ret * ann - risk_free_rate) / (std_ret * np.sqrt(ann) + 1e-10)
    sortino = (mean_ret * ann - risk_free_rate) / (downside_std * np.sqrt(ann) + 1e-10)

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

    return {
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "calmar": round(calmar, 4),
        "max_drawdown": round(max_dd, 4),
        "cagr": round(cagr, 4),
        "total_return": round(total_return, 4),
        "volatility": round(std_ret * np.sqrt(ann), 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "skew": round(returns.skew(), 4),
        "kurtosis": round(returns.kurtosis(), 4),
    }


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
