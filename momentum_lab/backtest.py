"""backtest.py - Vectorized backtest engine and evaluation metrics.

All functions accept pandas Series indexed by date. The backtest engine
uses previous-day positions to compute current-day returns (no look-ahead bias).
"""

import numpy as np
import pandas as pd

RISK_FREE_RATE: float = 0.0


def backtest(
    positions: pd.Series,
    prices: pd.Series,
    cost_bps: float = 1.0,
    vol_target: float | None = None,
    vol_lookback: int = 21,
    max_leverage: float = 2.0,
    annualization: float = 252,
    financing_rate: float = 0.0,
    borrow_bps: float = 0.0,
    slippage_bps: float = 0.0,
    cash_rate: float = 0.0,
    short_rebate_rate: float = 0.0,
    execution_lag: int = 0,
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
        financing_rate: Annual financing rate applied to absolute exposure
            above 1x, as a decimal (for example, 0.05 for 5%).
        borrow_bps: Annualized borrow fee for short exposure, in basis points.
        slippage_bps: Additional transaction slippage in basis points.
        cash_rate: Annual return earned by uninvested cash, as a decimal.
        short_rebate_rate: Annual rebate earned on short-sale collateral.
        execution_lag: Whole bars between observing a target and executing it.
            ``0`` models a same-close/MOC fill; ``1`` models a next-close fill.
            Search runs default to ``1`` so close-derived signals are not
            assumed to trade at the close that created them.

    Returns:
        dict with 'returns', 'equity', 'trades' keys.
    """
    if annualization <= 0:
        raise ValueError("annualization must be positive")
    if max_leverage <= 0:
        raise ValueError("max_leverage must be positive")
    if any(v < 0 for v in (cost_bps, borrow_bps, slippage_bps, financing_rate)):
        raise ValueError("cost, financing and slippage parameters cannot be negative")
    if not np.isfinite(cash_rate) or not np.isfinite(short_rebate_rate):
        raise ValueError("cash_rate and short_rebate_rate must be finite")
    if cash_rate <= -1 or short_rebate_rate <= -1:
        raise ValueError("cash_rate and short_rebate_rate must be greater than -1")
    if isinstance(execution_lag, bool) or not isinstance(execution_lag, (int, np.integer)) or execution_lag < 0:
        raise ValueError("execution_lag must be a non-negative integer")

    if prices.empty:
        empty = prices.astype(float).copy()
        return {"returns": empty, "equity": empty.copy(), "trades": empty.copy()}
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

    transaction_rate = (cost_bps + slippage_bps) / 10000.0
    annual_borrow_rate = borrow_bps / 10000.0
    strategy_returns = pd.Series(0.0, index=prices.index, dtype=float)
    trades = pd.Series(0.0, index=prices.index, dtype=float)
    previous_target = 0.0
    insolvent = False

    # A small stateful ledger is intentional.  Target positions are portfolio
    # weights, so market movement makes the held weight drift between fills.
    # Charging only ``target.diff()`` misses the turnover needed to rebalance a
    # constant fractional target (for example 50% invested).
    for i, idx in enumerate(prices.index):
        if insolvent:
            strategy_returns.iloc[i] = 0.0
            trades.iloc[i] = 0.0
            continue

        asset_return = float(returns.iloc[i])
        fraction = float(elapsed_years.iloc[i])
        held = previous_target
        long_exposure = max(held, 0.0)
        short_exposure = max(-held, 0.0)
        base_cash = max(1.0 - long_exposure, 0.0)
        borrowed_exposure = max(abs(held) - 1.0, 0.0)

        cash_pnl = base_cash * _period_rate(cash_rate, fraction)
        rebate_pnl = short_exposure * _period_rate(short_rebate_rate, fraction)
        financing_cost = borrowed_exposure * _period_rate(financing_rate, fraction)
        borrow_cost = short_exposure * _period_rate(annual_borrow_rate, fraction)
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
            turnover = abs(desired - drifted_weight)
            period_return = pre_trade_return - turnover * transaction_rate
            if period_return <= -1.0:
                period_return = -1.0
                insolvent = True

        trades.iloc[i] = turnover
        strategy_returns.iloc[i] = period_return
        previous_target = 0.0 if insolvent else desired

    equity = (1.0 + strategy_returns).cumprod().clip(lower=0.0)

    return {
        "returns": strategy_returns,
        "equity": equity,
        "trades": trades,
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
