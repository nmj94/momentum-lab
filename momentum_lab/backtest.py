"""backtest.py - Vectorized backtest engine and evaluation metrics.

All functions accept pandas Series indexed by date. The backtest engine
uses previous-day positions to compute current-day returns (no look-ahead bias).
"""

import numpy as np
import pandas as pd

RISK_FREE_RATE: float = 0.04


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
) -> dict:
    """Run a vectorized backtest.

    Args:
        positions: pd.Series of daily target positions (+1 long, 0 flat, -1 short).
        prices: pd.Series of daily close prices.
        cost_bps: Transaction cost in basis points per unit traded.
        vol_target: If not None, scale positions to target annualized volatility.
        vol_lookback: Lookback for volatility calculation.
        max_leverage: Maximum leverage cap.
        annualization: Number of return periods per year (252 for trading days,
            365 for continuously traded daily assets).
        financing_rate: Annual financing rate applied to held exposure, as a
            decimal (for example, 0.05 for 5%).
        borrow_bps: Annualized borrow fee for short exposure, in basis points.
        slippage_bps: Additional transaction slippage in basis points.

    Returns:
        dict with 'returns', 'equity', 'trades' keys.
    """
    if annualization <= 0:
        raise ValueError("annualization must be positive")
    if any(v < 0 for v in (cost_bps, borrow_bps, slippage_bps)):
        raise ValueError("cost and slippage parameters cannot be negative")

    if prices.empty:
        empty = prices.astype(float).copy()
        return {"returns": empty, "equity": empty.copy(), "trades": empty.copy()}
    if not np.isfinite(prices.to_numpy(dtype=float)).all() or (prices <= 0).any():
        raise ValueError("prices must be finite and positive")

    positions = positions.reindex(prices.index).ffill().fillna(0)
    returns = prices.pct_change().fillna(0)

    if vol_target is not None:
        realized_vol = returns.rolling(vol_lookback).std() * np.sqrt(annualization)
        # Stay flat until the rolling volatility estimate is available.  Leaving
        # the warm-up as NaN contaminates trades, returns, and cumulative equity.
        scaling = (vol_target / (realized_vol + 1e-10)).fillna(0.0)
        positions = positions * scaling
        positions = positions.clip(-max_leverage, max_leverage)

    trades = positions.diff().abs()
    trades.iloc[0] = abs(positions.iloc[0])
    cost = trades * ((cost_bps + slippage_bps) / 10000.0)

    held_positions = positions.shift(1).fillna(0)
    financing = held_positions.abs() * (financing_rate / annualization)
    borrow = held_positions.clip(upper=0).abs() * (borrow_bps / 10000.0 / annualization)

    # Fill the first price return before subtracting costs so an initial
    # position incurs its entry cost on the first bar instead of being
    # accidentally erased by a later fillna(0).
    strategy_returns = held_positions * returns - cost - financing - borrow

    equity = (1 + strategy_returns).cumprod()

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
    equity_peak = equity.cummax()
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


def evaluate_strategy(positions: pd.Series, prices: pd.Series, **bt_kwargs) -> dict:
    """Backtest + evaluate in one step."""
    result = backtest(positions, prices, **bt_kwargs)
    result["metrics"] = evaluate(result["returns"], annualization=bt_kwargs.get("annualization", 252))
    return result


def get_buy_and_hold(prices: pd.Series) -> pd.Series:
    """Buy and hold benchmark positions (always +1)."""
    return pd.Series(1.0, index=prices.index)
