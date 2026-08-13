"""backtest.py - Vectorized backtest engine and evaluation metrics.

All functions accept pandas Series indexed by date. The backtest engine
uses previous-day positions to compute current-day returns (no look-ahead bias).
"""

import numpy as np
import pandas as pd
from typing import Optional

RISK_FREE_RATE: float = 0.04


def backtest(
    positions: pd.Series,
    prices: pd.Series,
    cost_bps: float = 1.0,
    vol_target: Optional[float] = None,
    vol_lookback: int = 21,
    max_leverage: float = 2.0,
) -> dict:
    """Run a vectorized backtest.

    Args:
        positions: pd.Series of daily target positions (+1 long, 0 flat, -1 short).
        prices: pd.Series of daily close prices.
        cost_bps: Transaction cost in basis points per unit traded.
        vol_target: If not None, scale positions to target annualized volatility.
        vol_lookback: Lookback for volatility calculation.
        max_leverage: Maximum leverage cap.

    Returns:
        dict with 'returns', 'equity', 'trades' keys.
    """
    positions = positions.reindex(prices.index).ffill().fillna(0)
    returns = prices.pct_change()

    if vol_target is not None:
        realized_vol = returns.rolling(vol_lookback).std() * np.sqrt(252)
        scaling = vol_target / (realized_vol + 1e-10)
        positions = positions * scaling
        positions = positions.clip(-max_leverage, max_leverage)

    trades = positions.diff().abs()
    trades.iloc[0] = abs(positions.iloc[0])
    cost = trades * (cost_bps / 10000.0)

    strategy_returns = positions.shift(1) * returns - cost
    strategy_returns = strategy_returns.fillna(0)

    equity = (1 + strategy_returns).cumprod()

    return {
        "returns": strategy_returns,
        "equity": equity,
        "trades": trades,
    }


def evaluate(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> dict:
    """Compute comprehensive evaluation metrics.

    Args:
        returns: pd.Series of daily strategy returns.

    Returns:
        dict of metrics: sharpe, sortino, calmar, max_drawdown, cagr,
        total_return, volatility, win_rate, profit_factor, skew, kurtosis.
    """
    returns = returns.dropna()
    if len(returns) < 2 or returns.std() < 1e-10:
        return {k: 0.0 for k in [
            "sharpe", "sortino", "calmar", "max_drawdown", "cagr",
            "total_return", "volatility", "win_rate", "profit_factor",
            "skew", "kurtosis"
        ]}

    ann = 252
    mean_ret = returns.mean()
    std_ret = returns.std()
    downside_std = returns[returns < 0].std()

    sharpe = (mean_ret * ann - risk_free_rate) / (std_ret * np.sqrt(ann) + 1e-10)
    sortino = (mean_ret * ann - risk_free_rate) / (downside_std * np.sqrt(ann) + 1e-10)

    equity = (1 + returns).cumprod()
    drawdown = (equity / equity.cummax() - 1)
    max_dd = drawdown.min()

    total_return = equity.iloc[-1] - 1
    years = len(returns) / ann
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
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
    positions: pd.Series, prices: pd.Series, **bt_kwargs
) -> dict:
    """Backtest + evaluate in one step."""
    result = backtest(positions, prices, **bt_kwargs)
    result["metrics"] = evaluate(result["returns"])
    return result


def get_buy_and_hold(prices: pd.Series) -> pd.Series:
    """Buy and hold benchmark positions (always +1)."""
    return pd.Series(1.0, index=prices.index)
