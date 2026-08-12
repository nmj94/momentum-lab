"""Basic tests for momentum-lab."""

from momentum_lab.data import download_data, compute_features, prepare_data
from momentum_lab.backtest import backtest, evaluate, get_buy_and_hold
from momentum_lab.strategies import get_strategy, list_strategies, STRATEGY_REGISTRY
import pandas as pd
import numpy as np


def test_download_data():
    """Test data download with a small date range."""
    df = download_data("GLD", start="2024-01-01", end="2024-06-01", use_cache=False)
    assert len(df) > 100
    assert "close" in df.columns
    assert df["close"].iloc[0] > 0


def test_compute_features():
    """Test feature computation."""
    df = pd.DataFrame({"close": np.random.randn(300).cumsum() + 100,
                       "high": np.random.randn(300).cumsum() + 101,
                       "low": np.random.randn(300).cumsum() + 99,
                       "volume": np.random.randint(1000, 10000, 300)})
    feats = compute_features(df)
    assert len(feats) == len(df)
    assert "ret_5" in feats.columns
    assert "rsi_14" in feats.columns
    assert "vol_21" in feats.columns


def test_backtest():
    """Test backtest engine."""
    prices = pd.Series(np.random.randn(100).cumsum() + 100)
    positions = pd.Series(1.0, index=prices.index)
    result = backtest(positions, prices, cost_bps=1.0)
    assert "returns" in result
    assert "equity" in result
    assert len(result["returns"]) == len(prices)


def test_evaluate():
    """Test evaluation metrics."""
    returns = pd.Series(np.random.randn(252) * 0.01)
    metrics = evaluate(returns)
    assert "sharpe" in metrics
    assert "max_drawdown" in metrics
    assert "cagr" in metrics
    assert isinstance(metrics["sharpe"], float)


def test_strategies_registry():
    """Test strategy registry."""
    assert len(STRATEGY_REGISTRY) >= 26
    assert "tsmom" in STRATEGY_REGISTRY
    assert "regime_aware" in STRATEGY_REGISTRY
    assert "ml_xgb" in STRATEGY_REGISTRY


def test_get_strategy():
    """Test strategy instantiation."""
    s = get_strategy("tsmom")
    assert s.name == "tsmom"
    combos = s.get_param_combinations()
    assert len(combos) > 0


def test_tsmom_generate_positions():
    """Test TSMOM signal generation."""
    close = pd.Series(np.random.randn(300).cumsum() + 100)
    data = {"close": close, "high": close, "low": close, "open": close,
            "volume": pd.Series(1, index=close.index)}
    s = get_strategy("tsmom")
    pos = s.generate_positions(data, lookback=21, threshold=0.0, long_short=True)
    assert len(pos) == len(close)
    assert pos.isin([0.0, 1.0, -1.0]).all()


def test_regime_aware():
    """Test RegimeAware strategy."""
    close = pd.Series(np.random.randn(300).cumsum() + 100)
    data = {"close": close, "high": close + 1, "low": close - 1, "open": close,
            "volume": pd.Series(1, index=close.index)}
    s = get_strategy("regime_aware")
    pos = s.run(data, adx_trend_threshold=15, mom_lookback=63,
                vol_target_normal=0.12, position_size=2.0)
    assert len(pos) == len(close)
