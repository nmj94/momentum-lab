"""Basic tests for momentum-lab."""

from momentum_lab.data import download_data, compute_features, prepare_data
from momentum_lab.backtest import backtest, evaluate, get_buy_and_hold
from momentum_lab.strategies import get_strategy, list_strategies, STRATEGY_REGISTRY
import pandas as pd
import numpy as np


def test_download_data():
    """Test data download (falls back to cache if the API is rate-limited)."""
    try:
        df = download_data("GLD", start="2024-01-01", end="2024-06-01", use_cache=True)
    except ValueError:
        return  # no cache + network blocked: skip, not a code bug
    assert len(df) > 100
    assert "close" in df.columns
    assert df["close"].iloc[0] > 0


def test_download_data_range_respected():
    """Cached data must be sliced to the requested [start, end] window."""
    df = download_data("GLD", start="2023-01-01", end="2023-03-01", use_cache=True)
    assert df.index[0] >= pd.Timestamp("2023-01-01")
    assert df.index[-1] <= pd.Timestamp("2023-03-01")
    assert len(df) > 0


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


def test_evaluate_zero_volatility():
    """Evaluate() must not blow up on zero-volatility returns."""
    const = pd.Series([0.01] * 100)
    metrics = evaluate(const)
    assert metrics["sharpe"] == 0.0
    assert metrics["cagr"] == 0.0

    flat = pd.Series([0.0] * 100)
    assert evaluate(flat)["sharpe"] == 0.0


def test_ml_features_present():
    """ML strategies must work even when data already has 'features'."""
    from momentum_lab.strategies import get_strategy
    df = pd.DataFrame(np.random.randn(300, 3).cumsum(axis=0) + 100,
                      columns=["open", "high", "low"])
    df["close"] = df["high"] + df["low"]
    df.index = pd.date_range("2020-01-01", periods=300, freq="B")
    features = compute_features(df)
    data = {c: df[c] for c in df.columns}
    data["features"] = features
    s = get_strategy("ml_logreg")
    pos = s.run(data, lookback=21, forward=1, C=1.0, long_short=True)
    # Walk-forward predictions cover only the out-of-sample tail window.
    assert len(pos) > 0
    assert len(pos) <= len(df)


def test_perturb_params_int_float():
    """Perturbation must nudge ints by +-1 and floats by a fraction."""
    from momentum_lab.robustness import perturb_params
    params = {"regime_confirm": 1, "position_size": 2, "vol_target": 0.10}
    nb = perturb_params(params, frac=0.2)
    values = {k: {n[k] for n in nb} for k in params}
    assert 0 in values["regime_confirm"] and 2 in values["regime_confirm"]
    assert 1 in values["position_size"] and 3 in values["position_size"]
    assert 0.08 in values["vol_target"] or 0.12 in values["vol_target"]


def test_robustness_check_shape():
    """robustness_check returns a valid report dict."""
    from momentum_lab.robustness import robustness_check
    close = pd.Series(np.random.randn(300).cumsum() + 100)
    data = {"close": close, "high": close + 1, "low": close - 1, "open": close,
            "volume": pd.Series(1, index=close.index)}
    df = pd.DataFrame({"close": close}, index=close.index)
    n = len(df); s1 = int(n * 0.6); s2 = int(n * 0.8)
    periods = {"train": (df.index[0], df.index[s1]),
               "val": (df.index[s1], df.index[s2]),
               "test": (df.index[s2], df.index[-1])}
    report = robustness_check(data, df, periods, "tsmom",
                              {"lookback": 21, "threshold": 0.0,
                               "long_short": True, "position_size": 1.0},
                              cost_bps=1.0, frac=0.2)
    assert report["error"] is None
    assert report["grade"] in {"A", "B", "C", "D"}
    assert report["n_neighbors"] > 0
