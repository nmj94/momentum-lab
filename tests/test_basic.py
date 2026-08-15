"""Basic tests for momentum-lab."""

import numpy as np
import pandas as pd
import pytest

from momentum_lab.backtest import backtest, evaluate
from momentum_lab.data import _cache_covers_range, compute_features, download_data
from momentum_lab.search import _split_periods
from momentum_lab.strategies import STRATEGY_REGISTRY, get_strategy


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


def test_download_data_rejects_reversed_date_range():
    """Invalid date ranges must fail before consulting the cache or network."""
    with pytest.raises(ValueError, match="end must be on or after start"):
        download_data("GLD", start="2024-06-01", end="2024-01-01", use_cache=True)


def test_compute_features():
    """Test feature computation."""
    df = pd.DataFrame(
        {
            "close": np.random.randn(300).cumsum() + 100,
            "high": np.random.randn(300).cumsum() + 101,
            "low": np.random.randn(300).cumsum() + 99,
            "volume": np.random.randint(1000, 10000, 300),
        }
    )
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


def test_backtest_empty_prices_returns_empty_result():
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    result = backtest(empty, empty)
    assert set(result) == {"returns", "equity", "trades"}
    assert all(series.empty for series in result.values())


def test_backtest_cost_model():
    """Financing, borrow and slippage costs must reduce realized returns."""
    prices = pd.Series([100.0, 110.0, 100.0])
    long_positions = pd.Series([1.0, 1.0, 1.0])
    short_positions = pd.Series([-1.0, -1.0, -1.0])
    long_result = backtest(
        long_positions,
        prices,
        cost_bps=0.0,
        slippage_bps=10.0,
        financing_rate=0.252,
        annualization=252,
    )
    short_result = backtest(short_positions, prices, cost_bps=0.0, borrow_bps=252.0, annualization=252)
    assert long_result["returns"].iloc[0] < 0
    assert long_result["returns"].iloc[1] < 0.1
    assert short_result["returns"].iloc[1] < -0.1


def test_cache_coverage_respects_business_day_bounds():
    idx = pd.date_range("2024-01-02", "2024-01-31", freq="B")
    frame = pd.DataFrame({"close": 1.0}, index=idx)
    assert _cache_covers_range(frame, "2024-01-06", "2024-02-01")
    assert not _cache_covers_range(frame, "2023-12-01", "2024-02-01")
    assert not _cache_covers_range(frame, "2024-01-01", None)


def test_search_periods_do_not_overlap():
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    periods = _split_periods(idx)
    assert periods["train"][1] < periods["val"][0]
    assert periods["val"][1] < periods["test"][0]
    assert periods["train"][1] == idx[5]
    assert periods["val"][0] == idx[6]
    assert periods["val"][1] == idx[7]
    assert periods["test"][0] == idx[8]


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


def test_strategy_run_empty_data_returns_empty_positions():
    """Strategies must handle an empty price series without indexing errors."""
    close = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    positions = get_strategy("heikin_ashi").run({"close": close})
    assert positions.empty
    assert positions.index.equals(close.index)


def test_parameter_constraints_remove_incoherent_grids():
    ma_combos = get_strategy("ma_cross").get_param_combinations()
    triple_combos = get_strategy("triple_ma").get_param_combinations()
    accel_combos = get_strategy("acceleration").get_param_combinations()
    assert all(p["fast"] < p["slow"] for p in ma_combos)
    assert all(p["fast"] < p["medium"] < p["slow"] for p in triple_combos)
    assert all(p["short_lb"] < p["long_lb"] for p in accel_combos)


def test_tsmom_generate_positions():
    """Test TSMOM signal generation."""
    close = pd.Series(np.random.randn(300).cumsum() + 100)
    data = {"close": close, "high": close, "low": close, "open": close, "volume": pd.Series(1, index=close.index)}
    s = get_strategy("tsmom")
    pos = s.generate_positions(data, lookback=21, threshold=0.0, long_short=True)
    assert len(pos) == len(close)
    assert pos.isin([0.0, 1.0, -1.0]).all()


def test_regime_aware():
    """Test RegimeAware strategy."""
    close = pd.Series(np.random.randn(300).cumsum() + 100)
    data = {
        "close": close,
        "high": close + 1,
        "low": close - 1,
        "open": close,
        "volume": pd.Series(1, index=close.index),
    }
    s = get_strategy("regime_aware")
    pos = s.run(data, adx_trend_threshold=15, mom_lookback=63, vol_target_normal=0.12, position_size=2.0)
    assert len(pos) == len(close)


def test_evaluate_zero_volatility():
    """Evaluate() must not blow up on zero-volatility returns."""
    const = pd.Series([0.01] * 100)
    metrics = evaluate(const)
    assert metrics["sharpe"] == 0.0
    assert metrics["cagr"] == 0.0

    flat = pd.Series([0.0] * 100)
    assert evaluate(flat)["sharpe"] == 0.0


def test_evaluate_handles_non_positive_equity():
    """Leveraged losses must produce finite failure metrics, not NaN CAGR."""
    metrics = evaluate(pd.Series([-0.5, -1.5, 0.1, 0.1, 0.1]))
    assert metrics["cagr"] == -1.0
    assert np.isfinite(metrics["calmar"])


def test_ml_features_present():
    """ML strategies must work even when data already has 'features'."""
    from momentum_lab.strategies import get_strategy

    df = pd.DataFrame(np.random.randn(300, 3).cumsum(axis=0) + 100, columns=["open", "high", "low"])
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


def test_ml_logreg_l1_uses_elasticnet(monkeypatch):
    """The ML LogReg penalty grid must configure a real L1 estimator."""
    from momentum_lab.strategies import get_strategy

    close = pd.Series([100.0, 101.0, 99.0], dtype=float)
    data = {
        "close": close,
        "features": pd.DataFrame({"feature": [0.0, 1.0, 0.5]}),
    }
    strategy = get_strategy("ml_logreg")
    captured = {}

    def capture_model(feats, label, model_fn, **kwargs):
        captured["model"] = model_fn()
        return pd.Series(np.nan, index=feats.index)

    monkeypatch.setattr(strategy, "_walk_forward", capture_model)
    strategy.generate_positions(data, lookback=21, forward=1, penalty="l1")

    params = captured["model"].get_params()
    assert params["penalty"] == "elasticnet"
    assert params["l1_ratio"] == 1.0


def test_ml_logreg_uses_configured_lookback_for_training_window(monkeypatch):
    """ML lookback must control the walk-forward warm-up window."""
    close = pd.Series([100.0, 101.0, 99.0], dtype=float)
    data = {
        "close": close,
        "features": pd.DataFrame({"feature": [0.0, 1.0, 0.5]}),
    }
    strategy = get_strategy("ml_logreg")
    captured = {}

    def capture_model(feats, label, model_fn, **kwargs):
        captured.update(kwargs)
        return pd.Series(np.nan, index=feats.index)

    monkeypatch.setattr(strategy, "_walk_forward", capture_model)
    strategy.generate_positions(data, lookback=21, forward=1)

    assert captured["train_size"] == 21


def test_ml_labels_keep_unknown_future_tail_nan():
    df = _mk_data(600)
    features = compute_features(pd.DataFrame(df))
    data = {**df, "features": features}
    from momentum_lab.strategies import get_strategy

    _, label = get_strategy("ml_logreg")._prepare_data(data, lookback=21, forward=5)
    assert label.tail(5).isna().all()


def test_ml_walk_forward_purges_overlapping_labels():
    from momentum_lab.strategies import get_strategy

    feats = pd.DataFrame({"x": np.arange(30, dtype=float)})
    label = pd.Series(np.arange(30) % 2, dtype=float)
    fitted_sizes = []

    class RecordingModel:
        def fit(self, x, y):
            fitted_sizes.append(len(y))
            return self

        def predict(self, x):
            return np.zeros(len(x))

    get_strategy("ml_logreg")._walk_forward(
        feats,
        label,
        lambda: RecordingModel(),
        train_size=10,
        step=5,
        forward=3,
    )
    assert fitted_sizes[0] == 8


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
    data = {
        "close": close,
        "high": close + 1,
        "low": close - 1,
        "open": close,
        "volume": pd.Series(1, index=close.index),
    }
    df = pd.DataFrame({"close": close}, index=close.index)
    n = len(df)
    s1 = int(n * 0.6)
    s2 = int(n * 0.8)
    periods = {
        "train": (df.index[0], df.index[s1]),
        "val": (df.index[s1], df.index[s2]),
        "test": (df.index[s2], df.index[-1]),
    }
    report = robustness_check(
        data,
        df,
        periods,
        "tsmom",
        {"lookback": 21, "threshold": 0.0, "long_short": True, "position_size": 1.0},
        cost_bps=1.0,
        frac=0.2,
    )
    assert report["error"] is None
    assert report["grade"] in {"A", "B", "C", "D"}
    assert report["n_neighbors"] > 0


def _mk_data(n=400):
    rng = np.random.default_rng(42)
    idx = pd.date_range("2000-01-01", periods=n, freq="B")
    close = pd.Series(rng.normal(0, 1, n).cumsum() + 100, index=idx)
    return {
        "close": close,
        "high": close + 2 * rng.random(n),
        "low": close - 2 * rng.random(n),
        "open": close,
        "volume": pd.Series(1.0, index=idx),
    }


def test_wma_matches_reference():
    """Vectorized WMA must equal rolling().apply() reference."""
    from momentum_lab.strategies import _wma

    close = _mk_data()["close"]
    for p in [2, 5, 20, 100]:
        w = np.arange(1, p + 1)
        ref = close.rolling(p).apply(lambda x, w=w: np.dot(x, w) / w.sum(), raw=True)
        out = _wma(close, p)
        assert np.allclose(ref.fillna(0).to_numpy(), out.fillna(0).to_numpy(), equal_nan=True)


def test_heikin_ashi_matches_reference():
    """Vectorized Heikin Ashi must match the iterative construction."""
    data = _mk_data()
    s = get_strategy("heikin_ashi")
    c, op, hi, lo = data["close"], data["open"], data["high"], data["low"]
    ha_close = (op + hi + lo + c) / 4
    ha_open = pd.Series(index=c.index, dtype=float)
    ha_open.iloc[0] = (op.iloc[0] + c.iloc[0]) / 2
    for i in range(1, len(c)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
    ref_bull = ha_close > ha_open
    ref_pos = pd.Series(0.0, index=c.index)
    ref_pos[ref_bull] = 1.0
    new = s.generate_positions(data, smooth=1, confirmation=1, long_short=False)
    assert np.allclose(ref_pos.to_numpy(), new.to_numpy())


def test_supertrend_matches_reference():
    """Vectorized Supertrend must match the reference state machine."""
    data = _mk_data()
    s = get_strategy("supertrend")
    c, hi, lo = data["close"], data["high"], data["low"]
    atr_period, multiplier = 10, 3.0
    tr = pd.concat([hi - lo, (hi - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    upper = (hi + lo) / 2 + multiplier * atr
    lower = (hi + lo) / 2 - multiplier * atr
    trend = pd.Series(0, index=c.index)
    for i in range(1, len(c)):
        if pd.isna(upper.iloc[i - 1]) or pd.isna(lower.iloc[i - 1]):
            continue
        if c.iloc[i] > upper.iloc[i - 1]:
            trend.iloc[i] = 1
        elif c.iloc[i] < lower.iloc[i - 1]:
            trend.iloc[i] = -1
        elif trend.iloc[i - 1] == 0:
            trend.iloc[i] = 1
        else:
            trend.iloc[i] = trend.iloc[i - 1]
            if trend.iloc[i] == 1 and lower.iloc[i] < lower.iloc[i - 1]:
                lower.iloc[i] = lower.iloc[i - 1]
            if trend.iloc[i] == -1 and upper.iloc[i] > upper.iloc[i - 1]:
                upper.iloc[i] = upper.iloc[i - 1]
    ref = pd.Series(0.0, index=c.index)
    ref[trend == 1] = 1.0
    new = s.generate_positions(data, atr_period=atr_period, multiplier=multiplier, long_short=False)
    assert (new.iloc[:atr_period] == 0).all()
    assert np.allclose(ref.to_numpy(), new.to_numpy())
