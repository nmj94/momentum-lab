"""Property-based tests for momentum-lab core invariants.

These tests use hypothesis to express behavioral specifications that must
hold for arbitrary inputs, complementing the example-based regression
tests in test_basic.py.
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from momentum_lab.backtest import backtest, evaluate
from momentum_lab.robustness import perturb_params
from momentum_lab.search import _quick_sample, _split_periods
from momentum_lab.strategies import _wma, get_strategy

settings.register_profile("momentum", max_examples=50, deadline=None)
settings.load_profile("momentum")


def _price_series(rets, base=100.0):
    rets = np.asarray(rets, dtype=float)
    return pd.Series(base * np.cumprod(1.0 + rets))


_returns = st.lists(
    st.floats(min_value=-0.11, max_value=0.11, allow_nan=False, allow_infinity=False),
    min_size=30,
    max_size=220,
)


@st.composite
def _prices_and_positions(draw):
    n = draw(st.integers(min_value=30, max_value=220))
    rets = draw(
        st.lists(
            st.floats(min_value=-0.11, max_value=0.11, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    pos_vals = draw(
        st.lists(
            st.one_of(st.sampled_from([-1.0, 0.0, 1.0]), st.floats(min_value=-1.0, max_value=1.0)),
            min_size=n,
            max_size=n,
        )
    )
    return rets, pos_vals


@settings(max_examples=40)
@given(sample=_prices_and_positions(), use_vol=st.booleans(), cost=st.floats(0.0, 10.0))
def test_backtest_has_no_lookahead(sample, use_vol, cost):
    """Returns up to bar t must be identical when the future is truncated."""
    rets, pos_vals = sample
    prices = _price_series(rets)
    positions = pd.Series(pos_vals, index=prices.index)
    kwargs = {
        "cost_bps": cost,
        "slippage_bps": 1.0,
        "financing_rate": 0.02,
        "borrow_bps": 50.0,
    }
    if use_vol:
        kwargs.update(vol_target=0.15, vol_lookback=3, max_leverage=1.5)

    n = len(prices)
    cut = max(3, n // 2)
    full = backtest(positions, prices, **kwargs)
    truncated = backtest(positions.iloc[:cut], prices.iloc[:cut], **kwargs)

    assert np.allclose(full["returns"].iloc[:cut], truncated["returns"], atol=1e-12)
    assert np.allclose(full["equity"].iloc[:cut], truncated["equity"], atol=1e-9)


@settings(max_examples=40)
@given(rets=_returns, cost_low=st.floats(0.0, 5.0), cost_high=st.floats(5.0, 50.0))
def test_backtest_costs_weakly_reduce_equity(rets, cost_low, cost_high):
    """Raising transaction costs can never improve the equity curve."""
    prices = _price_series(rets)
    raw_positions = np.where(np.arange(len(prices)) % 7 < 3, 1.0, -0.5)
    positions = pd.Series(raw_positions, index=prices.index)

    lo = backtest(positions, prices, cost_bps=cost_low)["equity"]
    hi = backtest(positions, prices, cost_bps=cost_high)["equity"]

    assert (hi <= lo + 1e-12).all()


@given(rets=_returns)
def test_backtest_buy_and_hold_matches_price_returns(rets):
    """A cost-free constant +1 position must reproduce pure price returns."""
    prices = _price_series(rets)
    result = backtest(pd.Series(1.0, index=prices.index), prices, cost_bps=0.0)
    expected = prices.pct_change().fillna(0.0)
    assert np.allclose(result["returns"], expected, atol=1e-12)
    assert result["returns"].notna().all()


@settings(max_examples=40)
@given(
    rets=st.lists(
        st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=300,
    ),
    annualization=st.sampled_from([12, 52, 252, 365]),
    rf=st.floats(0.0, 0.08),
)
def test_evaluate_metric_ranges_and_sign_consistency(rets, annualization, rf):
    """Metrics must stay in their mathematically valid ranges for finite input."""
    metrics = evaluate(_price_series(rets).pct_change().fillna(0.0), risk_free_rate=rf, annualization=annualization)

    assert -1.0 - 1e-9 <= metrics["max_drawdown"] <= 0.0
    assert 0.0 <= metrics["win_rate"] <= 1.0
    assert metrics["volatility"] >= 0.0
    assert metrics["profit_factor"] >= 0.0
    assert np.isfinite(metrics["sharpe"]) and np.isfinite(metrics["calmar"])
    # |r| <= 0.1 keeps equity strictly positive, so CAGR and total return
    # must share a sign.  Both are rounded to 4 dp; a near-zero total return
    # can round to 0.0 while the annualized CAGR rounds up to 0.0001, so the
    # sign comparison only applies away from that rounding boundary.
    if abs(metrics["total_return"]) >= 1e-4:
        assert np.sign(metrics["cagr"]) == np.sign(metrics["total_return"])


@given(offsets=st.lists(st.integers(0, 500), min_size=3, max_size=60, unique=True))
def test_split_periods_partition_any_index(offsets):
    """Train/val/test must be consecutive, non-overlapping and covering."""
    idx = pd.DatetimeIndex([pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(o)) for o in sorted(offsets)])

    periods = _split_periods(idx)

    assert periods["train"][0] == idx[0]
    assert periods["test"][1] == idx[-1]
    assert periods["train"][1] < periods["val"][0]
    assert periods["val"][0] <= periods["val"][1] < periods["test"][0]


@settings(max_examples=40)
@given(
    ints=st.lists(st.integers(1, 300), min_size=1, max_size=3),
    floats=st.lists(st.floats(0.01, 10.0, allow_nan=False), min_size=1, max_size=3),
    frac=st.floats(0.05, 0.9),
)
def test_perturb_params_changes_exactly_one_numeric_key(ints, floats, frac):
    """Every neighbor differs from the base params in exactly one numeric key."""
    params = {f"i{k}": v for k, v in enumerate(ints)}
    params.update({f"f{k}": v for k, v in enumerate(floats)})
    params["flag"] = True
    params["mode"] = "sma"

    neighbors = perturb_params(params, frac=frac)

    assert len(neighbors) == 2 * (len(ints) + len(floats))
    for nb in neighbors:
        changed = [k for k in params if nb[k] != params[k]]
        assert len(changed) == 1
        assert not isinstance(params[changed[0]], bool)
        assert isinstance(params[changed[0]], (int, float))


@settings(max_examples=30)
@given(
    rets=st.lists(st.floats(min_value=-0.06, max_value=0.06, allow_nan=False), min_size=40, max_size=200),
    lookback=st.integers(1, 15),
    skip_recent=st.integers(0, 3),
    long_short=st.booleans(),
)
def test_tsmom_positions_are_bounded_and_flat_during_warmup(rets, lookback, skip_recent, long_short):
    close = _price_series(rets)
    pos = get_strategy("tsmom").generate_positions(
        {"close": close}, lookback=lookback, threshold=0.0, long_short=long_short, skip_recent=skip_recent
    )

    assert len(pos) == len(close)
    assert pos.isin([-1.0, 0.0, 1.0]).all()
    assert pos.iloc[: lookback + skip_recent].eq(0.0).all()


@settings(max_examples=30)
@given(
    rets=st.lists(st.floats(min_value=-0.05, max_value=0.05, allow_nan=False), min_size=80, max_size=220),
    momentum_lb=st.integers(5, 30),
    ma_filter=st.integers(10, 60),
)
def test_stacked_exit_filter_is_direction_aware(rets, momentum_lb, ma_filter):
    """Spec of the Stacked overlay, as a property over arbitrary price paths."""
    close = _price_series(rets)
    pos = get_strategy("stacked").generate_positions(
        {"close": close},
        momentum_lb=momentum_lb,
        ma_filter=ma_filter,
        base_strategy="tsmom",
        base_lookback=5,
        long_short=True,
        exit_on_neg=True,
    )

    mom = close.pct_change(momentum_lb)
    ma = close.rolling(ma_filter).mean()
    ready = mom.notna() & ma.notna()

    assert len(pos) == len(close)
    assert pos[~ready].eq(0.0).all()
    assert not ((pos > 0) & ready & ((mom <= 0) | (close < ma))).any()
    assert not ((pos < 0) & ready & ((mom >= 0) | (close > ma))).any()


@given(
    values=st.lists(st.floats(-500.0, 500.0, allow_nan=False), min_size=5, max_size=150),
    period=st.integers(2, 20),
)
def test_wma_stays_within_window_extremes(values, period):
    """A weighted average can never leave its own window's min/max range."""
    series = pd.Series(values)
    period = min(period, len(series))
    if period < 2:
        return

    out = _wma(series, period)

    lo = series.rolling(period).min()
    hi = series.rolling(period).max()
    valid = out.iloc[period - 1 :]
    tol = 1e-9 * max(1.0, float(np.abs(values).max()))
    assert (valid >= lo.iloc[period - 1 :] - tol).all()
    assert (valid <= hi.iloc[period - 1 :] + tol).all()


@settings(max_examples=20)
@given(k=st.integers(0, 6))
def test_quick_sample_size_distinctness_and_bounds(k):
    """Quick sampling must return min(k, total) distinct combos spanning the grid."""
    s = get_strategy("dual_momentum")
    total = s.count_param_combinations()
    combos = _quick_sample(s, k)

    assert len(combos) == min(k, total)
    unique = {tuple(sorted(c.items())) for c in combos}
    assert len(unique) == len(combos)
    assert combos == _quick_sample(s, k)
