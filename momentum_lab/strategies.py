"""strategies.py - 26 momentum strategies with exhaustive parameter grids.

Classes:
    BaseStrategy: Abstract base with universal search param (signal_smooth)
    TSMOM, MACross, MACD, RSI, ROC, Bollinger, Donchian, DualMomentum,
    TripleMA, VolScale, Accel, ZScore, HeikinAshi, Supertrend, MultiBreakout
    _MLBase, MLLogReg, MLRF, MLXGB, MLKNN, MLSVM, MLNB, MLAda, MLExtraTrees
    Ensemble, Stacked, RegimeAware
"""

from itertools import product
from typing import ClassVar

import numpy as np
import pandas as pd

from .indicators import IndicatorDAG, weighted_moving_average

_wma = weighted_moving_average


def _dag(data) -> IndicatorDAG | None:
    graph = data.get("_indicator_dag")
    return graph if isinstance(graph, IndicatorDAG) else None


def _returns(data, period=1, field="close", shift=0):
    graph = _dag(data)
    if graph is not None:
        return graph.returns(period, field, shift)
    return data[field].shift(shift).pct_change(period)


def _moving_average(data, period, kind="sma", field="close"):
    graph = _dag(data)
    if graph is not None:
        return graph.moving_average(period, kind, field)
    series = data[field]
    if kind == "sma":
        return series.rolling(period).mean()
    if kind == "ema":
        return series.ewm(span=period, adjust=False).mean()
    if kind == "wma":
        return _wma(series, period)
    if kind == "dema":
        first = series.ewm(span=period, adjust=False).mean()
        return 2 * first - first.ewm(span=period, adjust=False).mean()
    raise ValueError(f"unknown moving-average kind: {kind}")


def _rolling_std(data, period, field="close"):
    graph = _dag(data)
    return graph.rolling_std(period, field) if graph is not None else data[field].rolling(period).std()


def _rolling_max(data, period, field="high", shift=0):
    graph = _dag(data)
    source = data.get(field, data["close"])
    return graph.rolling_max(period, field, shift) if graph is not None else source.rolling(period).max().shift(shift)


def _rolling_min(data, period, field="low", shift=0):
    graph = _dag(data)
    source = data.get(field, data["close"])
    return graph.rolling_min(period, field, shift) if graph is not None else source.rolling(period).min().shift(shift)


class BaseStrategy:
    name = "base"
    param_grid: ClassVar[dict] = {}
    # Risk sizing is deliberately excluded from alpha selection.  Searching
    # ``position_size`` mechanically favors leverage when Sharpe subtracts a
    # fixed risk-free rate. Callers may still pass position_size to ``run``.
    UNIVERSAL_PARAMS: ClassVar[dict] = {"signal_smooth": [0, 2, 3, 5, 10]}

    def generate_positions(self, data, **params):
        raise NotImplementedError

    def run(self, data, **params):
        close = data["close"]
        if close.empty:
            return close.astype(float).copy()
        position_size = params.pop("position_size", 1.0)
        signal_smooth = params.pop("signal_smooth", 0)
        pos = self.generate_positions(data, **params)
        if signal_smooth > 0:
            pos = pos.ewm(span=signal_smooth, adjust=False).mean()
        pos = pos * position_size
        return pos

    def is_valid_params(self, params):
        """Return whether a parameter combination is internally coherent."""
        return True

    def iter_param_combinations(self):
        """Yield valid parameter combinations one at a time.

        The combined grids contain hundreds of thousands of combinations;
        materializing them as a list (as the previous implementation did) causes
        memory spikes just to iterate or count.
        """
        all_params = {**self.param_grid, **self.UNIVERSAL_PARAMS}
        keys = list(all_params.keys())
        if not keys:
            yield {}
            return
        vals = [all_params[k] for k in keys]
        for v in product(*vals):
            params = dict(zip(keys, v))
            if self.is_valid_params(params):
                yield params

    def count_param_combinations(self):
        """Count valid parameter combinations in O(1) memory."""
        return sum(1 for _ in self.iter_param_combinations())

    def get_param_combinations(self):
        return list(self.iter_param_combinations())


class TSMOM(BaseStrategy):
    name = "tsmom"
    param_grid: ClassVar[dict] = {
        "lookback": [3, 5, 7, 10, 14, 21, 28, 42, 55, 63, 84, 126, 168, 252],
        "threshold": [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05],
        "long_short": [True, False],
        "skip_recent": [0, 1, 2, 3, 5, 10, 21],
    }

    def generate_positions(self, data, lookback=21, threshold=0.0, long_short=True, skip_recent=0):
        close = data["close"]
        ret = _returns(data, lookback, shift=skip_recent)
        pos = pd.Series(0.0, index=close.index)
        pos[ret > threshold] = 1.0
        if long_short:
            pos[ret < -threshold] = -1.0
        return pos


class MACross(BaseStrategy):
    name = "ma_cross"
    param_grid: ClassVar[dict] = {
        "fast": [1, 2, 3, 5, 8, 10, 13, 15, 20, 25, 30],
        "slow": [10, 15, 20, 30, 50, 75, 100, 150, 200, 250],
        "long_short": [True, False],
        "ma_type": ["sma", "ema", "wma", "dema"],
    }

    def is_valid_params(self, params):
        return params["fast"] < params["slow"]

    def generate_positions(self, data, fast=10, slow=50, long_short=True, ma_type="sma"):
        close = data["close"]
        diff = _moving_average(data, fast, ma_type) - _moving_average(data, slow, ma_type)
        pos = pd.Series(0.0, index=close.index)
        pos[diff > 0] = 1.0
        if long_short:
            pos[diff < 0] = -1.0
        return pos


class MACD(BaseStrategy):
    name = "macd"
    param_grid: ClassVar[dict] = {
        "fast": [5, 6, 8, 10, 12, 16, 20],
        "slow": [12, 17, 19, 21, 26, 30, 35, 40, 50],
        "signal": [3, 5, 7, 9, 12, 15, 20],
        "long_short": [True, False],
        "mode": ["crossover", "histogram", "zero_filter"],
    }

    def is_valid_params(self, params):
        return params["fast"] < params["slow"]

    def generate_positions(self, data, fast=12, slow=26, signal=9, long_short=True, mode="crossover"):
        close = data["close"]
        graph = _dag(data)
        macd_line = _moving_average(data, fast, "ema") - _moving_average(data, slow, "ema")
        signal_line = (
            graph.node(
                ("macd_signal", int(fast), int(slow), int(signal)),
                lambda: macd_line.ewm(span=signal, adjust=False).mean(),
            )
            if graph is not None
            else macd_line.ewm(span=signal, adjust=False).mean()
        )
        diff = macd_line - signal_line
        pos = pd.Series(0.0, index=close.index)
        if mode == "crossover":
            pos[diff > 0] = 1.0
            if long_short:
                pos[diff < 0] = -1.0
        elif mode == "histogram":
            hist = diff.diff()
            pos[hist > 0] = 1.0
            if long_short:
                pos[hist < 0] = -1.0
        elif mode == "zero_filter":
            pos[(diff > 0) & (macd_line > 0)] = 1.0
            if long_short:
                pos[(diff < 0) & (macd_line < 0)] = -1.0
        return pos


class RSI(BaseStrategy):
    name = "rsi"
    param_grid: ClassVar[dict] = {
        "period": [3, 5, 7, 9, 10, 14, 21, 25, 28, 35, 50],
        "buy_threshold": [30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
        "sell_threshold": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "long_short": [True, False],
        "mode": ["momentum", "reversal"],
        "rsi_smooth": [1, 3, 5, 10],
    }

    def is_valid_params(self, params):
        # Momentum enters above buy_threshold and exits/shorts below
        # sell_threshold; reversal uses the opposite ordering.
        if params["mode"] == "momentum":
            return params["buy_threshold"] >= params["sell_threshold"]
        return params["buy_threshold"] <= params["sell_threshold"]

    def generate_positions(
        self, data, period=14, buy_threshold=50, sell_threshold=50, long_short=True, mode="momentum", rsi_smooth=1
    ):
        close = data["close"]
        graph = _dag(data)
        if graph is not None:
            rsi = graph.rsi(period)
        else:
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(period).mean()
            loss = (-delta.clip(upper=0)).rolling(period).mean()
            rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
        if rsi_smooth > 1:
            rsi = (
                graph.node(
                    ("rsi_smooth", int(period), int(rsi_smooth)),
                    lambda: rsi.ewm(span=rsi_smooth, adjust=False).mean(),
                )
                if graph is not None
                else rsi.ewm(span=rsi_smooth, adjust=False).mean()
            )
        pos = pd.Series(0.0, index=close.index)
        if mode == "momentum":
            pos[rsi > buy_threshold] = 1.0
            if long_short:
                pos[rsi < sell_threshold] = -1.0
        else:
            pos[rsi < buy_threshold] = 1.0
            if long_short:
                pos[rsi > sell_threshold] = -1.0
        return pos


class ROC(BaseStrategy):
    name = "roc"
    param_grid: ClassVar[dict] = {
        "period": [3, 5, 7, 10, 14, 21, 28, 42, 55, 63, 84, 126, 168, 252],
        "threshold": [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10],
        "long_short": [True, False],
        "smoothing": [0, 3, 5, 10],
    }

    def generate_positions(self, data, period=21, threshold=0.0, long_short=True, smoothing=0):
        close = data["close"]
        graph = _dag(data)
        roc = _returns(data, period)
        if smoothing > 0:
            roc = (
                graph.node(
                    ("roc_smooth", int(period), int(smoothing)),
                    lambda: roc.ewm(span=smoothing, adjust=False).mean(),
                )
                if graph is not None
                else roc.ewm(span=smoothing, adjust=False).mean()
            )
        pos = pd.Series(0.0, index=close.index)
        pos[roc > threshold] = 1.0
        if long_short:
            pos[roc < -threshold] = -1.0
        return pos


class Bollinger(BaseStrategy):
    name = "bollinger"
    param_grid: ClassVar[dict] = {
        "period": [5, 10, 15, 20, 30, 50, 75, 100],
        "num_std": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        "long_short": [True, False],
        "mode": ["breakout", "mean_revert"],
        "band_width_filter": [0, 0.1, 0.2],
    }

    def generate_positions(self, data, period=20, num_std=2.0, long_short=True, mode="breakout", band_width_filter=0):
        close = data["close"]
        ma = _moving_average(data, period)
        std = _rolling_std(data, period)
        upper = ma + num_std * std
        lower = ma - num_std * std
        bw = (upper - lower) / (ma + 1e-10)
        mask = bw > band_width_filter if band_width_filter > 0 else pd.Series(True, index=close.index)
        pos = pd.Series(0.0, index=close.index)
        if mode == "breakout":
            pos[(close > upper) & mask] = 1.0
            if long_short:
                pos[(close < lower) & mask] = -1.0
        else:
            pos[(close < lower) & mask] = 1.0
            if long_short:
                pos[(close > upper) & mask] = -1.0
        return pos


class Donchian(BaseStrategy):
    name = "donchian"
    param_grid: ClassVar[dict] = {
        "period": [3, 5, 7, 10, 15, 20, 30, 40, 55, 75, 100, 150, 200],
        "long_short": [True, False],
        "exit_period": [0, 3, 5, 10, 20, 30, 50],
        "confirmation": [1, 2, 3],
    }

    def generate_positions(self, data, period=20, long_short=True, exit_period=0, confirmation=1):
        close = data["close"]
        upper = _rolling_max(data, period, "high", 1)
        lower = _rolling_min(data, period, "low", 1)
        bu = close > upper
        bd = close < lower
        if confirmation > 1:
            bu = bu.rolling(confirmation).sum() >= confirmation
            bd = bd.rolling(confirmation).sum() >= confirmation
        # Donchian is a persistent breakout system: an entry remains active
        # until an exit channel is breached.  The previous implementation
        # emitted a one-bar entry signal, so the backtest immediately went
        # flat on the following bar and made ``exit_period`` ineffective.
        ep = exit_period if exit_period > 0 else period
        exit_upper = _rolling_max(data, ep, "high", 1)
        exit_lower = _rolling_min(data, ep, "low", 1)
        # Iterate over plain numpy buffers: pandas .iloc indexing in this
        # state machine dominates profile time for the 16k-combo grid.
        c = close.to_numpy(dtype=float)
        eu = exit_upper.to_numpy(dtype=float)
        el = exit_lower.to_numpy(dtype=float)
        bu_a = bu.to_numpy(dtype=bool)
        bd_a = bd.to_numpy(dtype=bool)
        pos = np.zeros(len(c))
        state = 0.0
        for i in range(len(c)):
            ci = c[i]
            if (state > 0 and ci < el[i]) or (state < 0 and ci > eu[i]):
                state = 0.0
            if state == 0.0:
                if bu_a[i]:
                    state = 1.0
                elif long_short and bd_a[i]:
                    state = -1.0
            pos[i] = state
        return pd.Series(pos, index=close.index)


class DualMomentum(BaseStrategy):
    name = "dual_momentum"
    param_grid: ClassVar[dict] = {
        "lookback": [5, 10, 21, 42, 63, 84, 126, 168, 252],
        "abs_threshold": [0.0, 0.005, 0.01, 0.02, 0.03, 0.05],
        "long_short": [True, False],
        "smoothing": [0, 5, 10],
    }

    def generate_positions(self, data, lookback=63, abs_threshold=0.0, long_short=False, smoothing=0):
        close = data["close"]
        graph = _dag(data)
        ret = _returns(data, lookback)
        if smoothing > 0:
            ret = (
                graph.node(
                    ("dual_momentum_smooth", int(lookback), int(smoothing)),
                    lambda: ret.ewm(span=smoothing, adjust=False).mean(),
                )
                if graph is not None
                else ret.ewm(span=smoothing, adjust=False).mean()
            )
        pos = pd.Series(0.0, index=close.index)
        pos[ret > abs_threshold] = 1.0
        if long_short:
            pos[ret < -abs_threshold] = -1.0
        return pos


class TripleMA(BaseStrategy):
    name = "triple_ma"
    param_grid: ClassVar[dict] = {
        "fast": [3, 5, 8, 10, 20],
        "medium": [20, 30, 50, 75, 100],
        "slow": [50, 75, 100, 150, 200, 250],
        "long_short": [True, False],
        "ma_type": ["sma", "ema", "wma"],
    }

    def is_valid_params(self, params):
        return params["fast"] < params["medium"] < params["slow"]

    def generate_positions(self, data, fast=10, medium=50, slow=200, long_short=True, ma_type="sma"):
        close = data["close"]
        fast_ma = _moving_average(data, fast, ma_type)
        medium_ma = _moving_average(data, medium, ma_type)
        slow_ma = _moving_average(data, slow, ma_type)
        bullish = (fast_ma > medium_ma) & (medium_ma > slow_ma)
        pos = pd.Series(0.0, index=close.index)
        pos[bullish] = 1.0
        if long_short:
            bearish = (fast_ma < medium_ma) & (medium_ma < slow_ma)
            pos[bearish] = -1.0
        return pos


class VolScale(BaseStrategy):
    name = "vol_scale_mom"
    param_grid: ClassVar[dict] = {
        "lookback": [5, 10, 21, 42, 63, 84, 126, 168, 252],
        "vol_lookback": [5, 10, 21, 42, 63, 84],
        "vol_target": [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30],
        "threshold": [0.0, 0.001, 0.005, 0.01, 0.02, 0.03],
    }

    def generate_positions(self, data, lookback=63, vol_lookback=21, vol_target=0.15, threshold=0.0):
        close = data["close"]
        # Annualize with the same horizon the backtest/metrics use (365 for
        # crypto); a hardcoded 252 would oversize positions by ~20% there.
        ann = data.get("annualization", 252)
        graph = _dag(data)
        ret = _returns(data, lookback)
        daily_returns = _returns(data)
        vol = (
            graph.node(
                ("return_volatility", int(vol_lookback), float(ann)),
                lambda: daily_returns.rolling(vol_lookback).std() * np.sqrt(ann),
            )
            if graph is not None
            else daily_returns.rolling(vol_lookback).std() * np.sqrt(ann)
        )
        raw = pd.Series(0.0, index=close.index)
        raw[ret > threshold] = 1.0
        raw[ret < -threshold] = -1.0
        scaling = (vol_target / (vol + 1e-10)).fillna(0.0)
        return (raw * scaling).clip(-2.0, 2.0)


class Accel(BaseStrategy):
    name = "acceleration"
    param_grid: ClassVar[dict] = {
        "short_lb": [1, 2, 3, 5, 7, 10, 14],
        "long_lb": [5, 10, 14, 21, 28, 42, 55, 63, 84, 126],
        "threshold": [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01],
        "long_short": [True, False],
    }

    def is_valid_params(self, params):
        return params["short_lb"] < params["long_lb"]

    def generate_positions(self, data, short_lb=5, long_lb=21, threshold=0.0, long_short=True):
        close = data["close"]
        accel = _returns(data, short_lb) - _returns(data, long_lb)
        pos = pd.Series(0.0, index=close.index)
        pos[accel > threshold] = 1.0
        if long_short:
            pos[accel < -threshold] = -1.0
        return pos


class ZScore(BaseStrategy):
    name = "zscore"
    param_grid: ClassVar[dict] = {
        "lookback": [5, 10, 14, 21, 28, 42, 55, 63, 84, 126],
        "entry_z": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
        "exit_z": [0.0, 0.25, 0.5, 0.75],
        "mode": ["momentum", "reversion"],
        "long_short": [True, False],
    }

    def is_valid_params(self, params):
        return params["entry_z"] >= params["exit_z"]

    def generate_positions(self, data, lookback=21, entry_z=1.0, exit_z=0.0, mode="momentum", long_short=True):
        close = data["close"]
        graph = _dag(data)
        z = (
            graph.zscore(lookback)
            if graph is not None
            else (close - close.rolling(lookback).mean()) / (close.rolling(lookback).std() + 1e-10)
        )
        is_reversion = mode != "momentum"
        # Plain numpy state machine: the pandas .iloc loop dominated profile
        # time for the 48k-combo grid.
        values = z.to_numpy(dtype=float)
        pos = np.zeros(len(values))
        state = 0.0
        for i, value in enumerate(values):
            if not np.isfinite(value):
                pos[i] = state
                continue
            long_entry = value < -entry_z if is_reversion else value > entry_z
            short_entry = value > entry_z if is_reversion else value < -entry_z
            if state == 0.0:
                if long_entry:
                    state = 1.0
                elif long_short and short_entry:
                    state = -1.0
            elif abs(value) <= exit_z:
                state = 0.0
            elif short_entry:
                state = -1.0 if long_short else 0.0
            elif state < 0 and long_entry:
                state = 1.0
            pos[i] = state
        return pd.Series(pos, index=close.index)


class HeikinAshi(BaseStrategy):
    name = "heikin_ashi"
    param_grid: ClassVar[dict] = {"smooth": [1, 2, 3, 5, 8, 10], "long_short": [True, False], "confirmation": [1, 2, 3]}

    def generate_positions(self, data, smooth=1, long_short=True, confirmation=1):
        close = data["close"]
        if close.empty:
            return close.astype(float).copy()
        op = data.get("open", close)
        hi = data.get("high", close)
        lo = data.get("low", close)
        ha_close = (op + hi + lo + close) / 4
        seed = (op.iloc[0] + close.iloc[0]) / 2
        ha_open = ha_close.shift(1).fillna(seed).ewm(alpha=0.5, adjust=False).mean()
        if smooth > 1:
            ha_close = ha_close.rolling(smooth).mean()
            ha_open = ha_open.rolling(smooth).mean()
        bull = ha_close > ha_open
        bear = ha_close < ha_open
        if confirmation > 1:
            bull = bull.rolling(confirmation).sum() >= confirmation
            bear = bear.rolling(confirmation).sum() >= confirmation
        pos = pd.Series(0.0, index=close.index)
        pos[bull] = 1.0
        if long_short:
            pos[bear] = -1.0
        return pos


class Supertrend(BaseStrategy):
    name = "supertrend"
    param_grid: ClassVar[dict] = {
        "atr_period": [5, 7, 10, 14, 21, 28, 42],
        "multiplier": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
        "long_short": [True, False],
    }

    def generate_positions(self, data, atr_period=10, multiplier=3.0, long_short=True):
        close = data["close"]
        hi = data.get("high", close)
        lo = data.get("low", close)
        graph = _dag(data)
        if graph is not None:
            atr = graph.atr(atr_period)
        else:
            tr = pd.concat([hi - lo, (hi - close.shift(1)).abs(), (lo - close.shift(1)).abs()], axis=1).max(axis=1)
            atr = tr.rolling(atr_period).mean()
        mid = (hi + lo) / 2
        upper = np.array(mid + multiplier * atr, dtype=float, copy=True)
        lower = np.array(mid - multiplier * atr, dtype=float, copy=True)
        c = np.array(close, dtype=float, copy=True)
        n = len(c)
        trend = np.zeros(n, dtype=int)
        for i in range(1, n):
            if not np.isfinite(upper[i - 1]) or not np.isfinite(lower[i - 1]):
                continue
            if c[i] > upper[i - 1]:
                trend[i] = 1
            elif c[i] < lower[i - 1]:
                trend[i] = -1
            elif trend[i - 1] == 0:
                trend[i] = 1
            else:
                trend[i] = trend[i - 1]
                if trend[i] == 1 and lower[i] < lower[i - 1]:
                    lower[i] = lower[i - 1]
                elif trend[i] == -1 and upper[i] > upper[i - 1]:
                    upper[i] = upper[i - 1]
        pos = pd.Series(0.0, index=close.index)
        pos[trend == 1] = 1.0
        if long_short:
            pos[trend == -1] = -1.0
        return pos


class MultiBreakout(BaseStrategy):
    name = "multi_breakout"
    param_grid: ClassVar[dict] = {
        "periods": [
            (5, 10, 20),
            (5, 20, 55),
            (10, 20, 55),
            (10, 55, 200),
            (20, 55, 100),
            (20, 100, 200),
            (5, 10, 20, 55, 100),
            (10, 20, 55, 100, 200),
            (5, 10, 21, 42, 63, 126, 252),
            (10, 20, 55, 126),
        ],
        "long_short": [True, False],
        "vote_threshold": [0.3, 0.4, 0.5, 0.6, 0.7],
    }

    def generate_positions(self, data, periods=(10, 20, 55), long_short=True, vote_threshold=0.5):
        close = data["close"]
        sigs = []
        for p in periods:
            u = _rolling_max(data, p, "high", 1)
            l = _rolling_min(data, p, "low", 1)
            s = pd.Series(0.0, index=close.index)
            s[close > u] = 1.0
            if long_short:
                s[close < l] = -1.0
            sigs.append(s)
        avg = pd.concat(sigs, axis=1).mean(axis=1)
        pos = pd.Series(0.0, index=close.index)
        pos[avg > vote_threshold] = 1.0
        if long_short:
            pos[avg < -vote_threshold] = -1.0
        return pos


class _MLBase(BaseStrategy):
    UNIVERSAL_PARAMS: ClassVar[dict] = {"signal_smooth": [0, 5]}

    def _prepare_data(self, data, lookback, forward, feature_cols=None):
        """Build the (features, label) pair for walk-forward training.

        ``lookback`` is not a feature window - the feature set uses fixed
        horizons.  Callers pass it straight through to ``_walk_forward`` as
        the initial ``train_size`` warm-up.
        """
        from .data import compute_features

        close = data["close"]
        if "features" not in data or data["features"] is None:
            feats = compute_features(pd.DataFrame({"close": close}), annualization=data.get("annualization", 252))
        else:
            feats = data["features"]
        if feature_cols:
            feats = feats[feature_cols]
        fwd_ret = close.pct_change(forward).shift(-forward)
        # Preserve the unknown tail as NaN.  Casting the comparison directly
        # to int turns those rows into class 0 and leaks fake labels into the
        # final walk-forward training windows.
        label = pd.Series(np.nan, index=close.index, dtype=float)
        known = fwd_ret.notna()
        label.loc[known] = (fwd_ret.loc[known] > 0).astype(float)
        # Feature availability and label availability are different concepts.
        # Unknown forward labels at the live tail must exclude rows from
        # training without also destroying otherwise valid prediction inputs.
        feats = feats.copy()
        feats[feats.isna().any(axis=1)] = np.nan
        return feats, label

    def _walk_forward(self, feats, label, model_fn, train_size=504, step=21, retrain=True, forward=1, embargo=0):
        """Generate predictions with purged expanding-window training.

        A label at time ``t`` uses prices through ``t + forward``.  Therefore
        rows whose label window overlaps the prediction start are removed from
        the training sample.  ``embargo`` can add an extra gap when a caller
        needs stricter separation between train and prediction windows.
        """
        from sklearn.preprocessing import StandardScaler

        if forward < 1 or embargo < 0:
            raise ValueError("forward must be >= 1 and embargo must be >= 0")

        original_index = feats.index
        feature_valid = ~feats.isna().any(axis=1)
        label_valid = feature_valid & label.notna()
        prediction_positions = np.flatnonzero(feature_valid.to_numpy())
        preds = pd.Series(np.nan, index=original_index)
        n = len(prediction_positions)
        i = train_size
        while i < n:
            # Use original bar offsets rather than compressed valid-row
            # offsets so intermittent feature gaps cannot weaken the purge.
            prediction_bar = prediction_positions[i]
            all_bars = np.arange(len(feats))
            train_mask = label_valid.to_numpy() & (all_bars + forward + embargo <= prediction_bar)
            if train_mask.sum() < 2:
                i += step
                continue
            train_labels = label.to_numpy()[train_mask]
            if np.unique(train_labels).size < 2:
                i += step
                continue
            scaler = StandardScaler()
            X = scaler.fit_transform(feats.to_numpy()[train_mask])
            model = model_fn()
            model.fit(X, train_labels)
            pe = min(i + step, n)
            block_positions = prediction_positions[i:pe]
            preds.iloc[block_positions] = model.predict(scaler.transform(feats.iloc[block_positions].values))
            if not retrain:
                rest_positions = prediction_positions[pe:]
                if len(rest_positions) > 0:
                    preds.iloc[rest_positions] = model.predict(scaler.transform(feats.iloc[rest_positions].values))
                break
            i = pe
        return preds

    def _preds_to_positions(self, preds, long_short=True):
        pos = pd.Series(0.0, index=preds.index)
        pos[preds == 1] = 1.0
        if long_short:
            pos[preds == 0] = -1.0
        return pos


_SKLEARN_PENALTY_DEPRECATED = None


def _sklearn_penalty_deprecated():
    """Probe (once) whether this scikit-learn version deprecates ``penalty``."""
    global _SKLEARN_PENALTY_DEPRECATED
    if _SKLEARN_PENALTY_DEPRECATED is None:
        from sklearn.linear_model import LogisticRegression

        _SKLEARN_PENALTY_DEPRECATED = LogisticRegression().get_params()["penalty"] == "deprecated"
    return _SKLEARN_PENALTY_DEPRECATED


class MLLogReg(_MLBase):
    name = "ml_logreg"
    param_grid: ClassVar[dict] = {
        "lookback": [252, 504, 756],
        "forward": [1, 3, 5, 10, 21],
        "C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
        "penalty": ["l2", "l1"],
        "long_short": [True, False],
        "retrain": [True, False],
    }

    def generate_positions(self, data, lookback=42, forward=5, C=1.0, penalty="l2", long_short=True, retrain=True):
        from sklearn.linear_model import LogisticRegression

        feats, label = self._prepare_data(data, lookback, forward)
        # Use explicit elastic-net ratios for both grid branches.  Newer
        # scikit-learn releases infer the penalty from l1_ratio; older
        # releases need the explicit elasticnet setting for L1 to take effect.
        l1_ratio = 1.0 if penalty == "l1" else 0.0
        model_kwargs = {
            "C": C,
            "l1_ratio": l1_ratio,
            "solver": "saga",
            "max_iter": 2000,
            "random_state": 42,
        }
        if not _sklearn_penalty_deprecated():
            model_kwargs["penalty"] = "elasticnet"
        fn = lambda: LogisticRegression(**model_kwargs)
        return self._preds_to_positions(
            self._walk_forward(feats, label, fn, train_size=lookback, retrain=retrain, forward=forward), long_short
        )


class MLRF(_MLBase):
    name = "ml_rf"
    param_grid: ClassVar[dict] = {
        "lookback": [252, 504, 756],
        "forward": [1, 5, 21],
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 5, 8, 10, None],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", None],
        "long_short": [True, False],
        "retrain": [True, False],
    }

    def generate_positions(
        self,
        data,
        lookback=42,
        forward=5,
        n_estimators=100,
        max_depth=5,
        min_samples_split=2,
        max_features="sqrt",
        long_short=True,
        retrain=True,
    ):
        from sklearn.ensemble import RandomForestClassifier

        feats, label = self._prepare_data(data, lookback, forward)
        fn = lambda: RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=42,
            n_jobs=1,
        )
        return self._preds_to_positions(
            self._walk_forward(feats, label, fn, train_size=lookback, retrain=retrain, forward=forward), long_short
        )


class MLXGB(_MLBase):
    name = "ml_xgb"
    param_grid: ClassVar[dict] = {
        "lookback": [252, 504, 756],
        "forward": [1, 5, 21],
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [2, 3, 5, 7, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "long_short": [True, False],
        "retrain": [True, False],
    }

    def generate_positions(
        self,
        data,
        lookback=42,
        forward=5,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        long_short=True,
        retrain=True,
    ):
        from xgboost import XGBClassifier

        feats, label = self._prepare_data(data, lookback, forward)
        fn = lambda: XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            n_jobs=1,
            eval_metric="logloss",
            verbosity=0,
        )
        return self._preds_to_positions(
            self._walk_forward(feats, label, fn, train_size=lookback, retrain=retrain, forward=forward), long_short
        )


class MLKNN(_MLBase):
    name = "ml_knn"
    param_grid: ClassVar[dict] = {
        "lookback": [252, 504, 756],
        "forward": [1, 5, 21],
        "n_neighbors": [3, 5, 7, 10, 15, 21, 30, 50],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
        "long_short": [True, False],
        "retrain": [True, False],
    }

    def is_valid_params(self, params):
        # At the first prediction block, purging removes ``forward`` labels
        # from the initial training window.  Reject configurations that cannot
        # satisfy KNN's sample-count requirement instead of running thousands
        # of experiments that are guaranteed to fail at predict time.
        available = params["lookback"] - params["forward"]
        while available < 2:
            available += 21
        return params["n_neighbors"] <= available

    def generate_positions(
        self, data, lookback=42, forward=5, n_neighbors=10, weights="uniform", p=2, long_short=True, retrain=True
    ):
        from sklearn.neighbors import KNeighborsClassifier

        feats, label = self._prepare_data(data, lookback, forward)
        fn = lambda: KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
        return self._preds_to_positions(
            self._walk_forward(feats, label, fn, train_size=lookback, retrain=retrain, forward=forward), long_short
        )


class MLSVM(_MLBase):
    name = "ml_svm"
    param_grid: ClassVar[dict] = {
        "lookback": [252, 504, 756],
        "forward": [1, 5, 21],
        "C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
        "kernel": ["rbf", "linear", "poly", "sigmoid"],
        "gamma": ["scale", "auto"],
        "long_short": [True, False],
        "retrain": [True],
    }

    def generate_positions(
        self, data, lookback=42, forward=5, C=1.0, kernel="rbf", gamma="scale", long_short=True, retrain=True
    ):
        from sklearn.svm import SVC

        feats, label = self._prepare_data(data, lookback, forward)
        fn = lambda: SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)
        return self._preds_to_positions(
            self._walk_forward(feats, label, fn, train_size=lookback, step=42, retrain=retrain, forward=forward),
            long_short,
        )


class MLNB(_MLBase):
    name = "ml_nb"
    param_grid: ClassVar[dict] = {
        "lookback": [252, 504, 756],
        "forward": [1, 5, 21],
        "var_smoothing": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
        "long_short": [True, False],
        "retrain": [True, False],
    }

    def generate_positions(self, data, lookback=42, forward=5, var_smoothing=1e-7, long_short=True, retrain=True):
        from sklearn.naive_bayes import GaussianNB

        feats, label = self._prepare_data(data, lookback, forward)
        fn = lambda: GaussianNB(var_smoothing=var_smoothing)
        return self._preds_to_positions(
            self._walk_forward(feats, label, fn, train_size=lookback, retrain=retrain, forward=forward), long_short
        )


class MLAda(_MLBase):
    name = "ml_ada"
    param_grid: ClassVar[dict] = {
        "lookback": [252, 504, 756],
        "forward": [1, 5, 21],
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0],
        "base_depth": [1, 2, 3, 5],
        "long_short": [True, False],
        "retrain": [True, False],
    }

    def generate_positions(
        self,
        data,
        lookback=42,
        forward=5,
        n_estimators=100,
        learning_rate=0.1,
        base_depth=3,
        long_short=True,
        retrain=True,
    ):
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier

        feats, label = self._prepare_data(data, lookback, forward)
        fn = lambda: AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=base_depth),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42,
        )
        return self._preds_to_positions(
            self._walk_forward(feats, label, fn, train_size=lookback, retrain=retrain, forward=forward), long_short
        )


class MLExtraTrees(_MLBase):
    name = "ml_extra_trees"
    param_grid: ClassVar[dict] = {
        "lookback": [252, 504, 756],
        "forward": [1, 5, 21],
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 5, 8, 10, None],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", None],
        "long_short": [True, False],
        "retrain": [True, False],
    }

    def generate_positions(
        self,
        data,
        lookback=42,
        forward=5,
        n_estimators=100,
        max_depth=5,
        min_samples_split=2,
        max_features="sqrt",
        long_short=True,
        retrain=True,
    ):
        from sklearn.ensemble import ExtraTreesClassifier

        feats, label = self._prepare_data(data, lookback, forward)
        fn = lambda: ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=42,
            n_jobs=1,
        )
        return self._preds_to_positions(
            self._walk_forward(feats, label, fn, train_size=lookback, retrain=retrain, forward=forward), long_short
        )


class Ensemble(BaseStrategy):
    name = "ensemble"
    param_grid: ClassVar[dict] = {
        "strategies": [
            ("tsmom_21", "ma_cross_10_50", "macd_12_26"),
            ("tsmom_63", "rsi_14", "donchian_20"),
            ("tsmom_21", "tsmom_63", "ma_cross_5_20", "rsi_14", "macd_12_26"),
            ("tsmom_42", "ma_cross_10_50", "bollinger_20_2", "donchian_55"),
            ("tsmom_21", "tsmom_63", "tsmom_126", "ma_cross_10_50", "ma_cross_5_20"),
            ("tsmom_5", "tsmom_10", "tsmom_21", "tsmom_42", "tsmom_63", "tsmom_126"),
            ("ma_cross_5_20", "ma_cross_10_50", "ma_cross_20_100", "donchian_20", "donchian_55"),
            ("tsmom_21", "donchian_20", "donchian_55", "bollinger_20_2", "rsi_14"),
        ],
        "vote_threshold": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "long_short": [True, False],
    }

    def generate_positions(
        self, data, strategies=("tsmom_21", "ma_cross_10_50", "macd_12_26"), vote_threshold=0.5, long_short=True
    ):
        close = data["close"]
        signals = []
        for s in strategies:
            parts = s.split("_")
            sn = parts[0]
            if sn == "tsmom":
                sig = TSMOM().generate_positions(data, lookback=int(parts[1]), threshold=0.0, long_short=long_short)
            elif sn == "ma" and parts[1] == "cross":
                sig = MACross().generate_positions(data, fast=int(parts[2]), slow=int(parts[3]), long_short=long_short)
            elif sn == "macd":
                sig = MACD().generate_positions(data, fast=int(parts[1]), slow=int(parts[2]), long_short=long_short)
            elif sn == "rsi":
                sig = RSI().generate_positions(
                    data, period=int(parts[1]), buy_threshold=50, sell_threshold=50, long_short=long_short
                )
            elif sn == "donchian":
                sig = Donchian().generate_positions(data, period=int(parts[1]), long_short=long_short)
            elif sn == "bollinger":
                sig = Bollinger().generate_positions(
                    data, period=int(parts[1]), num_std=float(parts[2]), long_short=long_short
                )
            else:
                # Silently dropping members would shrink the vote pool and
                # quietly change the meaning of vote_threshold.
                raise ValueError(f"Unknown ensemble member strategy: '{s}'")
            signals.append(sig)
        if not signals:
            return pd.Series(0.0, index=close.index)
        avg = pd.concat(signals, axis=1).mean(axis=1)
        pos = pd.Series(0.0, index=close.index)
        pos[avg > vote_threshold] = 1.0
        if long_short:
            pos[avg < -vote_threshold] = -1.0
        return pos


class Stacked(BaseStrategy):
    name = "stacked"
    param_grid: ClassVar[dict] = {
        "momentum_lb": [10, 21, 42, 63, 84, 126, 168, 252],
        "ma_filter": [10, 20, 50, 100, 150, 200],
        "base_strategy": ["tsmom", "ma_cross", "donchian", "roc", "acceleration"],
        "base_lookback": [5, 10, 21, 42, 55, 63, 84, 126],
        "long_short": [True, False],
        "exit_on_neg": [True, False],
    }

    def generate_positions(
        self,
        data,
        momentum_lb=63,
        ma_filter=50,
        base_strategy="tsmom",
        base_lookback=42,
        long_short=True,
        exit_on_neg=True,
    ):
        close = data["close"]
        if base_strategy == "tsmom":
            base = TSMOM().generate_positions(data, lookback=base_lookback, long_short=long_short)
        elif base_strategy == "ma_cross":
            base = MACross().generate_positions(
                data, fast=base_lookback // 4, slow=base_lookback, long_short=long_short
            )
        elif base_strategy == "donchian":
            base = Donchian().generate_positions(data, period=base_lookback, long_short=long_short)
        elif base_strategy == "roc":
            base = ROC().generate_positions(data, period=base_lookback, long_short=long_short)
        elif base_strategy == "acceleration":
            base = Accel().generate_positions(
                data, short_lb=max(3, base_lookback // 4), long_lb=base_lookback, long_short=long_short
            )
        else:
            base = pd.Series(0.0, index=close.index)
        mom = _returns(data, momentum_lb)
        ma = _moving_average(data, ma_filter)
        pos = base.copy()
        if exit_on_neg:
            # Direction-aware trend filter: longs are dropped once the
            # higher-timeframe momentum turns non-positive or price loses
            # the filter MA, shorts once momentum turns non-negative or
            # price reclaims the MA.  Warm-up (NaN) stays flat.  The old
            # single-rule form zeroed shorts exactly while momentum was
            # negative, i.e. only when shorts were supposed to be active.
            stale = mom.isna() | ma.isna()
            pos[(pos > 0) & (stale | (mom <= 0) | (close < ma))] = 0.0
            pos[(pos < 0) & (stale | (mom >= 0) | (close > ma))] = 0.0
        return pos


class RegimeAware(BaseStrategy):
    name = "regime_aware"
    param_grid: ClassVar[dict] = {
        "adx_trend_threshold": [15, 20],
        "adx_smooth": [0, 3],
        "regime_confirm": [1],
        "vol_fast": [5, 10],
        "crisis_vol_mult": [2.0],
        "mom_lookback": [21, 42, 63],
        "mom_threshold": [0.0],
        "vol_target_normal": [0.12, 0.15],
        "vol_target_crisis": [0.05],
        "choppy_bull_mode": ["full_vol"],
        "fast_exit_days": [3, 5, 10],
        "fast_exit_threshold": [-0.02, -0.03, -0.05],
        "bearish_mode": ["cash", "short"],
    }
    UNIVERSAL_PARAMS: ClassVar[dict] = {"signal_smooth": [0, 5]}

    def generate_positions(
        self,
        data,
        adx_trend_threshold=20,
        adx_smooth=3,
        regime_confirm=1,
        vol_fast=5,
        crisis_vol_mult=2.0,
        mom_lookback=42,
        mom_threshold=0.0,
        vol_target_normal=0.12,
        vol_target_crisis=0.05,
        choppy_bull_mode="full_vol",
        fast_exit_days=5,
        fast_exit_threshold=-0.03,
        bearish_mode="cash",
    ):
        close = data["close"]
        high = data.get("high", close)
        low = data.get("low", close)
        graph = _dag(data)
        adx_raw = graph.adx(14) if graph is not None else self._compute_adx(high, low, close, 14)
        if adx_smooth > 0:
            adx = (
                graph.node(
                    ("adx_smooth", 14, int(adx_smooth)),
                    lambda: adx_raw.ewm(span=adx_smooth, adjust=False).mean(),
                )
                if graph is not None
                else adx_raw.ewm(span=adx_smooth, adjust=False).mean()
            )
        else:
            adx = adx_raw
        ann = data.get("annualization", 252)
        dr = _returns(data)
        if graph is not None:
            vol_f = graph.node(
                ("return_volatility", int(vol_fast), float(ann)),
                lambda: dr.rolling(vol_fast).std() * np.sqrt(ann),
            )
            vol_s = graph.node(
                ("return_volatility", 63, float(ann)),
                lambda: dr.rolling(63).std() * np.sqrt(ann),
            )
        else:
            vol_f = dr.rolling(vol_fast).std() * np.sqrt(ann)
            vol_s = dr.rolling(63).std() * np.sqrt(ann)
        vol_ratio = vol_f / (vol_s + 1e-10)
        ma_f = _moving_average(data, 50)
        ma_s = _moving_average(data, 200)
        mom = _returns(data, mom_lookback)
        is_crisis = vol_ratio > crisis_vol_mult
        is_trending = adx > adx_trend_threshold
        if regime_confirm > 1:
            is_crisis = is_crisis.rolling(regime_confirm).sum() >= regime_confirm
            is_trending = is_trending.rolling(regime_confirm).sum() >= regime_confirm
        is_bullish = (close > ma_f) & (ma_f > ma_s) & (mom > mom_threshold)
        is_bearish = (close < ma_f) & (ma_f < ma_s) & (mom < -mom_threshold)
        pos = pd.Series(0.0, index=close.index)
        vpn = self._vol_scale_pos(mom, vol_f, vol_target_normal, mom_threshold, 2.0)
        vpc = self._vol_scale_pos(mom, vol_f, vol_target_crisis, mom_threshold, 2.0)
        pos[is_trending & is_bullish & ~is_crisis] = vpn[is_trending & is_bullish & ~is_crisis]
        pos[is_crisis & is_bullish] = vpc[is_crisis & is_bullish]
        mask = ~is_trending & is_bullish & ~is_crisis
        if choppy_bull_mode == "full_vol":
            pos[mask] = vpn[mask]
        elif choppy_bull_mode == "half_vol":
            pos[mask] = vpn[mask] * 0.5
        mask = is_trending & is_bearish & ~is_crisis
        if bearish_mode == "short":
            pos[mask] = vpn[mask]
        mask = ~is_trending & is_bearish & ~is_crisis
        if bearish_mode == "short":
            pos[mask] = vpn[mask] * 0.5
        pos[is_crisis & ~is_bullish] = vpc[is_crisis & ~is_bullish] * 0.3
        if fast_exit_days > 0 and fast_exit_threshold < 0:
            fr = _returns(data, fast_exit_days)
            pos[(fr < fast_exit_threshold) & (pos > 0)] *= 0.3
        return pos

    def _compute_adx(self, high, low, close, period):
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        um = high - high.shift(1)
        dm = low.shift(1) - low
        pdm = pd.Series(0.0, index=close.index)
        pdm[(um > dm) & (um > 0)] = um[(um > dm) & (um > 0)]
        mdm = pd.Series(0.0, index=close.index)
        mdm[(dm > um) & (dm > 0)] = dm[(dm > um) & (dm > 0)]
        atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
        pdi = 100 * (pdm.ewm(alpha=1.0 / period, adjust=False).mean() / (atr + 1e-10))
        mdi = 100 * (mdm.ewm(alpha=1.0 / period, adjust=False).mean() / (atr + 1e-10))
        dx = 100 * ((pdi - mdi).abs() / (pdi + mdi + 1e-10))
        return dx.ewm(alpha=1.0 / period, adjust=False).mean()

    def _vol_scale_pos(self, mom, vol, vol_target, threshold, max_lev):
        raw = pd.Series(0.0, index=mom.index)
        raw[mom > threshold] = 1.0
        raw[mom < -threshold] = -1.0
        return (raw * (vol_target / (vol + 1e-10))).clip(-max_lev, max_lev)


STRATEGY_REGISTRY = {
    "tsmom": TSMOM,
    "ma_cross": MACross,
    "macd": MACD,
    "rsi": RSI,
    "roc": ROC,
    "bollinger": Bollinger,
    "donchian": Donchian,
    "dual_momentum": DualMomentum,
    "triple_ma": TripleMA,
    "vol_scale_mom": VolScale,
    "acceleration": Accel,
    "zscore": ZScore,
    "heikin_ashi": HeikinAshi,
    "supertrend": Supertrend,
    "multi_breakout": MultiBreakout,
    "ml_logreg": MLLogReg,
    "ml_rf": MLRF,
    "ml_xgb": MLXGB,
    "ml_knn": MLKNN,
    "ml_svm": MLSVM,
    "ml_nb": MLNB,
    "ml_ada": MLAda,
    "ml_extra_trees": MLExtraTrees,
    "ensemble": Ensemble,
    "stacked": Stacked,
    "regime_aware": RegimeAware,
}

CLASSIC_STRATEGIES = [k for k in STRATEGY_REGISTRY if not k.startswith("ml_")]
ML_STRATEGIES = [k for k in STRATEGY_REGISTRY if k.startswith("ml_")]
COMBO_STRATEGIES = ["ensemble", "stacked", "regime_aware"]


def get_strategy(name):
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name]()


def list_strategies():
    print(f"\n{'=' * 70}")
    print(f"  Strategy Registry ({len(STRATEGY_REGISTRY)} strategies)")
    print(f"{'=' * 70}")
    total = 0
    for name, cls in STRATEGY_REGISTRY.items():
        n = cls().count_param_combinations()
        cat = "ML" if name.startswith("ml_") else ("combo" if name in COMBO_STRATEGIES else "classic")
        print(f"  [{cat:>5s}] {name:>20s}: {n:>8d} params")
        total += n
    print(f"  {'-' * 50}")
    print(f"  {'Total':>25s}: {total:>8d} experiments")
    print(f"{'=' * 70}\n")
    return total
