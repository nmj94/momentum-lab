"""strategies.py - 26 momentum strategies with exhaustive parameter grids.

Classes:
    BaseStrategy: Abstract base with universal params (position_size, signal_smooth)
    TSMOM, MACross, MACD, RSI, ROC, Bollinger, Donchian, DualMomentum,
    TripleMA, VolScale, Accel, ZScore, HeikinAshi, Supertrend, MultiBreakout
    _MLBase, MLLogReg, MLRF, MLXGB, MLKNN, MLSVM, MLNB, MLAda, MLExtraTrees
    Ensemble, Stacked, RegimeAware
"""

from itertools import product
from typing import ClassVar

import numpy as np
import pandas as pd


def _wma(series: pd.Series, period: int) -> pd.Series:
    """Vectorized weighted moving average (weight 1..period, newest heaviest).

    O(period) memory via convolution instead of pandas rolling().apply().
    """
    n = len(series)
    if period <= 1:
        return series.copy()
    weights = np.arange(1, period + 1, dtype=float)
    conv = np.convolve(series.to_numpy(dtype=float), weights[::-1])
    out = np.full(n, np.nan)
    out[period - 1 :] = conv[period - 1 : n] / weights.sum()
    return pd.Series(out, index=series.index)


class BaseStrategy:
    name = "base"
    param_grid: ClassVar[dict] = {}
    UNIVERSAL_PARAMS: ClassVar[dict] = {
        "position_size": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
        "signal_smooth": [0, 2, 3, 5, 10],
    }

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

    def get_param_combinations(self):
        all_params = {**self.param_grid, **self.UNIVERSAL_PARAMS}
        keys = list(all_params.keys())
        if not keys:
            return [{}]
        vals = [all_params[k] for k in keys]
        return [params for params in (dict(zip(keys, v)) for v in product(*vals)) if self.is_valid_params(params)]


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
        if skip_recent > 0:
            ret = close.shift(skip_recent).pct_change(lookback)
        else:
            ret = close.pct_change(lookback)
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

        def ma(series, period, mtype):
            if mtype == "sma":
                return series.rolling(period).mean()
            if mtype == "ema":
                return series.ewm(span=period, adjust=False).mean()
            if mtype == "wma":
                return _wma(series, period)
            if mtype == "dema":
                e1 = series.ewm(span=period, adjust=False).mean()
                e2 = e1.ewm(span=period, adjust=False).mean()
                return 2 * e1 - e2
            return series.rolling(period).mean()

        diff = ma(close, fast, ma_type) - ma(close, slow, ma_type)
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
        macd_line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
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
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
        if rsi_smooth > 1:
            rsi = rsi.ewm(span=rsi_smooth, adjust=False).mean()
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
        roc = close.pct_change(period)
        if smoothing > 0:
            roc = roc.ewm(span=smoothing, adjust=False).mean()
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
        ma = close.rolling(period).mean()
        std = close.rolling(period).std()
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
        high = data.get("high", close)
        low = data.get("low", close)
        upper = high.rolling(period).max().shift(1)
        lower = low.rolling(period).min().shift(1)
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
        exit_upper = high.rolling(ep).max().shift(1)
        exit_lower = low.rolling(ep).min().shift(1)
        pos = pd.Series(0.0, index=close.index)
        state = 0.0
        for i in range(len(close)):
            if (state > 0 and close.iloc[i] < exit_lower.iloc[i]) or (
                state < 0 and close.iloc[i] > exit_upper.iloc[i]
            ):
                state = 0.0
            if state == 0.0:
                if pd.notna(bu.iloc[i]) and bool(bu.iloc[i]):
                    state = 1.0
                elif long_short and pd.notna(bd.iloc[i]) and bool(bd.iloc[i]):
                    state = -1.0
            pos.iloc[i] = state
        return pos


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
        ret = close.pct_change(lookback)
        if smoothing > 0:
            ret = ret.ewm(span=smoothing, adjust=False).mean()
        pos = pd.Series(0.0, index=close.index)
        pos[ret > abs_threshold] = 1.0
        if long_short:
            pos[ret <= -abs_threshold] = -1.0
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

        def cm(s, p):
            if ma_type == "ema":
                return s.ewm(span=p, adjust=False).mean()
            if ma_type == "wma":
                return _wma(s, p)
            return s.rolling(p).mean()

        bullish = (cm(close, fast) > cm(close, medium)) & (cm(close, medium) > cm(close, slow))
        pos = pd.Series(0.0, index=close.index)
        pos[bullish] = 1.0
        if long_short:
            bearish = (cm(close, fast) < cm(close, medium)) & (cm(close, medium) < cm(close, slow))
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
        ret = close.pct_change(lookback)
        vol = close.pct_change().rolling(vol_lookback).std() * np.sqrt(252)
        raw = pd.Series(0.0, index=close.index)
        raw[ret > threshold] = 1.0
        raw[ret < -threshold] = -1.0
        return (raw * (vol_target / (vol + 1e-10))).clip(-2.0, 2.0)


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
        accel = close.pct_change(short_lb) - close.pct_change(long_lb)
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
        z = (close - close.rolling(lookback).mean()) / (close.rolling(lookback).std() + 1e-10)
        pos = pd.Series(0.0, index=close.index)
        if mode == "momentum":
            pos[z > entry_z] = 1.0
            pos[z.abs() < exit_z] = 0.0
            if long_short:
                pos[z < -entry_z] = -1.0
        else:
            pos[z < -entry_z] = 1.0
            pos[z.abs() < exit_z] = 0.0
            if long_short:
                pos[z > entry_z] = -1.0
        return pos


class HeikinAshi(BaseStrategy):
    name = "heikin_ashi"
    param_grid: ClassVar[dict] = {"smooth": [1, 2, 3, 5, 8, 10], "long_short": [True, False], "confirmation": [1, 2, 3]}

    def generate_positions(self, data, smooth=1, long_short=True, confirmation=1):
        close = data["close"]
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
        hi = data.get("high", close)
        lo = data.get("low", close)
        sigs = []
        for p in periods:
            u = hi.rolling(p).max().shift(1)
            l = lo.rolling(p).min().shift(1)
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
    UNIVERSAL_PARAMS: ClassVar[dict] = {"position_size": [0.5, 1.0, 2.0], "signal_smooth": [0, 5]}

    def _prepare_data(self, data, lookback, forward, feature_cols=None):
        from .data import compute_features

        close = data["close"]
        if "features" not in data or data["features"] is None:
            feats = compute_features(pd.DataFrame({"close": close}))
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
        feats = feats.copy()
        feats[label.isna() | feats.isna().any(axis=1)] = np.nan
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

        valid = ~(feats.isna().any(axis=1) | label.isna())
        original_positions = np.flatnonzero(valid.to_numpy())
        feats = feats[valid]
        label = label[valid]
        preds = pd.Series(np.nan, index=feats.index)
        n = len(feats)
        i = train_size
        while i < n:
            # Use original bar offsets rather than compressed valid-row
            # offsets so intermittent feature gaps cannot weaken the purge.
            prediction_bar = original_positions[i]
            train_mask = original_positions[:i] + forward + embargo <= prediction_bar
            if train_mask.sum() < 2:
                i += step
                continue
            scaler = StandardScaler()
            X = scaler.fit_transform(feats.iloc[:i].values[train_mask])
            model = model_fn()
            model.fit(X, label.iloc[:i].values[train_mask])
            pe = min(i + step, n)
            preds.iloc[i:pe] = model.predict(scaler.transform(feats.iloc[i:pe].values))
            if not retrain:
                rest = feats.iloc[pe:]
                if len(rest) > 0:
                    preds.iloc[pe:] = model.predict(scaler.transform(rest.values))
                break
            i = pe
        return preds

    def _preds_to_positions(self, preds, long_short=True):
        pos = pd.Series(0.0, index=preds.index)
        pos[preds == 1] = 1.0
        if long_short:
            pos[preds == 0] = -1.0
        return pos


class MLLogReg(_MLBase):
    name = "ml_logreg"
    param_grid: ClassVar[dict] = {
        "lookback": [21, 42, 63, 126],
        "forward": [1, 3, 5, 10, 21],
        "C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
        "penalty": ["l2", "l1"],
        "long_short": [True, False],
        "retrain": [True, False],
    }

    def generate_positions(self, data, lookback=42, forward=5, C=1.0, penalty="l2", long_short=True, retrain=True):
        from sklearn.linear_model import LogisticRegression

        feats, label = self._prepare_data(data, lookback, forward)
        # Use elastic-net with a full L1 ratio for the requested L1 variant;
        # passing l1_ratio alone leaves LogisticRegression's default penalty
        # as L2 and makes both grid branches equivalent.
        model_penalty = "elasticnet" if penalty == "l1" else "l2"
        l1_ratio = 1.0 if penalty == "l1" else None
        fn = lambda: LogisticRegression(
            C=C,
            penalty=model_penalty,
            l1_ratio=l1_ratio,
            solver="saga",
            max_iter=2000,
            random_state=42,
        )
        return self._preds_to_positions(
            self._walk_forward(feats, label, fn, train_size=lookback, retrain=retrain, forward=forward), long_short
        )


class MLRF(_MLBase):
    name = "ml_rf"
    param_grid: ClassVar[dict] = {
        "lookback": [21, 42, 63],
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
        "lookback": [21, 42, 63],
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
        "lookback": [21, 42, 63],
        "forward": [1, 5, 21],
        "n_neighbors": [3, 5, 7, 10, 15, 21, 30, 50],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
        "long_short": [True, False],
        "retrain": [True, False],
    }

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
        "lookback": [21, 42, 63],
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
        "lookback": [21, 42, 63],
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
        "lookback": [21, 42, 63],
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
        "lookback": [21, 42, 63],
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
                continue
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
        mom = close.pct_change(momentum_lb)
        ma = close.rolling(ma_filter).mean()
        pos = base.copy()
        if exit_on_neg:
            pos[(mom <= 0) | (close < ma)] = 0.0
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
    UNIVERSAL_PARAMS: ClassVar[dict] = {"position_size": [1.0, 1.5, 2.0], "signal_smooth": [0, 5]}

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
        position_size=2.0,
        signal_smooth=0,
    ):
        close = data["close"]
        high = data.get("high", close)
        low = data.get("low", close)
        adx_raw = self._compute_adx(high, low, close, 14)
        adx = adx_raw.ewm(span=adx_smooth, adjust=False).mean() if adx_smooth > 0 else adx_raw
        dr = close.pct_change()
        vol_f = dr.rolling(vol_fast).std() * np.sqrt(252)
        vol_s = dr.rolling(63).std() * np.sqrt(252)
        vol_ratio = vol_f / (vol_s + 1e-10)
        ma_f = close.rolling(50).mean()
        ma_s = close.rolling(200).mean()
        mom = close.pct_change(mom_lookback)
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
        pos[is_crisis & is_bullish] = vpn[is_crisis & is_bullish]
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
            pos[mask] = vpc[mask] * 0.5
        pos[is_crisis & ~is_bullish] = vpc[is_crisis & ~is_bullish] * 0.3
        if fast_exit_days > 0 and fast_exit_threshold < 0:
            fr = close.pct_change(fast_exit_days)
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
        n = len(cls().get_param_combinations())
        cat = "ML" if name.startswith("ml_") else ("combo" if name in COMBO_STRATEGIES else "classic")
        print(f"  [{cat:>5s}] {name:>20s}: {n:>8d} params")
        total += n
    print(f"  {'-' * 50}")
    print(f"  {'Total':>25s}: {total:>8d} experiments")
    print(f"{'=' * 70}\n")
    return total
