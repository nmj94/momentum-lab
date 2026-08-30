"""Bounded, reusable indicator dependency graph.

The search engine evaluates many strategies over the same immutable market
snapshot.  Indicator nodes are therefore safe to share across candidates as
long as each graph is scoped to one data snapshot.  ``IndicatorDAG`` keeps a
bounded LRU of those nodes so exhaustive searches do not trade CPU savings for
unbounded memory growth.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd


def weighted_moving_average(series: pd.Series, period: int) -> pd.Series:
    """Return a vectorized weighted moving average, newest value heaviest."""
    n = len(series)
    if period <= 1:
        return series.copy()
    weights = np.arange(1, period + 1, dtype=float)
    conv = np.convolve(series.to_numpy(dtype=float), weights[::-1])
    out = np.full(n, np.nan)
    out[period - 1 :] = conv[period - 1 : n] / weights.sum()
    return pd.Series(out, index=series.index)


class IndicatorDAG:
    """A data-snapshot-scoped, bounded LRU cache of indicator nodes."""

    def __init__(self, data: Mapping[str, Any], max_entries: int = 256):
        if isinstance(max_entries, bool) or not isinstance(max_entries, (int, np.integer)) or max_entries < 0:
            raise ValueError("max_entries must be a non-negative integer")
        self.data = data
        self.max_entries = int(max_entries)
        self._nodes: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def node(self, key: tuple[Any, ...], factory: Callable[[], Any]):
        """Return one node, computing and retaining it on a cache miss."""
        if key in self._nodes:
            self.hits += 1
            self._nodes.move_to_end(key)
            return self._nodes[key]
        self.misses += 1
        value = factory()
        if self.max_entries:
            self._nodes[key] = value
            self._nodes.move_to_end(key)
            while len(self._nodes) > self.max_entries:
                self._nodes.popitem(last=False)
                self.evictions += 1
        return value

    def series(self, field: str = "close", fallback: str | None = None) -> pd.Series:
        value = self.data.get(field)
        if value is None and fallback is not None:
            value = self.data[fallback]
        if not isinstance(value, pd.Series):
            raise TypeError(f"indicator source '{field}' must be a pandas Series")
        return value

    def returns(self, period: int = 1, field: str = "close", shift: int = 0) -> pd.Series:
        key = ("returns", field, int(period), int(shift))
        return self.node(key, lambda: self.series(field).shift(shift).pct_change(period))

    def difference(self, period: int = 1, field: str = "close") -> pd.Series:
        key = ("difference", field, int(period))
        return self.node(key, lambda: self.series(field).diff(period))

    def moving_average(self, period: int, kind: str = "sma", field: str = "close") -> pd.Series:
        period = int(period)
        key = ("moving_average", field, period, kind)

        def compute():
            series = self.series(field)
            if kind == "sma":
                return series.rolling(period).mean()
            if kind == "ema":
                return series.ewm(span=period, adjust=False).mean()
            if kind == "wma":
                return weighted_moving_average(series, period)
            if kind == "dema":
                first = self.moving_average(period, "ema", field)
                second_key = ("ema_of_ema", field, period)
                second = self.node(second_key, lambda: first.ewm(span=period, adjust=False).mean())
                return 2 * first - second
            raise ValueError(f"unknown moving-average kind: {kind}")

        return self.node(key, compute)

    def rolling_std(self, period: int, field: str = "close") -> pd.Series:
        key = ("rolling_std", field, int(period))
        return self.node(key, lambda: self.series(field).rolling(period).std())

    def rolling_max(self, period: int, field: str = "high", shift: int = 0) -> pd.Series:
        key = ("rolling_max", field, int(period), int(shift))
        return self.node(key, lambda: self.series(field, "close").rolling(period).max().shift(shift))

    def rolling_min(self, period: int, field: str = "low", shift: int = 0) -> pd.Series:
        key = ("rolling_min", field, int(period), int(shift))
        return self.node(key, lambda: self.series(field, "close").rolling(period).min().shift(shift))

    def true_range(self) -> pd.Series:
        def compute():
            close = self.series("close")
            high = self.series("high", "close")
            low = self.series("low", "close")
            return pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(
                axis=1
            )

        return self.node(("true_range",), compute)

    def atr(self, period: int, kind: str = "sma") -> pd.Series:
        period = int(period)
        key = ("atr", period, kind)

        def compute():
            true_range = self.true_range()
            if kind == "sma":
                return true_range.rolling(period).mean()
            if kind == "wilder":
                return true_range.ewm(alpha=1.0 / period, adjust=False).mean()
            raise ValueError(f"unknown ATR kind: {kind}")

        return self.node(key, compute)

    def rsi(self, period: int) -> pd.Series:
        period = int(period)
        key = ("rsi", period)

        def compute():
            delta = self.difference()
            gain_key = ("rsi_gain", period)
            loss_key = ("rsi_loss", period)
            gain = self.node(gain_key, lambda: delta.clip(lower=0).rolling(period).mean())
            loss = self.node(loss_key, lambda: (-delta.clip(upper=0)).rolling(period).mean())
            return 100 - (100 / (1 + gain / (loss + 1e-10)))

        return self.node(key, compute)

    def zscore(self, period: int, field: str = "close") -> pd.Series:
        period = int(period)
        key = ("zscore", field, period)
        return self.node(
            key,
            lambda: (
                (self.series(field) - self.moving_average(period, "sma", field))
                / (self.rolling_std(period, field) + 1e-10)
            ),
        )

    def adx(self, period: int = 14) -> pd.Series:
        period = int(period)
        key = ("adx", period)

        def compute():
            close = self.series("close")
            high = self.series("high", "close")
            low = self.series("low", "close")
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low
            positive = pd.Series(0.0, index=close.index)
            positive[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
            negative = pd.Series(0.0, index=close.index)
            negative[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]
            atr = self.atr(period, "wilder")
            positive_di = 100 * (positive.ewm(alpha=1.0 / period, adjust=False).mean() / (atr + 1e-10))
            negative_di = 100 * (negative.ewm(alpha=1.0 / period, adjust=False).mean() / (atr + 1e-10))
            dx = 100 * ((positive_di - negative_di).abs() / (positive_di + negative_di + 1e-10))
            return dx.ewm(alpha=1.0 / period, adjust=False).mean()

        return self.node(key, compute)

    def snapshot(self) -> dict[str, int | float]:
        requests = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "entries": len(self._nodes),
            "hit_rate": self.hits / requests if requests else 0.0,
        }

    def delta(self, before: Mapping[str, int | float]) -> dict[str, int]:
        """Return operational counter changes since ``before``."""
        return {key: int(self.snapshot()[key]) - int(before.get(key, 0)) for key in ("hits", "misses", "evictions")}
