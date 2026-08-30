"""Conditional return-level uncertainty, never a candidate-selection score.

Circular moving blocks preserve local time dependence and strategy/benchmark
pairs. They do not replay trading, correct selection bias, or repair test peeking.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd

BOOTSTRAP_SCHEMA_VERSION = 1
BOOTSTRAP_METHOD = "paired_circular_moving_block_percentile"
MAX_RESAMPLE_CELLS = 50_000_000
_BATCH_CELLS = 262_144
_MIN_STD = 1e-12
MIN_NOMINAL_BLOCKS = 5
BOOTSTRAP_WARNING = (
    "Post-selection, conditional on the fixed strategy and observed net returns; not selection-adjusted. "
    "Assumes approximately stationary, weakly dependent returns. Circular blocks may wrap within a window, "
    "never across validation/test boundaries. Annualized mean is arithmetic, not CAGR. "
    "Choose settings before viewing results; these intervals do not correct repeated test access."
)
STATISTIC_NAMES = (
    "strategy_annualized_mean",
    "strategy_sharpe",
    "benchmark_annualized_mean",
    "benchmark_sharpe",
    "annualized_mean_excess",
)


def validate_bootstrap_options(*, n_resamples, block_length, confidence_level, seed, min_observations):
    """Validate report-affecting options before data access or search execution."""
    bounds = {
        "n_resamples": (n_resamples, 200, 20_000),
        "block_length": (block_length, 1, 1_000_000),
        "seed": (seed, 0, 2**32 - 1),
        "min_observations": (min_observations, 2, 1_000_000),
    }
    for name, (value, lower, upper) in bounds.items():
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lower <= value <= upper
        ):
            raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    if (
        isinstance(confidence_level, (bool, np.bool_))
        or not isinstance(confidence_level, (int, float, np.integer, np.floating))
        or not math.isfinite(confidence_level)
        or not 0 < confidence_level < 1
    ):
        raise ValueError("confidence_level must be finite and in (0, 1)")
    if n_resamples * (1 - confidence_level) / 2 < 5 - 1e-10:
        raise ValueError("n_resamples must provide at least 5 expected draws per confidence tail")


def _returns_array(series, name):
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if isinstance(series.index, pd.MultiIndex):
        raise TypeError(f"{name} requires a single sorted index, not a MultiIndex")
    if not series.index.is_monotonic_increasing or not series.index.is_unique or series.index.hasnans:
        raise ValueError(f"{name} index must be sorted, unique and non-missing")
    if (
        not pd.api.types.is_numeric_dtype(series.dtype)
        or pd.api.types.is_bool_dtype(series.dtype)
        or pd.api.types.is_complex_dtype(series.dtype)
    ):
        raise ValueError(f"{name} must contain real numeric returns")
    values = series.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite; missing returns are not dropped or filled")
    return values


def _moments(values, annualization, risk_free_rate, axis=None):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean = np.mean(values, axis=axis) * annualization
        std = np.std(values, axis=axis, ddof=1)
        sharpe = np.where(std > _MIN_STD, (mean - risk_free_rate) / (std * math.sqrt(annualization)), np.nan)
    return mean, sharpe, std


def _finite(value):
    return float(value) if np.isfinite(value) else None


def paired_block_bootstrap(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    *,
    n_resamples: int = 2000,
    block_length: int = 10,
    confidence_level: float = 0.95,
    seed: int = 42,
    annualization: float = 252.0,
    risk_free_rate: float = 0.0,
    min_observations: int = 60,
) -> dict:
    """Percentile intervals from identically indexed circular moving blocks.

    Inputs must be already-realized, net simple returns in identical order.
    No alignment, missing-data removal, signal refitting or trade replay occurs.
    Means are arithmetic annualized returns and Sharpe uses sample standard
    deviation (ddof=1), without the display rounding or epsilon in ``evaluate``.
    Insufficient or degenerate data yields null intervals with explicit reasons.
    """
    validate_bootstrap_options(
        n_resamples=n_resamples,
        block_length=block_length,
        confidence_level=confidence_level,
        seed=seed,
        min_observations=min_observations,
    )
    # Normalize before workload arithmetic: NumPy integer multiplication can
    # overflow and otherwise bypass the resource guard for large inputs.
    n_resamples, block_length, seed, min_observations = map(int, (n_resamples, block_length, seed, min_observations))
    confidence_level = float(confidence_level)
    for name, value in (("annualization", annualization), ("risk_free_rate", risk_free_rate)):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{name} must be finite and numeric")
    annualization, risk_free_rate = float(annualization), float(risk_free_rate)
    if annualization <= 0:
        raise ValueError("annualization must be positive")

    strategy = _returns_array(strategy_returns, "strategy_returns")
    benchmark = None
    if benchmark_returns is not None:
        benchmark = _returns_array(benchmark_returns, "benchmark_returns")
        if not strategy_returns.index.equals(benchmark_returns.index):
            raise ValueError("strategy and benchmark indexes must match exactly; no implicit alignment")
    columns = {"strategy": strategy_returns}
    if benchmark_returns is not None:
        columns["benchmark"] = benchmark_returns
    digest = hashlib.sha256()
    digest.update(b"paired-return-snapshot-v1\0")
    digest.update(pd.util.hash_pandas_object(pd.DataFrame(columns), index=True).to_numpy(dtype="<u8").tobytes())
    n = len(strategy)
    required = max(int(min_observations), MIN_NOMINAL_BLOCKS * int(block_length))
    blocks = math.ceil(n / block_length)
    draws_per_sample = blocks * block_length
    statistics = {
        name: {
            "estimate": None,
            "ci": [None, None],
            "status": "not_provided",
            "reason": "Benchmark not provided",
            "valid_resamples": 0,
        }
        for name in STATISTIC_NAMES
    }
    report = {
        "schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "method": BOOTSTRAP_METHOD,
        "status": "unavailable",
        "warning": BOOTSTRAP_WARNING,
        "n_observations": n,
        "required_observations": required,
        "block_length": int(block_length),
        "nominal_nonoverlapping_blocks": n // int(block_length),
        "blocks_per_resample": blocks,
        "n_resamples": int(n_resamples),
        "completed_resamples": 0,
        "confidence_level": float(confidence_level),
        "seed": int(seed),
        "bit_generator": "PCG64",
        "quantile_method": "linear",
        "numpy_version": np.__version__,
        "annualization": float(annualization),
        "risk_free_rate": float(risk_free_rate),
        "paired": benchmark is not None,
        "returns_sha256": digest.hexdigest(),
        "start": str(strategy_returns.index[0]) if n else None,
        "end": str(strategy_returns.index[-1]) if n else None,
        "max_resample_cells": MAX_RESAMPLE_CELLS,
        "statistics": statistics,
    }
    if block_length == 1:
        report["warning"] += " Block length 1 is IID resampling and does not preserve serial dependence."

    sources = {"strategy": strategy}
    if benchmark is not None:
        sources["benchmark"] = benchmark
    original_std = {}
    for name, values in sources.items():
        mean, sharpe, std = (
            _moments(values, annualization, risk_free_rate)
            if n >= 2
            else (values[0] * annualization if n else np.nan, np.nan, np.nan)
        )
        for metric, estimate in ((f"{name}_annualized_mean", mean), (f"{name}_sharpe", sharpe)):
            statistics[metric].update(estimate=_finite(estimate), status="pending", reason=None)
            original_std[metric] = std
    active = None
    if benchmark is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            active = strategy - benchmark
        mean, _, std = (
            _moments(active, annualization, 0.0)
            if n >= 2
            else (active[0] * annualization if n else np.nan, np.nan, np.nan)
        )
        statistics["annualized_mean_excess"].update(estimate=_finite(mean), status="pending", reason=None)
        original_std["annualized_mean_excess"] = std

    unavailable = None
    if n < required:
        unavailable = (
            "insufficient_data",
            f"Need at least {required} observations, including 5 nominal blocks; got {n}",
        )
    elif n_resamples * draws_per_sample > MAX_RESAMPLE_CELLS:
        unavailable = (
            "resource_limit",
            "Resampling workload exceeds the recorded limit; reduce resamples or data length",
        )
    if unavailable:
        report["status"], report["reason"] = unavailable
        for metric in original_std:
            statistics[metric].update(status=unavailable[0], reason=unavailable[1])
        return report

    samples = {name: np.empty(n_resamples, dtype=float) for name in original_std}
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    offsets = np.arange(block_length)
    batch_size = max(1, _BATCH_CELLS // draws_per_sample)
    for first in range(0, n_resamples, batch_size):
        last = min(first + batch_size, n_resamples)
        starts = rng.integers(0, n, size=(last - first, blocks))
        indices = ((starts[:, :, None] + offsets) % n).reshape(last - first, draws_per_sample)[:, :n]
        for name, values in sources.items():
            mean, sharpe, _ = _moments(values[indices], annualization, risk_free_rate, axis=1)
            samples[f"{name}_annualized_mean"][first:last] = mean
            samples[f"{name}_sharpe"][first:last] = sharpe
        if active is not None:
            with np.errstate(over="ignore", invalid="ignore"):
                samples["annualized_mean_excess"][first:last] = active[indices].mean(axis=1) * annualization
    report["completed_resamples"] = int(n_resamples)
    quantiles = [(1 - confidence_level) / 2, (1 + confidence_level) / 2]
    for name, distribution in samples.items():
        statistic = statistics[name]
        valid = int(np.isfinite(distribution).sum())
        statistic["valid_resamples"] = valid
        if not np.isfinite(original_std[name]):
            status, reason = "undefined_statistic", "Statistic is undefined or numerically non-finite"
        elif original_std[name] <= _MIN_STD:
            status, reason = "zero_variance", "Constant or near-constant observed returns cannot estimate uncertainty"
        elif statistic["estimate"] is None:
            status, reason = "undefined_statistic", "Statistic is undefined or numerically non-finite"
        elif valid != n_resamples:
            status, reason = "degenerate_resamples", "Some resampled statistics are undefined; no draws were discarded"
        elif np.ptp(distribution) <= 32 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(distribution)))):
            status, reason = "degenerate_resamples", "Resampling distribution has no meaningful variation"
        else:
            statistic["ci"] = [float(value) for value in np.quantile(distribution, quantiles, method="linear")]
            status, reason = "ok", None
        statistic.update(status=status, reason=reason)
    successful = sum(statistics[name]["status"] == "ok" for name in original_std)
    report["status"] = "ok" if successful == len(original_std) else "partial" if successful else "unavailable"
    return report
