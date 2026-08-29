"""search.py - Exhaustive parameter search engine."""

import copy
import csv
import hashlib
import heapq
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import sqlite3
import subprocess
import time
import warnings
from collections import OrderedDict
from contextlib import closing
from datetime import datetime, timezone
from itertools import combinations, count
from pathlib import Path
from statistics import NormalDist
from uuid import uuid4

import numpy as np
import pandas as pd

from ._version import __version__
from .backtest import RISK_FREE_RATE, backtest, evaluate, get_buy_and_hold
from .config import load_search_config
from .data import infer_annualization, prepare_data
from .robustness import robustness_check
from .strategies import CLASSIC_STRATEGIES, STRATEGY_REGISTRY, get_strategy

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kw):
        return x


RESULT_DIR = Path("experiments")
ENGINE_SCHEMA_VERSION = 3
METRIC_KEYS = (
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
    "n_observations",
    "trade_count",
    "exposure",
    "turnover",
)
RESULT_COLUMNS = [
    "strategy",
    "params",
    "error",
    *(f"{period}_{metric}" for period in ("train", "val") for metric in METRIC_KEYS),
    "val_fold_sharpes",
]

# Shared state for parallel sub-processes (set via Pool initializer).
_POOL_STATE = None
_POOL_POSITION_CACHE = None


def _init_worker(
    data,
    df,
    periods,
    cost_bps,
    risk_free_rate,
    validation_folds,
    execution_price_column,
    backtest_kwargs,
):
    global _POOL_POSITION_CACHE, _POOL_STATE
    _POOL_STATE = (
        data,
        df,
        periods,
        cost_bps,
        risk_free_rate,
        validation_folds,
        execution_price_column,
        backtest_kwargs,
    )
    _POOL_POSITION_CACHE = OrderedDict()


def _worker_run(args):
    """Module-level worker that runs a single experiment (required on Windows)."""
    strategy_name, params = args
    data, df, periods, cost_bps, risk_free_rate, validation_folds, execution_price_column, backtest_kwargs = _POOL_STATE
    return run_single_experiment(
        strategy_name,
        params,
        data,
        df,
        periods,
        cost_bps,
        risk_free_rate,
        validation_folds=validation_folds,
        execution_price_column=execution_price_column,
        position_cache=_POOL_POSITION_CACHE,
        **backtest_kwargs,
    )


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        value = float(v)
        return value if np.isfinite(value) else None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(value) for k, value in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _params_to_str(params):
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except Exception:
            return params[:80]
    if not isinstance(params, dict):
        return str(params)[:80]
    parts = []
    for k, v in params.items():
        if isinstance(v, tuple):
            v = "_".join(str(x) for x in v)
        elif isinstance(v, bool):
            v = "T" if v else "F"
        elif isinstance(v, float):
            v = f"{v:.3f}"
        parts.append(f"{k}={v}")
    return ", ".join(parts)


def _git_revision():
    """Return the source revision, or ``unknown`` outside a Git checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    revision = completed.stdout.strip()
    return revision or "unknown"


def _source_fingerprint():
    """Hash package source so dirty or non-Git installs remain resume-safe."""
    package_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_dir.rglob("*.py"), key=lambda item: str(item.relative_to(package_dir))):
        relative = path.relative_to(package_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _data_snapshot(df):
    """Return a stable content hash for the prepared data frame."""
    digest = hashlib.sha256()
    digest.update("\x1f".join(map(str, df.columns)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _file_fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def _environment_manifest():
    packages = {}
    for distribution in ("numpy", "pandas", "scikit-learn", "xgboost", "yfinance"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _environment_fingerprint(manifest):
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_text_atomic(path, content):
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _write_frame_atomic(frame, path):
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(tmp_path, index=False, encoding="utf-8-sig")
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _results_rows(results):
    rows = []
    for r in results:
        row = {
            "strategy": r.get("strategy", ""),
            "params": json.dumps(
                r.get("params", {}), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_jsonable
            ),
            "error": r.get("error", ""),
        }
        for period in ["train", "val"]:
            m = r.get(f"{period}_metrics", {})
            for metric in METRIC_KEYS:
                row[f"{period}_{metric}"] = m.get(metric, np.nan)
        row["val_fold_sharpes"] = json.dumps(_jsonable(r.get("val_fold_sharpes", [])), separators=(",", ":"))
        rows.append(row)
    return rows


def _append_results_csv(results, path, write_header):
    """Write legacy/interop CSV rows; SQLite is the canonical checkpoint."""
    rows = _results_rows(results)
    if not rows:
        return
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(
        path,
        mode="a",
        header=write_header,
        index=False,
        # BOM only belongs at the very start of the file.
        encoding="utf-8-sig" if write_header else "utf-8",
    )


def _params_key(strategy, params):
    """Return a stable key for identifying a completed grid point."""
    canonical = json.dumps(
        _jsonable(params), sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=_jsonable
    )
    return f"{strategy}\0{canonical}"


def _result_from_row(row, source):
    raw_params = row.get("params", "{}")
    try:
        params = json.loads(raw_params) if isinstance(raw_params, str) else raw_params
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"checkpoint {source} contains invalid params JSON") from exc
    if not isinstance(params, dict):
        raise TypeError(f"checkpoint {source} contains non-object params")
    result = {
        "strategy": str(row.get("strategy", "")),
        "params": params,
        "train_metrics": {},
        "val_metrics": {},
    }
    error = row.get("error", "")
    if error is not None and str(error) and str(error).lower() != "nan":
        result["error"] = str(error)
    for period in ("train", "val"):
        prefix = f"{period}_"
        metrics = result[f"{period}_metrics"]
        for metric in METRIC_KEYS:
            value = row.get(f"{prefix}{metric}")
            if value in (None, "") or (isinstance(value, float) and math.isnan(value)):
                continue
            try:
                metrics[metric] = float(value)
            except (TypeError, ValueError):
                continue
    raw_folds = row.get("val_fold_sharpes", "[]")
    try:
        folds = json.loads(raw_folds) if isinstance(raw_folds, str) else raw_folds
    except (TypeError, json.JSONDecodeError):
        folds = []
    result["val_fold_sharpes"] = [float(value) for value in folds or [] if value is not None]
    return result


def _iter_csv_checkpoint(path):
    """Yield CSV rows without materializing a potentially 400 MB checkpoint."""
    if not path.exists():
        return
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ValueError(f"cannot read search checkpoint {path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        missing = {"strategy", "params"} - set(reader.fieldnames or [])
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"search checkpoint {path} is missing column(s): {names}")
        for line_number, row in enumerate(reader, start=2):
            if None in row or not row.get("strategy") or not row.get("params"):
                warnings.warn(
                    f"Ignoring incomplete checkpoint row {line_number} in {path}; it can be recomputed on resume.",
                    RuntimeWarning,
                )
                continue
            try:
                yield _result_from_row(row, path)
            except (TypeError, ValueError) as exc:
                warnings.warn(f"Ignoring malformed checkpoint row {line_number} in {path}: {exc}", RuntimeWarning)


def _load_checkpoint(path):
    """Compatibility helper; search resume uses the streaming iterator."""
    return list(_iter_csv_checkpoint(path))


def _open_result_store(path):
    connection = sqlite3.connect(path, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    metric_defs = ", ".join(
        f'"{column}" REAL'
        for column in RESULT_COLUMNS
        if column not in {"strategy", "params", "error", "val_fold_sharpes"}
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS results ("
        "strategy TEXT NOT NULL, params TEXT NOT NULL, error TEXT, "
        f"{metric_defs}, val_fold_sharpes TEXT NOT NULL DEFAULT '[]', "
        "PRIMARY KEY (strategy, params))"
    )
    connection.commit()
    return connection


def _store_results(results, path):
    """Commit one checkpoint batch transactionally to the canonical store."""
    rows = _results_rows(results)
    if not rows:
        return
    placeholders = ",".join("?" for _ in RESULT_COLUMNS)
    columns = ",".join(f'"{column}"' for column in RESULT_COLUMNS)
    values = [[row.get(column) for column in RESULT_COLUMNS] for row in rows]
    with closing(_open_result_store(path)) as connection, connection:
        connection.executemany(
            f"INSERT OR REPLACE INTO results ({columns}) VALUES ({placeholders})",
            values,
        )


def _iter_result_store(path, batch_size=10_000):
    if not path.exists():
        return
    with closing(_open_result_store(path)) as connection:
        cursor = connection.execute("SELECT * FROM results ORDER BY rowid")
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            for values in rows:
                yield _result_from_row(dict(zip(RESULT_COLUMNS, values, strict=True)), path)


def _export_result_store(store_path, csv_path):
    """Atomically refresh the human-readable CSV from the SQLite journal."""
    tmp_path = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            for result in _iter_result_store(store_path):
                writer.writerow(_results_rows([result])[0])
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(csv_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _store_counts(path):
    if not path.exists():
        return 0, 0
    with closing(_open_result_store(path)) as connection:
        total, errors = connection.execute(
            "SELECT COUNT(*), SUM(CASE WHEN error IS NOT NULL AND error != '' THEN 1 ELSE 0 END) FROM results"
        ).fetchone()
    return int(total or 0), int(errors or 0)


def _completed_params(path, strategy):
    if not path.exists():
        return set()
    with closing(_open_result_store(path)) as connection:
        rows = connection.execute("SELECT params FROM results WHERE strategy = ?", (strategy,))
        return {row[0] for row in rows}


def _migrate_csv_checkpoint(csv_path, store_path):
    batch = []
    for result in _iter_csv_checkpoint(csv_path):
        batch.append(result)
        if len(batch) >= 10_000:
            _store_results(batch, store_path)
            batch.clear()
    if batch:
        _store_results(batch, store_path)


# Checkpoint metrics are only comparable with a resumed run when source, data,
# search space, and cost/evaluation model match. Presentation-only options such
# as top_n, workers, and sensitivity reporting may differ.
_RESUME_COMPAT_FIELDS = (
    "ticker",
    "strategies",
    "quick",
    "data_snapshot",
    "package_version",
    "engine_schema_version",
    "source_fingerprint",
    "lock_fingerprint",
    "environment_fingerprint",
    "cost_bps",
    "slippage_bps",
    "financing_rate",
    "borrow_bps",
    "cash_rate",
    "short_rebate_rate",
    "max_leverage",
    "execution_model",
    "execution_lag",
    "annualization",
    "risk_free_rate",
    "validation_folds",
    "min_validation_bars",
    "min_validation_trades",
    "min_validation_exposure",
)


def _check_strategy_dependencies(strategy_names):
    """Fail once before a run if an explicitly requested ML extra is absent."""
    if any(name.startswith("ml_") for name in strategy_names) and importlib.util.find_spec("sklearn") is None:
        raise RuntimeError("ML strategies require the optional dependency set: pip install 'momentum-research-lab[ml]'")
    if "ml_xgb" in strategy_names and importlib.util.find_spec("xgboost") is None:
        raise RuntimeError(
            "ml_xgb requires the optional XGBoost dependency set: pip install 'momentum-research-lab[xgb]'"
        )


def _check_resume_compatibility(run_dir, metadata):
    """Reject a resume whose checkpoint was produced under a different config.

    The checkpoint store contains metrics computed with the previous run's data
    snapshot and cost model; resuming with different values would silently
    mix incomparable Sharpe/CAGR numbers in one ranking.  Must be called
    BEFORE ``run_config.json`` is rewritten, while the previous run's file
    is still on disk.
    """
    if not (run_dir / "all_results.csv").exists() and not (run_dir / "results.sqlite3").exists():
        return
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        warnings.warn(
            f"Cannot verify resume compatibility: {config_path} is missing.",
            RuntimeWarning,
        )
        return
    try:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read previous run config {config_path}: {exc}") from exc
    mismatched = [f for f in _RESUME_COMPAT_FIELDS if previous.get(f) != metadata.get(f)]
    if mismatched:
        names = ", ".join(mismatched)
        raise ValueError(
            f"resume configuration mismatch on {names}: the checkpoint in {run_dir} was produced "
            f"under a different configuration. Use a new run_id or restore the original settings."
        )


def _split_periods(index):
    """Split an ordered index into 40% train, 40% validation, 20% sealed test."""
    n = len(index)
    if n < 3:
        raise ValueError("at least 3 data points are required to split search periods")

    split1 = max(1, min(int(n * 0.4), n - 2))
    split2 = max(split1 + 1, min(int(n * 0.8), n - 1))
    return {
        "train": (index[0], index[split1 - 1]),
        "val": (index[split1], index[split2 - 1]),
        "test": (index[split2], index[-1]),
    }


def _quick_sample(strategy, k=5):
    """Pick a deterministic Latin-hypercube sample of the parameter grid."""
    total = strategy.count_param_combinations()
    if k <= 0:
        return []
    if total <= k:
        return list(strategy.iter_param_combinations())
    grid = {**strategy.param_grid, **strategy.UNIVERSAL_PARAMS}
    keys = list(grid)
    values = [grid[key] for key in keys]
    seed = int.from_bytes(hashlib.sha256(strategy.name.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    picked = []
    seen = set()
    # Draw repeated k-point LHS batches because coherence constraints (fast <
    # slow, ordered tuples, etc.) can reject otherwise well-spread points.
    sample_size = k
    for _ in range(64):
        permutations = [rng.permutation(sample_size) for _ in keys]
        jitters = rng.random((len(keys), sample_size))
        for row in range(sample_size):
            params = {}
            for dimension, key in enumerate(keys):
                quantile = (permutations[dimension][row] + jitters[dimension, row]) / sample_size
                choice = min(int(quantile * len(values[dimension])), len(values[dimension]) - 1)
                params[key] = values[dimension][choice]
            canonical = json.dumps(_jsonable(params), sort_keys=True, separators=(",", ":"), default=_jsonable)
            if canonical in seen or not strategy.is_valid_params(params):
                continue
            seen.add(canonical)
            picked.append(params)
            if len(picked) == k:
                return picked
    # A highly constrained custom grid may defeat the LHS attempts.  Fill the
    # remaining slots lazily and deterministically without materializing it.
    for params in strategy.iter_param_combinations():
        canonical = json.dumps(_jsonable(params), sort_keys=True, separators=(",", ":"), default=_jsonable)
        if canonical not in seen:
            picked.append(params)
            seen.add(canonical)
            if len(picked) == k:
                break
    return picked


def _period_metrics(bt, effective_positions, start, end, risk_free_rate, annualization):
    period_returns = bt["returns"].loc[start:end]
    period_trades = bt["trades"].loc[start:end]
    period_exposure = effective_positions.loc[start:end]
    metrics = evaluate(period_returns, risk_free_rate=risk_free_rate, annualization=annualization)
    metrics.update(
        {
            "n_observations": len(period_returns),
            "trade_count": int((period_trades > 1e-12).sum()),
            "exposure": float(period_exposure.abs().mean()) if len(period_exposure) else 0.0,
            "turnover": float(period_trades.sum()),
        }
    )
    if metrics["exposure"] <= 1e-12:
        metrics["sharpe"] = -99.0
    return metrics


def _validation_fold_sharpes(bt, effective_positions, bounds, folds, risk_free_rate, annualization):
    index = bt["returns"].loc[bounds[0] : bounds[1]].index
    if len(index) == 0:
        return []
    return [
        _period_metrics(
            bt,
            effective_positions,
            fold_index[0],
            fold_index[-1],
            risk_free_rate,
            annualization,
        )["sharpe"]
        for fold_index in np.array_split(index, min(folds, len(index)))
        if len(fold_index)
    ]


def _strategy_positions(strategy_name, params, data, position_cache=None, cache_size=32):
    """Reuse the expensive base signal across universal smoothing variants."""
    strategy = get_strategy(strategy_name)
    if not hasattr(strategy, "generate_positions"):
        return strategy.run(data, **params)
    base_params = dict(params)
    position_size = base_params.pop("position_size", 1.0)
    signal_smooth = base_params.pop("signal_smooth", 0)
    canonical = json.dumps(_jsonable(base_params), sort_keys=True, separators=(",", ":"), default=_jsonable)
    key = f"{strategy_name}\0{canonical}"
    raw = position_cache.get(key) if position_cache is not None else None
    if raw is None:
        raw = strategy.generate_positions(data, **base_params)
        if position_cache is not None:
            position_cache[key] = raw
            position_cache.move_to_end(key)
            while len(position_cache) > cache_size:
                position_cache.popitem(last=False)
    elif position_cache is not None:
        position_cache.move_to_end(key)
    positions = raw.ewm(span=signal_smooth, adjust=False).mean() if signal_smooth > 0 else raw
    return positions * position_size


def run_single_experiment(
    strategy_name,
    params,
    data,
    df,
    periods,
    cost_bps=1.0,
    risk_free_rate=RISK_FREE_RATE,
    validation_folds=4,
    execution_price_column="close",
    position_cache=None,
    **backtest_kwargs,
):
    try:
        positions = _strategy_positions(strategy_name, params, data, position_cache=position_cache)
    except Exception as e:
        return {
            "strategy": strategy_name,
            "params": params,
            "error": str(e),
            "train_metrics": {},
            "val_metrics": {},
            "val_fold_sharpes": [],
        }
    if execution_price_column not in df.columns:
        return {
            "strategy": strategy_name,
            "params": params,
            "error": f"execution price column '{execution_price_column}' is unavailable",
            "train_metrics": {"sharpe": -99.0},
            "val_metrics": {"sharpe": -99.0},
            "val_fold_sharpes": [],
        }
    prices = df[execution_price_column]
    results = {
        "strategy": strategy_name,
        "params": params,
        "train_metrics": {},
        "val_metrics": {},
        "val_fold_sharpes": [],
    }
    period_errors = []
    try:
        bt = backtest(positions, prices, cost_bps=cost_bps, **backtest_kwargs)
        execution_lag = int(backtest_kwargs.get("execution_lag", 0))
        effective_positions = positions.reindex(prices.index).ffill().fillna(0.0).shift(execution_lag + 1).fillna(0.0)
        effective_positions = effective_positions.clip(
            -backtest_kwargs.get("max_leverage", 2.0), backtest_kwargs.get("max_leverage", 2.0)
        )
    except Exception as exc:
        return {
            **results,
            "error": str(exc),
            "train_metrics": {"sharpe": -99.0},
            "val_metrics": {"sharpe": -99.0},
        }
    annualization = backtest_kwargs.get("annualization", 252)
    for pname in ("train", "val"):
        start, end = periods[pname]
        try:
            results[f"{pname}_metrics"] = _period_metrics(
                bt,
                effective_positions,
                start,
                end,
                risk_free_rate,
                annualization,
            )
        except Exception as exc:
            results[f"{pname}_metrics"] = {"sharpe": -99}
            period_errors.append(f"{pname}: {exc}")
    results["val_fold_sharpes"] = _validation_fold_sharpes(
        bt,
        effective_positions,
        periods["val"],
        validation_folds,
        risk_free_rate,
        annualization,
    )
    if period_errors:
        results["error"] = "; ".join(period_errors)
    return results


def _is_eligible(result, min_bars, min_trades, min_exposure):
    metrics = result.get("val_metrics", {})
    return (
        not result.get("error")
        and metrics.get("sharpe", -99) > -99
        and metrics.get("n_observations", 0) >= min_bars
        and metrics.get("trade_count", 0) >= min_trades
        and metrics.get("exposure", 0.0) >= min_exposure
    )


def _multiple_testing_hurdle(sharpe_std, trials):
    """Expected maximum annualized Sharpe under independent null trials."""
    if trials <= 1 or not np.isfinite(sharpe_std) or sharpe_std <= 0:
        return 0.0
    normal = NormalDist()
    euler_gamma = 0.5772156649015329
    expected_max_z = (1 - euler_gamma) * normal.inv_cdf(1 - 1 / trials)
    expected_max_z += euler_gamma * normal.inv_cdf(1 - 1 / (trials * math.e))
    return float(sharpe_std * expected_max_z)


def _deflated_sharpe_probability(metrics, hurdle, annualization):
    """Probability that Sharpe exceeds a multiple-testing-adjusted hurdle."""
    observations = int(metrics.get("n_observations", 0))
    if observations < 2:
        return 0.0
    sharpe = float(metrics.get("sharpe", -99.0)) / math.sqrt(annualization)
    benchmark = float(hurdle) / math.sqrt(annualization)
    skew = float(metrics.get("skew", 0.0) or 0.0)
    # pandas reports excess kurtosis; the probabilistic-Sharpe expression uses
    # ordinary kurtosis.
    kurtosis = float(metrics.get("kurtosis", 0.0) or 0.0) + 3.0
    variance_term = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe**2
    z_score = (sharpe - benchmark) * math.sqrt(observations - 1) / math.sqrt(max(variance_term, 1e-12))
    return float(NormalDist().cdf(z_score))


def _sharpe_confidence_interval(metrics, annualization, confidence_z=1.959963984540054):
    observations = int(metrics.get("n_observations", 0))
    if observations < 2:
        return [None, None]
    annual_sharpe = float(metrics.get("sharpe", 0.0))
    sharpe = annual_sharpe / math.sqrt(annualization)
    skew = float(metrics.get("skew", 0.0) or 0.0)
    kurtosis = float(metrics.get("kurtosis", 0.0) or 0.0) + 3.0
    variance_term = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe**2
    annual_standard_error = math.sqrt(max(variance_term, 1e-12) / (observations - 1)) * math.sqrt(annualization)
    return [
        annual_sharpe - confidence_z * annual_standard_error,
        annual_sharpe + confidence_z * annual_standard_error,
    ]


def _estimate_pbo(fold_rows):
    """Estimate CSCV Probability of Backtest Overfitting from validation folds."""
    if len(fold_rows) < 2:
        return None
    matrix = np.asarray(fold_rows, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2 or matrix.shape[1] % 2:
        return None
    logits = []
    all_columns = set(range(matrix.shape[1]))
    for in_sample_columns in combinations(range(matrix.shape[1]), matrix.shape[1] // 2):
        out_sample_columns = sorted(all_columns - set(in_sample_columns))
        in_scores = matrix[:, in_sample_columns].mean(axis=1)
        selected = int(np.argmax(in_scores))
        out_scores = matrix[:, out_sample_columns].mean(axis=1)
        percentile = float((out_scores <= out_scores[selected]).mean())
        percentile = min(max(percentile, 1e-9), 1 - 1e-9)
        logits.append(math.log(percentile / (1 - percentile)))
    return {
        "probability": float((np.asarray(logits) <= 0).mean()),
        "combinations": len(logits),
        "candidates": int(matrix.shape[0]),
        "folds": int(matrix.shape[1]),
    }


def _walk_forward_selection_diagnostic(fold_rows):
    """Replay expanding-fold parameter selection and score the next fold."""
    if len(fold_rows) < 2:
        return None
    matrix = np.asarray(fold_rows, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        return None
    steps = []
    for outer_fold in range(1, matrix.shape[1]):
        inner_scores = matrix[:, :outer_fold].mean(axis=1)
        selected = int(np.argmax(inner_scores))
        steps.append(
            {
                "outer_fold": outer_fold + 1,
                "selected_candidate_index": selected,
                "inner_mean_sharpe": float(inner_scores[selected]),
                "outer_sharpe": float(matrix[selected, outer_fold]),
            }
        )
    outer_scores = [step["outer_sharpe"] for step in steps]
    return {
        "steps": steps,
        "mean_outer_sharpe": float(np.mean(outer_scores)),
        "worst_outer_sharpe": float(np.min(outer_scores)),
        "positive_outer_fraction": float((np.asarray(outer_scores) > 0).mean()),
    }


def _select_from_store(
    store_path,
    top_n,
    annualization,
    min_bars,
    min_trades,
    min_exposure,
    validation_folds,
):
    valid_sharpes = []
    for result in _iter_result_store(store_path):
        sharpe = result.get("val_metrics", {}).get("sharpe", -99)
        if not result.get("error") and np.isfinite(sharpe) and sharpe > -99:
            valid_sharpes.append(float(sharpe))
    sharpe_std = float(np.std(valid_sharpes, ddof=1)) if len(valid_sharpes) > 1 else 0.0
    hurdle = _multiple_testing_hurdle(sharpe_std, len(valid_sharpes))

    heap = []
    sequence = count()
    eligible_count = 0
    fold_rows = []
    for result in _iter_result_store(store_path):
        if not _is_eligible(result, min_bars, min_trades, min_exposure):
            continue
        eligible_count += 1
        metrics = result["val_metrics"]
        probability = _deflated_sharpe_probability(metrics, hurdle, annualization)
        folds = [float(value) for value in result.get("val_fold_sharpes", []) if value > -99]
        fold_std = float(np.std(folds, ddof=1)) if len(folds) > 1 else 0.0
        result["selection_diagnostics"] = {
            "deflated_sharpe_probability": probability,
            "multiple_testing_sharpe_hurdle": hurdle,
            "validation_sharpe_ci_95": _sharpe_confidence_interval(metrics, annualization),
            "validation_fold_mean": float(np.mean(folds)) if folds else None,
            "validation_fold_std": fold_std,
            "validation_fold_worst": min(folds) if folds else None,
        }
        if len(folds) == validation_folds:
            fold_rows.append(folds)
        entry = (probability, float(metrics.get("sharpe", -99)), next(sequence), result)
        if len(heap) < top_n:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)

    top = [entry[3] for entry in sorted(heap, key=lambda item: (-item[0], -item[1]))]
    diagnostics = {
        "selection_metric": "deflated_sharpe_probability",
        "trials": len(valid_sharpes),
        "eligible_candidates": eligible_count,
        "observed_sharpe_std": sharpe_std,
        "multiple_testing_sharpe_hurdle": hurdle,
        "pbo": _estimate_pbo(fold_rows),
        "walk_forward_selection": _walk_forward_selection_diagnostic(fold_rows),
    }
    return top, diagnostics


def run_search(
    ticker="GLD",
    strategies=None,
    cost_bps=1.0,
    slippage_bps=0.0,
    financing_rate=0.0,
    borrow_bps=0.0,
    cash_rate=0.0,
    short_rebate_rate=0.0,
    max_leverage=2.0,
    execution_model="next_close",
    execution_lag=1,
    annualization=None,
    risk_free_rate=RISK_FREE_RATE,
    validation_folds=4,
    min_validation_bars=60,
    min_validation_trades=1,
    min_validation_exposure=0.01,
    workers=1,
    quick=True,
    top_n=50,
    start="2004-01-01",
    end=None,
    robust=True,
    robust_frac=0.2,
    use_cache=True,
    result_dir=None,
    run_id=None,
    keep_all_results=False,
    config=None,
    resume=False,
):
    """Compare momentum strategies for a ticker.

    Args:
        ticker: Yahoo Finance ticker (e.g. "GLD", "SPY", "BTC-USD").
        strategies: List of strategy names. None selects the non-ML strategies.
        cost_bps: Transaction cost in basis points.
        slippage_bps: Additional transaction slippage in basis points.
        financing_rate: Annual financing rate applied to exposure above 1x.
        borrow_bps: Annualized short borrow fee in basis points.
        cash_rate: Annual return earned by uninvested cash.
        short_rebate_rate: Annual rebate earned on short-sale collateral.
        max_leverage: Final absolute exposure cap applied by the backtest.
        execution_model: ``same_close``, ``next_close`` (default),
            ``next_open``, or ``delayed_close``.
        execution_lag: Delay used by ``delayed_close``; other models resolve
            their lag from the model name.
        annualization: Return periods per year. None infers 365 for common
            Yahoo crypto pairs and 252 otherwise.
        risk_free_rate: Annual risk-free rate as a decimal, used in Sharpe/Sortino.
        validation_folds: Even number of temporal validation folds used for
            stability reporting and CSCV/PBO diagnostics.
        min_validation_bars: Minimum validation observations for selection.
        min_validation_trades: Minimum validation trades for selection.
        min_validation_exposure: Minimum mean absolute validation exposure.
        workers: Number of parallel workers (1 = sequential).
        quick: If True (default), only test 5 params per strategy.
        top_n: Number of top results to keep.
        start: Data start date.
        end: Data end date. None = today.
        robust: If True, run local parameter-sensitivity analysis on the selected params.
        robust_frac: Perturbation fraction for the sensitivity analysis.
        use_cache: If True, reuse cached OHLCV data when available.
        result_dir: Parent directory for run artifacts. Defaults to ``experiments``.
        run_id: Optional stable run directory name. A unique ID is generated by default.
        keep_all_results: If True, retain every experiment in memory
            and return it as ``all_results``. Full grids can contain hundreds
            of thousands of experiments; pass False to
            commit results to SQLite and export ``all_results.csv`` without
            retaining every result in memory. ``all_results`` is then empty and
            ``n_results`` carries the experiment count.
        config: Optional :class:`SearchConfig`, mapping, or JSON path.  When
            provided, its fields are used as the complete search configuration.
        resume: If True, reuse the transactional SQLite journal for the explicit
            ``run_id`` and evaluate only missing parameter combinations.

    Returns:
        A dict containing results, the selected candidate, benchmark metrics,
        local parameter sensitivity, run paths, and experiment/error counts.
    """
    if config is not None:
        configured = load_search_config(config).to_kwargs()
        ticker = configured["ticker"]
        strategies = configured["strategies"]
        cost_bps = configured["cost_bps"]
        slippage_bps = configured["slippage_bps"]
        financing_rate = configured["financing_rate"]
        borrow_bps = configured["borrow_bps"]
        cash_rate = configured["cash_rate"]
        short_rebate_rate = configured["short_rebate_rate"]
        max_leverage = configured["max_leverage"]
        execution_model = configured["execution_model"]
        execution_lag = configured["execution_lag"]
        annualization = configured["annualization"]
        risk_free_rate = configured["risk_free_rate"]
        validation_folds = configured["validation_folds"]
        min_validation_bars = configured["min_validation_bars"]
        min_validation_trades = configured["min_validation_trades"]
        min_validation_exposure = configured["min_validation_exposure"]
        workers = configured["workers"]
        quick = configured["quick"]
        top_n = configured["top_n"]
        start = configured["start"]
        end = configured["end"]
        robust = configured["robust"]
        robust_frac = configured["robust_frac"]
        use_cache = configured["use_cache"]
        result_dir = configured["result_dir"]
        run_id = configured["run_id"]
        keep_all_results = configured["keep_all_results"]

    annualization = infer_annualization(ticker) if annualization is None else annualization

    # Validate up front: an invalid cost/annualization would otherwise be
    # swallowed by run_single_experiment's per-combo error handling and
    # silently turn the whole run into sentinel (-99) results.
    if cost_bps < 0 or slippage_bps < 0 or borrow_bps < 0 or financing_rate < 0:
        raise ValueError("cost, financing and slippage parameters cannot be negative")
    if annualization <= 0:
        raise ValueError("annualization must be positive")
    if max_leverage <= 0:
        raise ValueError("max_leverage must be positive")
    if not np.isfinite(cash_rate) or not np.isfinite(short_rebate_rate) or not np.isfinite(risk_free_rate):
        raise ValueError("cash_rate, short_rebate_rate and risk_free_rate must be finite")
    if cash_rate <= -1 or short_rebate_rate <= -1:
        raise ValueError("cash_rate and short_rebate_rate must be greater than -1")
    if isinstance(execution_lag, bool) or not isinstance(execution_lag, (int, np.integer)) or execution_lag < 0:
        raise ValueError("execution_lag must be a non-negative integer")
    if execution_model not in {"same_close", "next_close", "next_open", "delayed_close"}:
        raise ValueError("execution_model must be same_close, next_close, next_open, or delayed_close")
    if execution_model == "same_close":
        resolved_execution_lag = 0
        execution_price_column = "close"
    elif execution_model == "next_open":
        resolved_execution_lag = 1
        execution_price_column = "open"
    elif execution_model == "next_close":
        resolved_execution_lag = 1
        execution_price_column = "close"
    else:
        if execution_lag < 1:
            raise ValueError("delayed_close execution requires execution_lag >= 1")
        resolved_execution_lag = int(execution_lag)
        execution_price_column = "close"
    if (
        isinstance(validation_folds, bool)
        or not isinstance(validation_folds, (int, np.integer))
        or validation_folds < 2
        or validation_folds > 10
        or validation_folds % 2
    ):
        raise ValueError("validation_folds must be an even integer between 2 and 10")
    if min_validation_bars < 2:
        raise ValueError("min_validation_bars must be at least 2")
    if min_validation_trades < 0 or min_validation_exposure < 0:
        raise ValueError("minimum validation trades and exposure cannot be negative")
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if not 0 < robust_frac <= 1:
        raise ValueError("robust_frac must be in (0, 1]")
    if resume and not run_id:
        raise ValueError("resume requires an explicit run_id")

    base_result_dir = Path(result_dir) if result_dir is not None else RESULT_DIR
    safe_ticker = str(ticker).replace("/", "_").replace("^", "_")
    run_id = run_id or f"{safe_ticker}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    if run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("run_id must be a single directory name")
    run_dir = base_result_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    backtest_kwargs = {
        "annualization": annualization,
        "financing_rate": financing_rate,
        "borrow_bps": borrow_bps,
        "slippage_bps": slippage_bps,
        "cash_rate": cash_rate,
        "short_rebate_rate": short_rebate_rate,
        "max_leverage": max_leverage,
        "execution_lag": resolved_execution_lag,
    }
    print(f"momentum-lab: Comparing strategies for {ticker}")

    data, df = prepare_data(ticker, start=start, end=end, use_cache=use_cache, annualization=annualization)
    if execution_price_column not in df.columns:
        raise ValueError(f"execution model {execution_model} requires the '{execution_price_column}' price column")
    prices = df[execution_price_column]
    n = len(df)
    periods = _split_periods(df.index)
    # Workers receive no test boundary at all.  The coordinator releases it
    # only after one candidate has been selected from development metrics.
    selection_periods = {name: periods[name] for name in ("train", "val")}
    if strategies is None:
        # Safe default: a small quick run over non-ML strategies. ML remains
        # available when requested explicitly but no longer turns a one-line
        # command into days of model fitting and heavy optional dependencies.
        strategies = list(CLASSIC_STRATEGIES)
    elif isinstance(strategies, str):
        strategies = [name.strip() for name in strategies.split(",") if name.strip()]
    else:
        strategies = list(strategies)

    environment = _environment_manifest()
    project_root = Path(__file__).resolve().parent.parent
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "strategies": strategies,
        "cost_bps": cost_bps,
        **backtest_kwargs,
        "execution_model": execution_model,
        "execution_price_column": execution_price_column,
        "risk_free_rate": risk_free_rate,
        "validation_folds": int(validation_folds),
        "min_validation_bars": int(min_validation_bars),
        "min_validation_trades": int(min_validation_trades),
        "min_validation_exposure": float(min_validation_exposure),
        "workers": workers,
        "quick": quick,
        "top_n": top_n,
        "start": start,
        "end": end,
        "data_start": str(df.index[0]),
        "data_end": str(df.index[-1]),
        "n_bars": n,
        "git_sha": _git_revision(),
        "source_fingerprint": _source_fingerprint(),
        "lock_fingerprint": _file_fingerprint(project_root / "uv.lock"),
        "environment": environment,
        "environment_fingerprint": _environment_fingerprint(environment),
        "package_version": __version__,
        "engine_schema_version": ENGINE_SCHEMA_VERSION,
        "data_snapshot": _data_snapshot(df),
        "periods": {name: [str(bounds[0]), str(bounds[1])] for name, bounds in periods.items()},
        "robust": robust,
        "robust_frac": robust_frac,
        "use_cache": use_cache,
        "keep_all_results": keep_all_results,
        "resume": resume,
    }
    if resume:
        _check_resume_compatibility(run_dir, metadata)
    _write_text_atomic(run_dir / "run_config.json", json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"  Data: {df.index[0].date()} ~ {df.index[-1].date()}, {n} bars")
    print(f"  Train: {periods['train'][0].date()} ~ {periods['train'][1].date()}")
    print(f"  Val:   {periods['val'][0].date()} ~ {periods['val'][1].date()}")
    print(f"  Test:  {periods['test'][0].date()} ~ {periods['test'][1].date()} [SEALED UNTIL SELECTION]")

    known = [s for s in strategies if s in STRATEGY_REGISTRY]
    _check_strategy_dependencies(known)
    for s in strategies:
        if s not in STRATEGY_REGISTRY:
            print(f"  WARNING: Unknown strategy '{s}' skipped. Use --list to see available names.")
    counts = {s: get_strategy(s).count_param_combinations() for s in known}
    total = sum(min(5, c) if quick else c for c in counts.values())
    print(f"  Strategies: {len(known)} (of {len(strategies)} requested), Total experiments: {total}")

    all_results = []
    n_results = 0
    n_skipped = 0
    n_errors = 0
    t0 = time.time()
    all_csv = run_dir / "all_results.csv"
    store_path = run_dir / "results.sqlite3"
    if not resume and (all_csv.exists() or store_path.exists()):
        # Preserve previous artifacts, but start from an empty transactional
        # journal when a run-id is intentionally reused without --resume.
        stamp = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:6]}"
        moved = []
        if all_csv.exists():
            backup = run_dir / f"all_results.{stamp}.bak.csv"
            all_csv.rename(backup)
            moved.append(backup.name)
        if store_path.exists():
            with closing(_open_result_store(store_path)) as connection:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            backup = run_dir / f"results.{stamp}.bak.sqlite3"
            store_path.rename(backup)
            for suffix in ("-wal", "-shm"):
                sidecar = Path(f"{store_path}{suffix}")
                if sidecar.exists():
                    sidecar.rename(Path(f"{backup}{suffix}"))
            moved.append(backup.name)
        warnings.warn(
            f"Previous run artifacts moved to {', '.join(moved)}. Pass resume=True to continue them instead.",
            RuntimeWarning,
        )

    if resume:
        if not store_path.exists() and all_csv.exists():
            _migrate_csv_checkpoint(all_csv, store_path)
        n_skipped, n_errors = _store_counts(store_path)
        n_results = n_skipped
        if keep_all_results:
            all_results.extend(_iter_result_store(store_path))
        if n_skipped:
            print(f"  Resume: found {n_skipped} committed experiments in {store_path}")
    else:
        # Create the schema before work starts. SQLite transactions are the
        # canonical crash-safe checkpoint; CSV is an atomic exported view.
        with closing(_open_result_store(store_path)):
            pass

    use_workers = workers > 1 and len(known) > 0
    pool = None
    if use_workers:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        pool = ctx.Pool(
            workers,
            initializer=_init_worker,
            initargs=(
                data,
                df,
                selection_periods,
                cost_bps,
                risk_free_rate,
                validation_folds,
                execution_price_column,
                backtest_kwargs,
            ),
        )

    batch = []
    position_cache = OrderedDict()

    def _flush_batch():
        if batch:
            _store_results(batch, store_path)
            batch.clear()

    def _handle(result):
        nonlocal n_errors, n_results
        n_results += 1
        if result.get("error"):
            n_errors += 1
        if keep_all_results:
            all_results.append(result)
        batch.append(result)
        if len(batch) >= 10_000:
            _flush_batch()

    search_ok = False
    try:
        for sname in strategies:
            if sname not in STRATEGY_REGISTRY:
                continue
            s = get_strategy(sname)
            if quick:
                combos = _quick_sample(s, 5)
                n_combos = len(combos)
            else:
                # Iterate lazily: large grids contain hundreds of thousands
                # of combinations and must not be materialized as a list.
                combos = s.iter_param_combinations()
                n_combos = s.count_param_combinations()
            if resume:
                completed_for_strategy = _completed_params(store_path, sname)
                n_combos = max(0, n_combos - len(completed_for_strategy))
                combos = (
                    params
                    for params in combos
                    if json.dumps(
                        _jsonable(params),
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=_jsonable,
                    )
                    not in completed_for_strategy
                )
            cat = (
                "ML"
                if sname.startswith("ml_")
                else ("combo" if sname in ["ensemble", "stacked", "regime_aware"] else "classic")
            )
            print(f"\n  [{cat}] {sname} ({n_combos} params) ...")

            if pool is not None:
                job = pool.imap_unordered(_worker_run, ((sname, p) for p in combos), chunksize=16)
                for result in tqdm(job, desc=f"  {sname}", total=n_combos, leave=True):
                    _handle(result)
            else:
                for params in tqdm(combos, desc=f"  {sname}", total=n_combos, leave=True):
                    result = run_single_experiment(
                        sname,
                        params,
                        data,
                        df,
                        selection_periods,
                        cost_bps,
                        risk_free_rate,
                        validation_folds=validation_folds,
                        execution_price_column=execution_price_column,
                        position_cache=position_cache,
                        **backtest_kwargs,
                    )
                    _handle(result)

            _flush_batch()
            print(f"  [checkpoint] {sname} done, {n_results} total")
        search_ok = True
    finally:
        # Persist whatever was completed before an interruption so --resume
        # can pick up even when a strategy does not reach its normal checkpoint.
        _flush_batch()
        if pool is not None:
            # On failure, terminate: close()+join() would wait for millions of
            # queued tasks before propagating the error.
            if search_ok:
                pool.close()
            else:
                pool.terminate()
            pool.join()
        if store_path.exists():
            _export_result_store(store_path, all_csv)

    elapsed = time.time() - t0
    print(f"\n  Search complete! {n_results} results ({n_errors} errors) in {elapsed / 60:.1f} min")

    if n_results == 0:
        print("  WARNING: No experiments completed. Check strategy names and data.")
        return {
            "all_results": [],
            "top_results": [],
            "best": None,
            "robustness": None,
            "run_id": run_id,
            "result_dir": str(run_dir),
            "n_results": 0,
            "n_skipped": n_skipped,
            "n_errors": n_errors,
            "benchmark_metrics": None,
            "parameter_sensitivity": None,
        }

    # Phase 2: apply minimum-evidence constraints and rank by Deflated
    # Sharpe probability.  The second store scan keeps exhaustive runs
    # bounded in memory while accounting for the number of trials.
    top, selection_diagnostics = _select_from_store(
        store_path,
        top_n,
        annualization,
        min_validation_bars,
        min_validation_trades,
        min_validation_exposure,
        validation_folds,
    )
    if not keep_all_results:
        all_results = []

    # Phase 3: Test set evaluation
    benchmark_metrics = None
    if top:
        # Copy the selected candidate so neither the checkpoint nor the
        # validation-ranked top list is mutated with test-set information.
        best = copy.deepcopy(top[0])
        sname = best["strategy"]
        params = best.get("params", {})
        strategy = get_strategy(sname)
        positions = strategy.run(data, **params).reindex(prices.index).ffill().fillna(0.0)
        selected_bt = backtest(positions, prices, cost_bps=cost_bps, **backtest_kwargs)
        selected_effective = positions.shift(resolved_execution_lag + 1).fillna(0.0).clip(-max_leverage, max_leverage)
        test_m = _period_metrics(
            selected_bt,
            selected_effective,
            periods["test"][0],
            periods["test"][1],
            risk_free_rate,
            annualization,
        )
        best["test_metrics"] = test_m
        best["test_evaluated_at"] = datetime.now(timezone.utc).isoformat()

        benchmark_positions = get_buy_and_hold(prices)
        benchmark_bt = backtest(benchmark_positions, prices, cost_bps=cost_bps, **backtest_kwargs)
        benchmark_effective = benchmark_positions.shift(resolved_execution_lag + 1).fillna(0.0)
        benchmark_metrics = _period_metrics(
            benchmark_bt,
            benchmark_effective,
            periods["test"][0],
            periods["test"][1],
            risk_free_rate,
            annualization,
        )
        print(f"\n  Best: {sname}")
        print(f"  Params: {_params_to_str(params)}")
        print(f"  Val Sharpe:   {best['val_metrics'].get('sharpe', 0):.4f}")
        print(f"  Deflated Sharpe probability: {best['selection_diagnostics']['deflated_sharpe_probability']:.2%}")
        print(f"  Test Sharpe:  {test_m['sharpe']:.4f} (B&H: {benchmark_metrics['sharpe']:.4f})")
        print(f"  Test CAGR:    {test_m['cagr']:.2%} (B&H: {benchmark_metrics['cagr']:.2%})")
        print(f"  Test MaxDD:   {test_m['max_drawdown']:.2%} (B&H: {benchmark_metrics['max_drawdown']:.2%})")
    else:
        best = None
        print("  WARNING: No valid experiments remained after evaluation.")

    # Phase 4: Robustness check on the best parameters
    robustness = None
    if robust and best is not None:
        print(f"\n  [Phase 4] Parameter sensitivity (perturbing selected params by {robust_frac:.0%}) ...")
        robustness = robustness_check(
            data,
            df,
            periods,
            sname,
            params,
            cost_bps=cost_bps,
            frac=robust_frac,
            backtest_kwargs=backtest_kwargs,
            risk_free_rate=risk_free_rate,
        )
        if robustness.get("error"):
            print(f"    Skipped: {robustness['error']}")
        else:
            st = robustness["stats"]
            print(f"    Baseline val Sharpe: {robustness['baseline']:.4f}")
            print(f"    Neighbors evaluated: {robustness['n_neighbors']}")
            print(
                f"    Neighbor val Sharpe: mean={st['mean']:.4f} "
                f"median={st['median']:.4f} min={st['min']:.4f} "
                f"max={st['max']:.4f} std={st['std']:.4f}"
            )
            print(
                f"    Degraded (<50% base): {robustness['pct_degrade']:.1%} | "
                f"Positive neighbors: {robustness['pct_positive']:.1%}"
            )
            print(
                f"    Sensitivity grade: {robustness['grade']} "
                f"({robustness['verdict']})"
                + ("  [ISOLATED PEAK - fragile locally]" if robustness["isolated_peak"] else "")
            )
            robustness_frame = pd.DataFrame(
                [
                    {
                        "strategy": sname,
                        "params": json.dumps(params, ensure_ascii=False, default=_jsonable),
                        "baseline_val_sharpe": robustness["baseline"],
                        "n_neighbors": robustness["n_neighbors"],
                        "neighbor_mean": st["mean"],
                        "neighbor_median": st["median"],
                        "neighbor_std": st["std"],
                        "neighbor_min": st["min"],
                        "neighbor_max": st["max"],
                        "pct_degrade": robustness["pct_degrade"],
                        "pct_positive": robustness["pct_positive"],
                        "grade": robustness["grade"],
                        "verdict": robustness["verdict"],
                        "isolated_peak": robustness["isolated_peak"],
                    }
                ]
            )
            _write_frame_atomic(robustness_frame, run_dir / "robustness.csv")
            print(f"    Saved to {run_dir / 'robustness.csv'}")

    # Save validation-ranked candidates and the one final-selection summary.
    if top:
        rows = []
        for r in top:
            row = {
                "strategy": r["strategy"],
                "params": json.dumps(r.get("params", {}), ensure_ascii=False, default=_jsonable),
            }
            for p in ["train", "val"]:
                m = r.get(f"{p}_metrics", {})
                for k, v in m.items():
                    row[f"{p}_{k}"] = v
            diagnostics = r.get("selection_diagnostics", {})
            for key, value in diagnostics.items():
                row[f"selection_{key}"] = value
            rows.append(row)
        _write_frame_atomic(pd.DataFrame(rows), run_dir / "top_results.csv")

    summary = {
        "run_id": run_id,
        "best": _jsonable(best),
        "benchmark_metrics": _jsonable(benchmark_metrics),
        "selection_diagnostics": _jsonable(selection_diagnostics),
        "parameter_sensitivity": _jsonable(robustness),
        "n_results": n_results,
        "n_skipped": n_skipped,
        "n_errors": n_errors,
    }
    _write_text_atomic(
        run_dir / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
    )

    return {
        "all_results": all_results,
        "top_results": top,
        "best": best,
        "robustness": robustness,
        "parameter_sensitivity": robustness,
        "benchmark_metrics": benchmark_metrics,
        "selection_diagnostics": selection_diagnostics,
        "run_id": run_id,
        "result_dir": str(run_dir),
        "n_results": n_results,
        "n_skipped": n_skipped,
        "n_errors": n_errors,
    }
