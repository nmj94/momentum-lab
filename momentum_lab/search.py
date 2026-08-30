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
from .governance import StudyRegistry, validate_study_options
from .indicators import IndicatorDAG
from .reporting import render_html_report, render_markdown_report
from .robustness import robustness_check
from .strategies import CLASSIC_STRATEGIES, STRATEGY_REGISTRY, get_strategy
from .uncertainty import BOOTSTRAP_METHOD, BOOTSTRAP_WARNING, paired_block_bootstrap, validate_bootstrap_options

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kw):
        return x


RESULT_DIR = Path("experiments")
ENGINE_SCHEMA_VERSION = 5
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
    "requested_turnover",
    "fill_ratio",
    "transaction_cost_drag",
    "max_participation",
    "capacity_constrained_bars",
    "borrow_blocked_bars",
)
RESULT_COLUMNS = [
    "strategy",
    "params",
    "error",
    *(f"{period}_{metric}" for period in ("train", "val") for metric in METRIC_KEYS),
    "val_fold_sharpes",
]
STAGE_EXPORT_COLUMNS = [
    "strategy",
    "params",
    "stage",
    "resource_bars",
    "advanced",
    "score",
    "error",
    "val_sharpe",
    "val_n_observations",
    "val_trade_count",
    "val_exposure",
]

# Shared state for parallel sub-processes (set via Pool initializer).
_POOL_STATE = None
_POOL_POSITION_CACHE = None
_POOL_INDICATOR_CACHE = None


def _init_worker(
    data,
    df,
    periods,
    cost_bps,
    risk_free_rate,
    validation_folds,
    execution_price_column,
    backtest_kwargs,
    indicator_cache_size,
):
    global _POOL_INDICATOR_CACHE, _POOL_POSITION_CACHE, _POOL_STATE
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
    _POOL_INDICATOR_CACHE = IndicatorDAG(data, max_entries=indicator_cache_size)


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
        indicator_cache=_POOL_INDICATOR_CACHE,
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


def _canonical_params(params):
    return json.dumps(
        _jsonable(params),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_jsonable,
    )


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
    connection.execute(
        "CREATE TABLE IF NOT EXISTS stage_results ("
        "strategy TEXT NOT NULL, params TEXT NOT NULL, stage INTEGER NOT NULL, "
        "resource_bars INTEGER NOT NULL, advanced INTEGER NOT NULL DEFAULT 0, "
        "score REAL, result_json TEXT NOT NULL, "
        "PRIMARY KEY (strategy, params, stage))"
    )
    connection.commit()
    return connection


def _insert_results(connection, results):
    rows = _results_rows(results)
    if not rows:
        return
    placeholders = ",".join("?" for _ in RESULT_COLUMNS)
    columns = ",".join(f'"{column}"' for column in RESULT_COLUMNS)
    values = [[row.get(column) for column in RESULT_COLUMNS] for row in rows]
    connection.executemany(
        f"INSERT OR REPLACE INTO results ({columns}) VALUES ({placeholders})",
        values,
    )


def _store_results(results, path):
    """Commit one checkpoint batch transactionally to the canonical store."""
    with closing(_open_result_store(path)) as connection, connection:
        _insert_results(connection, results)


def _stage_values(stage_results):
    values = []
    for strategy, params, stage, resource_bars, result in stage_results:
        score = _stage_score(result)
        values.append(
            (
                strategy,
                _canonical_params(params),
                int(stage),
                int(resource_bars),
                score if np.isfinite(score) else None,
                json.dumps(_jsonable(result), ensure_ascii=False, sort_keys=True, allow_nan=False),
            )
        )
    return values


def _insert_stage_results(connection, stage_results):
    values = _stage_values(stage_results)
    if values:
        connection.executemany(
            "INSERT OR REPLACE INTO stage_results "
            "(strategy, params, stage, resource_bars, advanced, score, result_json) "
            "VALUES (?, ?, ?, ?, 0, ?, ?)",
            values,
        )


def _store_stage_results(stage_results, path, *, final=False):
    """Persist partial-resource evidence, optionally promoting it atomically."""
    if not stage_results:
        return
    with closing(_open_result_store(path)) as connection, connection:
        _insert_stage_results(connection, stage_results)
        if final:
            _insert_results(connection, [item[4] for item in stage_results])


def _load_stage_results(path, strategy, stage, resource_bars):
    if not path.exists():
        return {}
    with closing(_open_result_store(path)) as connection:
        rows = connection.execute(
            "SELECT params, result_json FROM stage_results WHERE strategy = ? AND stage = ? AND resource_bars = ?",
            (strategy, int(stage), int(resource_bars)),
        )
        loaded = {}
        for canonical, payload in rows:
            try:
                loaded[canonical] = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"corrupt staged result for {strategy} stage {stage}: {exc}") from exc
        return loaded


def _mark_stage_survivors(path, strategy, stage, survivor_params):
    survivor_keys = [_canonical_params(params) for params in survivor_params]
    with closing(_open_result_store(path)) as connection, connection:
        connection.execute(
            "UPDATE stage_results SET advanced = 0 WHERE strategy = ? AND stage = ?",
            (strategy, int(stage)),
        )
        connection.executemany(
            "UPDATE stage_results SET advanced = 1 WHERE strategy = ? AND params = ? AND stage = ?",
            [(strategy, canonical, int(stage)) for canonical in survivor_keys],
        )


def _iter_stage_store(path):
    if not path.exists():
        return
    with closing(_open_result_store(path)) as connection:
        rows = connection.execute(
            "SELECT strategy, params, stage, resource_bars, advanced, score, result_json "
            "FROM stage_results ORDER BY stage, strategy, rowid"
        )
        for strategy, params, stage, resource_bars, advanced, score, payload in rows:
            result = json.loads(payload)
            metrics = result.get("val_metrics", {})
            yield {
                "strategy": strategy,
                "params": params,
                "stage": stage,
                "resource_bars": resource_bars,
                "advanced": advanced,
                "score": score,
                "error": result.get("error", ""),
                "val_sharpe": metrics.get("sharpe"),
                "val_n_observations": metrics.get("n_observations"),
                "val_trade_count": metrics.get("trade_count"),
                "val_exposure": metrics.get("exposure"),
            }


def _export_stage_store(store_path, csv_path):
    tmp_path = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=STAGE_EXPORT_COLUMNS)
            writer.writeheader()
            writer.writerows(_iter_stage_store(store_path))
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(csv_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _stage_store_count(path):
    if not path.exists():
        return 0
    with closing(_open_result_store(path)) as connection:
        (count_rows,) = connection.execute("SELECT COUNT(*) FROM stage_results").fetchone()
    return int(count_rows or 0)


def _stage_score_dispersion(path, stage=0):
    """Estimate candidate Sharpe dispersion from one equal-resource stage."""
    if not path.exists():
        return None
    with closing(_open_result_store(path)) as connection:
        scores = [
            float(row[0])
            for row in connection.execute(
                "SELECT score FROM stage_results WHERE stage = ? AND score IS NOT NULL",
                (int(stage),),
            )
            if np.isfinite(row[0])
        ]
    return float(np.std(scores, ddof=1)) if len(scores) > 1 else None


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
    "search_method",
    "candidate_budget",
    "halving_factor",
    "halving_stages",
    "indicator_cache_size",
    "data_snapshot",
    "package_version",
    "engine_schema_version",
    "source_fingerprint",
    "lock_fingerprint",
    "environment_fingerprint",
    "cost_bps",
    "slippage_bps",
    "financing_rate",
    "financing_spread",
    "borrow_bps",
    "cash_rate",
    "short_rebate_rate",
    "spread_bps",
    "impact_bps",
    "impact_exponent",
    "impact_reference_participation",
    "max_participation",
    "initial_capital",
    "min_fee",
    "max_leverage",
    "execution_model",
    "execution_lag",
    "annualization",
    "risk_free_rate",
    "validation_folds",
    "min_validation_bars",
    "min_validation_trades",
    "min_validation_exposure",
    "bootstrap",
    "bootstrap_resamples",
    "bootstrap_block_length",
    "bootstrap_confidence",
    "bootstrap_seed",
    "bootstrap_min_observations",
    "study_id",
    "registry_path",
    "registry_id",
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


def _slice_temporal_mapping(values, end):
    """Slice every pandas object in a mapping without exposing later bars."""
    sliced = {}
    for key, value in values.items():
        if isinstance(value, (pd.Series, pd.DataFrame)):
            sliced[key] = value.loc[:end]
        else:
            sliced[key] = value
    return sliced


def _halving_resource_bars(validation_index, stages=3, factor=3, minimum=2):
    """Return strictly increasing validation prefixes ending at full resource."""
    total = len(validation_index)
    if total == 0:
        return []
    resources = []
    for remaining in range(stages - 1, -1, -1):
        proposed = math.ceil(total / (factor**remaining))
        resource = min(total, max(int(minimum), proposed))
        if not resources or resource > resources[-1]:
            resources.append(resource)
    if resources[-1] != total:
        resources.append(total)
    return resources


def _stage_score(result):
    """Validation-only score used to prune a staged search."""
    if result.get("error"):
        return -math.inf
    metrics = result.get("val_metrics", {})
    try:
        score = float(metrics.get("sharpe", -99.0))
        exposure = float(metrics.get("exposure", 0.0))
    except (TypeError, ValueError):
        return -math.inf
    if not np.isfinite(score) or exposure <= 1e-12:
        return -math.inf
    return score


def _stage_survivors(candidates, results_by_params, factor):
    """Deterministically retain the best ceil(n / factor) candidates."""
    ranked = sorted(
        candidates,
        key=lambda params: (
            -_stage_score(results_by_params[_canonical_params(params)]),
            _canonical_params(params),
        ),
    )
    keep = max(1, math.ceil(len(ranked) / factor)) if ranked else 0
    return ranked[:keep]


def _period_metrics(bt, effective_positions, start, end, risk_free_rate, annualization):
    period_returns = bt["returns"].loc[start:end]
    period_trades = bt["trades"].loc[start:end]
    requested_trades = bt.get("requested_trades", bt["trades"]).loc[start:end]
    transaction_costs = bt.get("transaction_costs", bt["trades"] * 0.0).loc[start:end]
    participation = bt.get("participation", bt["trades"] * 0.0).loc[start:end]
    capacity_constrained = bt.get("capacity_constrained", bt["trades"] > np.inf).loc[start:end]
    borrow_blocked = bt.get("borrow_blocked", bt["trades"] > np.inf).loc[start:end]
    period_exposure = effective_positions.loc[start:end]
    actual_turnover = float(period_trades.sum())
    requested_turnover = float(requested_trades.sum())
    metrics = evaluate(period_returns, risk_free_rate=risk_free_rate, annualization=annualization)
    metrics.update(
        {
            "n_observations": len(period_returns),
            "trade_count": int((period_trades > 1e-12).sum()),
            "exposure": float(period_exposure.abs().mean()) if len(period_exposure) else 0.0,
            "turnover": actual_turnover,
            "requested_turnover": requested_turnover,
            "fill_ratio": actual_turnover / requested_turnover if requested_turnover > 1e-12 else 1.0,
            "transaction_cost_drag": float(transaction_costs.sum()),
            "max_participation": float(participation.max()) if len(participation) else 0.0,
            "capacity_constrained_bars": int(capacity_constrained.sum()),
            "borrow_blocked_bars": int(borrow_blocked.sum()),
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


def _strategy_positions(
    strategy_name,
    params,
    data,
    position_cache=None,
    cache_size=32,
    indicator_cache=None,
):
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
        strategy_data = data
        if indicator_cache is not None:
            strategy_data = dict(data)
            strategy_data["_indicator_dag"] = indicator_cache
        raw = strategy.generate_positions(strategy_data, **base_params)
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
    indicator_cache=None,
    **backtest_kwargs,
):
    cache_before = indicator_cache.snapshot() if indicator_cache is not None else None
    try:
        positions = _strategy_positions(
            strategy_name,
            params,
            data,
            position_cache=position_cache,
            indicator_cache=indicator_cache,
        )
    except Exception as e:
        result = {
            "strategy": strategy_name,
            "params": params,
            "error": str(e),
            "train_metrics": {},
            "val_metrics": {},
            "val_fold_sharpes": [],
        }
        if cache_before is not None:
            result["indicator_cache"] = indicator_cache.delta(cache_before)
        return result
    if execution_price_column not in df.columns:
        result = {
            "strategy": strategy_name,
            "params": params,
            "error": f"execution price column '{execution_price_column}' is unavailable",
            "train_metrics": {"sharpe": -99.0},
            "val_metrics": {"sharpe": -99.0},
            "val_fold_sharpes": [],
        }
        if cache_before is not None:
            result["indicator_cache"] = indicator_cache.delta(cache_before)
        return result
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
        # ``bt['positions']`` contains the actually filled end-of-bar weight,
        # including partial fills and borrow blocks. Returns on a bar are earned
        # by the position held at the previous bar close.
        effective_positions = bt["positions"].shift(1).fillna(0.0)
    except Exception as exc:
        result = {
            **results,
            "error": str(exc),
            "train_metrics": {"sharpe": -99.0},
            "val_metrics": {"sharpe": -99.0},
        }
        if cache_before is not None:
            result["indicator_cache"] = indicator_cache.delta(cache_before)
        return result
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
    if cache_before is not None:
        results["indicator_cache"] = indicator_cache.delta(cache_before)
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
    trial_count_override=None,
    sharpe_std_override=None,
):
    valid_sharpes = []
    for result in _iter_result_store(store_path):
        sharpe = result.get("val_metrics", {}).get("sharpe", -99)
        if not result.get("error") and np.isfinite(sharpe) and sharpe > -99:
            valid_sharpes.append(float(sharpe))
    full_development_std = float(np.std(valid_sharpes, ddof=1)) if len(valid_sharpes) > 1 else 0.0
    use_override = sharpe_std_override is not None and np.isfinite(sharpe_std_override) and sharpe_std_override >= 0
    sharpe_std = float(sharpe_std_override) if use_override else full_development_std
    trial_count = max(len(valid_sharpes), int(trial_count_override or 0))
    hurdle = _multiple_testing_hurdle(sharpe_std, trial_count)

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
        "trials": trial_count,
        "full_development_trials": len(valid_sharpes),
        "eligible_candidates": eligible_count,
        "observed_sharpe_std": sharpe_std,
        "sharpe_std_source": "initial_halving_stage" if use_override else "full_development",
        "multiple_testing_sharpe_hurdle": hurdle,
        "pbo": _estimate_pbo(fold_rows),
        "walk_forward_selection": _walk_forward_selection_diagnostic(fold_rows),
    }
    return top, diagnostics


def _selected_ledgers(best, data, df, execution_price_column, cost_bps, backtest_kwargs):
    """Evaluate exactly one fixed strategy and its benchmark on the supplied window."""
    prices = df[execution_price_column]
    positions = get_strategy(best["strategy"]).run(data, **best.get("params", {}))
    positions = positions.reindex(prices.index).ffill().fillna(0.0)
    selected = backtest(positions, prices, cost_bps=cost_bps, **backtest_kwargs)
    benchmark = backtest(get_buy_and_hold(prices), prices, cost_bps=cost_bps, **backtest_kwargs)
    return selected, benchmark


def _bootstrap_windows(selected, benchmark, periods, labels, options, annualization, risk_free_rate):
    return {
        label: paired_block_bootstrap(
            selected["returns"].loc[periods[key][0] : periods[key][1]],
            benchmark["returns"].loc[periods[key][0] : periods[key][1]],
            annualization=annualization,
            risk_free_rate=risk_free_rate,
            **options,
        )
        for label, key in labels
    }


def _test_payload(
    best,
    data,
    df,
    periods,
    execution_price_column,
    cost_bps,
    backtest_kwargs,
    annualization,
    risk_free_rate,
    bootstrap_options,
    *,
    include_validation=False,
):
    """Caller must durably record exposure BEFORE passing full data here."""
    selected, benchmark = _selected_ledgers(best, data, df, execution_price_column, cost_bps, backtest_kwargs)
    test_metrics, benchmark_metrics = [
        _period_metrics(
            ledger, ledger["positions"].shift(1).fillna(0.0), *periods["test"], risk_free_rate, annualization
        )
        for ledger in (selected, benchmark)
    ]
    labels = [("validation", "val"), ("test", "test")] if include_validation else [("test", "test")]
    diagnostics = (
        _bootstrap_windows(selected, benchmark, periods, labels, bootstrap_options, annualization, risk_free_rate)
        if bootstrap_options is not None
        else {}
    )
    return _jsonable(
        {
            "test_metrics": test_metrics,
            "benchmark_metrics": benchmark_metrics,
            "test_evaluated_at": datetime.now(timezone.utc).isoformat(),
            "bootstrap_periods": diagnostics,
        }
    )


def run_search(
    ticker="GLD",
    strategies=None,
    cost_bps=1.0,
    slippage_bps=0.0,
    financing_rate=0.0,
    financing_spread=0.0,
    borrow_bps=0.0,
    cash_rate=0.0,
    short_rebate_rate=0.0,
    spread_bps=0.0,
    impact_bps=0.0,
    impact_exponent=0.5,
    impact_reference_participation=0.01,
    max_participation=None,
    initial_capital=1_000_000.0,
    min_fee=0.0,
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
    generate_report=True,
    config=None,
    resume=False,
    search_method="grid",
    candidate_budget=256,
    halving_factor=3,
    halving_stages=3,
    indicator_cache_size=256,
    bootstrap=True,
    bootstrap_resamples=2000,
    bootstrap_block_length=10,
    bootstrap_confidence=0.95,
    bootstrap_seed=42,
    bootstrap_min_observations=60,
    study_id=None,
    registry_path=None,
    reveal_test=False,
    allow_test_reuse=False,
    test_reuse_reason=None,
):
    """Compare momentum strategies for a ticker.

    Args:
        ticker: Yahoo Finance ticker (e.g. "GLD", "SPY", "BTC-USD").
        strategies: List of strategy names. None selects the non-ML strategies.
        cost_bps: Transaction cost in basis points.
        slippage_bps: Additional transaction slippage in basis points.
        financing_rate: Annual financing rate applied to exposure above 1x.
        financing_spread: Annual spread added to the financing base rate.
        borrow_bps: Annualized short borrow fee in basis points.
        cash_rate: Annual return earned by uninvested cash.
        short_rebate_rate: Annual rebate earned on short-sale collateral.
        spread_bps: Quoted full bid/ask spread; each unit traded pays half.
        impact_bps: One-way market impact at the reference participation rate.
        impact_exponent: Positive participation exponent for nonlinear impact.
        impact_reference_participation: Participation rate at which impact is quoted.
        max_participation: Optional maximum fraction of bar dollar volume traded.
        initial_capital: Starting NAV used by capacity and minimum-fee models.
        min_fee: Minimum currency fee charged per non-zero rebalance.
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
        search_method: ``grid`` for quick/exhaustive behavior or
            ``successive_halving`` for a deterministic budgeted search.
        candidate_budget: Initial candidates sampled per strategy by
            successive halving.
        halving_factor: Candidate reduction factor after each partial stage.
        halving_stages: Maximum number of validation-resource stages.
        indicator_cache_size: Maximum shared indicator nodes per process;
            zero disables the indicator DAG cache.
        bootstrap: Report post-selection paired block-bootstrap diagnostics;
            never used for candidate ranking or parameter sensitivity.
        bootstrap_resamples: Fixed number of resamples (200-20000).
        bootstrap_block_length: Circular block length in observations.
        bootstrap_confidence: Percentile interval confidence level in (0, 1).
        bootstrap_seed: Reproducible PCG64 seed, chosen before inspecting results.
        bootstrap_min_observations: Minimum observations; also requires at least
            five nominal non-overlapping blocks in each reported window.
        study_id: Fixed research protocol ID. Registered searches withhold test
            evaluation by default; complete a sealed search before revealing.
        registry_path: Shared SQLite observation registry, outside result dirs.
        reveal_test: Invocation-only acknowledgement to reveal an already-frozen
            study. Never enabled implicitly by a JSON configuration.
        allow_test_reuse: Invocation-only acknowledgement of known overlapping
            observations; cannot turn previously observed data into fresh data.
        test_reuse_reason: Required audit explanation when allowing test reuse.
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
        generate_report: Write self-contained Markdown and HTML research reports.
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
        financing_spread = configured["financing_spread"]
        borrow_bps = configured["borrow_bps"]
        cash_rate = configured["cash_rate"]
        short_rebate_rate = configured["short_rebate_rate"]
        spread_bps = configured["spread_bps"]
        impact_bps = configured["impact_bps"]
        impact_exponent = configured["impact_exponent"]
        impact_reference_participation = configured["impact_reference_participation"]
        max_participation = configured["max_participation"]
        initial_capital = configured["initial_capital"]
        min_fee = configured["min_fee"]
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
        search_method = configured["search_method"]
        candidate_budget = configured["candidate_budget"]
        halving_factor = configured["halving_factor"]
        halving_stages = configured["halving_stages"]
        indicator_cache_size = configured["indicator_cache_size"]
        bootstrap = configured["bootstrap"]
        bootstrap_resamples = configured["bootstrap_resamples"]
        bootstrap_block_length = configured["bootstrap_block_length"]
        bootstrap_confidence = configured["bootstrap_confidence"]
        bootstrap_seed = configured["bootstrap_seed"]
        bootstrap_min_observations = configured["bootstrap_min_observations"]
        study_id = configured["study_id"]
        registry_path = configured["registry_path"]
        top_n = configured["top_n"]
        start = configured["start"]
        end = configured["end"]
        robust = configured["robust"]
        robust_frac = configured["robust_frac"]
        use_cache = configured["use_cache"]
        result_dir = configured["result_dir"]
        run_id = configured["run_id"]
        keep_all_results = configured["keep_all_results"]
        generate_report = configured["generate_report"]

    validate_study_options(study_id, reveal_test, allow_test_reuse, test_reuse_reason)
    annualization = infer_annualization(ticker) if annualization is None else annualization

    if not isinstance(bootstrap, (bool, np.bool_)):
        raise TypeError("bootstrap must be boolean")
    bootstrap_options = {
        "n_resamples": bootstrap_resamples,
        "block_length": bootstrap_block_length,
        "confidence_level": bootstrap_confidence,
        "seed": bootstrap_seed,
        "min_observations": bootstrap_min_observations,
    }
    validate_bootstrap_options(**bootstrap_options)
    bootstrap_diagnostics = {
        "status": "no_selection" if bootstrap else "disabled",
        "method": BOOTSTRAP_METHOD,
        "used_for_selection": False,
        "warning": BOOTSTRAP_WARNING,
        "periods": {},
    }

    # Validate up front: an invalid cost/annualization would otherwise be
    # swallowed by run_single_experiment's per-combo error handling and
    # silently turn the whole run into sentinel (-99) results.
    nonnegative_inputs = (
        cost_bps,
        slippage_bps,
        financing_rate,
        financing_spread,
        borrow_bps,
        spread_bps,
        impact_bps,
        min_fee,
    )
    if any(isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value < 0 for value in nonnegative_inputs):
        raise ValueError("cost, financing and slippage parameters must be finite and cannot be negative")
    if isinstance(annualization, (bool, np.bool_)) or not np.isfinite(annualization) or annualization <= 0:
        raise ValueError("annualization must be positive")
    if isinstance(max_leverage, (bool, np.bool_)) or not np.isfinite(max_leverage) or max_leverage <= 0:
        raise ValueError("max_leverage must be positive")
    if isinstance(initial_capital, (bool, np.bool_)) or not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be finite and positive")
    if isinstance(impact_exponent, (bool, np.bool_)) or not np.isfinite(impact_exponent) or impact_exponent <= 0:
        raise ValueError("impact_exponent must be finite and positive")
    if (
        isinstance(impact_reference_participation, (bool, np.bool_))
        or not np.isfinite(impact_reference_participation)
        or not 0 < impact_reference_participation <= 1
    ):
        raise ValueError("impact_reference_participation must be in (0, 1]")
    if max_participation is not None and (
        isinstance(max_participation, (bool, np.bool_))
        or not np.isfinite(max_participation)
        or not 0 < max_participation <= 1
    ):
        raise ValueError("max_participation must be in (0, 1]")
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
    if search_method not in {"grid", "successive_halving"}:
        raise ValueError("search_method must be grid or successive_halving")
    integer_search_inputs = (candidate_budget, halving_factor, halving_stages, indicator_cache_size)
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in integer_search_inputs):
        raise TypeError("search budget, halving, and indicator-cache values must be integers")
    if candidate_budget < 1:
        raise ValueError("candidate_budget must be at least 1")
    if halving_factor < 2:
        raise ValueError("halving_factor must be at least 2")
    if halving_stages < 1:
        raise ValueError("halving_stages must be at least 1")
    if indicator_cache_size < 0:
        raise ValueError("indicator_cache_size cannot be negative")
    if not 0 < robust_frac <= 1:
        raise ValueError("robust_frac must be in (0, 1]")
    if not isinstance(generate_report, (bool, np.bool_)):
        raise TypeError("generate_report must be boolean")
    if resume and not run_id:
        raise ValueError("resume requires an explicit run_id")

    base_result_dir = Path(result_dir) if result_dir is not None else RESULT_DIR
    safe_ticker = str(ticker).replace("/", "_").replace("^", "_")
    run_id = run_id or f"{safe_ticker}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    if run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("run_id must be a single directory name")
    run_dir = base_result_dir / run_id
    previous_config_path = run_dir / "run_config.json"
    if study_id and not resume and run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("registered runs require a new empty run directory or resume=True")
    if study_id and resume and not previous_config_path.is_file():
        raise ValueError("registered resume requires its original run_config.json and registry")
    if previous_config_path.is_file():
        previous_config = json.loads(previous_config_path.read_text(encoding="utf-8"))
        if previous_config.get("study_id") and previous_config["study_id"] != study_id:
            raise ValueError("cannot change or disable study_id in an existing registered run")
    registry = StudyRegistry(registry_path, create=not resume and not reveal_test)
    if resume and previous_config_path.is_file() and previous_config.get("registry_id") != registry.registry_id:
        raise ValueError("resume registry identity mismatch; restore the original access history")
    if reveal_test:
        registry.require_reveal_ready(study_id)
    test_access = registry.unregistered_status()
    run_dir.mkdir(parents=True, exist_ok=True)

    backtest_config = {
        "annualization": annualization,
        "financing_rate": financing_rate,
        "financing_spread": financing_spread,
        "borrow_bps": borrow_bps,
        "slippage_bps": slippage_bps,
        "cash_rate": cash_rate,
        "short_rebate_rate": short_rebate_rate,
        "spread_bps": spread_bps,
        "impact_bps": impact_bps,
        "impact_exponent": impact_exponent,
        "impact_reference_participation": impact_reference_participation,
        "max_participation": max_participation,
        "initial_capital": initial_capital,
        "min_fee": min_fee,
        "max_leverage": max_leverage,
        "execution_lag": resolved_execution_lag,
    }
    backtest_kwargs = dict(backtest_config)
    print(f"momentum-lab: Comparing strategies for {ticker}")

    data, df = prepare_data(ticker, start=start, end=end, use_cache=use_cache, annualization=annualization)
    if execution_price_column not in df.columns:
        raise ValueError(f"execution model {execution_model} requires the '{execution_price_column}' price column")
    if impact_bps > 0 or max_participation is not None:
        if "volume" not in df.columns:
            raise ValueError("liquidity-aware execution requires a 'volume' column")
        backtest_kwargs["volume"] = df["volume"]
    n = len(df)
    periods = _split_periods(df.index)
    # Candidate workers receive neither test boundaries nor test observations.
    # The coordinator releases the full snapshot only after validation selects
    # one copied candidate.
    selection_periods = {name: periods[name] for name in ("train", "val")}
    development_end = periods["val"][1]
    development_data = _slice_temporal_mapping(data, development_end)
    development_df = df.loc[:development_end]
    development_backtest_kwargs = _slice_temporal_mapping(backtest_kwargs, development_end)
    validation_index = development_df.loc[periods["val"][0] : periods["val"][1]].index
    halving_resources = (
        _halving_resource_bars(validation_index, halving_stages, halving_factor, min_validation_bars)
        if search_method == "successive_halving"
        else []
    )
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
        **backtest_config,
        "execution_model": execution_model,
        "execution_price_column": execution_price_column,
        "risk_free_rate": risk_free_rate,
        "validation_folds": int(validation_folds),
        "min_validation_bars": int(min_validation_bars),
        "min_validation_trades": int(min_validation_trades),
        "min_validation_exposure": float(min_validation_exposure),
        "workers": workers,
        "quick": quick,
        "search_method": search_method,
        "candidate_budget": int(candidate_budget),
        "halving_factor": int(halving_factor),
        "halving_stages": int(halving_stages),
        "indicator_cache_size": int(indicator_cache_size),
        "bootstrap": bool(bootstrap),
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_block_length": int(bootstrap_block_length),
        "bootstrap_confidence": float(bootstrap_confidence),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_min_observations": int(bootstrap_min_observations),
        "study_id": study_id,
        "registry_path": str(registry.path),
        "registry_id": registry.registry_id,
        "reveal_test": reveal_test,
        "allow_test_reuse": allow_test_reuse,
        "test_reuse_reason": test_reuse_reason,
        "halving_resource_bars": halving_resources,
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
        "generate_report": generate_report,
        "resume": resume,
    }
    if resume:
        _check_resume_compatibility(run_dir, metadata)
    if study_id:
        unknown = [name for name in strategies if name not in STRATEGY_REGISTRY]
        if unknown:
            raise ValueError(f"registered study contains unknown strategies: {', '.join(unknown)}")
        protocol = {
            key: metadata[key]
            for key in _RESUME_COMPAT_FIELDS
            if key not in {"study_id", "registry_path", "registry_id"}
        }
        protocol.update(
            {
                "periods": metadata["periods"],
                "selection_rule": "deflated_sharpe_probability_v1",
                "split_rule": "ordered_40_40_20_v1",
                "strategy_space": {
                    name: _jsonable(
                        {"grid": get_strategy(name).param_grid, "universal": get_strategy(name).UNIVERSAL_PARAMS}
                    )
                    for name in strategies
                },
            }
        )
        test_access = registry.register(study_id, protocol)
        metadata["study_protocol_sha256"] = test_access["protocol_sha256"]
    _write_text_atomic(run_dir / "run_config.json", json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"  Data: {df.index[0].date()} ~ {df.index[-1].date()}, {n} bars")
    print(f"  Train: {periods['train'][0].date()} ~ {periods['train'][1].date()}")
    print(f"  Val:   {periods['val'][0].date()} ~ {periods['val'][1].date()}")
    test_policy = "EXPLICIT REVEAL REQUIRED" if study_id else "LEGACY AUTO-REVEAL; PRIOR HISTORY UNKNOWN"
    print(f"  Test:  {periods['test'][0].date()} ~ {periods['test'][1].date()} [{test_policy}]")

    known = [s for s in strategies if s in STRATEGY_REGISTRY]
    _check_strategy_dependencies(known)
    registry.record_development(
        ticker=ticker,
        start=periods["train"][0],
        end=periods["val"][1],
        data_snapshot=metadata["data_snapshot"],
        run_id=run_id,
        run_path=run_dir,
        study_id=study_id,
    )
    for s in strategies:
        if s not in STRATEGY_REGISTRY:
            print(f"  WARNING: Unknown strategy '{s}' skipped. Use --list to see available names.")
    counts = {s: get_strategy(s).count_param_combinations() for s in known}
    if search_method == "successive_halving":
        total = sum(min(candidate_budget, count_params) for count_params in counts.values())
        resources_text = " -> ".join(str(value) for value in halving_resources)
        print(
            f"  Strategies: {len(known)} (of {len(strategies)} requested), "
            f"Initial candidates: {total}, Validation bars: {resources_text}"
        )
    else:
        total = sum(min(5, count_params) if quick else count_params for count_params in counts.values())
        print(f"  Strategies: {len(known)} (of {len(strategies)} requested), Total experiments: {total}")

    all_results = []
    n_results = 0
    n_skipped = 0
    n_errors = 0
    t0 = time.time()
    all_csv = run_dir / "all_results.csv"
    stage_csv = run_dir / "search_stages.csv"
    store_path = run_dir / "results.sqlite3"
    if not resume and (all_csv.exists() or stage_csv.exists() or store_path.exists()):
        # Preserve previous artifacts, but start from an empty transactional
        # journal when a run-id is intentionally reused without --resume.
        stamp = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:6]}"
        moved = []
        if all_csv.exists():
            backup = run_dir / f"all_results.{stamp}.bak.csv"
            all_csv.rename(backup)
            moved.append(backup.name)
        if stage_csv.exists():
            backup = run_dir / f"search_stages.{stamp}.bak.csv"
            stage_csv.rename(backup)
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
        if keep_all_results and search_method == "grid":
            all_results.extend(_iter_result_store(store_path))
        if n_skipped:
            print(f"  Resume: found {n_skipped} committed experiments in {store_path}")
    else:
        # Create the schema before work starts. SQLite transactions are the
        # canonical crash-safe checkpoint; CSV is an atomic exported view.
        with closing(_open_result_store(store_path)):
            pass

    use_workers = workers > 1 and len(known) > 0
    cache_totals = {"hits": 0, "misses": 0, "evictions": 0, "observed_evaluations": 0}
    n_stage_evaluations = 0
    n_stage_skipped = 0
    halving_stage_rows = []
    halving_diagnostics = None

    def _observe_cache(result):
        stats = result.get("indicator_cache") or {}
        if stats:
            cache_totals["observed_evaluations"] += 1
            for key in ("hits", "misses", "evictions"):
                cache_totals[key] += int(stats.get(key, 0))

    def _make_pool(worker_data, worker_df, worker_periods, worker_backtest_kwargs):
        if not use_workers:
            return None
        import multiprocessing as mp

        return mp.get_context("spawn").Pool(
            workers,
            initializer=_init_worker,
            initargs=(
                worker_data,
                worker_df,
                worker_periods,
                cost_bps,
                risk_free_rate,
                validation_folds,
                execution_price_column,
                worker_backtest_kwargs,
                indicator_cache_size,
            ),
        )

    if search_method == "grid":
        pool = _make_pool(development_data, development_df, selection_periods, development_backtest_kwargs)
        batch = []
        position_cache = OrderedDict()
        indicator_cache = IndicatorDAG(development_data, max_entries=indicator_cache_size)

        def _flush_batch():
            if batch:
                _store_results(batch, store_path)
                batch.clear()

        def _handle(result):
            nonlocal n_errors, n_results
            n_results += 1
            if result.get("error"):
                n_errors += 1
            _observe_cache(result)
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
                strategy = get_strategy(sname)
                if quick:
                    combos = _quick_sample(strategy, 5)
                    n_combos = len(combos)
                else:
                    # Iterate lazily: large grids contain hundreds of thousands
                    # of combinations and must not be materialized as a list.
                    combos = strategy.iter_param_combinations()
                    n_combos = strategy.count_param_combinations()
                if resume:
                    completed_for_strategy = _completed_params(store_path, sname)
                    n_combos = max(0, n_combos - len(completed_for_strategy))
                    combos = (params for params in combos if _canonical_params(params) not in completed_for_strategy)
                cat = (
                    "ML"
                    if sname.startswith("ml_")
                    else ("combo" if sname in ["ensemble", "stacked", "regime_aware"] else "classic")
                )
                print(f"\n  [{cat}] {sname} ({n_combos} params) ...")

                if pool is not None:
                    job = pool.imap_unordered(_worker_run, ((sname, params) for params in combos), chunksize=16)
                    for result in tqdm(job, desc=f"  {sname}", total=n_combos, leave=True):
                        _handle(result)
                else:
                    for params in tqdm(combos, desc=f"  {sname}", total=n_combos, leave=True):
                        result = run_single_experiment(
                            sname,
                            params,
                            development_data,
                            development_df,
                            selection_periods,
                            cost_bps,
                            risk_free_rate,
                            validation_folds=validation_folds,
                            execution_price_column=execution_price_column,
                            position_cache=position_cache,
                            indicator_cache=indicator_cache,
                            **development_backtest_kwargs,
                        )
                        _handle(result)

                _flush_batch()
                print(f"  [checkpoint] {sname} done, {n_results} total")
            search_ok = True
        finally:
            # Persist whatever was completed before an interruption so --resume
            # can pick up even when a strategy does not reach its checkpoint.
            _flush_batch()
            if pool is not None:
                if search_ok:
                    pool.close()
                else:
                    pool.terminate()
                pool.join()
            if store_path.exists():
                _export_result_store(store_path, all_csv)
    else:
        survivors = {sname: _quick_sample(get_strategy(sname), min(candidate_budget, counts[sname])) for sname in known}
        initial_candidates = sum(len(values) for values in survivors.values())
        staged_search_ok = False
        try:
            for stage, resource_bars in enumerate(halving_resources):
                cutoff = validation_index[resource_bars - 1]
                stage_data = _slice_temporal_mapping(development_data, cutoff)
                stage_df = development_df.loc[:cutoff]
                stage_periods = {
                    "train": selection_periods["train"],
                    "val": (selection_periods["val"][0], cutoff),
                }
                stage_backtest_kwargs = _slice_temporal_mapping(development_backtest_kwargs, cutoff)
                stage_pool = _make_pool(stage_data, stage_df, stage_periods, stage_backtest_kwargs)
                stage_position_cache = OrderedDict()
                stage_indicator_cache = IndicatorDAG(stage_data, max_entries=indicator_cache_size)
                final_stage = stage == len(halving_resources) - 1
                stage_batch = []
                stage_ok = False
                stage_summary = {
                    "stage": stage + 1,
                    "resource_bars": resource_bars,
                    "validation_end": str(cutoff),
                    "input_candidates": 0,
                    "advanced_candidates": 0,
                    "new_evaluations": 0,
                    "resumed_evaluations": 0,
                    "errors": 0,
                    "strategies": [],
                }

                def _flush_stage_batch(items=stage_batch, promote=final_stage):
                    if items:
                        _store_stage_results(items, store_path, final=promote)
                        items.clear()

                try:
                    print(
                        f"\n  [halving {stage + 1}/{len(halving_resources)}] "
                        f"{resource_bars} validation bars through {cutoff.date()}"
                    )
                    for sname in known:
                        candidates = survivors[sname]
                        stage_summary["input_candidates"] += len(candidates)
                        restored = _load_stage_results(store_path, sname, stage, resource_bars) if resume else {}
                        results_by_params = {
                            _canonical_params(params): restored[_canonical_params(params)]
                            for params in candidates
                            if _canonical_params(params) in restored
                        }
                        missing = [params for params in candidates if _canonical_params(params) not in restored]
                        resumed_count = len(candidates) - len(missing)
                        n_stage_skipped += resumed_count
                        stage_summary["resumed_evaluations"] += resumed_count
                        stage_summary["errors"] += sum(
                            bool(result.get("error")) for result in results_by_params.values()
                        )
                        print(f"    {sname}: {len(candidates)} candidates ({len(missing)} to run)")

                        if stage_pool is not None:
                            job = stage_pool.imap_unordered(
                                _worker_run,
                                ((sname, params) for params in missing),
                                chunksize=16,
                            )
                            iterator = tqdm(job, desc=f"    {sname}", total=len(missing), leave=True)
                            for result in iterator:
                                canonical = _canonical_params(result["params"])
                                results_by_params[canonical] = result
                                stage_batch.append((sname, result["params"], stage, resource_bars, result))
                                n_stage_evaluations += 1
                                stage_summary["new_evaluations"] += 1
                                stage_summary["errors"] += int(bool(result.get("error")))
                                _observe_cache(result)
                                if len(stage_batch) >= 512:
                                    _flush_stage_batch()
                        else:
                            for params in tqdm(missing, desc=f"    {sname}", total=len(missing), leave=True):
                                result = run_single_experiment(
                                    sname,
                                    params,
                                    stage_data,
                                    stage_df,
                                    stage_periods,
                                    cost_bps,
                                    risk_free_rate,
                                    validation_folds=validation_folds,
                                    execution_price_column=execution_price_column,
                                    position_cache=stage_position_cache,
                                    indicator_cache=stage_indicator_cache,
                                    **stage_backtest_kwargs,
                                )
                                canonical = _canonical_params(params)
                                results_by_params[canonical] = result
                                stage_batch.append((sname, params, stage, resource_bars, result))
                                n_stage_evaluations += 1
                                stage_summary["new_evaluations"] += 1
                                stage_summary["errors"] += int(bool(result.get("error")))
                                _observe_cache(result)
                                if len(stage_batch) >= 512:
                                    _flush_stage_batch()

                        _flush_stage_batch()
                        if final_stage:
                            # Also repairs a manually damaged canonical table
                            # from the complete, validation-only stage journal.
                            promoted = [results_by_params[_canonical_params(params)] for params in candidates]
                            _store_results(promoted, store_path)
                            next_candidates = candidates
                        else:
                            next_candidates = _stage_survivors(candidates, results_by_params, halving_factor)
                        _mark_stage_survivors(store_path, sname, stage, next_candidates)
                        survivors[sname] = next_candidates
                        stage_summary["advanced_candidates"] += len(next_candidates)
                        best_stage_score = max(
                            (_stage_score(results_by_params[_canonical_params(params)]) for params in candidates),
                            default=None,
                        )
                        stage_summary["strategies"].append(
                            {
                                "strategy": sname,
                                "input_candidates": len(candidates),
                                "advanced_candidates": len(next_candidates),
                                "best_stage_sharpe": (
                                    best_stage_score
                                    if best_stage_score is not None and np.isfinite(best_stage_score)
                                    else None
                                ),
                            }
                        )
                    stage_ok = True
                finally:
                    _flush_stage_batch()
                    if stage_pool is not None:
                        if stage_ok:
                            stage_pool.close()
                        else:
                            stage_pool.terminate()
                        stage_pool.join()
                    _export_stage_store(store_path, stage_csv)
                    _export_result_store(store_path, all_csv)
                halving_stage_rows.append(stage_summary)
            staged_search_ok = True
        finally:
            if store_path.exists():
                _export_stage_store(store_path, stage_csv)
                _export_result_store(store_path, all_csv)

        if staged_search_ok:
            n_results, n_errors = _store_counts(store_path)
            if keep_all_results:
                all_results = list(_iter_result_store(store_path))
        halving_diagnostics = {
            "method": "successive_halving",
            "initial_candidates": initial_candidates,
            "resource_bars": halving_resources,
            "stages": halving_stage_rows,
            "new_stage_evaluations": n_stage_evaluations,
            "resumed_stage_evaluations": n_stage_skipped,
            "total_stage_evaluations": n_stage_evaluations + n_stage_skipped,
            "final_candidates": n_results,
            "eliminated_candidates": max(0, initial_candidates - n_results),
        }

    elapsed = time.time() - t0
    if halving_diagnostics is None:
        halving_diagnostics = {
            "method": "grid",
            "candidate_evaluations": n_results,
            "resumed_candidates": n_skipped,
        }
    cache_requests = cache_totals["hits"] + cache_totals["misses"]
    cache_diagnostics = {
        "max_entries_per_process": int(indicator_cache_size),
        **cache_totals,
        "hit_rate": cache_totals["hits"] / cache_requests if cache_requests else 0.0,
    }
    print(f"\n  Search complete! {n_results} results ({n_errors} errors) in {elapsed / 60:.1f} min")

    if n_results == 0:
        print("  WARNING: No experiments completed. Check strategy names and data.")
        test_access["status"] = "no_selection"
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
            "bootstrap_diagnostics": bootstrap_diagnostics,
            "test_access": test_access,
            "parameter_sensitivity": None,
            "search_diagnostics": halving_diagnostics,
            "indicator_cache": cache_diagnostics,
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
        trial_count_override=(
            halving_diagnostics.get("total_stage_evaluations") if search_method == "successive_halving" else None
        ),
        sharpe_std_override=(_stage_score_dispersion(store_path) if search_method == "successive_halving" else None),
    )
    if not keep_all_results:
        all_results = []

    # Phase 3: Freeze selection, then either remain sealed or reserve a reveal.
    benchmark_metrics = None
    if top:
        # Copy the selected candidate so neither the checkpoint nor the
        # validation-ranked top list is mutated with test-set information.
        best = copy.deepcopy(top[0])
        sname = best["strategy"]
        params = best.get("params", {})
        if study_id:
            registry.bind_selection(
                study_id,
                _jsonable(
                    {
                        "strategy": sname,
                        "params": params,
                        "val_metrics": best["val_metrics"],
                        "selection_diagnostics": best["selection_diagnostics"],
                    }
                ),
            )
            test_access = registry.status(study_id)
            if bootstrap:
                selected_dev, benchmark_dev = _selected_ledgers(
                    best,
                    development_data,
                    development_df,
                    execution_price_column,
                    cost_bps,
                    development_backtest_kwargs,
                )
                bootstrap_diagnostics["periods"].update(
                    _bootstrap_windows(
                        selected_dev,
                        benchmark_dev,
                        selection_periods,
                        [("validation", "val")],
                        bootstrap_options,
                        annualization,
                        risk_free_rate,
                    )
                )
        payload = None
        if not study_id or reveal_test:
            claim = registry.claim_test(
                ticker=ticker,
                start=periods["test"][0],
                end=periods["test"][1],
                data_snapshot=metadata["data_snapshot"],
                run_id=run_id,
                run_path=run_dir,
                study_id=study_id,
                allow_reuse=allow_test_reuse,
                reason=test_reuse_reason,
            )
            test_access.update(claim["access"])
            if claim["cached"]:
                payload = claim["payload"]
            else:
                event_id = test_access["event_id"]
                try:
                    payload = _test_payload(
                        best,
                        data,
                        df,
                        periods,
                        execution_price_column,
                        cost_bps,
                        backtest_kwargs,
                        annualization,
                        risk_free_rate,
                        bootstrap_options if bootstrap else None,
                        include_validation=not study_id,
                    )
                    registry.complete_test(event_id, payload)
                except BaseException as exc:
                    # An interrupted reservation remains possible exposure even
                    # if recording the failure itself cannot complete.
                    try:
                        registry.fail_test(event_id, f"{type(exc).__name__}: {exc}")
                    except Exception as audit_error:
                        warnings.warn(
                            f"Could not finalize failed reveal; reservation retained: {audit_error}", RuntimeWarning
                        )
                    raise
            best["test_metrics"] = payload["test_metrics"]
            best["test_evaluated_at"] = payload["test_evaluated_at"]
            benchmark_metrics = payload["benchmark_metrics"]
            bootstrap_diagnostics["periods"].update(payload["bootstrap_periods"])
            test_access["test_results_visible"] = True
        if bootstrap:
            statuses = [row["status"] for row in bootstrap_diagnostics["periods"].values()]
            bootstrap_diagnostics["status"] = (
                "ok"
                if all(status == "ok" for status in statuses)
                else "partial"
                if any(status in {"ok", "partial"} for status in statuses)
                else "unavailable"
            )
        print(f"\n  Best: {sname}")
        print(f"  Params: {_params_to_str(params)}")
        print(f"  Val Sharpe:   {best['val_metrics'].get('sharpe', 0):.4f}")
        print(f"  Deflated Sharpe probability: {best['selection_diagnostics']['deflated_sharpe_probability']:.2%}")
        print(f"  Test access: {test_access['status']} (history outside this registry remains unknown)")
        if payload is not None:
            test_m = payload["test_metrics"]
            print(f"  Test Sharpe:  {test_m['sharpe']:.4f} (B&H: {benchmark_metrics['sharpe']:.4f})")
            print(f"  Test CAGR:    {test_m['cagr']:.2%} (B&H: {benchmark_metrics['cagr']:.2%})")
            print(f"  Test MaxDD:   {test_m['max_drawdown']:.2%} (B&H: {benchmark_metrics['max_drawdown']:.2%})")
        else:
            print("  Test scores remain hidden. Explicitly reveal the frozen study when ready.")
    else:
        best = None
        test_access["status"] = "no_selection"
        print("  WARNING: No valid experiments remained after evaluation.")

    # Phase 4: Robustness check on the best parameters
    robustness = None
    if robust and best is not None:
        print(f"\n  [Phase 4] Parameter sensitivity (perturbing selected params by {robust_frac:.0%}) ...")
        robustness = robustness_check(
            development_data,
            development_df,
            selection_periods,
            sname,
            params,
            cost_bps=cost_bps,
            frac=robust_frac,
            backtest_kwargs=development_backtest_kwargs,
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
        "bootstrap_diagnostics": _jsonable(bootstrap_diagnostics),
        "test_access": _jsonable(test_access),
        "selection_diagnostics": _jsonable(selection_diagnostics),
        "search_diagnostics": _jsonable(halving_diagnostics),
        "indicator_cache": _jsonable(cache_diagnostics),
        "parameter_sensitivity": _jsonable(robustness),
        "n_results": n_results,
        "n_skipped": n_skipped,
        "n_errors": n_errors,
    }
    report_paths = {}
    if generate_report:
        markdown_path = run_dir / "report.md"
        html_path = run_dir / "report.html"
        _write_text_atomic(markdown_path, render_markdown_report(summary, metadata))
        _write_text_atomic(html_path, render_html_report(summary, metadata))
        report_paths = {"markdown": str(markdown_path), "html": str(html_path)}
        summary["reports"] = {"markdown": markdown_path.name, "html": html_path.name}
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
        "bootstrap_diagnostics": bootstrap_diagnostics,
        "test_access": test_access,
        "selection_diagnostics": selection_diagnostics,
        "search_diagnostics": halving_diagnostics,
        "indicator_cache": cache_diagnostics,
        "run_id": run_id,
        "result_dir": str(run_dir),
        "n_results": n_results,
        "n_skipped": n_skipped,
        "n_errors": n_errors,
        "reports": report_paths,
    }
