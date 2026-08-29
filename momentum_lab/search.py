"""search.py - Exhaustive parameter search engine."""

import hashlib
import heapq
import importlib.util
import json
import subprocess
import time
import warnings
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
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
ENGINE_SCHEMA_VERSION = 2
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
)
RESULT_COLUMNS = [
    "strategy",
    "params",
    "error",
    *(f"{period}_{metric}" for period in ("train", "val", "test") for metric in METRIC_KEYS),
]

# Shared state for parallel sub-processes (set via Pool initializer).
_POOL_STATE = None


def _init_worker(data, df, periods, cost_bps, risk_free_rate, backtest_kwargs):
    global _POOL_STATE
    _POOL_STATE = (data, df, periods, cost_bps, risk_free_rate, backtest_kwargs)


def _worker_run(args):
    """Module-level worker that runs a single experiment (required on Windows)."""
    strategy_name, params = args
    data, df, periods, cost_bps, risk_free_rate, backtest_kwargs = _POOL_STATE
    return run_single_experiment(strategy_name, params, data, df, periods, cost_bps, risk_free_rate, **backtest_kwargs)


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


def _results_rows(results):
    rows = []
    for r in results:
        row = {
            "strategy": r.get("strategy", ""),
            "params": json.dumps(r.get("params", {}), ensure_ascii=False, default=_jsonable),
            "error": r.get("error", ""),
        }
        for period in ["train", "val", "test"]:
            m = r.get(f"{period}_metrics", {})
            for metric in METRIC_KEYS:
                row[f"{period}_{metric}"] = m.get(metric, np.nan)
        rows.append(row)
    return rows


def _append_results_csv(results, path, write_header):
    """Incrementally flush checkpoint rows instead of rewriting the whole file.

    Full-search checkpoints approach 400 MB; rewriting the cumulative CSV
    after every strategy wastes multi-GB of I/O.
    """
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


def _load_checkpoint(path):
    """Load checkpoint rows written by ``_append_results_csv``.

    Checkpoints deliberately use a flat CSV so they remain inspectable without
    importing the package.  This parser restores the nested result shape used
    by the ranking phase and accepts older files that do not have ``error``.
    """
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"cannot read search checkpoint {path}: {exc}") from exc
    if frame.empty:
        return []
    required = {"strategy", "params"}
    missing = required - set(frame.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"search checkpoint {path} is missing column(s): {names}")

    results = []
    for row in frame.to_dict("records"):
        raw_params = row.get("params", "{}")
        try:
            params = json.loads(raw_params) if isinstance(raw_params, str) else raw_params
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"search checkpoint {path} contains invalid params JSON") from exc
        if not isinstance(params, dict):
            raise TypeError(f"search checkpoint {path} contains non-object params")
        result = {
            "strategy": str(row.get("strategy", "")),
            "params": params,
            "train_metrics": {},
            "val_metrics": {},
            "test_metrics": {},
        }
        error = row.get("error", "")
        if pd.notna(error) and str(error):
            result["error"] = str(error)
        for period in ("train", "val", "test"):
            prefix = f"{period}_"
            metrics = result[f"{period}_metrics"]
            for key, value in row.items():
                if not key.startswith(prefix) or pd.isna(value):
                    continue
                metric_name = key[len(prefix) :]
                metrics[metric_name] = float(value) if isinstance(value, (int, float, np.number)) else value
        results.append(result)
    return results


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
    "git_sha",
    "source_fingerprint",
    "cost_bps",
    "slippage_bps",
    "financing_rate",
    "borrow_bps",
    "cash_rate",
    "max_leverage",
    "annualization",
    "risk_free_rate",
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

    The checkpoint CSV stores metrics computed with the previous run's data
    snapshot and cost model; resuming with different values would silently
    mix incomparable Sharpe/CAGR numbers in one ranking.  Must be called
    BEFORE ``run_config.json`` is rewritten, while the previous run's file
    is still on disk.
    """
    if not (run_dir / "all_results.csv").exists():
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
    """Split an ordered index into non-overlapping train, validation, and test ranges."""
    n = len(index)
    if n < 3:
        raise ValueError("at least 3 data points are required to split search periods")

    split1 = max(1, min(int(n * 0.6), n - 2))
    split2 = max(split1 + 1, min(int(n * 0.8), n - 1))
    return {
        "train": (index[0], index[split1 - 1]),
        "val": (index[split1], index[split2 - 1]),
        "test": (index[split2], index[-1]),
    }


def _quick_sample(strategy, k=5):
    """Pick ``k`` evenly spaced parameter combinations without materializing the grid.

    Taking the first ``k`` combos biases the quick-mode sample toward the
    smallest lookbacks (and, for most grids, long_short=True), which is not
    representative of the search space.
    """
    total = strategy.count_param_combinations()
    if k <= 0:
        return []
    if total <= k:
        return list(strategy.iter_param_combinations())
    if k == 1:
        return [next(iter(strategy.iter_param_combinations()))]
    wanted = sorted({round(i * (total - 1) / (k - 1)) for i in range(k)})
    picked = []
    for i, combo in enumerate(strategy.iter_param_combinations()):
        if i == wanted[len(picked)]:
            picked.append(combo)
            if len(picked) == len(wanted):
                break
    return picked


def run_single_experiment(
    strategy_name, params, data, df, periods, cost_bps=1.0, risk_free_rate=RISK_FREE_RATE, **backtest_kwargs
):
    try:
        strategy = get_strategy(strategy_name)
        positions = strategy.run(data, **params)
    except Exception as e:
        return {
            "strategy": strategy_name,
            "params": params,
            "error": str(e),
            "train_metrics": {},
            "val_metrics": {},
            "test_metrics": {},
        }
    prices = df["close"]
    results = {"strategy": strategy_name, "params": params, "train_metrics": {}, "val_metrics": {}, "test_metrics": {}}
    period_errors = []
    for pname, (start, end) in periods.items():
        try:
            pp = positions.loc[start:end]
            pr = prices.loc[start:end]
            if len(pp) == 0 or pp.abs().sum() == 0:
                results[f"{pname}_metrics"] = {"sharpe": -99}
                continue
            bt = backtest(pp, pr, cost_bps=cost_bps, **backtest_kwargs)
            results[f"{pname}_metrics"] = evaluate(
                bt["returns"],
                risk_free_rate=risk_free_rate,
                annualization=backtest_kwargs.get("annualization", 252),
            )
        except Exception as exc:
            results[f"{pname}_metrics"] = {"sharpe": -99}
            period_errors.append(f"{pname}: {exc}")
    if period_errors:
        results["error"] = "; ".join(period_errors)
    return results


def run_search(
    ticker="GLD",
    strategies=None,
    cost_bps=1.0,
    slippage_bps=0.0,
    financing_rate=0.0,
    borrow_bps=0.0,
    cash_rate=0.0,
    max_leverage=2.0,
    annualization=None,
    risk_free_rate=RISK_FREE_RATE,
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
        max_leverage: Final absolute exposure cap applied by the backtest.
        annualization: Return periods per year. None infers 365 for common
            Yahoo crypto pairs and 252 otherwise.
        risk_free_rate: Annual risk-free rate as a decimal, used in Sharpe/Sortino.
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
            stream results to ``all_results.csv`` and keep only the top-N
            ranking in memory.  ``all_results`` is then an empty list and
            ``n_results`` carries the experiment count.
        config: Optional :class:`SearchConfig`, mapping, or JSON path.  When
            provided, its fields are used as the complete search configuration.
        resume: If True, reuse completed rows in ``all_results.csv`` for the
            explicit ``run_id`` and evaluate only missing parameter combinations.

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
        max_leverage = configured["max_leverage"]
        annualization = configured["annualization"]
        risk_free_rate = configured["risk_free_rate"]
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
    if not np.isfinite(cash_rate) or not np.isfinite(risk_free_rate):
        raise ValueError("cash_rate and risk_free_rate must be finite")
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
        "max_leverage": max_leverage,
    }
    print(f"momentum-lab: Comparing strategies for {ticker}")

    data, df = prepare_data(ticker, start=start, end=end, use_cache=use_cache, annualization=annualization)
    prices = df["close"]
    n = len(df)
    periods = _split_periods(df.index)
    if strategies is None:
        # Safe default: a small quick run over non-ML strategies. ML remains
        # available when requested explicitly but no longer turns a one-line
        # command into days of model fitting and heavy optional dependencies.
        strategies = list(CLASSIC_STRATEGIES)
    elif isinstance(strategies, str):
        strategies = [name.strip() for name in strategies.split(",") if name.strip()]
    else:
        strategies = list(strategies)

    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "strategies": strategies,
        "cost_bps": cost_bps,
        **backtest_kwargs,
        "risk_free_rate": risk_free_rate,
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
    (run_dir / "run_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Data: {df.index[0].date()} ~ {df.index[-1].date()}, {n} bars")
    print(f"  Train: {periods['train'][0].date()} ~ {periods['train'][1].date()}")
    print(f"  Val:   {periods['val'][0].date()} ~ {periods['val'][1].date()}")
    print(f"  Test:  {periods['test'][0].date()} ~ {periods['test'][1].date()}")

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
    # Bounded min-heap of (val_sharpe, seq, result) holding the current top-N.
    # Maintained incrementally so full grids never need every result resident.
    top_heap = []
    seq = count()
    t0 = time.time()
    all_csv = run_dir / "all_results.csv"
    if not resume and all_csv.exists():
        # A fresh run reusing an explicit run_id must not append to the old
        # checkpoint: resume dedup keeps the FIRST row per key, so stale rows
        # (possibly computed under a different config) would win over the new
        # ones.  Move the old checkpoint aside instead of silently mixing.
        stamp = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:6]}"
        backup = run_dir / f"all_results.{stamp}.bak.csv"
        warnings.warn(
            f"{all_csv} exists from a previous run; moved to {backup.name}. "
            f"Pass resume=True to continue that run instead.",
            RuntimeWarning,
        )
        all_csv.rename(backup)

    def _offer_top(result):
        sharpe = result.get("val_metrics", {}).get("sharpe", -99)
        if "error" not in result and sharpe > -99:
            entry = (sharpe, next(seq), result)
            if len(top_heap) < top_n:
                heapq.heappush(top_heap, entry)
            elif sharpe > top_heap[0][0]:
                heapq.heapreplace(top_heap, entry)

    completed_keys = set()
    if resume:
        checkpoint_results = _load_checkpoint(all_csv)
        for previous in checkpoint_results:
            key = _params_key(previous.get("strategy", ""), previous.get("params", {}))
            if key in completed_keys:
                continue
            completed_keys.add(key)
            n_skipped += 1
            n_results += 1
            if previous.get("error"):
                n_errors += 1
            _offer_top(previous)
            if keep_all_results:
                all_results.append(previous)
        if checkpoint_results:
            print(f"  Resume: loaded {n_skipped} completed experiments from {all_csv}")

    use_workers = workers > 1 and len(known) > 0
    pool = None
    if use_workers:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        pool = ctx.Pool(
            workers,
            initializer=_init_worker,
            initargs=(data, df, periods, cost_bps, risk_free_rate, backtest_kwargs),
        )

    batch = []

    def _flush_batch():
        if batch:
            _append_results_csv(batch, all_csv, write_header=not all_csv.exists())
            batch.clear()

    def _handle(result):
        nonlocal n_errors, n_results
        n_results += 1
        if result.get("error"):
            n_errors += 1
        completed_keys.add(_params_key(result.get("strategy", ""), result.get("params", {})))
        _offer_top(result)
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
                combos = (params for params in combos if _params_key(sname, params) not in completed_keys)
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
                        sname, params, data, df, periods, cost_bps, risk_free_rate, **backtest_kwargs
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

    # Phase 2: Rank by val Sharpe
    if keep_all_results:
        valid = [r for r in all_results if "error" not in r and r.get("val_metrics", {}).get("sharpe", -99) > -99]
        valid.sort(key=lambda r: -r.get("val_metrics", {}).get("sharpe", -99))
        top = valid[:top_n]
    else:
        top = [r for _, _, r in sorted(top_heap, key=lambda e: -e[0])]
        all_results = []

    # Phase 3: Test set evaluation
    benchmark_metrics = None
    if top:
        best = top[0]
        sname = best["strategy"]
        params = best.get("params", {})
        # The grid run already evaluated the test window with identical cost
        # settings; reuse those metrics instead of re-running the strategy.
        test_m = best.get("test_metrics") or {}
        if "cagr" not in test_m:
            # Only the sentinel was stored (degenerate test window, e.g. zero
            # positions); re-evaluate directly for a complete report.
            strategy = get_strategy(sname)
            positions = strategy.run(data, **params)
            test_m = evaluate(
                backtest(
                    positions.loc[periods["test"][0] : periods["test"][1]],
                    prices.loc[periods["test"][0] : periods["test"][1]],
                    cost_bps=cost_bps,
                    **backtest_kwargs,
                )["returns"],
                risk_free_rate=risk_free_rate,
                annualization=annualization,
            )
            best["test_metrics"] = test_m
        benchmark_metrics = evaluate(
            backtest(
                get_buy_and_hold(prices.loc[periods["test"][0] : periods["test"][1]]),
                prices.loc[periods["test"][0] : periods["test"][1]],
                # Charge the benchmark the same one-shot entry cost so the
                # comparison does not systematically flatter buy & hold.
                cost_bps=cost_bps,
                **backtest_kwargs,
            )["returns"],
            risk_free_rate=risk_free_rate,
            annualization=annualization,
        )
        print(f"\n  Best: {sname}")
        print(f"  Params: {_params_to_str(params)}")
        print(f"  Val Sharpe:   {best['val_metrics'].get('sharpe', 0):.4f}")
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
            pd.DataFrame(
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
            ).to_csv(run_dir / "robustness.csv", index=False, encoding="utf-8-sig")
            print(f"    Saved to {run_dir / 'robustness.csv'}")

    # Save summary (all_results.csv was already flushed incrementally)
    if top:
        rows = []
        for r in top:
            row = {
                "strategy": r["strategy"],
                "params": json.dumps(r.get("params", {}), ensure_ascii=False, default=_jsonable),
            }
            for p in ["train", "val", "test"]:
                m = r.get(f"{p}_metrics", {})
                for k, v in m.items():
                    row[f"{p}_{k}"] = v
            rows.append(row)
        pd.DataFrame(rows).to_csv(run_dir / "top_results.csv", index=False, encoding="utf-8-sig")

    summary = {
        "run_id": run_id,
        "best": _jsonable(best),
        "benchmark_metrics": _jsonable(benchmark_metrics),
        "parameter_sensitivity": _jsonable(robustness),
        "n_results": n_results,
        "n_skipped": n_skipped,
        "n_errors": n_errors,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    return {
        "all_results": all_results,
        "top_results": top,
        "best": best,
        "robustness": robustness,
        "parameter_sensitivity": robustness,
        "benchmark_metrics": benchmark_metrics,
        "run_id": run_id,
        "result_dir": str(run_dir),
        "n_results": n_results,
        "n_skipped": n_skipped,
        "n_errors": n_errors,
    }
