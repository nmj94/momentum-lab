"""search.py - Exhaustive parameter search engine."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .backtest import backtest, evaluate, get_buy_and_hold
from .data import prepare_data
from .robustness import robustness_check
from .strategies import STRATEGY_REGISTRY, get_strategy

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kw):
        return x


RESULT_DIR = Path("experiments")

# Shared state for parallel sub-processes (set via Pool initializer).
_POOL_STATE = None


def _init_worker(data, df, periods, cost_bps, backtest_kwargs):
    global _POOL_STATE
    _POOL_STATE = (data, df, periods, cost_bps, backtest_kwargs)


def _worker_run(args):
    """Module-level worker that runs a single experiment (required on Windows)."""
    strategy_name, params = args
    data, df, periods, cost_bps, backtest_kwargs = _POOL_STATE
    result = run_single_experiment(strategy_name, params, data, df, periods, cost_bps, **backtest_kwargs)
    return {k: v for k, v in result.items() if k != "positions"}


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
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


def _save_results_csv(results, path):
    rows = []
    for r in results:
        row = {
            "strategy": r.get("strategy", ""),
            "params": json.dumps(r.get("params", {}), ensure_ascii=False, default=_jsonable),
        }
        for period in ["train", "val", "test"]:
            m = r.get(f"{period}_metrics", {})
            for k, v in m.items():
                row[f"{period}_{k}"] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _normalize_results(results):
    normalized = []
    for r in results:
        if "val_metrics" not in r and "val_sharpe" in r:
            r = dict(r)
            for period in ["train", "val", "test"]:
                r[f"{period}_metrics"] = {}
                for k in [
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
                ]:
                    col = f"{period}_{k}"
                    if col in r:
                        try:
                            r[f"{period}_metrics"][k] = float(r[col])
                        except (ValueError, TypeError):
                            r[f"{period}_metrics"][k] = -99
            if "params" in r and isinstance(r["params"], str):
                try:
                    r["params"] = json.loads(r["params"])
                except Exception:
                    r["params"] = {}
        normalized.append(r)
    return normalized


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
    if total <= k:
        return list(strategy.iter_param_combinations())
    wanted = sorted({round(i * (total - 1) / (k - 1)) for i in range(k)})
    picked = []
    for i, combo in enumerate(strategy.iter_param_combinations()):
        if i == wanted[len(picked)]:
            picked.append(combo)
            if len(picked) == len(wanted):
                break
    return picked


def run_single_experiment(strategy_name, params, data, df, periods, cost_bps=1.0, **backtest_kwargs):
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
    for pname, (start, end) in periods.items():
        try:
            pp = positions.loc[start:end]
            pr = prices.loc[start:end]
            if len(pp) == 0 or pp.abs().sum() == 0:
                results[f"{pname}_metrics"] = {"sharpe": -99}
                continue
            bt = backtest(pp, pr, cost_bps=cost_bps, **backtest_kwargs)
            results[f"{pname}_metrics"] = evaluate(
                bt["returns"], annualization=backtest_kwargs.get("annualization", 252)
            )
        except Exception:
            results[f"{pname}_metrics"] = {"sharpe": -99}
    return results


def run_search(
    ticker="GLD",
    strategies=None,
    cost_bps=1.0,
    slippage_bps=0.0,
    financing_rate=0.0,
    borrow_bps=0.0,
    annualization=252,
    workers=1,
    quick=False,
    top_n=50,
    start="2004-01-01",
    end=None,
    robust=True,
    robust_frac=0.2,
    use_cache=True,
    result_dir=None,
    run_id=None,
):
    """Run exhaustive strategy search for any ticker.

    Args:
        ticker: Yahoo Finance ticker (e.g. "GLD", "SPY", "BTC-USD").
        strategies: List of strategy names. None = all.
        cost_bps: Transaction cost in basis points.
        slippage_bps: Additional transaction slippage in basis points.
        financing_rate: Annual financing rate applied to held exposure.
        borrow_bps: Annualized short borrow fee in basis points.
        annualization: Return periods per year (252 for trading days, 365 for crypto).
        workers: Number of parallel workers (1 = sequential).
        quick: If True, only test 5 params per strategy.
        top_n: Number of top results to keep.
        start: Data start date.
        end: Data end date. None = today.
        robust: If True, run a neighborhood robustness check on the best params.
        robust_frac: Perturbation fraction for the robustness check.
        use_cache: If True, reuse cached OHLCV data when available.
        result_dir: Parent directory for run artifacts. Defaults to ``experiments``.
        run_id: Optional stable run directory name. A unique ID is generated by default.

    Returns:
        dict with 'all_results', 'top_results', 'best', 'robustness',
        'run_id', and 'result_dir'.
    """
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
    }
    print(f"momentum-lab: Searching optimal strategies for {ticker}")

    data, df = prepare_data(ticker, start=start, end=end, use_cache=use_cache)
    prices = df["close"]
    n = len(df)
    periods = _split_periods(df.index)
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "strategies": strategies if strategies is not None else list(STRATEGY_REGISTRY),
        "cost_bps": cost_bps,
        **backtest_kwargs,
        "workers": workers,
        "quick": quick,
        "top_n": top_n,
        "start": start,
        "end": end,
        "data_start": str(df.index[0]),
        "data_end": str(df.index[-1]),
        "n_bars": n,
        "periods": {name: [str(bounds[0]), str(bounds[1])] for name, bounds in periods.items()},
    }
    (run_dir / "run_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Data: {df.index[0].date()} ~ {df.index[-1].date()}, {n} bars")
    print(f"  Train: {periods['train'][0].date()} ~ {periods['train'][1].date()}")
    print(f"  Val:   {periods['val'][0].date()} ~ {periods['val'][1].date()}")
    print(f"  Test:  {periods['test'][0].date()} ~ {periods['test'][1].date()}")

    if strategies is None:
        strategies = list(STRATEGY_REGISTRY.keys())

    known = [s for s in strategies if s in STRATEGY_REGISTRY]
    for s in strategies:
        if s not in STRATEGY_REGISTRY:
            print(f"  WARNING: Unknown strategy '{s}' skipped. Use --list to see available names.")
    counts = {s: get_strategy(s).count_param_combinations() for s in known}
    total = sum(min(5, c) if quick else c for c in counts.values())
    print(f"  Strategies: {len(known)} (of {len(strategies)} requested), Total experiments: {total}")

    all_results = []
    t0 = time.time()

    use_workers = workers > 1 and len(strategies) > 0
    pool = None
    if use_workers:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        pool = ctx.Pool(
            workers,
            initializer=_init_worker,
            initargs=(data, df, periods, cost_bps, backtest_kwargs),
        )

    try:
        for sname in strategies:
            if sname not in STRATEGY_REGISTRY:
                continue
            s = get_strategy(sname)
            if quick:
                combos = _quick_sample(s, 5)
                n_combos = len(combos)
            else:
                # Iterate lazily: the largest grids approach a million
                # combinations and must not be materialized as a list.
                combos = s.iter_param_combinations()
                n_combos = s.count_param_combinations()
            cat = (
                "ML"
                if sname.startswith("ml_")
                else ("combo" if sname in ["ensemble", "stacked", "regime_aware"] else "classic")
            )
            print(f"\n  [{cat}] {sname} ({n_combos} params) ...")

            if pool is not None:
                job = pool.imap_unordered(_worker_run, ((sname, p) for p in combos), chunksize=16)
                for result in tqdm(job, desc=f"  {sname}", total=n_combos, leave=True):
                    all_results.append(result)  # noqa: PERF402 - accumulates across strategies/checkpoints
            else:
                for i, params in enumerate(tqdm(combos, desc=f"  {sname}", total=n_combos, leave=True)):
                    result = run_single_experiment(sname, params, data, df, periods, cost_bps, **backtest_kwargs)
                    compact = {k: v for k, v in result.items() if k != "positions"}
                    all_results.append(compact)

            _save_results_csv(all_results, run_dir / "all_results.csv")
            print(f"  [checkpoint] {sname} done, {len(all_results)} total")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    elapsed = time.time() - t0
    print(f"\n  Search complete! {len(all_results)} results in {elapsed / 60:.1f} min")

    if not all_results:
        print("  WARNING: No experiments completed. Check strategy names and data.")
        return {"all_results": [], "top_results": [], "best": None, "run_id": run_id, "result_dir": str(run_dir)}

    # Phase 2: Rank by val Sharpe
    all_results = _normalize_results(all_results)
    valid = [r for r in all_results if "error" not in r and r.get("val_metrics", {}).get("sharpe", -99) > -99]
    valid.sort(key=lambda r: -r.get("val_metrics", {}).get("sharpe", -99))
    top = valid[:top_n]

    # Phase 3: Test set evaluation
    if top:
        best = top[0]
        sname = best["strategy"]
        params = best.get("params", {})
        strategy = get_strategy(sname)
        positions = strategy.run(data, **params)
        test_m = evaluate(
            backtest(
                positions.loc[periods["test"][0] : periods["test"][1]],
                prices.loc[periods["test"][0] : periods["test"][1]],
                cost_bps=cost_bps,
                **backtest_kwargs,
            )["returns"],
            annualization=annualization,
        )
        bh_m = evaluate(
            backtest(
                get_buy_and_hold(prices.loc[periods["test"][0] : periods["test"][1]]),
                prices.loc[periods["test"][0] : periods["test"][1]],
                # Charge the benchmark the same one-shot entry cost so the
                # comparison does not systematically flatter buy & hold.
                cost_bps=cost_bps,
                **backtest_kwargs,
            )["returns"],
            annualization=annualization,
        )
        print(f"\n  Best: {sname}")
        print(f"  Params: {_params_to_str(params)}")
        print(f"  Val Sharpe:   {best['val_metrics'].get('sharpe', 0):.4f}")
        print(f"  Test Sharpe:  {test_m['sharpe']:.4f} (B&H: {bh_m['sharpe']:.4f})")
        print(f"  Test CAGR:    {test_m['cagr']:.2%} (B&H: {bh_m['cagr']:.2%})")
        print(f"  Test MaxDD:   {test_m['max_drawdown']:.2%} (B&H: {bh_m['max_drawdown']:.2%})")
    else:
        best = None

    # Phase 4: Robustness check on the best parameters
    robustness = None
    if robust and best is not None:
        print(f"\n  [Phase 4] Robustness check (perturbing optimal params by {robust_frac:.0%}) ...")
        robustness = robustness_check(
            data,
            df,
            periods,
            sname,
            params,
            cost_bps=cost_bps,
            frac=robust_frac,
            backtest_kwargs=backtest_kwargs,
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
                f"    Robustness grade: {robustness['grade']} "
                f"({robustness['verdict']})"
                + ("  [ISOLATED PEAK - likely overfit]" if robustness["isolated_peak"] else "")
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

    # Save summary
    _save_results_csv(all_results, run_dir / "all_results.csv")
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

    return {
        "all_results": all_results,
        "top_results": top,
        "best": best,
        "robustness": robustness,
        "run_id": run_id,
        "result_dir": str(run_dir),
    }
