"""search.py - Exhaustive parameter search engine."""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import backtest, evaluate, get_buy_and_hold
from .data import prepare_data
from .strategies import STRATEGY_REGISTRY, get_strategy, list_strategies

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x

RESULT_DIR = Path("experiments")


def _jsonable(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, (np.bool_,)): return bool(v)
    if isinstance(v, (list, tuple)): return [_jsonable(x) for x in v]
    return v


def _params_to_str(params):
    if isinstance(params, str):
        try: params = json.loads(params)
        except Exception: return params[:80]
    if not isinstance(params, dict): return str(params)[:80]
    parts = []
    for k, v in params.items():
        if isinstance(v, tuple): v = "_".join(str(x) for x in v)
        elif isinstance(v, bool): v = "T" if v else "F"
        elif isinstance(v, float): v = f"{v:.3f}"
        parts.append(f"{k}={v}")
    return ", ".join(parts)


def _save_results_csv(results, path):
    rows = []
    for r in results:
        row = {"strategy": r.get("strategy", ""), "params": json.dumps(r.get("params", {}), ensure_ascii=False, default=_jsonable)}
        for period in ["train", "val", "test"]:
            m = r.get(f"{period}_metrics", {})
            for k, v in m.items(): row[f"{period}_{k}"] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _normalize_results(results):
    normalized = []
    for r in results:
        if "val_metrics" not in r and "val_sharpe" in r:
            r = dict(r)
            for period in ["train", "val", "test"]:
                r[f"{period}_metrics"] = {}
                for k in ["sharpe", "sortino", "calmar", "max_drawdown", "cagr", "total_return", "volatility", "win_rate", "profit_factor", "skew", "kurtosis"]:
                    col = f"{period}_{k}"
                    if col in r:
                        try: r[f"{period}_metrics"][k] = float(r[col])
                        except (ValueError, TypeError): r[f"{period}_metrics"][k] = -99
            if "params" in r and isinstance(r["params"], str):
                try: r["params"] = json.loads(r["params"])
                except Exception: r["params"] = {}
        normalized.append(r)
    return normalized


def run_single_experiment(strategy_name, params, data, df, periods, cost_bps=1.0):
    try:
        strategy = get_strategy(strategy_name)
        positions = strategy.run(data, **params)
    except Exception as e:
        return {"strategy": strategy_name, "params": params, "error": str(e),
                "train_metrics": {}, "val_metrics": {}, "test_metrics": {}}
    prices = df["close"]
    results = {"strategy": strategy_name, "params": params,
               "train_metrics": {}, "val_metrics": {}, "test_metrics": {}}
    for pname, (start, end) in periods.items():
        try:
            pp = positions.loc[start:end]; pr = prices.loc[start:end]
            if len(pp) == 0 or pp.abs().sum() == 0:
                results[f"{pname}_metrics"] = {"sharpe": -99}; continue
            bt = backtest(pp, pr, cost_bps=cost_bps)
            results[f"{pname}_metrics"] = evaluate(bt["returns"])
        except Exception:
            results[f"{pname}_metrics"] = {"sharpe": -99}
    return results


def run_search(ticker="GLD", strategies=None, cost_bps=1.0, workers=1,
               quick=False, top_n=50, start="2004-01-01", end=None):
    """Run exhaustive strategy search for any ticker.

    Args:
        ticker: Yahoo Finance ticker (e.g. "GLD", "SPY", "BTC-USD").
        strategies: List of strategy names. None = all.
        cost_bps: Transaction cost in basis points.
        workers: Number of parallel workers (1 = sequential).
        quick: If True, only test 5 params per strategy.
        top_n: Number of top results to keep.
        start: Data start date.
        end: Data end date. None = today.

    Returns:
        dict with 'all_results', 'top_results', 'best'.
    """
    RESULT_DIR.mkdir(exist_ok=True)
    print(f"momentum-lab: Searching optimal strategies for {ticker}")

    data, df = prepare_data(ticker, start=start, end=end)
    prices = df["close"]
    n = len(df)
    split1 = int(n * 0.6); split2 = int(n * 0.8)
    periods = {
        "train": (df.index[0], df.index[split1]),
        "val": (df.index[split1], df.index[split2]),
        "test": (df.index[split2], df.index[-1]),
    }
    print(f"  Data: {df.index[0].date()} ~ {df.index[-1].date()}, {n} bars")
    print(f"  Train: {periods['train'][0].date()} ~ {periods['train'][1].date()}")
    print(f"  Val:   {periods['val'][0].date()} ~ {periods['val'][1].date()}")
    print(f"  Test:  {periods['test'][0].date()} ~ {periods['test'][1].date()}")

    if strategies is None:
        strategies = list(STRATEGY_REGISTRY.keys())

    total = sum(len(get_strategy(s).get_param_combinations()[:5] if quick else get_strategy(s).get_param_combinations()) for s in strategies if s in STRATEGY_REGISTRY)
    print(f"  Strategies: {len(strategies)}, Total experiments: {total}")

    all_results = []
    t0 = time.time()

    for sname in strategies:
        if sname not in STRATEGY_REGISTRY: continue
        s = get_strategy(sname)
        combos = s.get_param_combinations()
        if quick: combos = combos[:5]
        cat = "ML" if sname.startswith("ml_") else ("combo" if sname in ["ensemble","stacked","regime_aware"] else "classic")
        print(f"\n  [{cat}] {sname} ({len(combos)} params) ...")

        for i, params in enumerate(tqdm(combos, desc=f"  {sname}", leave=True)):
            result = run_single_experiment(sname, params, data, df, periods, cost_bps)
            compact = {k: v for k, v in result.items() if k != "positions"}
            all_results.append(compact)

        _save_results_csv(all_results, RESULT_DIR / "all_results.csv")
        print(f"  [checkpoint] {sname} done, {len(all_results)} total")

    elapsed = time.time() - t0
    print(f"\n  Search complete! {len(all_results)} results in {elapsed/60:.1f} min")

    # Phase 2: Rank by val Sharpe
    all_results = _normalize_results(all_results)
    valid = [r for r in all_results if "error" not in r and r.get("val_metrics", {}).get("sharpe", -99) > -99]
    valid.sort(key=lambda r: -r.get("val_metrics", {}).get("sharpe", -99))
    top = valid[:top_n]

    # Phase 3: Test set evaluation
    if top:
        best = top[0]
        sname = best["strategy"]; params = best.get("params", {})
        strategy = get_strategy(sname)
        positions = strategy.run(data, **params)
        test_m = evaluate(backtest(positions.loc[periods["test"][0]:periods["test"][1]],
                                   prices.loc[periods["test"][0]:periods["test"][1]],
                                   cost_bps=cost_bps)["returns"])
        bh_m = evaluate(backtest(get_buy_and_hold(prices.loc[periods["test"][0]:periods["test"][1]]),
                                 prices.loc[periods["test"][0]:periods["test"][1]], cost_bps=0)["returns"])
        print(f"\n  Best: {sname}")
        print(f"  Params: {_params_to_str(params)}")
        print(f"  Val Sharpe:   {best['val_metrics'].get('sharpe', 0):.4f}")
        print(f"  Test Sharpe:  {test_m['sharpe']:.4f} (B&H: {bh_m['sharpe']:.4f})")
        print(f"  Test CAGR:    {test_m['cagr']:.2%} (B&H: {bh_m['cagr']:.2%})")
        print(f"  Test MaxDD:   {test_m['max_drawdown']:.2%} (B&H: {bh_m['max_drawdown']:.2%})")
    else:
        best = None

    # Save summary
    _save_results_csv(all_results, RESULT_DIR / "all_results.csv")
    if top:
        rows = []
        for r in top:
            row = {"strategy": r["strategy"], "params": json.dumps(r.get("params", {}), ensure_ascii=False, default=_jsonable)}
            for p in ["train", "val", "test"]:
                m = r.get(f"{p}_metrics", {})
                for k, v in m.items(): row[f"{p}_{k}"] = v
            rows.append(row)
        pd.DataFrame(rows).to_csv(RESULT_DIR / "top_results.csv", index=False, encoding="utf-8-sig")

    return {"all_results": all_results, "top_results": top, "best": best}
