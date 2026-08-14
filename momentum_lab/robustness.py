"""robustness.py - Overfitting detection via parameter neighborhood perturbation.

Given the best (strategy, params) found by the search, we nudge every numeric
parameter by +/- some fraction and re-evaluate the Validation Sharpe of each
neighbor. If the optimum sits on a wide "plateau" the score holds up; if it is
an isolated peak, neighbors collapse and the result is fragile.
"""

import numpy as np

from .backtest import backtest, evaluate
from .strategies import get_strategy

_NUMERIC = (int, float, np.integer, np.floating)


def perturb_params(params: dict, frac: float = 0.2) -> list:
    """Build neighbor parameter sets by perturbing numeric params by +/-frac.

    Booleans, strings and tuple/array params are kept fixed. Integer params are
    nudge by at least 1 (or by frac when the value is large).
    """
    neighbors = []
    for key, val in params.items():
        if isinstance(val, bool) or not isinstance(val, _NUMERIC):
            continue
        is_int = isinstance(val, (int, np.integer))
        val = float(val)
        step = max(abs(val) * frac, 1.0 if is_int else 0.0)
        if step <= 0:
            continue
        for delta in (-step, step):
            new = dict(params)
            nv = val + delta
            if is_int:
                nv = round(nv)
                if nv == int(val):
                    nv = int(val) + (1 if delta > 0 else -1)
            new[key] = nv
            neighbors.append(new)
    return neighbors


def _val_sharpe(strategy, data, prices, periods, params, cost_bps) -> float:
    try:
        positions = strategy.run(data, **params)
    except Exception:
        return -99.0
    pp = positions.loc[periods["val"][0] : periods["val"][1]]
    pr = prices.loc[periods["val"][0] : periods["val"][1]]
    if len(pp) == 0 or pp.abs().sum() == 0:
        return -99.0
    try:
        return evaluate(backtest(pp, pr, cost_bps=cost_bps)["returns"])["sharpe"]
    except Exception:
        return -99.0


def robustness_check(data, df, periods, sname, params, cost_bps=1.0, frac=0.2, min_neighbors=4):
    """Run neighborhood perturbation and summarize stability.

    Returns a dict with:
      baseline, neighbors (list of float sharpe), stats, grade, flags.
    """
    strategy = get_strategy(sname)
    prices = df["close"]

    base = _val_sharpe(strategy, data, prices, periods, params, cost_bps)
    neighbors = perturb_params(params, frac=frac)
    if len(neighbors) < min_neighbors:
        return {
            "error": f"Too few numeric params to perturb ({len(neighbors)} neighbors).",
            "baseline": base,
            "neighbors": neighbors,
            "n_neighbors": len(neighbors),
            "grade": "N/A",
        }

    vals = [_val_sharpe(strategy, data, prices, periods, nb, cost_bps) for nb in neighbors]
    vals = np.array([v for v in vals if v > -99], dtype=float)
    n_valid = len(vals)

    if n_valid == 0:
        return {
            "error": "All neighbors failed to evaluate.",
            "baseline": base,
            "neighbors": [],
            "n_neighbors": 0,
            "grade": "N/A",
        }

    stats = {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }
    pct_degrade = float((vals < base * 0.5).mean())
    pct_positive = float((vals > 0).mean())

    # Grade: A = robust plateau, B = moderate, C = fragile, D = overfit peak.
    if base > 0 and stats["median"] >= 0.8 * base and pct_degrade < 0.15:
        grade = "A"
    elif base > 0 and stats["median"] >= 0.6 * base and pct_degrade < 0.35:
        grade = "B"
    elif stats["median"] >= 0.4 * max(base, 0):
        grade = "C"
    else:
        grade = "D"

    peak_flag = base > 0 and (stats["median"] < 0.5 * base or pct_degrade > 0.4)
    return {
        "error": None,
        "baseline": base,
        "n_neighbors": n_valid,
        "stats": stats,
        "pct_degrade": pct_degrade,
        "pct_positive": pct_positive,
        "grade": grade,
        "isolated_peak": bool(peak_flag),
        "verdict": (
            "Robust"
            if grade == "A"
            else "Relatively stable"
            if grade == "B"
            else "Fragile"
            if grade == "C"
            else "Overfit / fragile"
        ),
    }
