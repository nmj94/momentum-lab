"""robustness.py - Local parameter-sensitivity analysis.

Given the best (strategy, params) found by the search, we nudge every numeric
parameter by +/- some fraction and re-evaluate the Validation Sharpe of each
neighbor. If the optimum sits on a wide "plateau" the score holds up; if it is
an isolated peak, neighbors collapse and the result is fragile.  This is not a
multiple-testing correction and must not be interpreted as proof that a search
result is free from backtest overfitting.
"""

import numpy as np

from .backtest import RISK_FREE_RATE, backtest, evaluate
from .strategies import get_strategy

_NUMERIC = (int, float, np.integer, np.floating)


def perturb_params(params: dict, frac: float = 0.2) -> list:
    """Build neighbor parameter sets by perturbing numeric params by +/-frac.

    Booleans, strings and tuple/array params are kept fixed. Integer params are
    nudged by at least 1 (or by frac when the value is large).
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
                # A non-negative original (window lengths, skip counts,
                # smoothing spans) has no meaningful negative neighbor: the
                # strategy either raises on it or - worse - treats it exactly
                # like zero, silently re-scoring the baseline and flattering
                # the robustness statistics.
                if val >= 0 and nv < 0:
                    continue
            new[key] = nv
            neighbors.append(new)
    return neighbors


def _val_sharpe(
    strategy, data, prices, periods, params, cost_bps, backtest_kwargs=None, risk_free_rate=RISK_FREE_RATE
) -> float:
    try:
        positions = strategy.run(data, **params)
    except Exception:
        return -99.0
    try:
        kwargs = dict(backtest_kwargs or {})
        full = backtest(positions, prices, cost_bps=cost_bps, **kwargs)
        val_returns = full["returns"].loc[periods["val"][0] : periods["val"][1]]
        lag = int(kwargs.get("execution_lag", 0))
        held = positions.reindex(prices.index).ffill().fillna(0.0).shift(lag + 1).fillna(0.0)
        val_held = held.loc[periods["val"][0] : periods["val"][1]]
        if len(val_returns) == 0 or val_held.abs().sum() == 0:
            return -99.0
        return evaluate(
            val_returns,
            risk_free_rate=risk_free_rate,
            annualization=kwargs.get("annualization", 252),
        )["sharpe"]
    except Exception:
        return -99.0


def robustness_check(
    data,
    df,
    periods,
    sname,
    params,
    cost_bps=1.0,
    frac=0.2,
    min_neighbors=4,
    backtest_kwargs=None,
    risk_free_rate=RISK_FREE_RATE,
):
    """Run neighborhood perturbation and summarize stability.

    Returns a dict with:
      baseline, neighbors (list of float sharpe), stats, grade, flags.
    """
    strategy = get_strategy(sname)
    prices = df["close"]

    base = _val_sharpe(strategy, data, prices, periods, params, cost_bps, backtest_kwargs, risk_free_rate)
    # Drop neighbors that violate the strategy's own coherence constraints
    # (e.g. fast >= slow after perturbation).  Scoring degenerate combos
    # biases the grade towards "fragile" for purely mechanical reasons.
    neighbors = [nb for nb in perturb_params(params, frac=frac) if strategy.is_valid_params(nb)]
    if len(neighbors) < min_neighbors:
        return {
            "error": f"Too few numeric params to perturb ({len(neighbors)} neighbors).",
            "baseline": base,
            "neighbors": neighbors,
            "n_neighbors": len(neighbors),
            "grade": "N/A",
            "verdict": "Skipped",
        }

    vals = [
        _val_sharpe(strategy, data, prices, periods, nb, cost_bps, backtest_kwargs, risk_free_rate) for nb in neighbors
    ]
    vals = np.array([v for v in vals if v > -99], dtype=float)
    n_valid = len(vals)

    if n_valid == 0:
        return {
            "error": "All neighbors failed to evaluate.",
            "baseline": base,
            "neighbors": [],
            "n_neighbors": 0,
            "grade": "N/A",
            "verdict": "Skipped",
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

    # Grade local smoothness only: A = plateau, B = moderate, C/D = unstable.
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
            else "Locally unstable"
        ),
    }
