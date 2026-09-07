"""Hash-locked independent turnover-budget software ledgers."""

import hashlib
import json
from importlib import resources

import numpy as np
import pandas as pd

from .portfolio import PortfolioError, backtest_portfolio

PORTFOLIO_TURNOVER_REFERENCE_SHA256 = "1ab11e798b4b3bde64c0b7772a3fa10ac28e29484a03c3b68d9cbb277a52f946"


def check_portfolio_turnover_reference():
    raw = resources.files("momentum_lab").joinpath("benchmark_data/portfolio_turnover_reference_v1.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != PORTFOLIO_TURNOVER_REFERENCE_SHA256:
        raise PortfolioError("Frozen portfolio turnover reference SHA-256 mismatch")
    suite = json.loads(raw)
    for case in suite["cases"]:
        index = pd.DatetimeIndex(case["dates"], name="date")
        prices = pd.DataFrame(case["prices"], index=index)
        targets = pd.DataFrame(case["targets"], index=index, columns=sorted(prices.columns), dtype=float)
        book = backtest_portfolio(targets, prices, **case["config"])
        for frame in ("ledger", "execution"):
            if list(book[frame].columns) != suite[f"{frame}_columns"]:
                raise PortfolioError(f"Frozen portfolio turnover columns changed: {frame}")
        for name, expected in case["expected"].items():
            actual, reference = book[name].to_numpy(dtype=float), np.asarray(expected, dtype=float)
            missing = name == "executed_targets"
            if (
                actual.shape != reference.shape
                or np.isinf(actual).any()
                or (not missing and np.isnan(actual).any())
                or not np.allclose(actual, reference, rtol=1e-11, atol=1e-11, equal_nan=missing)
            ):
                raise PortfolioError(f"Frozen portfolio turnover changed: {case['id']} / {name}")
    return {"status": "passed", "cases": len(suite["cases"]), "schema_version": suite["schema_version"]}
