"""Packaged independent membership and portfolio-study boundary regressions."""

import hashlib
import json
import tempfile
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from .portfolio import PortfolioError, portfolio_metrics
from .portfolio_research import PortfolioConfig, _compute_books
from .portfolio_study import _test_books
from .universe import load_membership

PORTFOLIO_STUDY_REFERENCE_SHA256 = "f08d381e294e83af4584e72df1991609c84fff4892ae43867a1c7ae2576cb093"


def _compare(actual, expected, label, *, missing=False):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if (
        actual.shape != expected.shape
        or np.isinf(actual).any()
        or np.isinf(expected).any()
        or (not missing and (np.isnan(actual).any() or np.isnan(expected).any()))
        or not np.allclose(actual, expected, rtol=1e-11, atol=1e-11, equal_nan=missing)
    ):
        raise PortfolioError(f"Frozen portfolio study changed: {label}")


def check_portfolio_study_reference():
    raw = resources.files("momentum_lab").joinpath("benchmark_data/portfolio_study_reference_v1.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != PORTFOLIO_STUDY_REFERENCE_SHA256:
        raise PortfolioError("Frozen portfolio study reference SHA-256 mismatch")
    suite = json.loads(raw)
    with tempfile.TemporaryDirectory(prefix="momentum-study-benchmark-") as folder:
        manifest = Path(folder) / "membership.json"
        for case in suite["cases"]:
            index = pd.DatetimeIndex(case["dates"], name="date")
            prices = pd.DataFrame(case["prices"], index=index)
            manifest.write_text(json.dumps(case["membership"]), encoding="utf-8")
            eligibility, _ = load_membership(manifest, index, prices.columns)
            _compare(eligibility, case["eligibility"], f"{case['id']} / membership")
            books = _compute_books(PortfolioConfig(datasets={}, **case["config"]), prices, eligibility)
            for name, expected in (
                ("scores", case["scores"]),
                ("targets", case["targets"]),
                ("rebalance", case["flags"]),
            ):
                _compare(books["plan"][name], expected, f"{case['id']} / {name}", missing=name != "rebalance")
            tests = _test_books(books, index[case["test_split"] - 1])
            for phase, expected in ((books, case["expected"]), (tests, case["test_expected"])):
                for account in ("result", "benchmark"):
                    if list(phase[account]["ledger"].columns) != suite["ledger_columns"]:
                        raise PortfolioError("Frozen portfolio study ledger columns changed")
                    for name, values in expected[account].items():
                        _compare(
                            phase[account][name],
                            values,
                            f"{case['id']} / {account} / {name}",
                            missing=name == "executed_targets",
                        )
            for account in ("result", "benchmark"):
                actual = portfolio_metrics(tests[account])
                actual["starting_nav"] = float(tests[account]["ledger"]["nav"].iloc[0])
                for name, expected in case["test_metrics"][account].items():
                    if expected is None:
                        if actual[name] is not None:
                            raise PortfolioError(f"Frozen portfolio study metric changed: {name}")
                    else:
                        _compare(actual[name], expected, f"{case['id']} / {account} / {name}")
    return {"status": "passed", "cases": len(suite["cases"]), "schema_version": suite["schema_version"]}
