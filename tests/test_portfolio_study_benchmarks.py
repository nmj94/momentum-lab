"""Packaged independent oracle, frozen hash and public CLI regression gate."""

import copy
import hashlib
import json
import math
import subprocess
import sys
from importlib import resources
from pathlib import Path

import pytest

from momentum_lab import PortfolioError, cli
from momentum_lab import portfolio_benchmarks as pb


def assert_oracle_reference(actual, expected):
    """Keep inputs/books exact; allow only a few ULPs for stdlib statistics.

    Python 3.10 statistics.stdev rounds differently from 3.11+ by 1-2 ULPs.
    Do not rebaseline the fixture or relax the production numeric-change gate.
    """
    actual, expected = copy.deepcopy(actual), copy.deepcopy(expected)
    assert len(actual["cases"]) == len(expected["cases"])
    for observed, frozen in zip(actual["cases"], expected["cases"]):
        for account in ("result", "benchmark"):
            for statistic in ("sharpe", "volatility"):
                value = observed["test_metrics"][account].pop(statistic)
                reference = frozen["test_metrics"][account].pop(statistic)
                if reference is None:
                    assert value is None
                else:
                    assert type(value) is float and math.isfinite(value)
                    assert abs(value - reference) <= 8 * math.ulp(reference), (frozen["id"], account, statistic)
    assert actual == expected


def test_independent_oracle_matches_reviewed_reference():
    path = Path(__file__).resolve().parents[1] / "scripts" / "portfolio_study_reference_oracle.py"
    output = subprocess.run([sys.executable, str(path)], check=True, capture_output=True, text=True, timeout=30)
    raw = resources.files("momentum_lab").joinpath("benchmark_data/portfolio_study_reference_v1.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == pb.PORTFOLIO_STUDY_REFERENCE_SHA256
    assert_oracle_reference(json.loads(output.stdout), json.loads(raw))
    assert pb.check_portfolio_study_reference() == {"status": "passed", "cases": 2, "schema_version": 1}


def fixture_copy():
    raw = resources.files("momentum_lab").joinpath("benchmark_data/portfolio_study_reference_v1.json").read_bytes()
    return json.loads(raw)


def test_oracle_comparison_accepts_machine_rounding_only_in_derived_statistics():
    expected = fixture_copy()
    actual = copy.deepcopy(expected)
    metrics = actual["cases"][0]["test_metrics"]["result"]
    metrics["sharpe"] = math.nextafter(metrics["sharpe"], math.inf)
    metrics["volatility"] = math.nextafter(metrics["volatility"], math.inf)
    assert_oracle_reference(actual, expected)


@pytest.mark.parametrize("mutation", ["ledger", "input", "statistic", "nonfinite", "metadata"])
def test_oracle_comparison_still_rejects_real_drift(mutation):
    expected = fixture_copy()
    actual = copy.deepcopy(expected)
    case = actual["cases"][0]
    if mutation == "ledger":
        values = case["expected"]["result"]["ledger"][0]
        values[0] = math.nextafter(values[0], math.inf)
    elif mutation == "input":
        case["prices"]["AAA"][0] += 0.001
    elif mutation == "statistic":
        case["test_metrics"]["result"]["volatility"] *= 1 + 1e-8
    elif mutation == "nonfinite":
        case["test_metrics"]["result"]["volatility"] = float("nan")
    else:
        case["id"] = "changed-case"
    with pytest.raises(AssertionError):
        assert_oracle_reference(actual, expected)


def test_undefined_oracle_statistics_are_not_coerced_to_zero():
    expected = fixture_copy()
    expected["cases"][0]["test_metrics"]["result"]["sharpe"] = None
    actual = copy.deepcopy(expected)
    assert_oracle_reference(actual, expected)
    actual["cases"][0]["test_metrics"]["result"]["sharpe"] = 0.0
    with pytest.raises(AssertionError):
        assert_oracle_reference(actual, expected)


def test_hash_and_causal_membership_regressions_fail_closed(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(pb, "PORTFOLIO_STUDY_REFERENCE_SHA256", "0" * 64)
        with pytest.raises(PortfolioError, match="SHA-256"):
            pb.check_portfolio_study_reference()
    load = pb.load_membership

    def changed(*args):
        frame, source = load(*args)
        frame.iloc[0, 0] = not frame.iloc[0, 0]
        return frame, source

    monkeypatch.setattr(pb, "load_membership", changed)
    with pytest.raises(PortfolioError, match="membership"):
        pb.check_portfolio_study_reference()


def test_benchmark_cli_and_failure_exit(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["momentum-lab", "portfolio", "study", "benchmark"])
    assert cli.main() == 0
    assert "passed (2 cases)" in capsys.readouterr().out
    monkeypatch.setattr(pb, "PORTFOLIO_STUDY_REFERENCE_SHA256", "0" * 64)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
