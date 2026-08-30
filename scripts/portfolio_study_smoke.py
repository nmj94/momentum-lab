"""Installed core-wheel membership/study lifecycle; no live market data or ML."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import momentum_lab
from momentum_lab import PortfolioStudyRegistry, StudyRegistry


def main():
    assert "site-packages" in Path(momentum_lab.__file__).resolve().parts, "smoke must use the installed wheel"
    assert momentum_lab.__version__ == "0.14.0"
    assert importlib.util.find_spec("sklearn") is None, "portfolio study core must not require optional ML"
    with tempfile.TemporaryDirectory(prefix="momentum-portfolio-study-smoke-") as folder:
        root = Path(folder)
        registry_path = root / "registry.sqlite3"
        environment = {
            **os.environ,
            "MOMENTUM_LAB_REGISTRY_PATH": str(registry_path),
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "HTTP_PROXY": "http://127.0.0.1:9",
        }

        def invoke(*args, success=True):
            completed = subprocess.run(
                [sys.executable, "-m", "momentum_lab", *map(str, args)],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            assert completed.returncode == (0 if success else 2), completed.stdout + completed.stderr
            return completed.stdout + completed.stderr

        assert "passed (2 cases)" in invoke("portfolio", "study", "benchmark")
        assert not registry_path.exists()
        index = pd.date_range("2024-01-02", periods=90, freq="B", name="date")
        for number, ticker in enumerate(("AAA", "BBB")):
            close = 100 * np.exp(0.001 * np.arange(len(index)) + 0.1 * np.sin(np.arange(len(index)) / 7 + number))
            frame = pd.DataFrame(
                {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close}, index=index
            )
            csv = root / f"{ticker}.csv"
            frame.to_csv(csv)
            invoke(
                "data",
                "import",
                csv,
                "--output",
                root / ticker,
                "--ticker",
                ticker,
                "--source",
                "Project-generated synthetic study smoke",
                "--license",
                "MIT; software tests only",
                "--currency",
                "USD",
                "--calendar",
                "exchange",
                "--price-adjustment",
                "split_and_dividend_adjusted",
            )
        membership = {
            "schema_version": 1,
            "universe_id": "synthetic-smoke",
            "source": "Synthetic membership events",
            "license": "MIT",
            "coverage_start": "2024-01-01",
            "coverage_end": "2024-12-31",
            "initial_known_on": "2023-12-29",
            "initial_members": ["AAA"],
            "events": [
                {
                    "ticker": "BBB",
                    "known_on": str(index[18].date()),
                    "effective_on": str(index[20].date()),
                    "action": "add",
                },
                {
                    "ticker": "AAA",
                    "known_on": str(index[73].date()),
                    "effective_on": str(index[75].date()),
                    "action": "remove",
                },
            ],
        }
        (root / "membership.json").write_text(json.dumps(membership), encoding="utf-8")
        config = {
            "datasets": {"AAA": "AAA/manifest.json", "BBB": "BBB/manifest.json"},
            "universe": "membership.json",
            "study_id": "installed-study",
            "test_start": str(index[60].date()),
            "lookback": 10,
            "top_k": 1,
            "rebalance": "weekly",
            "max_weight": 0.8,
            "cost_bps": 5,
            "slippage_bps": 2,
            "spread_bps": 4,
            "cash_rate": 0.02,
            "initial_capital": 10000,
            "result_dir": str(root / "runs"),
        }
        path = root / "study.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        prefix = ("portfolio", "study", "--config", path)
        assert "registry not found" in invoke(*prefix, "--run-id", "too-early", "--reveal-test", success=False)
        assert not registry_path.exists() and not (root / "runs").exists()
        invoke(*prefix, "--run-id", "sealed")

        def summary(name):
            return json.loads((root / "runs" / name / "summary.json").read_text())

        sealed = summary("sealed")
        assert sealed["test"] is None and not sealed["test_access"]["test_results_visible"]
        assert not list((root / "runs" / "sealed").glob("test_*.csv"))
        assert sealed["development"]["n_bars"] == 60
        status = json.loads(invoke("portfolio", "study", "status", "installed-study"))
        assert status["status"] == "sealed" and "metrics" not in status
        registry = PortfolioStudyRegistry(registry_path, create=False)
        assert registry.registry_id == StudyRegistry(registry_path, create=False).registry_id
        assert len(registry.history()) == 2
        invoke(*prefix, "--run-id", "revealed", "--reveal-test")
        revealed = summary("revealed")
        assert revealed["test_access"]["status"] == "first_recorded_reveal"
        assert revealed["test"]["metrics"]["return_intervals"] == 30
        book = pd.read_csv(root / "runs" / "revealed" / "test_ledger.csv", index_col="date")
        values = pd.read_csv(root / "runs" / "revealed" / "test_asset_values.csv", index_col="date")
        development = pd.read_csv(root / "runs" / "sealed" / "development_ledger.csv", index_col="date")
        assert len(book) == 31 and book.index[0] == development.index[-1]
        np.testing.assert_allclose(book["nav"], book["cash"] + values.sum(axis=1))
        np.testing.assert_allclose(revealed["test"]["metrics"]["starting_nav"], development["nav"].iloc[-1])
        assert book["return"].iloc[0] == 0 and book["transaction_cost"].iloc[0] == 0
        invoke(*prefix, "--run-id", "replayed", "--reveal-test")
        replay = summary("replayed")
        assert replay["test"] == revealed["test"] and replay["test_access"]["cached"]
        assert replay["artifact_scope"] == "cached_summary_only"
        assert not list((root / "runs" / "replayed").glob("*.csv"))
        assert len(registry.history()) == 6
        config["study_id"] = "overlapping-study"
        path.write_text(json.dumps(config), encoding="utf-8")
        invoke(*prefix, "--run-id", "second-development")
        assert "overlap" in invoke(*prefix, "--run-id", "blocked", "--reveal-test", success=False)
        invoke(
            *prefix,
            "--run-id",
            "acknowledged",
            "--reveal-test",
            "--allow-test-reuse",
            "--test-reuse-reason",
            "Acknowledged prior portfolio study access",
        )
        assert summary("acknowledged")["test_access"]["status"] == "repeated_use"
        assert len(registry.history()) == 10
    print(
        "Portfolio study smoke: passed (core wheel, frozen cases, membership, sealed development, reveal, carried NAV, cached replay, reuse audit)"
    )


if __name__ == "__main__":
    main()
