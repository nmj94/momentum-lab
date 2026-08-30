"""Installed-wheel multi-asset CLI, cash book, audit and frozen-ledger smoke."""

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
from momentum_lab import StudyRegistry


def main():
    assert "site-packages" in Path(momentum_lab.__file__).resolve().parts, "smoke must use the installed wheel"
    assert importlib.util.find_spec("sklearn") is None, "portfolio core must not require optional ML"
    with tempfile.TemporaryDirectory(prefix="momentum-portfolio-smoke-") as folder:
        root = Path(folder)
        environment = {**os.environ, "MOMENTUM_LAB_REGISTRY_PATH": str(root / "registry.sqlite3")}
        environment.update(HTTPS_PROXY="http://127.0.0.1:9", HTTP_PROXY="http://127.0.0.1:9")

        def invoke(*args, success=True):
            completed = subprocess.run(
                [sys.executable, "-m", "momentum_lab", *map(str, args)],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
            assert completed.returncode == (0 if success else 2), completed.stdout + completed.stderr
            return completed.stdout + completed.stderr

        assert "passed (6 cases)" in invoke("portfolio", "benchmark")
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
                "Project-generated synthetic smoke",
                "--license",
                "MIT; synthetic software tests",
                "--currency",
                "USD",
                "--calendar",
                "exchange",
                "--price-adjustment",
                "split_and_dividend_adjusted",
            )
        config = {
            "datasets": {"AAA": "AAA/manifest.json", "BBB": "BBB/manifest.json"},
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
            "run_id": "portfolio-smoke",
        }
        path = root / "portfolio.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        assert "acknowledge_history" in invoke("portfolio", "--config", path, success=False)
        assert not (root / "runs").exists() and not (root / "registry.sqlite3").exists()
        invoke("portfolio", "--config", path, "--acknowledge-history")
        output = root / "runs" / "portfolio-smoke"
        summary = json.loads((output / "summary.json").read_text())
        assert summary["research_status"] == "exploratory_full_history"
        assert summary["metrics"]["return_intervals"] == 89
        book = pd.read_csv(output / "ledger.csv", index_col=0)
        assets = pd.read_csv(output / "asset_values.csv", index_col=0)
        weights = pd.read_csv(output / "weights.csv", index_col=0)
        trades = pd.read_csv(output / "trades.csv", index_col=0)
        np.testing.assert_allclose(book["nav"], assets.sum(axis=1) + book["cash"])
        np.testing.assert_allclose(weights.sum(axis=1) + book["cash_weight"], 1)
        np.testing.assert_allclose(book["transaction_cost"], trades.abs().sum(axis=1) * 0.0009, atol=1e-12)
        assert book["cash"].min() >= 0
        assert np.flatnonzero(book["rebalance_executed"])[0] == 11
        registry = StudyRegistry(environment["MOMENTUM_LAB_REGISTRY_PATH"], create=False)
        history = registry.history()
        assert len(history) == 2 and {event["ticker"] for event in history} == {"AAA", "BBB"}
        assert all(event["kind"] == "development" and event["start_date"] == "2024-01-02" for event in history)
        assert all(event["end_date"] == str(index[-1].date()) for event in history)
        assert "not a sealed test" in (output / "report.html").read_text()
        assert "Final holdings and last signal" in (output / "report.md").read_text()
        before = (output / "summary.json").read_bytes()
        assert "already exists" in invoke("portfolio", "--config", path, "--acknowledge-history", success=False)
        assert before == (output / "summary.json").read_bytes()
        assert len(registry.history()) == 2
    print(
        "Portfolio smoke: passed (core wheel, six frozen ledgers, offline CLI, delayed fills, cash/fees, audit, no overwrite)"
    )


if __name__ == "__main__":
    main()
