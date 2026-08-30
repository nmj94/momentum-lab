"""Installed-wheel offline CSV import, CLI search, reveal and portability smoke."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import momentum_lab
from momentum_lab import DatasetError, StudyRegistry, load_dataset


def main():
    assert "site-packages" in Path(momentum_lab.__file__).resolve().parts, "smoke must use the installed wheel"
    assert importlib.util.find_spec("sklearn") is None, "offline core must not require optional ML"
    rng = np.random.Generator(np.random.PCG64(771))
    index = pd.date_range("2021-01-04", periods=500, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.012, len(index))))
    frame = pd.DataFrame(
        {"open": close * 0.999, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 100000.0},
        index=index,
    )
    with tempfile.TemporaryDirectory(prefix="momentum-dataset-smoke-") as folder:
        root = Path(folder)
        csv_path = root / "synthetic.csv"
        frame.to_csv(csv_path, index_label="date")
        environment = {**os.environ, "MOMENTUM_LAB_REGISTRY_PATH": str(root / "registry.sqlite3")}
        # A local dead proxy also makes accidental network fallback observable.
        environment.update(HTTPS_PROXY="http://127.0.0.1:9", HTTP_PROXY="http://127.0.0.1:9")

        def invoke(*args):
            result = subprocess.run(
                [sys.executable, "-m", "momentum_lab", *map(str, args)],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            return result.stdout

        invoke(
            "data",
            "import",
            csv_path,
            "--output",
            root / "snapshot",
            "--ticker",
            "SYNTHETIC",
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
        manifest = root / "snapshot" / "manifest.json"
        assert json.loads(invoke("data", "inspect", manifest))["rows"] == 500
        config = {
            "ticker": "SYNTHETIC",
            "dataset": "snapshot/manifest.json",
            "start": "2021-01-04",
            "strategies": ["tsmom"],
            "robust": False,
            "bootstrap_resamples": 200,
            "workers": 2,
            "study_id": "offline-smoke",
            "run_id": "offline-smoke",
            "result_dir": str(root / "runs"),
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        invoke("--config", config_path)
        summary_path = root / "runs" / "offline-smoke" / "summary.json"
        sealed = json.loads(summary_path.read_text())
        assert sealed["best"] and "test_metrics" not in sealed["best"]
        assert sealed["test_access"]["status"] == "sealed"
        assert sealed["data_provenance"]["provider"] == "local_csv"
        assert set(sealed["bootstrap_diagnostics"]["periods"]) == {"validation"}
        shutil.copytree(manifest.parent, root / "relocated")
        config["dataset"] = "relocated/manifest.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        invoke("--config", config_path, "--resume", "--reveal-test")
        revealed = json.loads(summary_path.read_text())
        assert revealed["test_access"]["status"] == "first_recorded_reveal"
        invoke("--config", config_path, "--resume", "--reveal-test")
        replay = json.loads(summary_path.read_text())
        assert replay["test_access"]["status"] == "previously_revealed"
        assert replay["best"]["test_evaluated_at"] == revealed["best"]["test_evaluated_at"]
        assert sealed["data_provenance"] == replay["data_provenance"]
        assert "Data provenance" in (summary_path.parent / "report.html").read_text()
        assert "test_metrics" not in json.dumps(
            StudyRegistry(environment["MOMENTUM_LAB_REGISTRY_PATH"], create=False).history()
        )
        prices = root / "relocated" / "prices.csv"
        prices.write_bytes(prices.read_bytes() + b"\n")
        try:
            load_dataset(root / "relocated" / "manifest.json")
        except DatasetError as exc:
            assert "SHA-256 mismatch" in str(exc)
        else:
            raise AssertionError("Changed CSV must fail checksum validation")
    print("Offline dataset smoke: passed (import, inspect, parallel search, seal, relocate, reveal, replay, tamper)")


if __name__ == "__main__":
    main()
