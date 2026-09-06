"""Installed-core preflight, exit codes, offline isolation and report safety."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import momentum_lab
from momentum_lab import import_dataset, preflight_dataset


def main():
    assert "site-packages" in Path(momentum_lab.__file__).resolve().parts
    assert importlib.util.find_spec("sklearn") is None
    with tempfile.TemporaryDirectory(prefix="momentum-preflight-smoke-") as folder:
        root = Path(folder)
        env = {
            **os.environ,
            "MOMENTUM_LAB_REGISTRY_PATH": str(root / "registry.sqlite3"),
            "MOMENTUM_LAB_DATA_DIR": str(root / "cache"),
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "HTTP_PROXY": "http://127.0.0.1:9",
        }
        dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
        for ticker, price in (("AAA", "987654.321"), ("BBB", "123456.789")):
            csv = root / f"{ticker}.csv"
            csv.write_text(
                "date,open,high,low,close\n" + "".join(f"{day},{price},{price},{price},{price}\n" for day in dates),
                encoding="utf-8",
            )
            import_dataset(
                csv,
                root / ticker,
                ticker=ticker,
                source="Private synthetic test",
                license="MIT synthetic fixture",
                currency="USD",
                calendar="exchange",
                price_adjustment="split_and_dividend_adjusted",
            )
        sessions = root / "sessions.json"
        sessions.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "calendar_id": "synthetic-v1",
                    "source": "synthetic test",
                    "license": "MIT",
                    "coverage_start": "2024-01-01",
                    "coverage_end": "2024-01-10",
                    "sessions": dates,
                }
            ),
            encoding="utf-8",
        )
        recipe = root / "portfolio.json"
        recipe.write_text(
            json.dumps(
                {
                    "datasets": {"AAA": "AAA/manifest.json", "BBB": "BBB/manifest.json"},
                    "lookback": 2,
                    "result_dir": str(root / "unused-runs"),
                }
            ),
            encoding="utf-8",
        )

        def invoke(*args, code=0):
            result = subprocess.run(
                [sys.executable, "-m", "momentum_lab", *map(str, args)],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=45,
                check=False,
            )
            assert result.returncode == code, result.stdout + result.stderr
            assert "987654" not in result.stdout and "123456" not in result.stdout
            return result

        manifest = root / "AAA" / "manifest.json"
        assert json.loads(invoke("data", "check", manifest, code=1).stdout)["status"] == "warning"
        checked = json.loads(
            invoke("data", "check", manifest, "--sessions", sessions, "--output", root / "report").stdout
        )
        assert checked == preflight_dataset(manifest, sessions=sessions)
        assert checked == json.loads((root / "report" / "report.json").read_text())
        invoke("data", "check", manifest, "--output", root / "report", code=2)
        portfolio = json.loads(invoke("portfolio", "preflight", "--config", recipe, "--sessions", sessions).stdout)
        assert portfolio["status"] == "passed" and len(portfolio["assets"]) == 2
        bad = json.loads(invoke("data", "check", root / "missing.json", code=2).stdout)
        assert bad["status"] == "error" and bad["issues"][0]["code"] == "invalid_dataset"
        assert all(not (root / name).exists() for name in ("registry.sqlite3", "unused-runs", "cache"))
    print("Preflight smoke: passed (installed core, read-only, calendar, portfolio, exits, privacy, exclusive reports)")


if __name__ == "__main__":
    main()
