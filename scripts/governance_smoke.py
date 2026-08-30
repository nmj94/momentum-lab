"""Deterministic offline, installed-wheel verification of the research lifecycle."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import momentum_lab
from momentum_lab import StudyRegistry, TestReuseError, run_search


def main():
    assert "site-packages" in Path(momentum_lab.__file__).resolve().parts, "smoke must use the installed wheel"
    rng = np.random.Generator(np.random.PCG64(771))
    index = pd.date_range("2021-01-04", periods=500, freq="B")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0008, 0.012, len(index)))), index=index)
    frame = pd.DataFrame({"close": close, "volume": 100000.0})
    data = {"close": close, "volume": frame["volume"], "annualization": 252}
    with tempfile.TemporaryDirectory(prefix="momentum-governance-smoke-") as folder:
        root = Path(folder)
        settings = {
            "ticker": "SYNTHETIC",
            "strategies": ["tsmom"],
            "quick": True,
            "robust": False,
            "bootstrap_resamples": 200,
            "result_dir": root / "runs",
            "run_id": "first",
            "study_id": "first",
            "registry_path": root / "registry.sqlite3",
        }
        with patch("momentum_lab.search.prepare_data", return_value=(data, frame)):
            sealed = run_search(**settings)
            assert "test_metrics" not in sealed["best"]
            assert sealed["test_access"]["status"] == "sealed"
            assert set(sealed["bootstrap_diagnostics"]["periods"]) == {"validation"}
            revealed = run_search(**settings, resume=True, reveal_test=True)
            assert revealed["test_access"]["status"] == "first_recorded_reveal"
            assert revealed["best"]["test_metrics"]
            with patch("momentum_lab.search._test_payload", side_effect=AssertionError("must reuse cache")):
                replay = run_search(**settings, resume=True, reveal_test=True)
            assert replay["test_access"]["status"] == "previously_revealed"
            assert replay["best"]["test_evaluated_at"] == revealed["best"]["test_evaluated_at"]
            other = {**settings, "run_id": "second", "study_id": "second"}
            run_search(**other)
            try:
                run_search(**other, resume=True, reveal_test=True)
            except TestReuseError:
                pass
            else:
                raise AssertionError("cross-study overlap must block an unacknowledged reveal")
            reused = run_search(
                **other,
                resume=True,
                reveal_test=True,
                allow_test_reuse=True,
                test_reuse_reason="Synthetic governance smoke",
            )
            assert reused["test_access"]["status"] == "repeated_use"
        registry = StudyRegistry(settings["registry_path"], create=False)
        assert "test_metrics" not in json.dumps(registry.history())
        assert registry.status("first")["test_results_visible"] is False
        assert len(registry.list_studies()) == 2
        json.dumps(reused, allow_nan=False)
    print("Governance smoke: passed (sealed, reveal, cached replay, cross-study overlap, acknowledged reuse)")


if __name__ == "__main__":
    main()
