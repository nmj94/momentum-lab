"""Independent reproductions for the v0.14.1 integrity audit."""

import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numpy as np
import pandas as pd
import pytest

from momentum_lab import cli, data, governance, search
from momentum_lab import config as config_module
from momentum_lab.artifacts import atomic_text_output
from momentum_lab.backtest import backtest, evaluate
from momentum_lab.config import SearchConfig
from momentum_lab.governance import StudyRegistry
from momentum_lab.robustness import robustness_check


def _provider_frame(value=1.0):
    index = pd.date_range("2024-01-02", periods=3, freq="B")
    return pd.DataFrame({name: value for name in ("Open", "High", "Low", "Close")}, index=index)


def _market(monkeypatch):
    index = pd.date_range("2020-01-01", periods=120, freq="B")
    time = np.arange(len(index))
    close = 100 * np.exp(0.001 * time + 0.06 * np.sin(time / 5))
    opening = close * (1 + 0.03 * np.cos(time / 3))
    frame = pd.DataFrame({"close": close, "open": opening, "volume": 1000.0}, index=index)
    market = {name: frame[name] for name in frame}
    monkeypatch.setattr(search, "prepare_data", lambda *a, **k: (market, frame))
    monkeypatch.setattr(search, "_quick_sample", lambda *a: [{"lookback": 5, "threshold": 0.001, "long_short": False}])
    return market, frame


def _search_config(tmp_path, **changes):
    return {
        "ticker": "FAKE",
        "strategies": ["tsmom"],
        "result_dir": str(tmp_path),
        "run_id": "audit",
        "robust": False,
        "bootstrap": False,
        "min_validation_bars": 10,
        **changes,
    }


@pytest.mark.parametrize("rate", [0.01, -0.01, 0.0])
def test_constant_returns_retain_realized_performance(rate):
    result = evaluate(pd.Series([rate] * 100), annualization=252)
    assert result["total_return"] == pytest.approx(round((1 + rate) ** 100 - 1, 4))
    assert result["cagr"] == pytest.approx(round((1 + rate) ** 252 - 1, 4))
    assert result["max_drawdown"] == pytest.approx(round(min(0.0, (1 + rate) ** 100 - 1), 4))
    assert result["win_rate"] == float(rate > 0)
    assert result["sharpe"] == 0.0  # Legacy undefined-ratio sentinel, not evidence of no gain/loss.


@pytest.mark.parametrize("value", [-1.0, -0.1, 0.1])
def test_single_return_is_not_reported_as_no_change(value):
    result = evaluate(pd.Series([value]), annualization=1)
    assert result["total_return"] == value
    assert result["cagr"] == value
    assert result["max_drawdown"] == min(0, value)
    assert all(np.isfinite(metric) for metric in result.values())


def test_evaluate_insolvency_is_absorbing_and_does_not_mutate_input():
    values = pd.Series([-2.0, -2.0, 0.5])
    original = values.copy()
    result = evaluate(values)
    assert result["total_return"] == result["max_drawdown"] == result["cagr"] == -1.0
    pd.testing.assert_series_equal(values, original)


@pytest.mark.parametrize("tickers", [("^GSPC", "_GSPC"), ("EURUSD=X", "EURUSD_X")])
def test_symbolic_tickers_never_share_a_cache(tickers, tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    calls = []

    def download(ticker, **kwargs):
        calls.append(ticker)
        return _provider_frame(float(len(calls)))

    monkeypatch.setattr(data.yf, "download", download)
    first = data.download_data(tickers[0], "2024-01-02", "2024-01-04")
    second = data.download_data(tickers[1], "2024-01-02", "2024-01-04")
    assert calls == list(tickers)
    assert first["close"].iloc[0] == 1.0
    assert second["close"].iloc[0] == 2.0
    assert len(list(tmp_path.glob("*_daily.csv"))) == 2


def test_cache_roundtrip_preserves_exact_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data.yf, "download", lambda *a, **k: _provider_frame(0.12345678901234566))
    fresh = data.download_data("GLD", "2024-01-02", "2024-01-04")
    monkeypatch.setattr(data.yf, "download", lambda *a, **k: pytest.fail("expected a cache hit"))
    cached = data.download_data("GLD", "2024-01-02", "2024-01-04")
    pd.testing.assert_frame_equal(fresh, cached, check_exact=True, check_freq=False)
    assert search._data_snapshot(fresh) == search._data_snapshot(cached)


@pytest.mark.parametrize("change", [{"ticker": "SPY"}, {"earliest_available": "not-a-date"}])
def test_invalid_cache_metadata_is_a_miss_not_wrong_data_or_a_crash(change, tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data.yf, "download", lambda *a, **k: _provider_frame())
    data.download_data("GLD", "2024-01-02", "2024-01-04")
    meta_path = tmp_path / "GLD_daily.meta.json"
    metadata = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps({**metadata, **change}), encoding="utf-8")
    monkeypatch.setattr(data.yf, "download", lambda *a, **k: _provider_frame(2.0))
    with pytest.warns(RuntimeWarning, match="cache metadata"):
        result = data.download_data("GLD", "2024-01-02", "2024-01-04")
    assert result["close"].iloc[0] == 2.0


def test_resume_without_provenance_refuses_unverified_checkpoints(tmp_path):
    (tmp_path / "all_results.csv").write_text('strategy,params\ntsmom,"{}"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="run_config.json"):
        search._check_resume_compatibility(tmp_path, {})


def test_rerun_preserves_config_reports_and_removes_stale_rankings(tmp_path, monkeypatch):
    _market(monkeypatch)
    config = _search_config(tmp_path)
    first = search.run_search(config=config)
    assert first["best"] is not None
    output = tmp_path / "audit"
    before = {
        path.name: path.read_bytes() for path in output.iterdir() if path.suffix in {".json", ".csv", ".md", ".html"}
    }
    with pytest.warns(RuntimeWarning, match="moved to"):
        second = search.run_search(
            config={**config, "cost_bps": 2.0, "min_validation_bars": 1000, "generate_report": False}
        )
    assert second["best"] is None
    for filename, content in before.items():
        name = Path(filename)
        backups = list(output.glob(f"{name.stem}.*.bak{name.suffix}"))
        assert len(backups) == 1, filename
        assert backups[0].read_bytes() == content, filename
    assert not (output / "top_results.csv").exists()
    assert not (output / "report.html").exists()


def test_next_open_sensitivity_uses_the_selected_execution_prices(monkeypatch):
    market, frame = _market(monkeypatch)
    periods = search._split_periods(frame.index)
    params = {"lookback": 5, "threshold": 0.001, "long_short": False}
    options = {"execution_lag": 1}
    expected = search.run_single_experiment(
        "tsmom", params, market, frame, periods, execution_price_column="open", **options
    )["val_metrics"]["sharpe"]
    result = robustness_check(
        market,
        frame,
        periods,
        "tsmom",
        params,
        min_neighbors=0,
        backtest_kwargs=options,
        execution_price_column="open",
    )
    assert result["baseline"] == expected
    close_result = robustness_check(market, frame, periods, "tsmom", params, min_neighbors=0, backtest_kwargs=options)
    assert close_result["baseline"] != expected


def test_search_forwards_next_open_execution_to_sensitivity(tmp_path, monkeypatch):
    _market(monkeypatch)
    seen = {}

    def check(*args, **kwargs):
        seen.update(kwargs)
        return {"error": "audit spy"}

    monkeypatch.setattr(search, "robustness_check", check)
    search.run_search(config=_search_config(tmp_path, robust=True, execution_model="next_open"))
    assert seen["execution_price_column"] == "open"


def test_first_completed_reveal_remains_pinned_when_older_claim_finishes(tmp_path, monkeypatch):
    registry = StudyRegistry(tmp_path / "registry.sqlite3")
    protocol = {
        "ticker": "FAKE",
        "data_snapshot": "a" * 64,
        "periods": {
            "train": ["2020-01-01", "2020-01-31"],
            "val": ["2020-02-01", "2020-02-29"],
            "test": ["2020-03-01", "2020-03-31"],
        },
    }
    registry.register("ordered", protocol)
    registry.bind_selection("ordered", {"strategy": "tsmom"})
    context = {
        "ticker": "FAKE",
        "data_snapshot": "a" * 64,
        "study_id": "ordered",
        "start": "2020-03-01",
        "end": "2020-03-31",
        "run_id": "a",
        "run_path": tmp_path,
    }
    older = registry.claim_test(**context)
    newer = registry.claim_test(**{**context, "run_id": "b"}, allow_reuse=True, reason="retry interrupted work")
    # Deliberately identical clocks: cache identity cannot depend on timestamp ordering.
    monkeypatch.setattr(governance, "_now", lambda: "2020-04-01T00:00:00+00:00")
    registry.complete_test(newer["access"]["event_id"], {"score": 2.0})
    registry.complete_test(older["access"]["event_id"], {"score": 1.0})
    replay = registry.claim_test(**context)
    assert replay["payload"] == {"score": 2.0}
    assert replay["access"]["event_id"] == newer["access"]["event_id"]
    assert registry.status("ordered")["event_id"] == newer["access"]["event_id"]


@pytest.mark.parametrize(
    "payload",
    [
        '{"cost_bps": 10, "cost_bps": 0}',
        '{"cash_rate": NaN}',
        '{"cash_rate": Infinity}',
        '{"cash_rate": 1e999}',
        '{"cash_rate": -1e999}',
    ],
)
def test_ambiguous_or_nonstandard_json_config_is_rejected(payload, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        SearchConfig.from_json(path)


@pytest.mark.parametrize(
    "options",
    [
        {"quick": "false"},
        {"robust": "false"},
        {"resume": "false"},
        {"workers": True},
        {"top_n": 1.5},
        {"min_validation_bars": np.nan},
        {"min_validation_trades": 0.5},
        {"min_validation_exposure": np.nan},
        {"risk_free_rate": True},
        {"cash_rate": True},
    ],
)
def test_invalid_search_controls_fail_before_data_access(options, tmp_path, monkeypatch):
    monkeypatch.setattr(search, "prepare_data", lambda *a, **k: pytest.fail("invalid controls reached data access"))
    with pytest.raises((TypeError, ValueError)):
        search.run_search(result_dir=tmp_path, run_id="invalid", **options)


@pytest.mark.parametrize(
    "options",
    [{"vol_target": -0.1}, {"vol_target": np.nan}, {"vol_target": True}, {"vol_target": 0.1, "vol_lookback": 1}],
)
def test_invalid_volatility_target_does_not_flip_or_erase_positions(options):
    prices = pd.Series([100.0, 102.0, 101.0, 103.0])
    with pytest.raises((TypeError, ValueError)):
        backtest(pd.Series(1.0, index=prices.index), prices, **options)


def test_atomic_writes_do_not_share_temporary_files_in_one_process(tmp_path, monkeypatch):
    original_replace = Path.replace
    barrier = Barrier(2)
    temporary_paths = []

    def replace(path, target):
        temporary_paths.append(path)
        barrier.wait(timeout=5)
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", replace)
    destination = tmp_path / "report.txt"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(search._write_text_atomic, destination, content) for content in ("a" * 100, "b" * 100)]
        for future in futures:
            future.result(timeout=10)
    assert len(set(temporary_paths)) == 2
    assert destination.read_text() in {"a" * 100, "b" * 100}
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("writer", ["text", "frame", "cache", "metadata", "results", "stages"])
def test_atomic_export_failure_preserves_previous_file(writer, tmp_path, monkeypatch):
    destination = tmp_path / "result.csv"
    target = data._metadata_path(destination) if writer == "metadata" else destination
    target.write_bytes(b"previous verified result\n")

    def fail(*args, **kwargs):
        raise OSError("injected replacement failure")

    monkeypatch.setattr(Path, "replace", fail)
    frame = pd.DataFrame({"value": [1.0]})
    callbacks = {
        "text": lambda: search._write_text_atomic(destination, "new result"),
        "frame": lambda: search._write_frame_atomic(frame, destination),
        "cache": lambda: data._write_cache_atomic(frame, destination),
        "metadata": lambda: data._write_cache_metadata_atomic({"ticker": "GLD"}, destination),
        "results": lambda: search._export_result_store(tmp_path / "absent.sqlite3", destination),
        "stages": lambda: search._export_stage_store(tmp_path / "absent.sqlite3", destination),
    }
    with pytest.raises(OSError, match="injected"):
        callbacks[writer]()
    assert target.read_bytes() == b"previous verified result\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_writer_exception_preserves_previous_file(tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("old")
    with pytest.raises(RuntimeError, match="interrupted"), atomic_text_output(path) as handle:
        handle.write("incomplete")
        raise RuntimeError("interrupted")
    assert path.read_text() == "old"
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("values", [[0.1, np.nan], [0.1, np.inf], [True, False], [1j, 2j], ["0.1", "0.2"]])
def test_evaluate_never_coerces_invalid_return_observations(values):
    with pytest.raises((TypeError, ValueError)):
        evaluate(pd.Series(values))


@pytest.mark.parametrize(
    "options",
    [{"annualization": np.nan}, {"annualization": True}, {"risk_free_rate": np.inf}, {"risk_free_rate": True}],
)
def test_evaluate_rejects_nonfinite_or_boolean_assumptions(options):
    with pytest.raises((TypeError, ValueError)):
        evaluate(pd.Series([0.1, -0.1]), **options)


def test_undefined_sharpe_is_not_eligible_even_with_real_exposure():
    index = pd.date_range("2024-01-01", periods=100)
    returns = pd.Series(-0.01, index=index)
    book = {"returns": returns, "trades": pd.Series(1.0, index=index)}
    metrics = search._period_metrics(book, pd.Series(1.0, index=index), index[0], index[-1], 0.0, 252)
    assert metrics["total_return"] == pytest.approx(round(0.99**100 - 1, 4))
    assert metrics["sharpe"] == -99.0
    assert not search._is_eligible({"val_metrics": metrics}, 2, 0, 0)


def test_two_observations_have_finite_undefined_higher_moments():
    result = evaluate(pd.Series([0.1, -0.1]))
    assert result["skew"] == result["kurtosis"] == 0.0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("values", [[True, False], [0.1j, 0.2j], [0.1, np.nan], [0.1, np.inf]])
def test_invalid_rate_schedules_are_not_coerced(values):
    with pytest.raises(ValueError):
        backtest(pd.Series([0.0, 0.0]), pd.Series([100.0, 100.0]), cash_rate=pd.Series(values))


def test_extreme_prices_cannot_produce_nonfinite_books():
    with pytest.raises(ValueError, match="numerical range"):
        backtest(pd.Series([1.0, 1.0]), pd.Series([1e-300, 1e300]))


@pytest.mark.parametrize("ticker", ["GLD\n", "X" * 65, True, None])
def test_invalid_ticker_is_rejected_before_provider_or_cache(ticker, tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path / "cache")
    monkeypatch.setattr(data.yf, "download", lambda *a, **k: pytest.fail("invalid ticker reached network"))
    with pytest.raises(ValueError, match="Invalid ticker"):
        data.download_data(ticker, "2024-01-02", "2024-01-04")
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("start", ["NaT", "2024-01-01 12:00", "2024-01-01T00:00:00Z", None, 20240101])
def test_invalid_session_bounds_are_rejected_before_provider(start, monkeypatch):
    monkeypatch.setattr(data.yf, "download", lambda *a, **k: pytest.fail("invalid date reached network"))
    with pytest.raises(ValueError, match="daily date"):
        data.download_data("GLD", start, "2024-01-04")


def test_timezone_daily_provider_labels_keep_local_session_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    provider = _provider_frame()
    provider.index = provider.index.tz_localize("Pacific/Auckland")
    monkeypatch.setattr(data.yf, "download", lambda *a, **k: provider.copy())
    result = data.download_data("GLD", "2024-01-02", "2024-01-04")
    assert result.index.equals(provider.index.tz_localize(None))


@pytest.mark.parametrize("volume", [-1, np.inf, -np.inf])
def test_invalid_provider_volume_never_reaches_cache(volume, tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    frame = _provider_frame()
    frame["Volume"] = volume
    monkeypatch.setattr(data.yf, "download", lambda *a, **k: frame)
    with pytest.raises(ValueError, match="volume"):
        data.download_data("GLD", "2024-01-02", "2024-01-04")
    assert not list(tmp_path.glob("*_daily.csv"))


def test_old_lossy_symbol_cache_is_not_reinterpreted_as_a_different_asset(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    old = tmp_path / "_GSPC_daily.csv"
    _provider_frame().rename(columns=str.lower).to_csv(old)
    original = old.read_bytes()
    monkeypatch.setattr(data.yf, "download", lambda *a, **k: _provider_frame(2.0))
    result = data.download_data("_GSPC", "2024-01-02", "2024-01-04")
    assert result["close"].iloc[0] == 2.0
    assert old.read_bytes() == original


def test_ticker_case_alias_uses_one_canonical_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    calls = []

    def provider(ticker, **kwargs):
        calls.append(ticker)
        return _provider_frame()

    monkeypatch.setattr(data.yf, "download", provider)
    data.download_data("gld", "2024-01-02", "2024-01-04")
    data.download_data("GLD", "2024-01-02", "2024-01-04")
    assert calls == ["GLD"]


def test_config_read_is_bounded_before_json_parse(tmp_path, monkeypatch):
    path = tmp_path / "oversized.json"
    path.write_text('{"ticker":"GLD"}')
    monkeypatch.setattr(config_module, "MAX_CONFIG_BYTES", 4)
    with pytest.raises(ValueError, match="limit"):
        SearchConfig.from_json(path)


def test_cli_validation_errors_are_actionable_without_tracebacks(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["momentum-lab", "GLD", "--cost", "-1"])
    monkeypatch.setattr(search, "prepare_data", lambda *a, **k: pytest.fail("invalid controls reached data"))
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "cannot be negative" in error
    assert "Traceback" not in error


def test_resume_without_config_never_changes_existing_artifacts(tmp_path, monkeypatch):
    _market(monkeypatch)
    output = tmp_path / "audit"
    output.mkdir()
    marker = output / "all_results.csv"
    marker.write_text("previous evidence")
    with pytest.raises(ValueError, match="run_config.json"):
        search.run_search(config=_search_config(tmp_path), resume=True)
    assert marker.read_text() == "previous evidence"
    assert not (output / "run_config.json").exists()


def test_archiving_retains_journal_contents_and_ignores_unrelated_files(tmp_path):
    store = tmp_path / "results.sqlite3"
    with sqlite3.connect(store) as connection:
        connection.execute("CREATE TABLE original (value TEXT)")
        connection.execute("INSERT INTO original VALUES ('preserved')")
    (tmp_path / "research_notes.md").write_text("user notes")
    with pytest.warns(RuntimeWarning, match="moved to"):
        search._archive_run_artifacts(tmp_path)
    archived = list(tmp_path.glob("results.*.bak.sqlite3"))
    assert len(archived) == 1
    with sqlite3.connect(archived[0]) as connection:
        assert connection.execute("SELECT value FROM original").fetchone()[0] == "preserved"
    assert (tmp_path / "research_notes.md").read_text() == "user notes"
