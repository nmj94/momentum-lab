"""Offline CLI, full-history consent, durable audit and portable portfolio reports."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from momentum_lab import (
    PortfolioConfig,
    PortfolioError,
    RunBusyError,
    StudyRegistry,
    TestReuseError,
    cli,
    data,
    inspect_run,
    run_portfolio,
)
from momentum_lab import portfolio_research as pr
from momentum_lab.datasets import import_dataset
from momentum_lab.governance import RegistryError


@pytest.fixture(autouse=True)
def no_downloads(monkeypatch):
    monkeypatch.setattr(data, "download_data", lambda *args, **kwargs: pytest.fail("Portfolio must stay offline"))


@pytest.fixture
def config(tmp_path):
    index = pd.date_range("2024-01-02", periods=45, freq="B", name="date")
    assets = {}
    for number, ticker in enumerate(("AAA", "BBB", "CCC")):
        close = 100 * np.exp(0.004 * np.arange(len(index)) + 0.12 * np.sin(np.arange(len(index)) / 6 + number))
        frame = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close}, index=index)
        csv = tmp_path / f"{ticker}.csv"
        frame.to_csv(csv)
        assets[ticker] = str(
            import_dataset(
                csv,
                tmp_path / ticker,
                ticker=ticker,
                source="Project-generated synthetic fixture",
                license="MIT; synthetic software tests only",
                currency="USD",
                calendar="exchange",
                price_adjustment="split_and_dividend_adjusted",
            )
        )
    return PortfolioConfig(
        datasets=assets,
        lookback=5,
        skip_recent=1,
        top_k=2,
        rebalance="weekly",
        max_weight=0.4,
        initial_capital=10000,
        cost_bps=5,
        slippage_bps=2,
        spread_bps=4,
        cash_rate=0.02,
        result_dir=str(tmp_path / "runs"),
        run_id="first",
    )


def metadata(config):
    return json.loads((Path(config.result_dir) / config.run_id / "run_config.json").read_text())


def test_portfolio_run_receipt_and_contention_preserve_one_complete_output(config, monkeypatch):
    compute = pr._compute_books

    def checked(*args, **kwargs):
        before = StudyRegistry(create=False).history()
        with pytest.raises(RunBusyError):
            run_portfolio(config, acknowledge_history=True)
        assert before == StudyRegistry(create=False).history()
        return compute(*args, **kwargs)

    monkeypatch.setattr(pr, "_compute_books", checked)
    result = run_portfolio(config, acknowledge_history=True)
    state = inspect_run(result["result_dir"], verify=True)
    assert state["status"] == "completed"
    assert state["integrity"] == "verified"
    assert state["attempt"]["workflow"] == "portfolio"
    assert len(state["history"]) == 1
    assert "metrics" not in json.dumps(state)


@pytest.mark.parametrize("failure", ["computation", "publication"])
def test_portfolio_failures_have_recovery_state_without_erasing_exposure(config, monkeypatch, failure):
    def fail(*args, **kwargs):
        raise OSError("simulated interruption")

    monkeypatch.setattr(pr, "_compute_books" if failure == "computation" else "_write_frame_atomic", fail)
    with pytest.raises(OSError, match="simulated interruption"):
        run_portfolio(config, acknowledge_history=True)
    state = inspect_run(Path(config.result_dir) / config.run_id, verify=True)
    assert state["status"] == "failed"
    assert state["integrity"] == "unavailable"
    assert "new run_id" in state["recovery"]
    assert len(StudyRegistry(create=False).history()) == len(config.datasets)


def change_manifest(config, ticker="BBB", **changes):
    path = Path(config.datasets[ticker])
    value = json.loads(path.read_text())
    value.update(changes)
    path.write_text(json.dumps(value))


def assert_no_calculation(monkeypatch):
    monkeypatch.setattr(pr, "cross_sectional_momentum", lambda *args, **kwargs: pytest.fail("No scoring permitted"))
    monkeypatch.setattr(pr, "backtest_portfolio", lambda *args, **kwargs: pytest.fail("No book permitted"))


@pytest.mark.parametrize("acknowledgement", [False, None, 1, "true"])
def test_consent_is_explicit_and_precedes_all_data_registry_and_output_access(config, monkeypatch, acknowledgement):
    monkeypatch.setattr(pr, "load_dataset", lambda *args, **kwargs: pytest.fail("No data access before consent"))
    monkeypatch.setattr(pr, "StudyRegistry", lambda *args, **kwargs: pytest.fail("No registry before consent"))
    with pytest.raises(PortfolioError, match="acknowledge_history=True"):
        run_portfolio(config, acknowledge_history=acknowledgement)
    assert not Path(config.result_dir).exists()


def test_all_assets_reserved_before_any_signal_or_return_calculation(config, monkeypatch):
    original = pr.cross_sectional_momentum
    before = config.to_dict()

    def checked(prices, **kwargs):
        events = StudyRegistry(create=False).history()
        assert len(events) == 3
        assert {event["ticker"] for event in events} == {"AAA", "BBB", "CCC"}
        assert {event["kind"] for event in events} == {"development"}
        assert all(
            event["start_date"] == str(prices.index[0].date()) and event["end_date"] == str(prices.index[-1].date())
            for event in events
        )
        assert all(event["study_id"] is None for event in events)
        return original(prices, **kwargs)

    monkeypatch.setattr(pr, "cross_sectional_momentum", checked)
    result = run_portfolio(config, acknowledge_history=True)
    assert config.to_dict() == before
    assert result["research_status"] == "exploratory_full_history"
    assert result["history_acknowledged"] is True
    assert "not a sealed test" in result["history_notice"]
    assert metadata(config)["registry_id"] == StudyRegistry(create=False).registry_id


def test_complete_book_exports_reconcile_and_baseline_uses_same_warmup(config):
    summary = run_portfolio(config, acknowledge_history=True)
    output = Path(summary["result_dir"])
    expected = {
        "run_config.json",
        "summary.json",
        "report.html",
        "report.md",
        "ledger.csv",
        "weights.csv",
        "holdings.csv",
        "asset_values.csv",
        "trades.csv",
        "scores.csv",
        "targets.csv",
        "executed_targets.csv",
        "benchmark_ledger.csv",
        "benchmark_weights.csv",
    }
    assert {path.name for path in output.iterdir()} == expected
    book = pd.read_csv(output / "ledger.csv", index_col="date")
    values = pd.read_csv(output / "asset_values.csv", index_col="date")
    weights = pd.read_csv(output / "weights.csv", index_col="date")
    benchmark = pd.read_csv(output / "benchmark_ledger.csv", index_col="date")
    np.testing.assert_allclose(book["nav"], book["cash"] + values.sum(axis=1))
    np.testing.assert_allclose(weights.sum(axis=1) + book["cash_weight"], 1)
    assert book["cash"].min() >= 0
    assert summary["metrics"]["final_nav"] == pytest.approx(book["nav"].iloc[-1])
    assert summary["metrics"]["transaction_costs"] == pytest.approx(book["transaction_cost"].sum())
    assert summary["metrics"]["return_intervals"] == len(book) - 1
    first = np.flatnonzero(book["rebalance_executed"])[0]
    assert first == config.lookback + 1
    assert np.flatnonzero(benchmark["rebalance_executed"]).tolist() == [first]
    pd.testing.assert_frame_equal(book.iloc[:first], benchmark.iloc[:first])
    assert summary["assets"] == ["AAA", "BBB", "CCC"]
    manifest = metadata(config)
    assert manifest["portfolio_engine_schema"] == 1
    assert manifest["cash_convention"] == "effective annual rate, ACT/365"
    assert manifest["observation_scope"] == "entire_evaluated_history_is_development"
    assert manifest["recipe"]["lookback"] == config.lookback
    json.dumps(summary, allow_nan=False)


def test_current_signal_and_realized_allocations_are_separate(config):
    config.rebalance = "daily"
    summary = run_portfolio(config, acknowledge_history=True)
    assert summary["last_signal_date"] == summary["data_end"]
    output = Path(summary["result_dir"])
    targets = pd.read_csv(output / "targets.csv", index_col=0)
    executed = pd.read_csv(output / "executed_targets.csv", index_col=0)
    np.testing.assert_allclose(executed.iloc[-1], targets.iloc[-2])
    assert summary["last_signal_targets"] == targets.iloc[-1].to_dict()
    assert "last signal may still be pending" in (output / "report.md").read_text()


def test_portfolio_history_prevents_a_later_fresh_single_asset_test_claim(config):
    registry = StudyRegistry()
    protocol = {
        "ticker": "AAA",
        "data_snapshot": "a" * 64,
        "periods": {
            "train": ["2022-01-01", "2022-06-30"],
            "val": ["2023-01-01", "2023-12-31"],
            "test": ["2024-02-01", "2024-02-28"],
        },
    }
    registry.register("sealed-before-portfolio", protocol)
    registry.bind_selection("sealed-before-portfolio", {"strategy": "tsmom", "params": {"lookback": 20}})
    assert registry.status("sealed-before-portfolio")["status"] == "sealed"
    run_portfolio(config, acknowledge_history=True)
    assert registry.status("sealed-before-portfolio")["status"] == "known_prior_exposure"
    context = {
        "study_id": "sealed-before-portfolio",
        "ticker": "AAA",
        "data_snapshot": "a" * 64,
        "start": "2024-02-01",
        "end": "2024-02-28",
        "run_id": "later",
        "run_path": Path(config.result_dir) / "later",
    }
    with pytest.raises(TestReuseError):
        registry.claim_test(**context)
    revealed = registry.claim_test(**context, allow_reuse=True, reason="Portfolio already inspected this history")
    assert revealed["access"]["status"] == "repeated_use"


def test_failed_second_reservation_prevents_all_calculations_and_keeps_first_exposure(config, monkeypatch):
    record = StudyRegistry.record_development

    def fail_second(self, **kwargs):
        if kwargs["ticker"] == "BBB":
            raise RegistryError("synthetic reservation failure")
        return record(self, **kwargs)

    monkeypatch.setattr(StudyRegistry, "record_development", fail_second)
    assert_no_calculation(monkeypatch)
    with pytest.raises(RegistryError, match="reservation failure"):
        run_portfolio(config, acknowledge_history=True)
    assert [event["ticker"] for event in StudyRegistry(create=False).history()] == ["AAA"]
    assert not (Path(config.result_dir) / config.run_id / "summary.json").exists()


@pytest.mark.parametrize("stage", ["calculation", "export"])
def test_failures_keep_all_history_and_never_write_completion_marker(config, monkeypatch, stage):
    def fail(*args, **kwargs):
        raise OSError("synthetic interruption")

    monkeypatch.setattr(pr, "backtest_portfolio" if stage == "calculation" else "_write_frame_atomic", fail)
    with pytest.raises(OSError, match="interruption"):
        run_portfolio(config, acknowledge_history=True)
    assert len(StudyRegistry(create=False).history()) == 3
    assert not (Path(config.result_dir) / config.run_id / "summary.json").exists()
    with pytest.raises(PortfolioError, match="already exists"):
        run_portfolio(config, acknowledge_history=True)


def test_existing_run_is_not_overwritten_or_recorded_twice(config):
    summary = run_portfolio(config, acknowledge_history=True)
    before = {path.name: path.read_bytes() for path in Path(summary["result_dir"]).iterdir()}
    with pytest.raises(PortfolioError, match="already exists"):
        run_portfolio(config, acknowledge_history=True)
    assert len(StudyRegistry(create=False).history()) == 3
    assert {path.name: path.read_bytes() for path in Path(summary["result_dir"]).iterdir()} == before


@pytest.mark.parametrize(
    "changes",
    [{"currency": "EUR"}, {"calendar": "continuous"}, {"annualization": 365}, {"price_adjustment": "unadjusted"}],
)
def test_mixed_conventions_are_rejected_without_calculation(config, monkeypatch, changes):
    change_manifest(config, **changes)
    assert_no_calculation(monkeypatch)
    with pytest.raises(PortfolioError, match="share currency"):
        run_portfolio(config, acknowledge_history=True)
    assert not Path(config.result_dir).exists()


def test_exact_sessions_required_no_silent_union_intersection_or_fill(config, monkeypatch):
    original = pr.load_dataset

    def missing_session(path, **kwargs):
        frame, provenance = original(path, **kwargs)
        if kwargs["ticker"] == "BBB":
            frame = frame.iloc[1:]
        return frame, provenance

    monkeypatch.setattr(pr, "load_dataset", missing_session)
    assert_no_calculation(monkeypatch)
    with pytest.raises(PortfolioError, match="session dates must match exactly"):
        run_portfolio(config, acknowledge_history=True)


def test_explicit_slice_limits_exposure_dates_and_preserves_full_snapshot_provenance(config):
    config.start, config.end = "2024-01-15", "2024-02-15"
    summary = run_portfolio(config, acknowledge_history=True)
    assert summary["data_start"] == config.start and summary["data_end"] == config.end
    for event in StudyRegistry(create=False).history():
        assert event["start_date"] == config.start and event["end_date"] == config.end
    assert summary["data_provenance"]["AAA"]["first_date"] == "2024-01-02"
    assert len(metadata(config)["evaluated_snapshots"]["AAA"]) == 64


@pytest.mark.parametrize(
    "change",
    [
        {"lookback": 44},
        {"top_k": 4},
        {"result_dir": ""},
        {"run_id": "../escape"},
        {"run_id": ".."},
        {"run_id": True},
        {"run_id": "a\\b"},
        {"risk_free_rate": -1},
    ],
)
def test_invalid_configuration_fails_before_calculation(config, monkeypatch, change):
    assert_no_calculation(monkeypatch)
    with pytest.raises(PortfolioError):
        run_portfolio({**config.to_dict(), **change}, acknowledge_history=True)


@pytest.mark.parametrize(
    "mapping", [None, [], {}, {"AAA": "a"}, {"AAA": "a", "aaa": "b"}, {"AAA": "", "BBB": "b"}, {"AAA": 4, "BBB": "b"}]
)
def test_invalid_asset_mappings(config, mapping):
    with pytest.raises(PortfolioError):
        run_portfolio({**config.to_dict(), "datasets": mapping}, acknowledge_history=True)


def test_work_budget_is_checked_before_allocation(config, monkeypatch):
    monkeypatch.setattr(pr, "MAX_PORTFOLIO_CELLS", 50)
    assert_no_calculation(monkeypatch)
    with pytest.raises(PortfolioError, match="cell"):
        run_portfolio(config, acknowledge_history=True)


@pytest.mark.parametrize("field", ["acknowledge_history", "reveal_test", "study_id", "workers", "leverage"])
def test_consent_and_unsupported_search_settings_cannot_be_embedded_in_config(config, field):
    with pytest.raises(PortfolioError, match="Unknown"):
        PortfolioConfig.from_mapping({**config.to_dict(), field: True})


@pytest.mark.parametrize(
    "payload", ["[]", "null", "{", "{}", '{"datasets":{},"datasets":{}}', '{"datasets":{"AAA":"a","AAA":"b"}}']
)
def test_json_requires_valid_unique_object_fields(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(payload)
    with pytest.raises(PortfolioError):
        PortfolioConfig.from_json(path)


def test_json_relative_dataset_paths_and_cli_dispatch(config, tmp_path, monkeypatch, capsys):
    values = config.to_dict()
    values["datasets"] = {ticker.lower(): f"{ticker}/manifest.json" for ticker in values["datasets"]}
    path = tmp_path / "portfolio.json"
    path.write_text(json.dumps(values))
    monkeypatch.setattr(sys, "argv", ["momentum-lab", "portfolio", "--config", str(path)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "acknowledge_history" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--acknowledge-history"])
    assert cli.main() == 0
    assert "Portfolio completed" in capsys.readouterr().out
    assert set(metadata(config)["config"]["datasets"]) == {"AAA", "BBB", "CCC"}


def test_relocated_datasets_change_paths_not_research_contract(config, tmp_path):
    first = run_portfolio(config, acknowledge_history=True)
    for ticker, manifest in list(config.datasets.items()):
        destination = tmp_path / "relocated" / ticker
        shutil.copytree(Path(manifest).parent, destination)
        config.datasets[ticker] = str(destination / "manifest.json")
    config.run_id = "relocated"
    second = run_portfolio(config, acknowledge_history=True)
    assert first["contract_sha256"] == second["contract_sha256"]
    assert first["metrics"] == second["metrics"]
    assert len(StudyRegistry(create=False).history()) == 6


def test_html_markdown_escape_untrusted_labels_and_show_limits(config):
    # Exercise published reports with a name that exists on every platform.
    config.run_id = "report & note"
    change_manifest(config, source='<script>alert("source")</script>|note')
    result = run_portfolio(config, acknowledge_history=True)
    output = Path(result["result_dir"])
    for name in ("report.md", "report.html"):
        text = (output / name).read_text()
        assert "<img src=x" not in text and "<script>" not in text
        assert "&lt;script&gt;" in text
        assert "report &amp; note" in text
        assert "point-in-time" in text and "not a sealed test" in text
        assert "ACT/365.25" in text
    assert "\\|note" in (output / "report.md").read_text()
    # Angle brackets cannot be Windows directory names, but they remain valid
    # untrusted renderer inputs. Do not skip their escaping checks on Windows.
    untrusted = {**result, "run_id": "<img src=x onerror=alert(1)>"}
    for render in (pr.render_portfolio_markdown, pr.render_portfolio_html):
        text = render(untrusted, metadata(config))
        assert "<img src=x" not in text and "&lt;img src=x" in text


def test_all_cash_report_and_metrics_remain_well_defined(config):
    config.absolute_threshold = 1e20
    result = run_portfolio(config, acknowledge_history=True)
    assert result["latest_cash_weight"] == 1
    assert result["metrics"]["total_return"] > 0
    assert result["metrics"]["transaction_costs"] == 0
    assert "Uninvested cash" in (Path(result["result_dir"]) / "report.html").read_text()


def test_frozen_benchmark_cli_and_failure_exit(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["momentum-lab", "portfolio", "benchmark"])
    assert cli.main() == 0
    assert "passed (6 cases)" in capsys.readouterr().out

    def fail():
        raise PortfolioError("synthetic regression")

    monkeypatch.setattr(pr, "check_portfolio_reference", fail)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "synthetic regression" in capsys.readouterr().err
