"""Offline portfolio development/reveal boundaries and interruption recovery."""

import copy
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
    PortfolioStudyConfig,
    PortfolioStudyRegistry,
    RegistryError,
    StudyRegistry,
    TestReuseError,
    cli,
    data,
    run_portfolio,
    run_portfolio_study,
)
from momentum_lab import portfolio_research as pr
from momentum_lab import portfolio_study as ps
from momentum_lab.datasets import import_dataset


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(data, "download_data", lambda *a, **k: pytest.fail("No network fallback"))


@pytest.fixture
def config(tmp_path):
    index = pd.date_range("2024-01-02", periods=50, freq="B", name="date")
    datasets = {}
    for number, ticker in enumerate(("AAA", "BBB", "CCC")):
        close = 100 * np.exp(0.004 * np.arange(len(index)) + 0.12 * np.sin(np.arange(len(index)) / 6 + number))
        frame = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close}, index=index)
        path = tmp_path / f"{ticker}.csv"
        frame.to_csv(path)
        datasets[ticker] = str(
            import_dataset(
                path,
                tmp_path / ticker,
                ticker=ticker,
                source="Project-generated synthetic test fixture",
                license="MIT; software tests",
                currency="USD",
                calendar="exchange",
                price_adjustment="split_and_dividend_adjusted",
            )
        )
    membership = {
        "schema_version": 1,
        "universe_id": "synthetic-test",
        "source": "Synthetic events",
        "license": "MIT",
        "coverage_start": "2024-01-01",
        "coverage_end": "2024-12-31",
        "initial_known_on": "2023-12-29",
        "initial_members": ["AAA", "BBB"],
        "events": [
            {
                "ticker": "CCC",
                "known_on": str(index[18].date()),
                "effective_on": str(index[20].date()),
                "action": "add",
            },
            {
                "ticker": "AAA",
                "known_on": str(index[38].date()),
                "effective_on": str(index[40].date()),
                "action": "remove",
            },
        ],
    }
    path = tmp_path / "membership.json"
    path.write_text(json.dumps(membership))
    return PortfolioStudyConfig(
        datasets=datasets,
        universe=str(path),
        study_id="study-alpha",
        test_start=str(index[35].date()),
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
        run_id="development",
    )


def run(config, name="reveal", **options):
    value = config.to_dict()
    value["run_id"] = name
    return run_portfolio_study(value, **options)


def ledger(result, name):
    return pd.read_csv(Path(result["result_dir"]) / name, index_col="date", parse_dates=True)


def metadata(result):
    return json.loads((Path(result["result_dir"]) / "run_config.json").read_text())


def no_evaluation(monkeypatch):
    monkeypatch.setattr(ps, "_compute_books", lambda *a, **k: pytest.fail("No portfolio evaluation permitted"))


def test_development_evaluator_receives_only_prefix_and_sanitized_rules(config, monkeypatch):
    original = ps._compute_books
    before = copy.deepcopy(config.to_dict())
    calls = []

    def checked(recipe, prices, eligibility):
        events = StudyRegistry(create=False).history()
        assert len(events) == 3 and {row["kind"] for row in events} == {"portfolio_development"}
        assert {row["ticker"] for row in events} == {"AAA", "BBB", "CCC"}
        assert all(row["end_date"] < config.test_start for row in events)
        assert len(prices) == len(eligibility) == 35 and str(prices.index[-1].date()) < config.test_start
        assert type(recipe) is PortfolioConfig and recipe.datasets == {} and recipe.universe is None
        assert not hasattr(recipe, "test_start") and not hasattr(recipe, "study_id")
        assert recipe.start is None and recipe.end is None and recipe.registry_path is None
        calls.append(len(prices))
        return original(recipe, prices, eligibility)

    monkeypatch.setattr(ps, "_compute_books", checked)
    result = run_portfolio_study(config)
    assert calls == [35] and config.to_dict() == before
    assert result["test"] is None and result["test_access"]["status"] == "sealed"
    assert result["test_access"]["test_results_visible"] is False
    assert result["artifact_scope"] == "development_ledgers_only"
    assert not list(Path(result["result_dir"]).glob("test_*.csv"))
    assert len(ledger(result, "development_scores.csv")) == 35
    assert result["development"]["n_bars"] == 35
    assert "test hidden" in result["development"]["history_notice"]
    assert "development only; test hidden" in (Path(result["result_dir"]) / "report.md").read_text()
    assert metadata(result)["registry_id"] == StudyRegistry(create=False).registry_id


def test_reveal_requires_prior_completed_invocation_before_reading_data(config, monkeypatch):
    monkeypatch.setattr(ps, "_load_universe", lambda *a: pytest.fail("No files before reveal ready"))
    with pytest.raises(RegistryError, match="registry not found"):
        run(config, reveal_test=True)
    assert not Path(config.result_dir).exists()


def test_all_asset_test_reservations_precede_full_evaluation(config, monkeypatch):
    run_portfolio_study(config)
    original = ps._compute_books
    calls = []

    def checked(recipe, prices, eligibility):
        events = StudyRegistry(create=False).history()
        claims = [event for event in events if event["kind"] == "portfolio_reveal"]
        assert len(claims) == 3 and {row["status"] for row in claims} == {"reserved"}
        assert all(row["start_date"] == config.test_start for row in claims)
        assert {row["ticker"] for row in claims} == {"AAA", "BBB", "CCC"}
        assert len(prices) == len(eligibility) == 50
        assert recipe.datasets == {} and not hasattr(recipe, "test_start")
        calls.append(50)
        return original(recipe, prices, eligibility)

    monkeypatch.setattr(ps, "_compute_books", checked)
    result = run(config, reveal_test=True)
    assert calls == [50] and result["test_access"]["test_results_visible"]
    assert result["test_access"]["status"] == "first_recorded_reveal"
    assert {row["status"] for row in StudyRegistry(create=False).history() if row["kind"] == "portfolio_reveal"} == {
        "completed"
    }


def test_carried_test_book_includes_first_return_and_excludes_development_costs(config):
    sealed = run_portfolio_study(config)
    revealed = run(config, reveal_test=True)
    development = ledger(sealed, "development_ledger.csv")
    test = ledger(revealed, "test_ledger.csv")
    baseline = ledger(revealed, "test_benchmark_ledger.csv")
    values = ledger(revealed, "test_asset_values.csv")
    shares = ledger(revealed, "test_holdings.csv")
    trades = ledger(revealed, "test_trades.csv")
    assert len(test) == 16 and test.index[0] == development.index[-1]
    np.testing.assert_allclose(shares.iloc[0], ledger(sealed, "development_holdings.csv").iloc[-1])
    np.testing.assert_allclose(test["nav"], values.sum(axis=1) + test["cash"])
    assert test["return"].iloc[0] == 0 and test["transaction_cost"].iloc[0] == 0
    assert not test["rebalance_executed"].iloc[0] and trades.iloc[0].eq(0).all()
    np.testing.assert_allclose(test["return"].iloc[1:], test["nav"].pct_change().iloc[1:])
    metrics = revealed["test"]["metrics"]
    assert metrics["starting_nav"] == pytest.approx(development["nav"].iloc[-1])
    assert metrics["total_return"] == pytest.approx(test["nav"].iloc[-1] / test["nav"].iloc[0] - 1)
    assert metrics["return_intervals"] == 15 and revealed["test"]["n_bars"] == 15
    assert metrics["transaction_costs"] == pytest.approx(test["transaction_cost"].sum())
    assert metrics["rebalances"] == test["rebalance_executed"].sum()
    assert baseline["equity"].iloc[0] == 1 and test["equity"].iloc[0] == 1
    assert revealed["test"]["benchmark_metrics"]["starting_nav"] == pytest.approx(baseline["nav"].iloc[0])
    assert revealed["test"]["warmup_bars"] == 0
    assert revealed["test"]["last_signal_scores"]["AAA"] is None
    assert "Period starting NAV" in (Path(revealed["result_dir"]) / "report.html").read_text()
    assert (
        "test reports use the documented prior-close anchor"
        in (Path(revealed["result_dir"]) / "report.html").read_text()
    )
    assert not list(Path(revealed["result_dir"]).glob("development_*.csv"))


def test_short_test_without_new_signals_carries_holdings(config):
    config.universe = None
    config.rebalance = "monthly"
    config.test_start = "2024-03-08"
    run_portfolio_study(config)
    result = run(config, reveal_test=True)
    assert result["test"]["metrics"]["rebalances"] == 0
    assert result["test"]["last_signal_date"] < config.test_start
    assert result["test"]["metrics"]["return_intervals"] == 2


def test_cached_replay_does_not_recompute_or_fabricate_missing_ledgers(config, monkeypatch):
    sealed = run_portfolio_study(config)
    first = run(config, reveal_test=True)
    original_test = copy.deepcopy(first["test"])
    no_evaluation(monkeypatch)
    replay = run(config, "replay", reveal_test=True)
    assert replay["test"] == original_test and replay["development"] == sealed["development"]
    assert replay["artifact_scope"] == "cached_summary_only" and replay["original_test_output"] == first["result_dir"]
    assert replay["test_access"]["status"] == "previously_revealed" and replay["test_access"]["cached"]
    assert {path.name for path in Path(replay["result_dir"]).iterdir()} == {
        "run_config.json",
        "summary.json",
        "report.md",
        "report.html",
    }
    assert len(StudyRegistry(create=False).history()) == 9
    assert "Cached summary replay only" in (Path(replay["result_dir"]) / "report.md").read_text()


def test_non_reveal_after_prior_reveal_remains_hidden(config):
    run_portfolio_study(config)
    run(config, reveal_test=True)
    later = run(config, "development-again")
    assert later["test"] is None and later["test_access"]["test_results_visible"] is False
    assert later["test_access"]["status"] == "previously_revealed"
    assert not list(Path(later["result_dir"]).glob("test_*.csv"))
    assert "test hidden" in (Path(later["result_dir"]) / "report.md").read_text()


def test_whole_history_v13_exposure_requires_explicit_reuse(config, monkeypatch):
    old = PortfolioConfig.from_mapping(
        {key: value for key, value in config.to_dict().items() if key not in {"study_id", "test_start"}}
    )
    old.run_id = "exploratory"
    run_portfolio(old, acknowledge_history=True)
    sealed = run_portfolio_study(config)
    assert sealed["test_access"]["status"] == "known_prior_exposure"
    with monkeypatch.context() as patch:
        no_evaluation(patch)
        with pytest.raises(TestReuseError):
            run(config, reveal_test=True)
    result = run(
        config,
        "reused",
        reveal_test=True,
        allow_test_reuse=True,
        test_reuse_reason="Already inspected the full history",
    )
    assert result["test_access"]["status"] == "repeated_use"


@pytest.mark.parametrize(
    "flags",
    [
        {"reveal_test": "true"},
        {"reveal_test": 1},
        {"allow_test_reuse": True},
        {"reveal_test": True, "allow_test_reuse": True},
        {"test_reuse_reason": "why"},
        {"reveal_test": True, "allow_test_reuse": True, "test_reuse_reason": " "},
    ],
)
def test_invalid_invocation_consent_fails_before_data_access(config, monkeypatch, flags):
    monkeypatch.setattr(ps, "_load_universe", lambda *a: pytest.fail("No data for invalid flags"))
    with pytest.raises(RegistryError):
        run_portfolio_study(config, **flags)


@pytest.mark.parametrize(
    "changes",
    [
        {"study_id": None},
        {"study_id": "bad/id"},
        {"test_start": None},
        {"test_start": "2024-01-03"},
        {"test_start": "2024-02-24"},
        {"test_start": "2024-03-11"},
        {"test_start": "2024-02-20T00:00:00"},
        {"universe": ""},
        {"universe": 1},
        {"lookback": True},
    ],
)
def test_invalid_study_config_fails_without_evaluation(config, monkeypatch, changes):
    no_evaluation(monkeypatch)
    with pytest.raises((RegistryError, PortfolioError)):
        run_portfolio_study({**config.to_dict(), **changes})


@pytest.mark.parametrize("field", ["reveal_test", "allow_test_reuse", "test_reuse_reason", "acknowledge_history"])
def test_consent_cannot_be_persisted_in_config(config, field):
    with pytest.raises(PortfolioError, match="Unknown"):
        PortfolioStudyConfig.from_mapping({**config.to_dict(), field: True})


@pytest.mark.parametrize("mutation", ["recipe", "source", "membership", "snapshot", "software"])
def test_frozen_input_revision_rejected_before_evaluation(config, monkeypatch, mutation):
    run_portfolio_study(config)
    if mutation == "recipe":
        config.max_weight = 0.3
    elif mutation == "source":
        path = Path(config.datasets["BBB"])
        value = json.loads(path.read_text())
        value["source"] += " revised"
        path.write_text(json.dumps(value))
    elif mutation == "membership":
        path = Path(config.universe)
        path.write_text(json.dumps(json.loads(path.read_text()), indent=2))
    elif mutation == "snapshot":
        original = pr.load_dataset

        def changed(*args, **kwargs):
            frame, provenance = original(*args, **kwargs)
            if kwargs["ticker"] == "BBB":
                frame = frame.copy()
                frame.iloc[-1, frame.columns.get_loc("close")] *= 1.001
            return frame, provenance

        monkeypatch.setattr(pr, "load_dataset", changed)
    else:
        monkeypatch.setattr(pr, "_source_fingerprint", lambda: "different-software")
    no_evaluation(monkeypatch)
    with pytest.raises(RegistryError, match="protocol mismatch"):
        run(config, reveal_test=True)
    assert len(StudyRegistry(create=False).history()) == 3


def test_relocation_and_numeric_path_api_normalization_preserve_contract(config, tmp_path):
    config.universe = Path(config.universe)
    config.lookback = np.int64(config.lookback)
    config.max_weight = np.float64(config.max_weight)
    first = run_portfolio_study(config)
    config.lookback = int(config.lookback)
    config.initial_capital = float(config.initial_capital)
    for ticker, path in list(config.datasets.items()):
        destination = tmp_path / "moved" / ticker
        shutil.copytree(Path(path).parent, destination)
        config.datasets[ticker] = str(destination / "manifest.json")
    moved = tmp_path / "moved-membership.json"
    moved.write_bytes(config.universe.read_bytes())
    config.universe = str(moved)
    revealed = run(config, reveal_test=True)
    assert first["contract_sha256"] == revealed["contract_sha256"]


@pytest.mark.parametrize("stage", ["calculation", "csv"])
def test_failed_development_never_freezes_or_marks_completed(config, monkeypatch, stage):
    def fail(*a, **k):
        raise OSError("synthetic interruption")

    monkeypatch.setattr(ps, "_compute_books" if stage == "calculation" else "_write_frame_atomic", fail)
    with pytest.raises(OSError):
        run_portfolio_study(config)
    assert len(StudyRegistry(create=False).history()) == 3
    with pytest.raises(RegistryError, match="frozen development"):
        PortfolioStudyRegistry(create=False).require_reveal_ready(config.study_id)
    assert not (Path(config.result_dir) / "development" / "summary.json").exists()


@pytest.mark.parametrize("stage", ["calculation", "csv"])
def test_failed_test_exposure_is_retained_and_acknowledged_retry_required(config, monkeypatch, stage):
    run_portfolio_study(config)

    def fail(*a, **k):
        raise OSError("synthetic test interruption")

    with monkeypatch.context() as patch:
        patch.setattr(ps, "_compute_books" if stage == "calculation" else "_write_frame_atomic", fail)
        with pytest.raises(OSError):
            run(config, reveal_test=True)
    claims = [event for event in StudyRegistry(create=False).history() if event["kind"] == "portfolio_reveal"]
    assert len(claims) == 3 and {row["status"] for row in claims} == {"failed"}
    assert not (Path(config.result_dir) / "reveal" / "summary.json").exists()
    with pytest.raises(TestReuseError):
        run(config, "blocked-retry", reveal_test=True)
    retry = run(
        config,
        "retry",
        reveal_test=True,
        allow_test_reuse=True,
        test_reuse_reason="Acknowledge interrupted test access",
    )
    assert retry["test_access"]["status"] == "repeated_use"


def test_report_failure_after_cache_commit_recovers_without_recalculation(config, monkeypatch):
    run_portfolio_study(config)

    def fail(*a, **k):
        raise OSError("synthetic report failure")

    with monkeypatch.context() as patch:
        patch.setattr(ps, "render_portfolio_study_html", fail)
        with pytest.raises(OSError):
            run(config, reveal_test=True)
    assert not (Path(config.result_dir) / "reveal" / "summary.json").exists()
    no_evaluation(monkeypatch)
    replay = run(config, "recovery", reveal_test=True)
    assert replay["test_access"]["cached"] and replay["test"] is not None


def test_existing_output_never_overwritten_or_rerecorded(config):
    result = run_portfolio_study(config)
    output = Path(result["result_dir"])
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(PortfolioError, match="already exists"):
        run_portfolio_study(config)
    assert len(StudyRegistry(create=False).history()) == 3
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}


def test_defensive_reports_do_not_disclose_injected_test_without_access(config):
    result = run_portfolio_study(config)
    result["test"] = copy.deepcopy(result["development"])
    result["test"]["run_id"] = "DO_NOT_DISCLOSE_TEST"
    result["test"]["metrics"]["final_nav"] = 987654321.123
    manifest = metadata(result)
    for renderer in (ps.render_portfolio_study_markdown, ps.render_portfolio_study_html):
        text = renderer(result, manifest)
        assert "987654321" not in text and "DO_NOT_DISCLOSE_TEST" not in text
        assert "test hidden" in text


def test_cli_relative_paths_status_list_reveal_replay(config, tmp_path, monkeypatch, capsys):
    values = config.to_dict()
    values["datasets"] = {ticker: f"{ticker}/manifest.json" for ticker in config.datasets}
    values["universe"] = "membership.json"
    path = tmp_path / "study.json"
    path.write_text(json.dumps(values))

    def invoke(*args):
        monkeypatch.setattr(sys, "argv", ["momentum-lab", "portfolio", "study", *map(str, args)])
        return cli.main()

    assert invoke("--config", path) == 0
    assert "visible: False" in capsys.readouterr().out
    assert invoke("status", config.study_id) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "sealed" and "metrics" not in status
    assert invoke("list") == 0
    assert json.loads(capsys.readouterr().out)["test_results_visible"] is False
    assert invoke("--config", path, "--run-id", "cli-reveal", "--reveal-test") == 0
    assert "visible: True" in capsys.readouterr().out
    assert invoke("--config", path, "--run-id", "cli-replay", "--reveal-test") == 0
    assert "previously_revealed" in capsys.readouterr().out


def test_empty_membership_has_json_safe_null_scores_and_cash_results(config):
    path = Path(config.universe)
    value = json.loads(path.read_text())
    value.update(initial_members=[], events=[])
    path.write_text(json.dumps(value))
    config.absolute_threshold = None
    run_portfolio_study(config)
    result = run(config, reveal_test=True)
    assert result["test"]["latest_cash_weight"] == 1
    assert set(result["test"]["last_signal_scores"].values()) == {None}
    assert result["test"]["metrics"]["transaction_costs"] == 0
    json.dumps(result, allow_nan=False)
