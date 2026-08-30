"""Research protocol locks, durable observations and sealed-output contracts."""

import copy
import json
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from inspect import signature
from pathlib import Path
from threading import Barrier

import numpy as np
import pandas as pd
import pytest

from momentum_lab import cli, search
from momentum_lab.config import SearchConfig
from momentum_lab.governance import (
    RegistryError,
    StudyRegistry,
    TestReuseError,
    registry_path,
    validate_study_options,
)
from momentum_lab.reporting import render_html_report, render_markdown_report


def _protocol(*, snapshot="a" * 64, test=("2020-04-01", "2020-04-30"), ticker="FAKE"):
    return {
        "ticker": ticker,
        "data_snapshot": snapshot,
        "periods": {"train": ["2019-01-01", "2019-06-30"], "val": ["2019-07-01", "2019-12-31"], "test": list(test)},
        "strategies": ["tsmom"],
        "cost_bps": 1.0,
        "source_fingerprint": "source-v1",
    }


def _registered(registry, study_id="first", **kwargs):
    protocol = _protocol(**kwargs)
    registry.register(study_id, protocol)
    registry.bind_selection(study_id, {"strategy": "tsmom", "params": {"lookback": 20}})
    return {
        "study_id": study_id,
        "ticker": protocol["ticker"],
        "data_snapshot": protocol["data_snapshot"],
        "start": protocol["periods"]["test"][0],
        "end": protocol["periods"]["test"][1],
        "run_id": study_id + "-run",
        "run_path": registry.path.parent / study_id,
    }


def _events(registry):
    with sqlite3.connect(registry.path) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute("SELECT * FROM observations ORDER BY id")]


def test_registry_identity_persists_and_readers_do_not_create_or_modify_files(tmp_path):
    missing = tmp_path / "missing" / "registry.sqlite3"
    with pytest.raises(RegistryError, match="not found"):
        StudyRegistry(missing, create=False)
    assert not missing.parent.exists()
    registry = StudyRegistry()
    _registered(registry)
    before = registry.path.read_bytes()
    reopened = StudyRegistry(create=False)
    assert reopened.registry_id == registry.registry_id
    assert reopened.status("first")["status"] == "sealed"
    assert reopened.list_studies()[0]["study_id"] == "first"
    assert registry.path.read_bytes() == before


def test_path_override_is_shared_and_evaluated_at_call_time(tmp_path, monkeypatch):
    first = tmp_path / "first.sqlite3"
    second = tmp_path / "second.sqlite3"
    monkeypatch.setenv("MOMENTUM_LAB_REGISTRY_PATH", str(first))
    assert registry_path() == first
    monkeypatch.setenv("MOMENTUM_LAB_REGISTRY_PATH", str(second))
    assert registry_path() == second
    assert registry_path(first) == first
    with pytest.raises(RegistryError, match="non-empty"):
        registry_path("")


@pytest.mark.parametrize(
    "changed",
    [
        {"cost_bps": 2},
        {"data_snapshot": "b" * 64},
        {"strategies": ["rsi"]},
        {"source_fingerprint": "source-v2"},
        {
            "periods": {
                "train": ["2018-01-01", "2019-06-30"],
                "val": ["2019-07-01", "2019-12-31"],
                "test": ["2020-04-01", "2020-04-30"],
            }
        },
    ],
)
def test_protocol_is_immutable_and_failed_update_preserves_it(changed):
    registry = StudyRegistry()
    protocol = _protocol()
    first = registry.register("locked", protocol)
    assert registry.register("locked", dict(reversed(list(protocol.items())))) == first
    with pytest.raises(RegistryError, match="protocol mismatch"):
        registry.register("locked", {**protocol, **changed})
    assert registry.status("locked")["protocol_sha256"] == first["protocol_sha256"]


def test_selection_freezes_once_and_reveal_requires_prior_selection():
    registry = StudyRegistry()
    with pytest.raises(RegistryError, match="not registered"):
        registry.require_reveal_ready("missing")
    registry.register("locked", _protocol())
    with pytest.raises(RegistryError, match="no frozen selection"):
        registry.require_reveal_ready("locked")
    selected = {"strategy": "tsmom", "params": {"lookback": 20}}
    registry.bind_selection("locked", selected)
    registry.bind_selection("locked", selected)
    registry.require_reveal_ready("locked")
    with pytest.raises(RegistryError, match="selection changed"):
        registry.bind_selection("locked", {**selected, "params": {"lookback": 30}})


@pytest.mark.parametrize(
    "options",
    [
        {"study_id": "../escape"},
        {"study_id": ""},
        {"study_id": True},
        {"study_id": "a" * 129},
        {"study_id": None, "reveal_test": True},
        {"study_id": "s", "reveal_test": 1},
        {"study_id": "s", "allow_test_reuse": True},
        {"study_id": "s", "reveal_test": True, "allow_test_reuse": True},
        {"study_id": "s", "reveal_test": True, "allow_test_reuse": True, "test_reuse_reason": " "},
        {"study_id": "s", "reveal_test": True, "allow_test_reuse": True, "test_reuse_reason": "a" * 2001},
        {"study_id": "s", "test_reuse_reason": "not acknowledged"},
    ],
)
def test_invalid_governance_options_fail_before_market_data(options, monkeypatch):
    monkeypatch.setattr(search, "prepare_data", lambda *a, **k: pytest.fail("data must not be read"))
    with pytest.raises(RegistryError):
        search.run_search(**options)


@pytest.mark.parametrize(
    "bounds",
    [
        (None, "2020-04-30"),
        (20200401, 20200430),
        ("today", "today"),
        ("NaT", "2020-04-30"),
        ("2020-05-01", "2020-04-30"),
    ],
)
def test_invalid_observation_dates_are_not_coerced(bounds):
    with pytest.raises(RegistryError):
        StudyRegistry().register("bad", _protocol(test=bounds))


def test_invalid_snapshot_and_overlapping_protocol_periods_are_rejected():
    registry = StudyRegistry()
    with pytest.raises(RegistryError, match="SHA-256"):
        registry.register("bad", _protocol(snapshot="bad"))
    with pytest.raises(RegistryError, match="disjoint"):
        registry.register("bad", _protocol(test=("2019-12-31", "2020-01-31")))


@pytest.mark.parametrize(
    "bounds",
    [
        ("2020-04-01", "2020-04-30"),
        ("2020-03-25", "2020-04-02"),
        ("2020-04-30", "2020-05-20"),
        ("2020-01-01", "2020-12-31"),
    ],
)
def test_overlap_ignores_run_study_case_and_data_version(bounds):
    registry = StudyRegistry()
    first = registry.claim_test(**_registered(registry))
    registry.complete_test(first["access"]["event_id"], {"score": 0.123})
    context = _registered(registry, "another", snapshot="b" * 64, test=bounds, ticker="fake")
    with pytest.raises(TestReuseError, match="overlap"):
        registry.claim_test(**context)
    acknowledged = registry.claim_test(**context, allow_reuse=True, reason="Historical comparison, not fresh evidence")
    assert acknowledged["access"]["status"] == "repeated_use"
    assert acknowledged["access"]["prior_overlap_count"] == 1
    assert acknowledged["access"]["reuse_reason"] == "Historical comparison, not fresh evidence"


def test_disjoint_daily_intervals_and_distinct_tickers_remain_distinct():
    registry = StudyRegistry()
    registry.claim_test(**_registered(registry))
    adjacent = registry.claim_test(**_registered(registry, "next", test=("2020-05-01", "2020-05-31")))
    other_asset = registry.claim_test(**_registered(registry, "other", ticker="OTHER"))
    assert adjacent["access"]["status"] == other_asset["access"]["status"] == "first_recorded_reveal"


def test_previous_development_observations_also_prevent_a_fresh_test_claim():
    registry = StudyRegistry()
    context = _registered(registry, "new", test=("2020-02-01", "2020-02-28"))
    registry.record_development(
        ticker="fake",
        start="2020-01-01",
        end="2020-03-31",
        data_snapshot="b" * 64,
        run_id="old-development",
        run_path=registry.path.parent / "old",
    )
    with pytest.raises(TestReuseError):
        registry.claim_test(**context)
    assert registry.status("new")["prior_overlaps"][0]["kind"] == "development"


def test_completed_reveal_is_reused_across_run_ids_without_new_computation_or_fresh_label():
    registry = StudyRegistry()
    context = _registered(registry)
    first = registry.claim_test(**context)
    payload = {"score": 0.123456789, "test_evaluated_at": "original-time"}
    registry.complete_test(first["access"]["event_id"], payload)
    before = len(_events(registry))
    repeated = registry.claim_test(**{**context, "run_id": "new-run", "run_path": registry.path.parent / "elsewhere"})
    assert repeated["cached"] is True
    assert repeated["payload"] == payload
    assert repeated["access"]["status"] == "previously_revealed"
    assert repeated["access"]["event_id"] == first["access"]["event_id"]
    assert len(_events(registry)) == before + 1
    assert _events(registry)[-1]["kind"] == "reveal_replay"
    assert _events(registry)[-1]["source_event_id"] == first["access"]["event_id"]
    status = registry.status("first")
    assert status["test_results_visible"] is False
    assert "0.123456789" not in json.dumps(status)
    assert "score" not in json.dumps(registry.list_studies())


@pytest.mark.parametrize("mark_failed", [False, True])
def test_reserved_and_failed_reveals_are_never_forgotten(mark_failed):
    registry = StudyRegistry()
    context = _registered(registry)
    claim = registry.claim_test(**context)
    if mark_failed:
        registry.fail_test(claim["access"]["event_id"], "interrupted before publishing")
    reopened = StudyRegistry(create=False)
    with pytest.raises(TestReuseError):
        reopened.claim_test(**context)
    assert reopened.status("first")["prior_overlap_count"] == 1
    retry = reopened.claim_test(
        **context, allow_reuse=True, reason="Retry interrupted evaluation; possible prior exposure"
    )
    assert retry["access"]["status"] == "repeated_use"


def test_concurrent_claims_cannot_both_claim_first_reveal():
    registry = StudyRegistry()
    context = _registered(registry)
    barrier = Barrier(2)

    def claim():
        local = StudyRegistry(create=False)
        barrier.wait(timeout=5)
        try:
            return local.claim_test(**context)["access"]["status"]
        except TestReuseError:
            return "blocked"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(claim) for _ in range(2)]
        statuses = [future.result(timeout=10) for future in futures]
    assert sorted(statuses) == ["blocked", "first_recorded_reveal"]
    assert len(_events(registry)) == 1


def test_cached_payload_and_protocol_corruption_fail_closed():
    registry = StudyRegistry()
    context = _registered(registry)
    claim = registry.claim_test(**context)
    registry.complete_test(claim["access"]["event_id"], {"score": 0.1})
    with sqlite3.connect(registry.path) as connection:
        connection.execute("UPDATE observations SET result_json='{}'")
    with pytest.raises(RegistryError, match="integrity"):
        registry.claim_test(**context)
    with sqlite3.connect(registry.path) as connection:
        connection.execute("UPDATE studies SET protocol_json='{}'")
    with pytest.raises(RegistryError, match="integrity"):
        registry.status("first")


def test_registry_replacement_and_unsupported_schema_do_not_reset_history(tmp_path):
    registry = StudyRegistry()
    _registered(registry)
    registry.path.rename(tmp_path / "saved-registry.sqlite3")
    replacement = StudyRegistry()
    assert replacement.registry_id != registry.registry_id
    with pytest.raises(RegistryError, match="identity changed"):
        registry.status("first")
    with sqlite3.connect(replacement.path) as connection:
        connection.execute("PRAGMA user_version=999")
    before = replacement.path.read_bytes()
    with pytest.raises(RegistryError, match="schema"):
        StudyRegistry()
    assert replacement.path.read_bytes() == before


@pytest.fixture
def market(monkeypatch):
    index = pd.date_range("2021-01-04", periods=500, freq="B")
    rng = np.random.Generator(np.random.PCG64(771))
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0008, 0.012, len(index)))), index=index)
    frame = pd.DataFrame({"close": close, "open": close * 0.999, "volume": 100000.0})
    data = {**{key: frame[key] for key in frame}, "annualization": 252}
    monkeypatch.setattr(search, "prepare_data", lambda *a, **k: (data, frame))
    params = [
        {"lookback": size, "threshold": 0.002, "long_short": True, "skip_recent": 1, "signal_smooth": 0}
        for size in (12, 24)
    ]
    monkeypatch.setattr(search, "_quick_sample", lambda *a, **k: params)
    return data, frame


def _run(tmp_path, run_id="sealed", study_id="study", **kwargs):
    options = {
        "ticker": "FAKE",
        "strategies": ["tsmom"],
        "quick": True,
        "robust": False,
        "bootstrap_resamples": 200,
        "result_dir": tmp_path / "experiments",
        "run_id": run_id,
        "study_id": study_id,
        "keep_all_results": True,
    }
    options.update(kwargs)
    return search.run_search(**options)


def test_registered_run_never_evaluates_test_and_sensitivity_only_sees_development(
    tmp_path, market, monkeypatch, capsys
):
    _, frame = market
    val_end = search._split_periods(frame.index)["val"][1]
    original_ledgers, original_robustness = search._selected_ledgers, search.robustness_check
    touched = []

    def ledgers(best, supplied, df, *args):
        assert df.index[-1] == val_end
        assert supplied["close"].index[-1] == val_end
        return original_ledgers(best, supplied, df, *args)

    def sensitivity(supplied, df, periods, *args, **kwargs):
        assert df.index[-1] == val_end
        assert "test" not in periods
        assert supplied["close"].index[-1] == val_end
        touched.append(True)
        return original_robustness(supplied, df, periods, *args, **kwargs)

    monkeypatch.setattr(search, "_selected_ledgers", ledgers)
    monkeypatch.setattr(search, "robustness_check", sensitivity)
    monkeypatch.setattr(search, "_test_payload", lambda *a, **k: pytest.fail("sealed means no test evaluation"))
    result = _run(tmp_path, robust=True)
    assert touched == [True]
    assert result["best"] is not None and "test_metrics" not in result["best"]
    assert result["benchmark_metrics"] is None
    assert set(result["bootstrap_diagnostics"]["periods"]) == {"validation"}
    assert result["test_access"]["status"] == "sealed"
    assert result["test_access"]["test_results_visible"] is False
    assert "Test Sharpe:" not in capsys.readouterr().out
    summary = json.loads((Path(result["result_dir"]) / "summary.json").read_text())
    assert "test_metrics" not in json.dumps(summary)
    assert "Test (withheld)" in (Path(result["result_dir"]) / "report.md").read_text()
    assert [event["kind"] for event in _events(StudyRegistry(create=False))] == ["development"]


def test_reveal_is_logged_before_evaluation_then_reused_on_resume_and_new_run(tmp_path, market, monkeypatch):
    sealed = _run(tmp_path)
    registry = StudyRegistry(create=False)
    original = search._test_payload
    evaluations = []

    def evaluate(*args, **kwargs):
        events = _events(registry)
        assert events[-1]["kind"] == "registered_reveal"
        assert events[-1]["status"] == "reserved"
        evaluations.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(search, "_test_payload", evaluate)
    revealed = _run(tmp_path, resume=True, reveal_test=True)
    assert revealed["test_access"]["status"] == "first_recorded_reveal"
    assert revealed["test_access"]["test_results_visible"] is True
    assert revealed["top_results"] == sealed["top_results"]
    # Transient indicator-cache counters are not part of persisted candidate evidence.
    assert search._results_rows(revealed["all_results"]) == search._results_rows(sealed["all_results"])
    assert (
        revealed["bootstrap_diagnostics"]["periods"]["validation"]
        == sealed["bootstrap_diagnostics"]["periods"]["validation"]
    )
    again = _run(tmp_path, resume=True, reveal_test=True)
    elsewhere = _run(tmp_path, run_id="another-run", result_dir=tmp_path / "elsewhere", reveal_test=True)
    for replay in (again, elsewhere):
        assert replay["test_access"]["status"] == "previously_revealed"
        assert replay["test_access"]["cached"] is True
        assert replay["best"]["test_metrics"] == revealed["best"]["test_metrics"]
        assert replay["best"]["test_evaluated_at"] == revealed["best"]["test_evaluated_at"]
    assert evaluations == [True]
    hidden_again = _run(tmp_path, resume=True)
    assert "test_metrics" not in hidden_again["best"]
    assert hidden_again["test_access"]["status"] == "previously_revealed"
    assert hidden_again["test_access"]["test_results_visible"] is False


def test_new_study_and_changed_data_version_do_not_bypass_known_exposure(tmp_path, market, monkeypatch):
    _run(tmp_path)
    _run(tmp_path, resume=True, reveal_test=True)
    data, frame = market
    changed = frame.copy()
    changed.iloc[-10:, changed.columns.get_loc("close")] *= 1.001
    changed_data = {**data, "close": changed["close"]}
    monkeypatch.setattr(search, "prepare_data", lambda *a, **k: (changed_data, changed))
    other = _run(tmp_path, run_id="new-data", study_id="new-data")
    assert other["test_access"]["status"] == "known_prior_exposure"
    with pytest.raises(TestReuseError):
        _run(tmp_path, run_id="new-data", study_id="new-data", resume=True, reveal_test=True)
    acknowledged = _run(
        tmp_path,
        run_id="new-data",
        study_id="new-data",
        resume=True,
        reveal_test=True,
        allow_test_reuse=True,
        test_reuse_reason="Revised historical prices",
    )
    assert acknowledged["test_access"]["status"] == "repeated_use"
    assert acknowledged["test_access"]["reuse_reason"] == "Revised historical prices"


def test_failed_reveal_keeps_possible_exposure_and_requires_acknowledged_retry(tmp_path, market, monkeypatch):
    _run(tmp_path)
    original = search._test_payload

    def failure(*args, **kwargs):
        raise RuntimeError("simulated evaluation interruption")

    monkeypatch.setattr(search, "_test_payload", failure)
    with pytest.raises(RuntimeError, match="interruption"):
        _run(tmp_path, resume=True, reveal_test=True)
    events = _events(StudyRegistry(create=False))
    assert events[-1]["status"] == "failed"
    monkeypatch.setattr(search, "_test_payload", original)
    with pytest.raises(TestReuseError):
        _run(tmp_path, resume=True, reveal_test=True)
    retry = _run(
        tmp_path, resume=True, reveal_test=True, allow_test_reuse=True, test_reuse_reason="Retry failed reveal"
    )
    assert retry["test_access"]["status"] == "repeated_use"


def test_unknown_legacy_history_is_recorded_and_blocks_a_later_registered_reveal(tmp_path, market):
    legacy = _run(tmp_path, run_id="legacy", study_id=None)
    assert legacy["test_access"]["status"] == "history_unknown"
    assert "test_metrics" in legacy["best"]
    assert legacy["test_access"]["mode"] == "legacy"
    sealed = _run(tmp_path)
    assert legacy["top_results"] == sealed["top_results"]
    assert sealed["test_access"]["status"] == "known_prior_exposure"
    with pytest.raises(TestReuseError):
        _run(tmp_path, resume=True, reveal_test=True)


def test_registered_protocol_changes_and_directory_reuse_are_rejected_before_search(tmp_path, market, monkeypatch):
    _run(tmp_path)
    monkeypatch.setattr(search, "run_single_experiment", lambda *a, **k: pytest.fail("must reject before evaluation"))
    with pytest.raises(RegistryError, match="cost_bps"):
        _run(tmp_path, run_id="changed-cost", cost_bps=2)
    with pytest.raises(ValueError, match="empty run directory"):
        _run(tmp_path)
    with pytest.raises(ValueError, match="disable study_id"):
        _run(tmp_path, study_id=None)
    with pytest.raises(ValueError, match="original run_config"):
        _run(tmp_path, run_id="missing-run", resume=True)


def test_registered_resume_detects_a_replaced_or_missing_registry(tmp_path, market):
    _run(tmp_path)
    path = registry_path()
    path.rename(tmp_path / "preserved-registry.sqlite3")
    with pytest.raises(RegistryError, match="not found"):
        _run(tmp_path, resume=True)
    StudyRegistry()
    with pytest.raises(ValueError, match="identity mismatch"):
        _run(tmp_path, resume=True)


def test_reveal_is_invocation_only_and_config_does_not_overwrite_explicit_consent(tmp_path, market):
    config = SearchConfig(
        ticker="FAKE",
        strategies=["tsmom"],
        study_id="config-study",
        result_dir=str(tmp_path / "config"),
        run_id="run",
        robust=False,
        bootstrap=False,
    )
    first = search.run_search(config=config)
    assert first["test_access"]["status"] == "sealed"
    shown = search.run_search(config=config, resume=True, reveal_test=True)
    assert shown["test_access"]["status"] == "first_recorded_reveal"
    with pytest.raises(ValueError, match="unknown search config"):
        SearchConfig.from_mapping({"reveal_test": True})
    for obj in (SearchConfig, search.run_search):
        names = list(signature(obj).parameters)
        assert names.index("study_id") > names.index("bootstrap_min_observations")


def test_no_winner_does_not_freeze_or_reveal(tmp_path, market):
    result = _run(tmp_path, min_validation_trades=1_000_000)
    assert result["best"] is None
    assert result["test_access"]["status"] == "no_selection"
    with pytest.raises(RegistryError, match="no frozen selection"):
        _run(tmp_path, min_validation_trades=1_000_000, resume=True, reveal_test=True)
    assert all(event["kind"] == "development" for event in _events(StudyRegistry(create=False)))


def test_legacy_import_is_idempotent_preserves_artifacts_and_blocks_reuse(tmp_path, market):
    original = _run(tmp_path, run_id="old", study_id=None)
    run_dir = Path(original["result_dir"])
    before = {name: (run_dir / name).read_bytes() for name in ("summary.json", "run_config.json")}
    registry = StudyRegistry(tmp_path / "imported.sqlite3")
    imported = registry.import_legacy(run_dir)
    assert imported["status"] == "history_unknown"
    assert imported["already_imported"] is False
    copied = tmp_path / "copy"
    shutil.copytree(run_dir, copied)
    assert registry.import_legacy(copied)["already_imported"] is True
    assert len(_events(registry)) == 2
    assert before == {name: (run_dir / name).read_bytes() for name in before}
    _run(tmp_path, run_id="new", study_id="new", registry_path=str(registry.path))
    with pytest.raises(TestReuseError):
        _run(tmp_path, run_id="new", study_id="new", registry_path=str(registry.path), resume=True, reveal_test=True)


def test_sealed_results_cannot_be_imported_as_observed_test_results(tmp_path, market):
    result = _run(tmp_path)
    with pytest.raises(RegistryError, match="no visible legacy test"):
        StudyRegistry().import_legacy(result["result_dir"])


def test_reports_mask_test_scores_and_escape_reuse_reasons():
    summary = {
        "best": {"val_metrics": {"sharpe": 1.2}, "test_metrics": {"sharpe": "SECRET_TEST_SCORE"}},
        "benchmark_metrics": {"sharpe": "SECRET_BENCHMARK_SCORE"},
        "bootstrap_diagnostics": {"periods": {"test": {"n_observations": "SECRET_TEST_BOOTSTRAP"}}},
        "test_access": {
            "status": "previously_revealed",
            "test_results_visible": False,
            "reuse_reason": "<script>example</script>|reason",
        },
    }
    for renderer in (render_markdown_report, render_html_report):
        output = renderer(copy.deepcopy(summary), {})
        assert "SECRET_" not in output
        assert "Test (withheld)" in output
        assert "previously_revealed" in output
        assert "<script>" not in output
        assert "outside this registry is unknown" in output
    assert "\\|reason" in render_markdown_report(summary, {})


def test_cli_forwards_transient_reveal_options_and_status_never_discloses_scores(monkeypatch, capsys):
    registry = StudyRegistry()
    context = _registered(registry)
    claim = registry.claim_test(**context)
    registry.complete_test(claim["access"]["event_id"], {"secret_score": 12345})
    monkeypatch.setattr("sys.argv", ["momentum-lab", "study", "status", "first"])
    assert cli.main() == 0
    output = capsys.readouterr().out
    assert "previously_revealed" in output
    assert "secret_score" not in output
    captured = {}
    monkeypatch.setattr(search, "run_search", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(cli, "run_search", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        "sys.argv",
        [
            "momentum-lab",
            "FAKE",
            "--study-id",
            "first",
            "--registry",
            str(registry.path),
            "--resume",
            "--reveal-test",
            "--allow-test-reuse",
            "--test-reuse-reason",
            "audit",
        ],
    )
    cli.main()
    assert captured["study_id"] == "first"
    assert captured["registry_path"] == str(registry.path)
    assert captured["reveal_test"] is True and captured["allow_test_reuse"] is True
    assert captured["test_reuse_reason"] == "audit"


def test_cli_missing_registry_is_an_error_not_a_new_empty_history(tmp_path, monkeypatch):
    path = tmp_path / "missing.sqlite3"
    monkeypatch.setattr("sys.argv", ["momentum-lab", "study", "--registry", str(path), "list"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert not path.exists()


def test_valid_acknowledgement_options():
    validate_study_options("trial-1", True, True, "Known historical reuse")


def test_concurrent_registry_initialization_preserves_one_identity():
    barrier = Barrier(2)

    def initialize():
        barrier.wait(timeout=5)
        return StudyRegistry().registry_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(initialize) for _ in range(2)]
        identities = [future.result(timeout=10) for future in futures]
    assert identities[0] == identities[1]


def test_reveal_claim_must_match_the_frozen_protocol_and_can_only_finalize_once():
    registry = StudyRegistry()
    context = _registered(registry)
    for changed in ({"data_snapshot": "b" * 64}, {"ticker": "OTHER"}, {"end": "2020-05-01"}):
        with pytest.raises(RegistryError, match="does not match"):
            registry.claim_test(**{**context, **changed})
    claim = registry.claim_test(**context)
    registry.complete_test(claim["access"]["event_id"], {"score": 1.0})
    with pytest.raises(RegistryError, match="already finalized"):
        registry.complete_test(claim["access"]["event_id"], {"score": 2.0})


def test_history_is_bounded_filtered_and_score_free(capsys, monkeypatch):
    registry = StudyRegistry()
    first = registry.claim_test(**_registered(registry))
    registry.complete_test(first["access"]["event_id"], {"secret_score": 100})
    registry.claim_test(**_registered(registry, "other", ticker="OTHER"))
    assert len(registry.history("fake")) == 1
    assert registry.history("fake")[0]["ticker"] == "FAKE"
    assert len(registry.history(limit=1)) == 1
    assert "secret_score" not in json.dumps(registry.history())
    for limit in (0, 1001, True, 1.5):
        with pytest.raises(RegistryError, match="limit"):
            registry.history(limit=limit)
    monkeypatch.setattr("sys.argv", ["momentum-lab", "study", "history", "--ticker", "fake"])
    cli.main()
    assert "secret_score" not in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["momentum-lab", "study", "list"])
    cli.main()
    assert "first" in capsys.readouterr().out


def test_empty_search_does_not_create_a_reveal(tmp_path, market, monkeypatch):
    monkeypatch.setattr(search, "_quick_sample", lambda *a, **k: [])
    result = _run(tmp_path)
    assert result["n_results"] == 0
    assert result["test_access"]["status"] == "no_selection"
    assert StudyRegistry().status("study")["selection_sha256"] is None


def test_registered_successive_halving_preserves_selection_on_reveal(tmp_path, market):
    options = {
        "search_method": "successive_halving",
        "candidate_budget": 2,
        "halving_factor": 2,
        "halving_stages": 2,
        "bootstrap": False,
    }
    sealed = _run(tmp_path, **options)
    assert sealed["best"] is not None
    shown = _run(tmp_path, **options, resume=True, reveal_test=True)
    assert shown["top_results"] == sealed["top_results"]
    assert shown["test_access"]["status"] == "first_recorded_reveal"


def test_failed_export_does_not_reset_a_completed_reveal(tmp_path, market, monkeypatch):
    _run(tmp_path)
    original = search._write_text_atomic

    def fail_report(path, text):
        if path.name == "report.md":
            raise OSError("simulated report export failure")
        return original(path, text)

    monkeypatch.setattr(search, "_write_text_atomic", fail_report)
    with pytest.raises(OSError, match="export failure"):
        _run(tmp_path, resume=True, reveal_test=True)
    monkeypatch.setattr(search, "_write_text_atomic", original)
    monkeypatch.setattr(search, "_test_payload", lambda *a, **k: pytest.fail("must reuse completed reveal"))
    restored = _run(tmp_path, resume=True, reveal_test=True)
    assert restored["test_access"]["status"] == "previously_revealed"


def test_reservation_survives_when_failure_logging_also_fails(tmp_path, market, monkeypatch):
    _run(tmp_path)

    def fail_evaluation(*args, **kwargs):
        raise KeyboardInterrupt("simulated interrupted process")

    def fail_audit(*args, **kwargs):
        raise OSError("simulated unavailable audit write")

    monkeypatch.setattr(search, "_test_payload", fail_evaluation)
    monkeypatch.setattr(StudyRegistry, "fail_test", fail_audit)
    with pytest.warns(RuntimeWarning, match="reservation retained"), pytest.raises(KeyboardInterrupt):
        _run(tmp_path, resume=True, reveal_test=True)
    assert _events(StudyRegistry(create=False))[-1]["status"] == "reserved"


def test_cli_imports_legacy_and_reuse_errors_are_actionable(tmp_path, market, monkeypatch, capsys):
    old = _run(tmp_path, study_id=None)
    capsys.readouterr()
    monkeypatch.setattr("sys.argv", ["momentum-lab", "study", "import-legacy", old["result_dir"]])
    cli.main()
    assert json.loads(capsys.readouterr().out)["status"] == "history_unknown"

    def rejected(**kwargs):
        raise TestReuseError("recorded observation overlap; explicit acknowledgement required")

    monkeypatch.setattr(cli, "run_search", rejected)
    monkeypatch.setattr("sys.argv", ["momentum-lab", "FAKE"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "explicit acknowledgement" in capsys.readouterr().err


def test_corrupt_or_unrelated_registry_is_never_reinitialized(tmp_path):
    path = tmp_path / "unrelated.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE important_user_data (value TEXT)")
    before = path.read_bytes()
    with pytest.raises(RegistryError, match="schema"):
        StudyRegistry(path)
    assert path.read_bytes() == before
    corrupt = tmp_path / "corrupt.sqlite3"
    corrupt.write_bytes(b"not a sqlite database")
    with pytest.raises(RegistryError, match="cannot open"):
        StudyRegistry(corrupt)
    assert corrupt.read_bytes() == b"not a sqlite database"
