"""All-asset transactions, shared history, frozen protocols and cached replay."""

import copy
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from momentum_lab import PortfolioStudyRegistry, RegistryError, StudyRegistry, TestReuseError
from momentum_lab import portfolio_governance as pg


@pytest.fixture
def protocol():
    return {
        "kind": "fixed_rule_portfolio_v1",
        "assets": {"AAA": "a" * 64, "BBB": "b" * 64, "CCC": "c" * 64},
        "periods": {"development": ["2023-01-02", "2023-12-29"], "test": ["2024-01-02", "2024-03-29"]},
        "recipe": {"lookback": 20, "top_k": 2},
    }


@pytest.fixture
def registry(tmp_path, protocol):
    registry = PortfolioStudyRegistry(tmp_path / "registry.sqlite3")
    registry.register("alpha", protocol)
    return registry


def freeze(registry, study="alpha"):
    registry.complete_development(study, {"rule": "fixed"}, {"metrics": {"return": 0.123}})


def claim(registry, tmp_path, run="reveal", **kwargs):
    return registry.claim_test("alpha", run, tmp_path / run, **kwargs)


def sql(registry, command, params=()):
    with sqlite3.connect(registry.path) as connection:
        return connection.execute(command, params).fetchall()


def test_extension_preserves_identity_base_version_and_existing_history(tmp_path):
    path = tmp_path / "audit.sqlite3"
    old = StudyRegistry(path)
    old.record_development(
        ticker="AAA", start="2024-01-02", end="2024-01-31", data_snapshot="d" * 64, run_id="old", run_path=tmp_path
    )
    new = PortfolioStudyRegistry(path)
    assert old.registry_id == new.registry_id
    assert old.history() == new.history()
    assert sql(new, "PRAGMA user_version") == [(1,)]
    assert sql(new, "SELECT schema_version FROM portfolio_registry_info") == [(1,)]


def test_read_only_does_not_create_registry_or_extension(tmp_path):
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(RegistryError, match="registry not found"):
        PortfolioStudyRegistry(missing, create=False)
    assert not missing.exists()
    base = StudyRegistry(tmp_path / "base.sqlite3")
    before = base.path.read_bytes()
    with pytest.raises(RegistryError, match="extension"):
        PortfolioStudyRegistry(base.path, create=False)
    assert base.path.read_bytes() == before


def test_status_list_are_score_free_and_read_only(registry):
    freeze(registry)
    before = registry.path.read_bytes()
    reader = PortfolioStudyRegistry(registry.path, create=False)
    text = json.dumps({"status": reader.status("alpha"), "list": reader.list_studies()})
    assert "metrics" not in text and "0.123" not in text
    assert reader.status("alpha")["test_results_visible"] is False
    assert reader.status("alpha")["status"] == "sealed"
    assert registry.path.read_bytes() == before


def test_protocol_selection_and_first_development_are_immutable(registry, protocol):
    before = copy.deepcopy(protocol)
    registry.register("alpha", protocol)
    registry.require_protocol("alpha", protocol)
    assert protocol == before
    changed = {**protocol, "recipe": {"lookback": 10}}
    with pytest.raises(RegistryError, match="mismatch"):
        registry.register("alpha", changed)
    with pytest.raises(RegistryError, match="mismatch"):
        registry.require_protocol("alpha", changed)
    freeze(registry)
    original = registry.development_payload("alpha")
    registry.complete_development("alpha", {"rule": "fixed"}, {"metrics": {"return": 99}})
    assert registry.development_payload("alpha") == original
    original["metrics"]["return"] = -100
    assert registry.development_payload("alpha")["metrics"]["return"] == 0.123
    with pytest.raises(RegistryError, match="selection"):
        registry.complete_development("alpha", {"rule": "different"}, {})


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "other"},
        {"assets": {"AAA": "a" * 64}},
        {"assets": {"AAA": "bad", "BBB": "b" * 64}},
        {"assets": {"AAA": "a" * 64, "aaa": "b" * 64}},
        {"assets": []},
        {"periods": {}},
        {"periods": {"development": ["2024-02-01", "2024-03-01"], "test": ["2024-01-01", "2024-02-01"]}},
        {"periods": {"development": ["2023-02-01", "2023-03-01"], "test": ["2024-01-01T12:00", "2024-02-01"]}},
        {"periods": {"development": ["2023-02-01", "2023-03-01"], "test": ["2024-02-01", "2024-01-01"]}},
        {"periods": {"development": ["2023-02-01", "2023-03-01"], "test": ["2024-01-01"]}},
        {"recipe": {"bad": float("nan")}},
    ],
)
def test_bad_protocols_fail_closed(tmp_path, protocol, changes):
    with pytest.raises(RegistryError):
        PortfolioStudyRegistry(tmp_path / "bad.sqlite3").register("bad", {**protocol, **changes})


def test_reveal_needs_prior_frozen_development(registry, tmp_path):
    for call in (
        lambda: registry.require_reveal_ready("alpha"),
        lambda: registry.development_payload("alpha"),
        lambda: claim(registry, tmp_path),
    ):
        with pytest.raises(RegistryError, match="frozen development"):
            call()
    assert registry.history() == []


def test_development_is_atomic_deduplicated_and_in_shared_history(registry, tmp_path):
    registry.record_portfolio_development("alpha", "dev", tmp_path)
    registry.record_portfolio_development("alpha", "dev", tmp_path)
    history = StudyRegistry(registry.path, create=False).history()
    assert len(history) == 3
    assert {event["ticker"] for event in history} == {"AAA", "BBB", "CCC"}
    assert {event["kind"] for event in history} == {"portfolio_development"}
    assert all(event["end_date"] == "2023-12-29" and event["study_id"] is None for event in history)
    assert registry.status("alpha")["status"] == "sealed"


@pytest.mark.parametrize("kind", ["development", "reveal"])
def test_second_asset_insert_failure_rolls_back_whole_group(registry, tmp_path, kind):
    freeze(registry)
    sql(
        registry,
        "CREATE TRIGGER reject_second BEFORE INSERT ON observations WHEN NEW.ticker='BBB' BEGIN SELECT RAISE(ABORT,'synthetic failure'); END",
    )
    with pytest.raises(sqlite3.DatabaseError, match="synthetic failure"):
        if kind == "development":
            registry.record_portfolio_development("alpha", "dev", tmp_path)
        else:
            claim(registry, tmp_path)
    assert registry.history() == []
    assert sql(registry, "SELECT COUNT(*) FROM portfolio_access_batches") == [(0,)]
    assert sql(registry, "SELECT COUNT(*) FROM portfolio_batch_events") == [(0,)]


def test_group_reservation_completion_and_cached_replay(registry, tmp_path):
    freeze(registry)
    fresh = claim(registry, tmp_path)
    assert fresh["cached"] is False and fresh["access"]["status"] == "first_recorded_reveal"
    assert fresh["access"]["test_results_visible"] is False
    assert set(fresh["access"]["event_ids"]) == {"AAA", "BBB", "CCC"}
    assert {item["status"] for item in registry.history()} == {"reserved"}
    payload = {"test": {"metrics": {"return": 99}}, "original_test_output": str(tmp_path)}
    registry.complete_test(fresh["access"]["batch_id"], payload)
    assert {item["status"] for item in registry.history()} == {"completed"}
    with pytest.raises(RegistryError, match="finalized"):
        registry.complete_test(fresh["access"]["batch_id"], {})
    replay = claim(registry, tmp_path, "replay")
    assert replay["cached"] and replay["payload"] == payload
    assert replay["access"]["status"] == "previously_revealed"
    assert replay["access"]["test_results_visible"]
    assert len(registry.history()) == 6
    for event in registry.history()[:3]:
        assert (
            event["kind"] == "portfolio_replay"
            and event["source_event_id"] == fresh["access"]["event_ids"][event["ticker"]]
        )
    assert '"metrics"' not in json.dumps(registry.status("alpha"))
    assert registry.status("alpha")["test_results_visible"] is False


def test_one_asset_prior_exposure_blocks_entire_claim_across_snapshots(registry, tmp_path):
    freeze(registry)
    StudyRegistry(registry.path).record_development(
        ticker="bbb", start="2024-02-01", end="2024-02-02", data_snapshot="f" * 64, run_id="external", run_path=tmp_path
    )
    with pytest.raises(TestReuseError):
        claim(registry, tmp_path)
    assert len(registry.history()) == 1
    status = registry.status("alpha")
    assert status["status"] == "known_prior_exposure" and status["prior_overlap_counts"] == {
        "AAA": 0,
        "BBB": 1,
        "CCC": 0,
    }
    reused = claim(registry, tmp_path, allow_reuse=True, reason="History already inspected")
    assert reused["access"]["status"] == "repeated_use" and len(registry.history()) == 4


def test_portfolio_claim_blocks_fresh_single_asset_study(registry, tmp_path):
    freeze(registry)
    claim(registry, tmp_path)
    single = StudyRegistry(registry.path)
    single.register(
        "alpha",
        {
            "ticker": "AAA",
            "data_snapshot": "f" * 64,
            "periods": {
                "train": ["2022-01-01", "2022-12-31"],
                "val": ["2023-01-01", "2023-12-31"],
                "test": ["2024-01-02", "2024-03-29"],
            },
        },
    )
    single.bind_selection("alpha", {"strategy": "tsmom"})
    with pytest.raises(TestReuseError):
        single.claim_test(
            ticker="AAA",
            start="2024-01-02",
            end="2024-03-29",
            data_snapshot="f" * 64,
            run_id="single",
            run_path=tmp_path,
            study_id="alpha",
        )
    assert len(single.list_studies()) == len(registry.list_studies()) == 1  # Separate namespaces, shared history.


@pytest.mark.parametrize("failed", [False, True])
def test_interrupted_or_failed_claim_is_never_fresh_again(registry, tmp_path, failed):
    freeze(registry)
    first = claim(registry, tmp_path)
    if failed:
        registry.fail_test(first["access"]["batch_id"], "interrupted")
        assert {row["status"] for row in registry.history()} == {"failed"}
    with pytest.raises(TestReuseError):
        claim(registry, tmp_path, "retry")
    second = claim(registry, tmp_path, "retry", allow_reuse=True, reason="Acknowledge interruption")
    assert second["access"]["status"] == "repeated_use" and len(registry.history()) == 6


@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE portfolio_access_batches SET result_json='{}' WHERE kind='reveal'",
        "DELETE FROM portfolio_batch_events WHERE event_id=(SELECT event_id FROM portfolio_batch_events LIMIT 1)",
        "UPDATE observations SET data_snapshot='ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff' WHERE ticker='AAA'",
        "UPDATE observations SET run_id='changed' WHERE ticker='BBB'",
        "UPDATE observations SET status='failed' WHERE ticker='CCC'",
        "UPDATE portfolio_studies SET protocol_json='{}'",
        "UPDATE portfolio_studies SET development_json='{}'",
        "UPDATE portfolio_studies SET selected_at=NULL",
    ],
)
def test_corrupt_cache_or_group_refuses_silent_recomputation(registry, tmp_path, mutation):
    freeze(registry)
    first = claim(registry, tmp_path)
    registry.complete_test(first["access"]["batch_id"], {"test": {"metric": 1}})
    sql(registry, mutation)
    with pytest.raises(RegistryError):
        claim(registry, tmp_path, "replay")
    assert sql(registry, "SELECT COUNT(*) FROM portfolio_access_batches") == [(1,)]


def test_partial_completion_failure_rolls_back_then_fail_marks_all(registry, tmp_path):
    freeze(registry)
    first = claim(registry, tmp_path)
    sql(
        registry,
        "CREATE TRIGGER reject_complete BEFORE UPDATE ON observations WHEN NEW.status='completed' AND NEW.ticker='BBB' BEGIN SELECT RAISE(ABORT,'synthetic completion'); END",
    )
    with pytest.raises(sqlite3.DatabaseError):
        registry.complete_test(first["access"]["batch_id"], {})
    assert {event["status"] for event in registry.history()} == {"reserved"}
    registry.fail_test(first["access"]["batch_id"], "completion interrupted")
    assert {event["status"] for event in registry.history()} == {"failed"}


def test_late_older_claim_cannot_replace_first_completed_cache(registry, tmp_path):
    freeze(registry)
    older = claim(registry, tmp_path, "slow-first")
    newer = claim(
        registry, tmp_path, "fast-second", allow_reuse=True, reason="Concurrent reuse explicitly acknowledged"
    )
    registry.complete_test(newer["access"]["batch_id"], {"test": {"result": "first completed"}})
    first = claim(registry, tmp_path, "read-before-slow-finishes")
    registry.complete_test(older["access"]["batch_id"], {"test": {"result": "late older claim"}})
    second = claim(registry, tmp_path, "read-after-slow-finishes")
    assert first["payload"] == second["payload"] == {"test": {"result": "first completed"}}
    assert first["access"]["source_batch_id"] == second["access"]["source_batch_id"] == newer["access"]["batch_id"]


@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE portfolio_studies SET cache_batch_id=NULL",
        "UPDATE portfolio_studies SET cache_batch_id='missing'",
        "UPDATE portfolio_access_batches SET status='failed' WHERE kind='reveal'",
    ],
)
def test_missing_or_corrupt_cache_pointer_fails_closed(registry, tmp_path, mutation):
    freeze(registry)
    first = claim(registry, tmp_path)
    registry.complete_test(first["access"]["batch_id"], {"test": {}})
    sql(registry, mutation)
    with pytest.raises(RegistryError, match="cache pointer"):
        claim(registry, tmp_path, "replay")


def test_cache_is_bounded_finite_and_schema_is_checked(registry, tmp_path, monkeypatch):
    freeze(registry)
    first = claim(registry, tmp_path)
    for value in ({"bad": float("nan")}, [], {"bad": object()}):
        with pytest.raises(RegistryError):
            registry.complete_test(first["access"]["batch_id"], value)
    monkeypatch.setattr(pg, "MAX_CACHED_SUMMARY_BYTES", 20)
    with pytest.raises(RegistryError, match="limit"):
        registry.complete_test(first["access"]["batch_id"], {"large": "x" * 100})
    sql(registry, "UPDATE portfolio_registry_info SET schema_version=999")
    with pytest.raises(RegistryError, match="schema"):
        PortfolioStudyRegistry(registry.path)


def test_overlapping_concurrent_portfolios_only_one_first_reveal(tmp_path, protocol):
    registry = PortfolioStudyRegistry(tmp_path / "parallel.sqlite3")
    for name, assets in (("one", {"AAA": "a" * 64, "BBB": "b" * 64}), ("two", {"BBB": "b" * 64, "CCC": "c" * 64})):
        registry.register(name, {**protocol, "assets": assets})
        freeze(registry, name)

    def access(name):
        try:
            local = PortfolioStudyRegistry(registry.path)
            return local.claim_test(name, name, tmp_path / name)["access"]["status"]
        except TestReuseError:
            return "blocked"

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(access, ("one", "two"))) == ["blocked", "first_recorded_reveal"]
    assert len(registry.history()) == 2
    assert sql(registry, "SELECT COUNT(*) FROM portfolio_access_batches") == [(1,)]
