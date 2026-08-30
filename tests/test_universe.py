"""Declared membership timing, strict schema, causal signals and no invented quotes."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from momentum_lab import PortfolioConfig, PortfolioError, cross_sectional_momentum, load_membership, universe
from momentum_lab.portfolio_research import _compute_books


@pytest.fixture
def manifest():
    return {
        "schema_version": 1,
        "universe_id": "synthetic",
        "source": "Synthetic test fixture",
        "license": "MIT",
        "coverage_start": "2024-01-01",
        "coverage_end": "2024-03-31",
        "initial_known_on": "2023-12-29",
        "initial_members": ["aaa"],
        "events": [
            {"ticker": "bbb", "known_on": "2024-01-03", "effective_on": "2024-01-06", "action": "add"},
            {"ticker": "aaa", "known_on": "2024-01-05", "effective_on": "2024-01-10", "action": "remove"},
        ],
    }


@pytest.fixture
def prices():
    index = pd.date_range("2024-01-02", periods=12, freq="B", name="date")
    return pd.DataFrame({"AAA": 100 + np.arange(12), "BBB": 100 + 3 * np.arange(12)}, index=index)


def write(tmp_path, manifest):
    path = tmp_path / "membership.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_effective_weekend_and_sliced_prior_state(manifest, prices, tmp_path):
    path = write(tmp_path, manifest)
    before = copy.deepcopy(manifest)
    mask, source = load_membership(path, prices.index, prices.columns)
    assert not mask.loc[:"2024-01-05", "BBB"].any()
    assert mask.loc["2024-01-08":, "BBB"].all()
    assert mask.loc[:"2024-01-09", "AAA"].all()
    assert not mask.loc["2024-01-10":, "AAA"].any()
    sliced, same = load_membership(path, prices.index[7:], prices.columns)
    pd.testing.assert_frame_equal(sliced, mask.iloc[7:])
    assert source == same and manifest == before
    assert len(source["manifest_sha256"]) == len(source["canonical_sha256"]) == 64
    assert "not independently verified" in source["note"]


def test_changes_force_delayed_monthly_trades_for_strategy_and_baseline(manifest, prices, tmp_path):
    mask, _ = load_membership(write(tmp_path, manifest), prices.index, prices.columns)
    config = PortfolioConfig(datasets={}, lookback=2, rebalance="monthly", top_k=1, absolute_threshold=None, cost_bps=0)
    books = _compute_books(config, prices, mask)
    dates = list(books["plan"]["rebalance"].loc[lambda s: s].index.strftime("%Y-%m-%d"))
    assert dates == ["2024-01-04", "2024-01-08", "2024-01-10"]
    assert books["plan"]["targets"].loc["2024-01-04", "BBB"] == 0
    assert books["plan"]["targets"].loc["2024-01-08", "BBB"] == 1
    for name in ("result", "benchmark"):
        assert books[name]["holdings"].loc[:"2024-01-08", "BBB"].eq(0).all()
        assert books[name]["holdings"].loc["2024-01-09", "BBB"] > 0
        assert books[name]["holdings"].loc["2024-01-11":, "AAA"].eq(0).all()
    assert books["benchmark"]["executed_targets"].loc["2024-01-09", "AAA"] == 0.5
    assert books["benchmark"]["executed_targets"].loc["2024-01-11", "BBB"] == 1


def test_future_membership_and_quotes_cannot_change_prefix(manifest, prices, tmp_path):
    path = write(tmp_path, manifest)
    mask, _ = load_membership(path, prices.index, prices.columns)
    rule = {"lookback": 2, "rebalance": "monthly", "absolute_threshold": None}
    original = cross_sectional_momentum(prices, eligibility=mask, **rule)
    changed = prices.copy()
    changed.iloc[7:] *= 20
    manifest["events"].append(
        {"ticker": "AAA", "known_on": "2024-01-08", "effective_on": "2024-01-12", "action": "add"}
    )
    revised, _ = load_membership(write(tmp_path, manifest), prices.index, prices.columns)
    later = cross_sectional_momentum(changed, eligibility=revised, **rule)
    prefix = cross_sectional_momentum(prices.iloc[:7], eligibility=mask.iloc[:7], **rule)
    for key in ("scores", "targets"):
        pd.testing.assert_frame_equal(original[key].iloc[:7], later[key].iloc[:7])
        pd.testing.assert_frame_equal(prefix[key], original[key].iloc[:7])


def test_empty_universe_keeps_cash_without_selecting_nan_scores(manifest, prices, tmp_path):
    manifest.update(initial_members=[], events=[])
    mask, _ = load_membership(write(tmp_path, manifest), prices.index, prices.columns)
    books = _compute_books(PortfolioConfig(datasets={}, lookback=2, absolute_threshold=None), prices, mask)
    assert books["plan"]["scores"].isna().all().all()
    assert books["plan"]["targets"].dropna().eq(0).all().all()
    assert books["result"]["ledger"]["cash_weight"].eq(1).all()
    assert books["benchmark"]["ledger"]["cash_weight"].eq(1).all()


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"extra": 1},
        {"source": ""},
        {"license": "a\nb"},
        {"universe_id": " padded "},
        {"source": "x" * 2049},
        {"initial_members": "AAA"},
        {"initial_members": ["AAA", "aaa"]},
        {"initial_members": ["BAD SPACE"]},
        {"initial_members": [False]},
        {"initial_known_on": "2024-01-02"},
        {"coverage_end": "2023-12-31"},
        {"coverage_start": "2024-02-30"},
        {"events": {}},
        {"coverage_start": "2024-1-01"},
        {"coverage_start": "2024-01-01T00:00:00"},
    ],
)
def test_invalid_manifest_fields_rejected(manifest, prices, tmp_path, changes):
    manifest.update(changes)
    with pytest.raises(PortfolioError):
        load_membership(write(tmp_path, manifest), prices.index, prices.columns)


@pytest.mark.parametrize(
    "changes",
    [
        {"known_on": "2024-01-09"},
        {"effective_on": "2024-01-01"},
        {"effective_on": "2024-04-01"},
        {"action": "sell"},
        {"known_on": None},
        {"ticker": "A/B"},
        {"extra": 2},
        {"action": []},
    ],
)
def test_invalid_events_rejected(manifest, prices, tmp_path, changes):
    manifest["events"][0].update(changes)
    with pytest.raises(PortfolioError):
        load_membership(write(tmp_path, manifest), prices.index, prices.columns)


@pytest.mark.parametrize(
    "event",
    [
        {"ticker": "BBB", "known_on": "2024-01-02", "effective_on": "2024-01-06", "action": "remove"},
        {"ticker": "AAA", "known_on": "2024-01-02", "effective_on": "2024-01-05", "action": "add"},
        {"ticker": "BBB", "known_on": "2024-01-02", "effective_on": "2024-01-05", "action": "remove"},
    ],
)
def test_conflicting_or_redundant_events_rejected(manifest, prices, tmp_path, event):
    manifest["events"].append(event)
    with pytest.raises(PortfolioError):
        load_membership(write(tmp_path, manifest), prices.index, prices.columns)


def test_hashes_relocation_raw_format_and_limits(manifest, prices, tmp_path, monkeypatch):
    path = write(tmp_path, manifest)
    _, original = load_membership(path, prices.index, prices.columns)
    moved = tmp_path / "moved.json"
    moved.write_bytes(path.read_bytes())
    assert load_membership(moved, prices.index, prices.columns)[1] == original
    moved.write_text(json.dumps(manifest, indent=2))
    reformatted = load_membership(moved, prices.index, prices.columns)[1]
    assert original["canonical_sha256"] == reformatted["canonical_sha256"]
    assert original["manifest_sha256"] != reformatted["manifest_sha256"]
    monkeypatch.setattr(universe, "MAX_UNIVERSE_BYTES", 5)
    with pytest.raises(PortfolioError, match="limit"):
        load_membership(path, prices.index, prices.columns)
    monkeypatch.setattr(universe, "MAX_UNIVERSE_BYTES", 2 * 1024 * 1024)
    monkeypatch.setattr(universe, "MAX_UNIVERSE_EVENTS", 1)
    with pytest.raises(PortfolioError, match="events"):
        load_membership(path, prices.index, prices.columns)


@pytest.mark.parametrize("raw", ["[]", "null", "{", '{"source":"a","source":"b"}', "{}"])
def test_malformed_json_fails_closed(prices, tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_text(raw)
    with pytest.raises(PortfolioError):
        load_membership(path, prices.index, prices.columns)


def test_missing_file_unknown_ticker_and_insufficient_coverage(manifest, prices, tmp_path):
    with pytest.raises(PortfolioError, match="Cannot read"):
        load_membership(tmp_path / "missing.json", prices.index, prices.columns)
    path = write(tmp_path, manifest)
    with pytest.raises(PortfolioError, match="supplied"):
        load_membership(path, prices.index, ["AAA", "CCC"])
    with pytest.raises(PortfolioError, match="coverage"):
        load_membership(path, pd.date_range("2023-12-01", periods=2), prices.columns)


@pytest.mark.parametrize(
    "index",
    [
        pd.DatetimeIndex([]),
        pd.DatetimeIndex(["2024-01-02", "2024-01-02"]),
        pd.DatetimeIndex(["2024-01-03", "2024-01-02"]),
        pd.DatetimeIndex(["2024-01-02 12:00"]),
        pd.DatetimeIndex(["2024-01-02"], tz="UTC"),
        pd.Index(["2024-01-02"]),
    ],
)
def test_bad_session_index_rejected(manifest, tmp_path, index):
    with pytest.raises(PortfolioError, match="daily session"):
        load_membership(write(tmp_path, manifest), index, ["AAA", "BBB"])


@pytest.mark.parametrize("kind", ["numeric", "missing", "wrong_dates", "wrong_assets", "duplicate_assets"])
def test_eligibility_is_exact_complete_boolean(prices, kind):
    mask = prices > 0
    if kind == "numeric":
        mask = mask.astype(int)
    elif kind == "missing":
        mask = mask.astype("boolean")
        mask.iloc[0, 0] = pd.NA
    elif kind == "wrong_dates":
        mask = mask.iloc[1:]
    elif kind == "wrong_assets":
        mask.columns = ["AAA", "CCC"]
    else:
        mask.columns = ["AAA", "aaa"]
    with pytest.raises(PortfolioError):
        cross_sectional_momentum(prices, eligibility=mask, lookback=2)


def test_inactive_asset_still_requires_real_prices(prices):
    mask = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    broken = prices.astype(float)
    broken.iloc[5, 1] = np.nan
    with pytest.raises(PortfolioError, match="finite and positive"):
        cross_sectional_momentum(broken, eligibility=mask, lookback=2)


@pytest.mark.parametrize("invalid", [b"\xff", b'{"source":"\\ud800"}'])
def test_invalid_unicode_is_rejected_as_a_portfolio_error(manifest, prices, tmp_path, invalid):
    path = tmp_path / "unicode.json"
    if invalid.startswith(b"{"):
        manifest["source"] = "\ud800"
        path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        path.write_bytes(invalid)
    with pytest.raises(PortfolioError):
        load_membership(path, prices.index, prices.columns)


def test_membership_work_limit_is_checked_before_matrix_allocation(manifest, prices, tmp_path, monkeypatch):
    monkeypatch.setattr(universe, "MAX_PORTFOLIO_CELLS", 1)
    with pytest.raises(PortfolioError, match="work limit"):
        load_membership(write(tmp_path, manifest), prices.index, prices.columns)
