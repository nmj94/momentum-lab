"""Structural diagnostics must not become another route to strategy/test scores."""

import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from momentum_lab import (
    DatasetError,
    PortfolioConfig,
    cli,
    data,
    datasets,
    governance,
    portfolio_research,
    preflight,
    preflight_dataset,
    preflight_portfolio,
    search,
    write_preflight_report,
)


@pytest.fixture(autouse=True)
def structural_only(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Preflight must not download, evaluate, reserve or create operational state")

    monkeypatch.setattr(data.yf, "download", forbidden)
    monkeypatch.setattr(search, "run_search", forbidden)
    monkeypatch.setattr(portfolio_research, "_compute_books", forbidden)
    monkeypatch.setattr(governance.StudyRegistry, "__init__", forbidden)
    monkeypatch.setattr(portfolio_research.RunSession, "__init__", forbidden)
    monkeypatch.setenv("MOMENTUM_LAB_REGISTRY_PATH", str(tmp_path / "unused-registry.sqlite3"))


def snapshot(tmp_path, ticker="AAA", *, dates=None, value=987654.321, **declarations):
    dates = pd.date_range("2024-01-02", periods=6, freq="B") if dates is None else pd.DatetimeIndex(dates)
    frame = pd.DataFrame({name: value for name in ("open", "high", "low", "close")}, index=dates)
    csv = tmp_path / f"{ticker}.csv"
    frame.to_csv(csv, index_label="date")
    return datasets.import_dataset(
        csv,
        tmp_path / ticker,
        ticker=ticker,
        source="PRIVATE_PROVIDER_DECLARATION",
        license="PRIVATE_LICENSE",
        currency=declarations.get("currency", "USD"),
        calendar=declarations.get("calendar", "exchange"),
        price_adjustment=declarations.get("price_adjustment", "split_and_dividend_adjusted"),
        annualization=declarations.get("annualization"),
    )


def calendar(tmp_path, **changes):
    obj = {
        "schema_version": 1,
        "calendar_id": "synthetic-sessions-v1",
        "source": "synthetic fixture",
        "license": "MIT",
        "coverage_start": "2024-01-01",
        "coverage_end": "2024-01-10",
        "sessions": [str(value.date()) for value in pd.date_range("2024-01-02", periods=6, freq="B")],
        **changes,
    }
    path = tmp_path / "sessions.json"
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def codes(report):
    return [issue["code"] for issue in report["issues"]]


def files_under(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_exact_calendar_matches_dates_and_never_exports_prices_or_private_declarations(tmp_path):
    manifest, sessions = snapshot(tmp_path), calendar(tmp_path)
    before = files_under(tmp_path)
    report = preflight_dataset(manifest, sessions=sessions)
    assert report["status"] == "passed" and report["error_count"] == report["warning_count"] == 0
    assert report["assets"]["AAA"]["selected_rows"] == 6
    assert report["session_calendar"]["manifest_sha256"] == hashlib.sha256(sessions.read_bytes()).hexdigest()
    assert report["report_sha256"] == preflight._digest({k: v for k, v in report.items() if k != "report_sha256"})
    assert files_under(tmp_path) == before
    encoded = json.dumps(report)
    for secret in ("987654", "PRIVATE_", str(tmp_path), "test_metrics", "sharpe"):
        assert secret not in encoded


def test_exchange_without_explicit_calendar_never_claims_verified_completeness(tmp_path):
    report = preflight_dataset(snapshot(tmp_path))
    assert report["status"] == "warning"
    assert codes(report) == ["calendar_unverified"]


@pytest.mark.parametrize("omit", [[], [2]])
def test_continuous_calendar_counts_days_without_exchange_assumptions(tmp_path, omit):
    dates = pd.date_range("2024-01-01", periods=7).delete(omit)
    report = preflight_dataset(snapshot(tmp_path, dates=dates, calendar="continuous"))
    assert report["status"] == ("error" if omit else "passed")
    if omit:
        issue = report["issues"][0]
        assert issue["code"] == "missing_sessions" and issue["sample_dates"] == ["2024-01-03"]


def test_custom_calendar_cannot_override_continuous_declaration(tmp_path):
    manifest = snapshot(tmp_path, calendar="continuous")
    report = preflight_dataset(manifest, sessions=calendar(tmp_path))
    assert "calendar_declaration_conflict" in codes(report) and report["status"] == "error"


def test_holiday_is_not_fabricated_and_missing_session_is_exact(tmp_path):
    sessions = calendar(tmp_path)
    dates = pd.date_range("2024-01-02", periods=6, freq="B").delete(2)
    manifest = snapshot(tmp_path, dates=dates)
    report = preflight_dataset(manifest, start="2024-01-01", sessions=sessions)
    assert codes(report) == ["late_start", "missing_sessions"]
    assert report["issues"][1]["sample_dates"] == ["2024-01-04"]
    assert "2024-01-01" not in report["issues"][1]["sample_dates"]


def test_unexpected_sessions_and_output_samples_are_bounded(tmp_path):
    dates = pd.date_range("2024-01-01", periods=50)
    manifest = snapshot(tmp_path, dates=dates)
    sessions = calendar(tmp_path, coverage_end="2024-03-01", sessions=["2024-01-02"])
    report = preflight_dataset(manifest, sessions=sessions)
    issue = next(issue for issue in report["issues"] if issue["code"] == "unexpected_sessions")
    assert issue["count"] == 49 and len(issue["sample_dates"]) == 10


@pytest.mark.parametrize(
    "changes,code",
    [
        ({"start": "2024-01-01"}, "late_start"),
        ({"end": "2024-01-10"}, "truncated_end"),
        ({"start": "2024-02-01"}, "empty_range"),
        ({"end": "2023-12-31"}, "empty_range"),
    ],
)
def test_range_diagnostics_do_not_clip_silently(tmp_path, changes, code):
    report = preflight_dataset(snapshot(tmp_path), **changes)
    assert code in codes(report)


def test_calendar_coverage_must_cover_the_requested_range_not_only_existing_bars(tmp_path):
    report = preflight_dataset(snapshot(tmp_path), start="2023-12-31", sessions=calendar(tmp_path))
    assert "calendar_coverage" in codes(report)


@pytest.mark.parametrize(
    "changes",
    [
        {"start": "2024-01-10", "end": "2024-01-01"},
        {"start": "2024-1-2"},
        {"start": True},
        {"end": "2024-01-01T00:00:00Z"},
        {"start": "2024-02-31"},
    ],
)
def test_invalid_invocation_dates_raise_before_loading_files(changes, monkeypatch):
    monkeypatch.setattr(preflight, "load_dataset", lambda *a, **k: pytest.fail("No reads"))
    with pytest.raises((DatasetError, ValueError)):
        preflight_dataset("not-read.json", **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"extra": 1},
        {"calendar_id": ""},
        {"source": " bad"},
        {"license": "x\n"},
        {"source": "\ud800"},
        {"sessions": []},
        {"sessions": "2024-01-01"},
        {"sessions": ["2024-01-03", "2024-01-02"]},
        {"sessions": ["2024-01-02", "2024-01-02"]},
        {"sessions": ["2024-01-11"]},
        {"sessions": ["2024-01-02T00:00:00"]},
        {"coverage_start": "2024-01-11"},
    ],
)
def test_invalid_calendars_are_rejected_without_echoing_file_values(tmp_path, changes):
    path = calendar(tmp_path, **changes)
    with pytest.raises(DatasetError, match="Invalid session calendar"):
        preflight_dataset("unused.json", sessions=path)


@pytest.mark.parametrize("payload", [b"[]", b"\xff", b'{"schema_version":1,"schema_version":1}', b"BAD_PRICE_SECRET"])
def test_malformed_calendar_cannot_leak_its_contents(tmp_path, payload):
    path = tmp_path / "sessions.json"
    path.write_bytes(payload)
    with pytest.raises(DatasetError) as exc:
        preflight_dataset("unused.json", sessions=path)
    assert "BAD_PRICE_SECRET" not in str(exc.value)


def test_calendar_size_and_count_limits(tmp_path, monkeypatch):
    path = calendar(tmp_path)
    monkeypatch.setattr(preflight, "MAX_SESSION_BYTES", 10)
    with pytest.raises(DatasetError):
        preflight_dataset("unused", sessions=path)
    monkeypatch.setattr(preflight, "MAX_SESSION_BYTES", 10000)
    monkeypatch.setattr(preflight, "MAX_SESSIONS", 2)
    with pytest.raises(DatasetError):
        preflight_dataset("unused", sessions=path)


def test_calendar_symlink_is_rejected(tmp_path):
    path = calendar(tmp_path)
    alias = tmp_path / "alias.json"
    try:
        alias.symlink_to(path)
    except OSError:
        pytest.skip("Symlinks require OS permission")
    with pytest.raises(DatasetError):
        preflight_dataset("unused.json", sessions=alias)


@pytest.mark.parametrize("damage", ["hash", "manifest", "rows", "missing", "outside_slice"])
def test_invalid_snapshot_is_reported_without_any_price_echo(tmp_path, damage):
    manifest = snapshot(tmp_path)
    prices = manifest.parent / "prices.csv"
    if damage == "manifest":
        manifest.write_text("PRICE_SECRET", encoding="utf-8")
    elif damage == "missing":
        prices.unlink()
    else:
        prices.write_text(prices.read_text().replace("987654.321", "PRICE_SECRET", 1), encoding="utf-8")
        if damage in {"rows", "outside_slice"}:
            obj = json.loads(manifest.read_text())
            obj["sha256"] = hashlib.sha256(prices.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(obj), encoding="utf-8")
    report = preflight_dataset(manifest, start="2024-01-05" if damage == "outside_slice" else None)
    assert report["status"] == "error" and codes(report) == ["invalid_dataset"]
    assert "PRICE_SECRET" not in json.dumps(report)


@pytest.mark.parametrize("adjustment", ["unadjusted", "split_adjusted"])
def test_adjustment_warnings_are_structured_without_data_warnings(tmp_path, adjustment):
    report = preflight_dataset(snapshot(tmp_path, price_adjustment=adjustment), sessions=calendar(tmp_path))
    assert report["status"] == "warning" and codes(report) == ["adjustment_risk"]


def recipe(tmp_path, **second_options):
    first = snapshot(tmp_path)
    second = snapshot(tmp_path, "BBB", value=123.456, **second_options)
    return PortfolioConfig(
        datasets={"BBB": str(second), "AAA": str(first)}, lookback=2, result_dir=str(tmp_path / "must-not-create")
    )


def test_portfolio_preflight_is_deterministic_portable_and_read_only(tmp_path):
    config, sessions = recipe(tmp_path), calendar(tmp_path)
    original = copy.deepcopy(config)
    before = files_under(tmp_path)
    report = preflight_portfolio(config, sessions=sessions)
    assert report["status"] == "passed" and list(report["assets"]) == ["AAA", "BBB"]
    assert config == original and files_under(tmp_path) == before
    copied = tmp_path / "copy"
    copied.mkdir()
    for ticker in ("AAA", "BBB"):
        shutil.copytree(tmp_path / ticker, copied / ticker)
    moved = PortfolioConfig(
        datasets={ticker: str(copied / ticker / "manifest.json") for ticker in ("AAA", "BBB")}, lookback=2
    )
    assert preflight_portfolio(moved, sessions=sessions) == report
    moved.lookback = 3
    assert preflight_portfolio(moved, sessions=sessions)["checked_recipe_sha256"] != report["checked_recipe_sha256"]


@pytest.mark.parametrize(
    "change",
    [{"currency": "EUR"}, {"calendar": "continuous"}, {"annualization": 250}, {"price_adjustment": "unadjusted"}],
)
def test_all_portfolio_convention_mismatches_are_reported(tmp_path, change):
    report = preflight_portfolio(recipe(tmp_path, **change))
    assert report["status"] == "error" and "convention_mismatch" in codes(report)


def test_portfolio_does_not_hide_invalid_assets_or_intersect_sessions(tmp_path):
    config = recipe(tmp_path, dates=pd.date_range("2024-01-03", periods=5, freq="B"))
    report = preflight_portfolio(config)
    issue = next(issue for issue in report["issues"] if issue["code"] == "session_mismatch")
    assert issue["sample_dates"] == ["2024-01-02"]
    config.datasets["AAA"] = str(tmp_path / "missing.json")
    report = preflight_portfolio(config)
    assert report["assets"]["AAA"]["integrity"] == "invalid"
    assert report["assets"]["BBB"]["integrity"] == "verified"


def test_identical_bytes_warn_about_security_identity_without_asserting_it(tmp_path):
    config = recipe(tmp_path)
    first = Path(config.datasets["AAA"])
    second = Path(config.datasets["BBB"])
    (second.parent / "prices.csv").write_bytes((first.parent / "prices.csv").read_bytes())
    manifest = json.loads(second.read_text())
    manifest["sha256"] = json.loads(first.read_text())["sha256"]
    second.write_text(json.dumps(manifest), encoding="utf-8")
    report = preflight_portfolio(config, sessions=calendar(tmp_path))
    assert codes(report) == ["identical_price_files"] and report["status"] == "warning"


def test_portfolio_history_and_work_limits(tmp_path, monkeypatch):
    config = recipe(tmp_path)
    config.lookback = 5
    assert codes(preflight_portfolio(config)).count("insufficient_history") == 2
    monkeypatch.setattr(preflight, "MAX_PORTFOLIO_CELLS", 10)
    with pytest.raises(DatasetError, match="work limit"):
        preflight_portfolio(config)


def test_continuous_calendar_work_limit(tmp_path, monkeypatch):
    manifest = snapshot(tmp_path, calendar="continuous")
    monkeypatch.setattr(preflight, "MAX_SESSIONS", 2)
    assert "calendar_work_limit" in codes(preflight_dataset(manifest))


@pytest.mark.parametrize("case", ["valid", "invalid", "unaligned", "all_invalid"])
def test_membership_preflight_never_computes_signals(tmp_path, case):
    config = (
        recipe(tmp_path, dates=pd.date_range("2024-01-03", periods=6, freq="B"))
        if case == "unaligned"
        else recipe(tmp_path)
    )
    membership = tmp_path / "membership.json"
    membership.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "universe_id": "test-v1",
                "source": "private",
                "license": "MIT",
                "coverage_start": "2024-01-01",
                "coverage_end": "2024-02-01",
                "initial_known_on": "2024-01-01",
                "initial_members": ["AAA", "BBB"],
                "events": [],
            }
        ),
        encoding="utf-8",
    )
    config.universe = str(membership)
    if case == "invalid":
        membership.write_text("SECRET", encoding="utf-8")
    if case == "all_invalid":
        config.datasets = {ticker: str(tmp_path / "missing") for ticker in ("AAA", "BBB")}
    report = preflight_portfolio(config)
    if case == "valid":
        assert report["membership"]["event_count"] == 0
    else:
        assert report["membership"] is None
        assert ("invalid_membership" if case == "invalid" else "membership_not_checked") in codes(report)
    assert "SECRET" not in json.dumps(report)


def test_reports_are_non_overwriting_and_digest_checked(tmp_path):
    report = preflight_dataset(snapshot(tmp_path), sessions=calendar(tmp_path))
    path = write_preflight_report(report, tmp_path / "report")
    assert json.loads(path.read_text()) == report
    before = files_under(path.parent)
    with pytest.raises(DatasetError, match="new directory"):
        write_preflight_report(report, path.parent)
    assert files_under(path.parent) == before
    report["status"] = "error"
    with pytest.raises(DatasetError, match="digest"):
        write_preflight_report(report, tmp_path / "invalid-report")
    assert not (tmp_path / "invalid-report").exists()


def test_gap_report_lists_bounded_samples_and_retains_partial_exports(tmp_path, monkeypatch):
    manifest = snapshot(tmp_path, dates=pd.date_range("2024-01-02", periods=6, freq="B").delete(2))
    report = preflight_dataset(manifest, sessions=calendar(tmp_path))
    target = tmp_path / "gap-report"
    original = Path.open

    def fail_json(path, *args, **kwargs):
        if path == target / "report.json":
            raise OSError("injected write failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_json)
    with pytest.raises(DatasetError, match="new directory"):
        write_preflight_report(report, target)
    assert "2024-01-04" in (target / "report.md").read_text()
    assert not (target / "report.json").exists()
    monkeypatch.setattr(Path, "open", original)
    with pytest.raises(DatasetError, match="new directory"):
        write_preflight_report(report, target)


@pytest.mark.parametrize(
    "changes",
    [
        {"datasets": {"AAA": "unused"}},
        {"lookback": 0},
        {"datasets": {"AAA": "a", "aaa": "b"}},
        {"test_start": "2024-01-05"},
    ],
)
def test_invalid_or_study_recipes_fail_before_snapshot_reads(changes, monkeypatch):
    monkeypatch.setattr(preflight, "load_dataset", lambda *a, **k: pytest.fail("No dataset reads"))
    with pytest.raises(ValueError):
        preflight_portfolio({"datasets": {"AAA": "a", "BBB": "b"}, **changes})


@pytest.mark.parametrize("sessions,expected", [(False, 1), (True, 0)])
def test_cli_dataset_status_codes_and_pure_json(tmp_path, monkeypatch, capsys, sessions, expected):
    manifest = snapshot(tmp_path)
    argv = ["momentum-lab", "data", "check", str(manifest), "--output", str(tmp_path / "report")]
    if sessions:
        argv += ["--sessions", str(calendar(tmp_path))]
    monkeypatch.setattr(sys, "argv", argv)
    assert cli.main() == expected
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads((tmp_path / "report" / "report.json").read_text())


def test_cli_errors_are_json_for_bad_datasets_but_usage_errors_are_argparse(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["momentum-lab", "data", "check", str(tmp_path / "missing.json")])
    assert cli.main() == 2
    assert json.loads(capsys.readouterr().out)["status"] == "error"
    monkeypatch.setattr(sys, "argv", ["momentum-lab", "data", "check", "unused", "--output", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


def test_cli_portfolio_relative_paths_and_no_consent_or_registry(tmp_path, monkeypatch, capsys):
    config = recipe(tmp_path).to_dict()
    config["datasets"] = {ticker: f"{ticker}/manifest.json" for ticker in ("AAA", "BBB")}
    path = tmp_path / "portfolio.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.chdir(tmp_path.parent)
    monkeypatch.setattr(
        sys,
        "argv",
        ["momentum-lab", "portfolio", "preflight", "--config", str(path), "--sessions", str(calendar(tmp_path))],
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["workflow"] == "portfolio"
    assert not (tmp_path / "must-not-create").exists()
    assert not (tmp_path / "unused-registry.sqlite3").exists()


def test_direct_command_entrypoints_use_sys_argv(tmp_path, monkeypatch, capsys):
    config = recipe(tmp_path)
    monkeypatch.setattr(sys, "argv", ["data", "check", config.datasets["AAA"]])
    assert datasets.main() == 1
    assert json.loads(capsys.readouterr().out)["workflow"] == "dataset"
    path = tmp_path / "portfolio.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["portfolio", "preflight", "--config", str(path)])
    assert portfolio_research.main() == 1
    assert json.loads(capsys.readouterr().out)["workflow"] == "portfolio"
