"""Strict offline CSV contracts and integration with sealed research."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from momentum_lab import DatasetError, SearchConfig, StudyRegistry, TestReuseError, cli, data, search
from momentum_lab import datasets as ds
from momentum_lab.reporting import render_html_report, render_markdown_report

FIXTURE = Path(__file__).parent / "fixtures" / "daily_ohlcv_v1.csv"
DECLARATIONS = {
    "ticker": "SYNTHETIC",
    "source": "Project-generated synthetic software fixture",
    "license": "MIT; synthetic software tests only",
    "currency": "USD",
    "calendar": "exchange",
    "price_adjustment": "split_and_dividend_adjusted",
}


@pytest.fixture(autouse=True)
def no_downloads(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Offline datasets must never access Yahoo downloads or caches")

    monkeypatch.setattr(data, "download_data", forbidden)


@pytest.fixture
def snapshot(tmp_path):
    return ds.import_dataset(FIXTURE, tmp_path / "snapshot", **DECLARATIONS)


def rewrite_manifest(path, **changes):
    value = json.loads(path.read_text())
    value.update(changes)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def test_frozen_csv_preserves_bytes_values_dates_and_hash(snapshot):
    original = FIXTURE.read_bytes()
    assert hashlib.sha256(original).hexdigest() == "1180425f47b28c14a2f7bc61ff4bd80be17415dbd5e4cddbd5c4e365d2d8aeb0"
    assert (snapshot.parent / "prices.csv").read_bytes() == original
    frame, provenance = ds.load_dataset(snapshot, ticker="synthetic")
    index = pd.DatetimeIndex(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"], name="date"
    )
    expected = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105],
            "high": [102, 103, 104, 105, 106, 107],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [101, 102, 103, 104, 105, 106],
            "volume": [1000, 1200, 0, 1400, 1500, 1600],
        },
        index=index,
        dtype=float,
    )
    pd.testing.assert_frame_equal(frame, expected)
    assert provenance["sha256"] == hashlib.sha256(original).hexdigest()
    assert provenance["rows"] == 6 and provenance["has_volume"] is True
    assert provenance["annualization"] == 252
    assert provenance["first_date"] == "2024-01-02" and provenance["last_date"] == "2024-01-09"
    assert provenance["provider"] == "local_csv" and len(provenance["contract_sha256"]) == 64
    assert "csv_file" not in provenance


def test_copy_and_json_formatting_do_not_change_contract(snapshot, tmp_path):
    _, before = ds.load_dataset(snapshot)
    moved = tmp_path / "moved"
    shutil.copytree(snapshot.parent, moved)
    rewrite_manifest(moved / "manifest.json", annualization=252.0, csv_file="renamed.csv")
    (moved / "prices.csv").rename(moved / "renamed.csv")
    _, after = ds.load_dataset(moved / "manifest.json")
    assert before == after


def test_original_export_can_disappear_after_import(tmp_path):
    original = tmp_path / "original.csv"
    original.write_bytes(FIXTURE.read_bytes())
    snapshot = ds.import_dataset(original, tmp_path / "copied", **DECLARATIONS)
    original.unlink()
    frame, _ = ds.load_dataset(snapshot)
    assert frame["close"].iloc[-1] == 106


def test_full_precision_decimal_parsing(tmp_path):
    original = tmp_path / "precise.csv"
    original.write_text("date,open,high,low,close\n2024-01-02,100,101,99,100.00000000000001\n")
    manifest = ds.import_dataset(original, tmp_path / "snapshot", **DECLARATIONS)
    frame, _ = ds.load_dataset(manifest)
    assert frame["close"].iloc[0] == float("100.00000000000001")
    assert frame["close"].iloc[0] > 100.0


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", True),
        ("schema_version", 99),
        ("ticker", "spy"),
        ("ticker", "A/B"),
        ("dataset_id", "a/b"),
        ("frequency", "1h"),
        ("calendar", "guess"),
        ("source", ""),
        ("source", None),
        ("source", " whitespace "),
        ("source", "line\nbreak"),
        ("source", "\ud800"),
        ("source", "x" * 2049),
        ("license", ""),
        ("currency", "usd"),
        ("price_adjustment", "unknown"),
        ("annualization", 0),
        ("annualization", -1),
        ("annualization", True),
        ("annualization", "252"),
        ("annualization", float("nan")),
        ("annualization", float("inf")),
        ("annualization", 10**400),
        ("csv_file", "../escape.csv"),
        ("csv_file", "/tmp/escape.csv"),
        ("csv_file", "C:\\escape.csv"),
        ("csv_file", "https://example.com/a.csv"),
        ("csv_file", "prices.json"),
        ("sha256", "bad"),
    ],
)
def test_invalid_manifest_declarations_fail_closed(snapshot, key, value):
    rewrite_manifest(snapshot, **{key: value})
    with pytest.raises(DatasetError):
        ds.load_dataset(snapshot)


@pytest.mark.parametrize("payload", ["[]", "null", "{", '{"schema_version":1,"schema_version":1}', '{"surprise":1}'])
def test_invalid_json_schema_and_duplicate_fields(snapshot, payload):
    snapshot.write_text(payload)
    with pytest.raises(DatasetError):
        ds.load_dataset(snapshot)


def test_unknown_fields_and_missing_required_metadata(snapshot):
    original = snapshot.read_text()
    rewrite_manifest(snapshot, source_url="https://example.com")
    with pytest.raises(DatasetError, match="unknown"):
        ds.load_dataset(snapshot)
    manifest = json.loads(original)
    del manifest["license"]
    snapshot.write_text(json.dumps(manifest))
    with pytest.raises(DatasetError, match="missing"):
        ds.load_dataset(snapshot)


def test_ticker_mismatch_and_missing_files(snapshot, tmp_path):
    with pytest.raises(DatasetError, match="ticker"):
        ds.load_dataset(snapshot, ticker="OTHER")
    with pytest.raises(DatasetError, match="Cannot read"):
        ds.load_dataset(tmp_path / "absent.json")
    (snapshot.parent / "prices.csv").unlink()
    with pytest.raises(DatasetError, match="Cannot read"):
        ds.load_dataset(snapshot)


def test_hash_checked_before_parsing(snapshot, monkeypatch):
    (snapshot.parent / "prices.csv").write_bytes(FIXTURE.read_bytes() + b"\n")
    monkeypatch.setattr(ds, "_parse_csv", lambda *args: pytest.fail("Checksum must precede parsing"))
    with pytest.raises(DatasetError, match="SHA-256 mismatch"):
        ds.load_dataset(snapshot)


def test_path_escape_via_symlink_rejected(snapshot, tmp_path):
    target = tmp_path / "outside.csv"
    target.write_bytes(FIXTURE.read_bytes())
    (snapshot.parent / "prices.csv").unlink()
    (snapshot.parent / "prices.csv").symlink_to(target)
    with pytest.raises(DatasetError, match="symbolic link"):
        ds.load_dataset(snapshot)


def test_bounded_file_reads(snapshot, monkeypatch):
    monkeypatch.setattr(ds, "MAX_MANIFEST_BYTES", 10)
    with pytest.raises(DatasetError, match="byte limit"):
        ds.load_dataset(snapshot)
    monkeypatch.setattr(ds, "MAX_MANIFEST_BYTES", 65536)
    monkeypatch.setattr(ds, "MAX_CSV_BYTES", 10)
    with pytest.raises(DatasetError, match="byte limit"):
        ds.load_dataset(snapshot)


HEADER = "date,open,high,low,close,volume\n"
VALID_ROW = "2024-01-02,100,102,99,101,1000\n"


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\xff",
        HEADER.encode(),
        b"date,close\n2024-01-02,100\n",
        b"date,open,high,low,close,adj_close\n2024-01-02,100,102,99,101,101\n",
        b"date,open,high,low,close, Close \n2024-01-02,100,102,99,101,101\n",
        (HEADER + VALID_ROW.replace(",1000", ",1000,extra")).encode(),
        (HEADER + VALID_ROW.replace(",1000", "")).encode(),
        (HEADER + VALID_ROW + "\n").encode(),
        (HEADER + VALID_ROW.replace("100,", "bad,", 1)).encode(),
        (HEADER + VALID_ROW.replace("100,", "0,", 1)).encode(),
        (HEADER + VALID_ROW.replace("100,", "-1,", 1)).encode(),
        (HEADER + VALID_ROW.replace("101,", "NaN,")).encode(),
        (HEADER + VALID_ROW.replace("100,", "inf,", 1)).encode(),
        (HEADER + VALID_ROW.replace("102,", "100,")).encode(),
        (HEADER + VALID_ROW.replace("99,", "101,")).encode(),
        (HEADER + VALID_ROW.replace("1000", "-1")).encode(),
        (HEADER + VALID_ROW.replace("1000", "NaN")).encode(),
        (HEADER + VALID_ROW.replace("1000", "inf")).encode(),
        (HEADER + VALID_ROW.replace("1000", "")).encode(),
        (HEADER + VALID_ROW.replace("100,", "100\x00bad,", 1)).encode(),
        (HEADER + VALID_ROW.replace("2024-01-02", "2024-02-30")).encode(),
        (HEADER + VALID_ROW.replace("2024-01-02", "2024-1-2")).encode(),
        (HEADER + VALID_ROW.replace("2024-01-02", "2024-01-02T00:00:00Z")).encode(),
        (HEADER + VALID_ROW + VALID_ROW).encode(),
        (HEADER + VALID_ROW.replace("2024-01-02", "2024-01-03") + VALID_ROW).encode(),
    ],
)
def test_invalid_csv_is_never_repaired_or_imported(tmp_path, payload):
    original = tmp_path / "bad.csv"
    original.write_bytes(payload)
    output = tmp_path / "rejected"
    with pytest.raises(DatasetError):
        ds.import_dataset(original, output, **DECLARATIONS)
    assert not output.exists()
    assert original.read_bytes() == payload


def test_valid_utf8_bom_and_no_volume(tmp_path):
    original = tmp_path / "no-volume.csv"
    original.write_bytes(b"\xef\xbb\xbfDATE, OPEN , HIGH , LOW , CLOSE\r\n2024-01-02,100,102,99,101\r\n")
    manifest = ds.import_dataset(original, tmp_path / "snapshot", **DECLARATIONS)
    frame, provenance = ds.load_dataset(manifest)
    assert list(frame.columns) == ["open", "high", "low", "close"]
    assert provenance["has_volume"] is False


def test_new_directory_only_and_existing_data_preserved(snapshot):
    before = {path.name: path.read_bytes() for path in snapshot.parent.iterdir()}
    with pytest.raises(DatasetError, match="new directory"):
        ds.import_dataset(FIXTURE, snapshot.parent, **DECLARATIONS)
    assert {path.name: path.read_bytes() for path in snapshot.parent.iterdir()} == before


def test_dates_are_inclusive_end_none_is_snapshot_not_today(snapshot):
    frame, provenance = ds.load_dataset(snapshot, start="2024-01-03", end="2024-01-05")
    assert list(frame.index.day) == [3, 4, 5]
    assert provenance["rows"] == 6  # Provenance identifies the complete source.
    frame, _ = ds.load_dataset(snapshot, start="2024-01-08")
    assert list(frame.index.day) == [8, 9]
    with pytest.warns(RuntimeWarning, match="later than requested start"):
        frame, _ = ds.load_dataset(snapshot, start="2020-01-01")
    assert len(frame) == 6


@pytest.mark.parametrize(
    "bounds",
    [
        {"end": "2024-01-10"},
        {"start": "2024-01-05", "end": "2024-01-03"},
        {"start": "2024-01-06", "end": "2024-01-07"},
        {"start": "NaT"},
        {"start": "not-a-date"},
        {"end": "2024-01-09T12:00:00"},
        {"end": "2024-01-09T00:00:00Z"},
    ],
)
def test_invalid_or_uncovered_ranges(snapshot, bounds):
    with pytest.raises(DatasetError):
        ds.load_dataset(snapshot, **bounds)


def test_invalid_row_outside_requested_slice_is_not_ignored(snapshot):
    payload = FIXTURE.read_bytes().replace(b"2024-01-09,105", b"2024-01-09,0")
    (snapshot.parent / "prices.csv").write_bytes(payload)
    rewrite_manifest(snapshot, sha256=hashlib.sha256(payload).hexdigest())
    with pytest.raises(DatasetError, match="non-positive"):
        ds.load_dataset(snapshot, end="2024-01-05")


@pytest.mark.parametrize("adjustment", ["unadjusted", "split_adjusted"])
def test_non_total_return_data_warns_without_changing_prices(snapshot, adjustment):
    rewrite_manifest(snapshot, price_adjustment=adjustment)
    with pytest.warns(RuntimeWarning, match="not reconstructed"):
        frame, _ = ds.load_dataset(snapshot)
    assert frame["close"].tolist() == [101, 102, 103, 104, 105, 106]


def test_declared_calendar_controls_features_and_rejects_conflicting_override(snapshot):
    rewrite_manifest(snapshot, calendar="continuous", annualization=365)
    prepared, frame = data.prepare_data("SYNTHETIC", start="2024-01-02", dataset=snapshot)
    assert prepared["annualization"] == 365
    pd.testing.assert_frame_equal(prepared["features"], data.compute_features(frame, annualization=365))
    for annualization in (252, True, float("nan"), "365"):
        with pytest.raises(DatasetError, match="annualization conflicts"):
            data.prepare_data("SYNTHETIC", start="2024-01-02", dataset=snapshot, annualization=annualization)


@pytest.fixture
def research_dataset(tmp_path, monkeypatch):
    rng = np.random.Generator(np.random.PCG64(771))
    index = pd.date_range("2021-01-04", periods=500, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.012, len(index))))
    frame = pd.DataFrame(
        {"open": close * 0.999, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 100000.0},
        index=index,
    )
    original = tmp_path / "research.csv"
    frame.to_csv(original, index_label="date")
    manifest = ds.import_dataset(original, tmp_path / "research-data", **DECLARATIONS)
    params = [
        {"lookback": size, "threshold": 0.002, "long_short": True, "skip_recent": 1, "signal_smooth": 0}
        for size in (12, 24)
    ]
    monkeypatch.setattr(search, "_quick_sample", lambda *args, **kwargs: params)
    return SearchConfig(
        ticker="SYNTHETIC",
        dataset=str(manifest),
        start="2021-01-04",
        strategies=["tsmom"],
        robust=False,
        bootstrap_resamples=200,
        result_dir=str(tmp_path / "runs"),
        study_id="local-study",
        run_id="local-run",
    )


def test_real_offline_search_seals_test_and_records_provenance(research_dataset, monkeypatch):
    original = search.run_single_experiment
    calls = []

    def candidate(strategy, params, supplied, frame, *args, **kwargs):
        assert "data_provenance" not in supplied and not frame.attrs
        assert len(frame) == 400
        calls.append(1)
        return original(strategy, params, supplied, frame, *args, **kwargs)

    monkeypatch.setattr(search, "run_single_experiment", candidate)
    sealed = search.run_search(config=research_dataset)
    assert calls and sealed["test_access"]["status"] == "sealed"
    assert "test_metrics" not in sealed["best"]
    assert set(sealed["bootstrap_diagnostics"]["periods"]) == {"validation"}
    metadata = json.loads((Path(sealed["result_dir"]) / "run_config.json").read_text())
    assert metadata["data_provenance"] == sealed["data_provenance"]
    assert metadata["data_provenance"]["rows"] == 500
    with StudyRegistry(create=False)._connect() as connection:
        protocol_json = connection.execute(
            "SELECT protocol_json FROM studies WHERE study_id=?", ("local-study",)
        ).fetchone()[0]
    assert json.loads(protocol_json)["data_provenance"] == sealed["data_provenance"]
    for filename in ("report.html", "report.md"):
        report = (Path(sealed["result_dir"]) / filename).read_text()
        assert "Data provenance" in report and "Raw CSV SHA-256" in report
        assert sealed["data_provenance"]["contract_sha256"] in report


def test_relocated_snapshot_resume_and_cached_reveal(research_dataset, tmp_path, monkeypatch):
    sealed = search.run_search(config=research_dataset)
    moved = tmp_path / "relocated"
    shutil.copytree(Path(research_dataset.dataset).parent, moved)
    research_dataset.dataset = str(moved / "manifest.json")
    revealed = search.run_search(config=research_dataset, resume=True, reveal_test=True)
    assert revealed["test_access"]["status"] == "first_recorded_reveal"
    assert revealed["data_provenance"] == sealed["data_provenance"]
    monkeypatch.setattr(search, "_test_payload", lambda *args: pytest.fail("Must reuse cached result"))
    replay = search.run_search(config=research_dataset, resume=True, reveal_test=True)
    assert replay["test_access"]["status"] == "previously_revealed"
    assert replay["best"]["test_evaluated_at"] == revealed["best"]["test_evaluated_at"]


@pytest.mark.parametrize("change", ["source", "license", "prices", "unused_rows", "bytes_only"])
def test_changed_bytes_or_declarations_reject_resume_without_overwriting(research_dataset, change):
    if change == "unused_rows":
        frame, _ = ds.load_dataset(research_dataset.dataset)
        research_dataset.end = str(frame.index[-5].date())
    first = search.run_search(config=research_dataset)
    path = Path(first["result_dir"]) / "run_config.json"
    previous = path.read_bytes()
    manifest = Path(research_dataset.dataset)
    if change in {"source", "license"}:
        rewrite_manifest(manifest, **{change: "A different declaration"})
    else:
        csv_path = manifest.parent / "prices.csv"
        if change == "bytes_only":
            payload = csv_path.read_bytes().replace(b"\n", b"\r\n")
        else:
            lines = csv_path.read_bytes().splitlines(keepends=True)
            last_row = lines[-1].decode().strip().split(",")
            last_row[1:5] = [str(float(value) * 1.001) for value in last_row[1:5]]
            lines[-1] = (",".join(last_row) + "\n").encode()
            payload = b"".join(lines)
        csv_path.write_bytes(payload)
        rewrite_manifest(manifest, sha256=hashlib.sha256(payload).hexdigest())
        if change in {"unused_rows", "bytes_only"}:
            selected, _ = ds.load_dataset(manifest, start=research_dataset.start, end=research_dataset.end)
            assert search._data_snapshot(selected) == json.loads(previous)["data_snapshot"]
    with pytest.raises(ValueError, match="resume configuration mismatch.*data_provenance"):
        search.run_search(config=research_dataset, resume=True)
    assert path.read_bytes() == previous


def test_another_source_with_same_ticker_does_not_erase_exposure(research_dataset):
    search.run_search(config=research_dataset)
    search.run_search(config=research_dataset, resume=True, reveal_test=True)
    rewrite_manifest(Path(research_dataset.dataset), source="Second synthetic source", dataset_id="second")
    research_dataset.study_id, research_dataset.run_id = "other-study", "other-run"
    sealed = search.run_search(config=research_dataset)
    assert sealed["test_access"]["status"] == "known_prior_exposure"
    with pytest.raises(TestReuseError):
        search.run_search(config=research_dataset, resume=True, reveal_test=True)


def test_dataset_config_path_is_relative_to_json_not_cwd(snapshot, tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"ticker": "SYNTHETIC", "dataset": "snapshot/manifest.json"}))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = SearchConfig.from_json(config_path)
    assert config.dataset == str(snapshot.resolve())
    assert SearchConfig.from_mapping({"dataset": "snapshot/manifest.json"}).dataset == "snapshot/manifest.json"
    config_path.write_text('{"dataset": ""}')
    with pytest.raises(ValueError, match="non-empty"):
        SearchConfig.from_json(config_path)


def test_cli_import_inspect_and_dispatch(tmp_path, monkeypatch, capsys):
    args = [
        "momentum-lab",
        "data",
        "import",
        str(FIXTURE),
        "--output",
        str(tmp_path / "cli-data"),
        "--ticker",
        "SYNTHETIC",
        "--source",
        "Synthetic test",
        "--license",
        "MIT",
        "--currency",
        "USD",
        "--calendar",
        "exchange",
        "--price-adjustment",
        "split_and_dividend_adjusted",
    ]
    monkeypatch.setattr("sys.argv", args)
    assert cli.main() == 0
    assert "Created dataset" in capsys.readouterr().out
    manifest = tmp_path / "cli-data" / "manifest.json"
    monkeypatch.setattr("sys.argv", ["momentum-lab", "data", "inspect", str(manifest)])
    assert cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["rows"] == 6 and "close" not in output and "test_metrics" not in output
    captured = {}
    monkeypatch.setattr(cli, "run_search", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr("sys.argv", ["momentum-lab", "SYNTHETIC", "--dataset", str(manifest)])
    cli.main()
    assert captured["dataset"] == str(manifest)


def test_cli_errors_are_actionable_not_tracebacks(tmp_path, monkeypatch, capsys):
    missing = str(tmp_path / "missing.json")
    for args in (["data", "inspect", missing], ["SYNTHETIC", "--dataset", missing]):
        monkeypatch.setattr("sys.argv", ["momentum-lab", *args])
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 2
        assert "Cannot read dataset" in capsys.readouterr().err


def test_reports_escape_declarations_and_support_missing_provenance():
    config = {"data_provenance": {"source": "<script>alert(1)</script>|unsafe", "price_adjustment": "unadjusted"}}
    for render in (render_markdown_report, render_html_report):
        report = render({}, config)
        assert "<script>" not in report and "&lt;script&gt;" in report
        assert "artificial jumps" in report and "not independently verified" in report
        assert "Unavailable in this run" in render({}, {})


def test_liquidity_search_requires_supplied_volume(research_dataset, tmp_path):
    frame, _ = ds.load_dataset(research_dataset.dataset)
    original = tmp_path / "prices-without-volume.csv"
    frame.drop(columns="volume").to_csv(original)
    research_dataset.dataset = str(ds.import_dataset(original, tmp_path / "without-volume", **DECLARATIONS))
    research_dataset.max_participation = 0.01
    with pytest.raises(ValueError, match="requires a 'volume' column"):
        search.run_search(config=research_dataset)
    assert StudyRegistry(create=False).list_studies() == []


def test_yahoo_provenance_does_not_invent_csv_hashes(monkeypatch):
    frame = ds._parse_csv(FIXTURE.read_bytes())
    monkeypatch.setattr(data, "download_data", lambda *args, **kwargs: frame)
    prepared, _ = data.prepare_data("BTC-USD")
    provenance = prepared["data_provenance"]
    assert provenance["provider"] == "yahoo" and provenance["price_adjustment"] == "yfinance_auto_adjust"
    assert provenance["annualization"] == 365 and provenance["calendar"] == "continuous"
    assert "sha256" not in provenance and "contract_sha256" not in provenance
