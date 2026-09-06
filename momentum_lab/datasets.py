"""Offline daily OHLCV snapshots with explicit, checksum-bound declarations.

Checksums identify bytes and declarations, not data quality, legal rights,
point-in-time availability or previously unseen observations. No function in
this module downloads data or silently repairs financial observations.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .data import _validate_ohlcv

DATASET_SCHEMA_VERSION = 1
MAX_MANIFEST_BYTES = 64 * 1024
MAX_CSV_BYTES = 64 * 1024 * 1024
PRICE_ADJUSTMENTS = ("unadjusted", "split_adjusted", "split_and_dividend_adjusted")
CALENDARS = {"exchange": 252.0, "continuous": 365.0}
PROVENANCE_NOTE = (
    "Source, licensing, calendar and price-adjustment fields are declarations, not verified facts. "
    "Checksums bind bytes and declarations; they do not establish completeness, point-in-time availability, "
    "correct corporate-action treatment, legal permission or previously unseen data. "
    "Corporate actions and dividend cashflows are not reconstructed by the engine."
)
_FIELDS = {
    "schema_version",
    "dataset_id",
    "ticker",
    "source",
    "license",
    "currency",
    "calendar",
    "frequency",
    "price_adjustment",
    "annualization",
    "csv_file",
    "sha256",
}
_OHLC = ["open", "high", "low", "close"]


class DatasetError(ValueError):
    """An offline snapshot is unreadable, incompatible or invalid."""


def _read_bytes(path, limit):
    try:
        if not Path(path).is_file():
            raise DatasetError(f"Cannot read dataset file {path!s}: expected a regular file")
        with Path(path).open("rb") as stream:
            payload = stream.read(limit + 1)
    except (OSError, ValueError, TypeError) as exc:
        raise DatasetError(f"Cannot read dataset file {path!s}: {exc}") from exc
    if len(payload) > limit:
        raise DatasetError(f"Dataset file exceeds the {limit}-byte limit: {path!s}")
    return payload


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DatasetError(f"Duplicate manifest field: {key}")
        result[key] = value
    return result


def _validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise DatasetError("Dataset manifest must be a JSON object")
    missing, unknown = sorted(_FIELDS - manifest.keys()), sorted(manifest.keys() - _FIELDS)
    if missing or unknown:
        raise DatasetError(f"Invalid manifest fields; missing={missing}, unknown={unknown}")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != DATASET_SCHEMA_VERSION:
        raise DatasetError("Unsupported dataset schema_version")
    for key in _FIELDS - {"schema_version", "annualization"}:
        value = manifest[key]
        if not isinstance(value, str) or not value.strip() or len(value) > 2048:
            raise DatasetError(f"Manifest {key} must be a non-empty string of at most 2048 characters")
        if value != value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise DatasetError(f"Manifest {key} cannot contain surrounding whitespace or control characters")
        try:
            value.encode("utf-8")
        except UnicodeError as exc:
            raise DatasetError(f"Manifest {key} must contain valid Unicode") from exc
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", manifest["dataset_id"]):
        raise DatasetError("Invalid dataset_id; use 1-128 letters, digits, '.', '_' or '-'")
    if not re.fullmatch(r"[A-Z0-9._^=-]{1,64}", manifest["ticker"]):
        raise DatasetError("Manifest ticker must be an uppercase canonical ticker (at most 64 characters)")
    if not re.fullmatch(r"[A-Z]{3,12}", manifest["currency"]):
        raise DatasetError("Manifest currency must be an uppercase currency code")
    if manifest["calendar"] not in CALENDARS:
        raise DatasetError("Manifest calendar must be exchange or continuous")
    if manifest["frequency"] != "1d":
        raise DatasetError("Only daily session-date data (frequency=1d) is supported")
    if manifest["price_adjustment"] not in PRICE_ADJUSTMENTS:
        raise DatasetError(f"Manifest price_adjustment must be one of {PRICE_ADJUSTMENTS}")
    annualization = manifest["annualization"]
    if (
        isinstance(annualization, bool)
        or not isinstance(annualization, (int, float))
        or not 0 < annualization <= 366
        or not math.isfinite(annualization)
    ):
        raise DatasetError("Daily dataset annualization must be finite and in (0, 366]")
    name = manifest["csv_file"]
    if (
        name in {".", ".."}
        or any(char in name for char in "/\\:")
        or Path(name).name != name
        or not name.lower().endswith(".csv")
    ):
        raise DatasetError("csv_file must be a CSV filename inside the manifest directory")
    if not re.fullmatch(r"[a-f0-9]{64}", manifest["sha256"]):
        raise DatasetError("Manifest sha256 must be a lowercase SHA-256 digest")
    return manifest


def _parse_csv(payload):
    try:
        text = payload.decode("utf-8-sig")
        if "\x00" in text:
            raise DatasetError("CSV cannot contain NUL bytes")
        reader = csv.reader(io.StringIO(text), strict=True)
        header = next(reader)
        columns = [name.strip().lower() for name in header]
        if len(columns) != len(set(columns)):
            raise DatasetError("Duplicate CSV columns (including case/whitespace aliases)")
        required = {"date", *_OHLC}
        missing, unknown = sorted(required - set(columns)), sorted(set(columns) - required - {"volume"})
        if missing or unknown:
            raise DatasetError(f"Invalid CSV columns; missing={missing}, unknown={unknown}")
        # pandas can otherwise reinterpret extra leading cells as an index,
        # or ignore blank rows. Reject every malformed row before parsing.
        for number, row in enumerate(reader, start=2):
            if len(row) != len(columns):
                raise DatasetError(f"CSV row {number} has {len(row)} cells; expected {len(columns)}")
        frame = pd.read_csv(
            io.StringIO(text),
            header=0,
            names=columns,
            dtype=str,
            keep_default_na=False,
            skip_blank_lines=False,
        )
        if frame.empty:
            raise DatasetError("CSV contains no observations")
        if not frame["date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
            raise DatasetError("CSV dates must be YYYY-MM-DD session dates, without time or timezone")
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("date"), format="%Y-%m-%d", errors="raise"))
        frame.index.name = "date"
        for name in frame.columns:
            # Convert decimal text directly to binary64. pd.to_numeric's fast
            # parser can round away meaningful final digits before the cast.
            frame[name] = frame[name].astype(float)
        frame = frame[_OHLC + (["volume"] if "volume" in frame else [])]
        _validate_ohlcv(frame, "local CSV")
        if "volume" in frame:
            volume = frame["volume"].to_numpy()
            if not np.isfinite(volume).all() or (volume < 0).any():
                raise DatasetError("CSV volume must be finite, non-negative units of the asset")
    except DatasetError:
        raise
    except (UnicodeError, csv.Error, StopIteration, ValueError, TypeError, OverflowError) as exc:
        raise DatasetError(f"Invalid daily OHLCV CSV: {exc}") from exc
    return frame


def _date_bound(value, name):
    try:
        bound = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise DatasetError(f"Invalid {name} session date") from exc
    if pd.isna(bound) or bound.tzinfo is not None or bound != bound.normalize():
        raise DatasetError(f"{name} must be a timezone-free daily session date")
    return bound


def read_dataset_manifest(manifest_path):
    """Read validated declarations only; this does NOT verify CSV bytes or rows."""
    payload = _read_bytes(manifest_path, MAX_MANIFEST_BYTES)
    try:
        manifest = json.loads(payload.decode("utf-8-sig"), object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise DatasetError(f"Invalid dataset manifest: {exc}") from exc
    return _validate_manifest(manifest)


def resolve_dataset_annualization(declarations, requested=None):
    """Use the declared calendar convention, rejecting contradictory overrides."""
    declared = float(declarations["annualization"])
    if requested is not None and (isinstance(requested, (bool, np.bool_)) or requested != declared):
        raise DatasetError("annualization conflicts with the dataset declaration; import a new declared snapshot")
    return declared


def load_dataset(manifest_path, *, ticker=None, start=None, end=None):
    """Verify a complete snapshot, then return ``(OHLCV frame, provenance)``.

    Bounds are inclusive. ``end=None`` means the snapshot's last observation,
    not today. An early requested start warns; a later end or empty slice fails.
    Every original row is validated, including rows outside the requested slice.
    Moving the snapshot does not change its contract; rewriting CSV bytes does.
    """
    manifest = read_dataset_manifest(manifest_path)
    if ticker is not None and str(ticker).upper() != manifest["ticker"]:
        raise DatasetError(f"Dataset ticker {manifest['ticker']} does not match requested ticker {ticker}")
    csv_path = Path(manifest_path).resolve().parent / manifest["csv_file"]
    if csv_path.is_symlink():
        raise DatasetError("Dataset CSV cannot be a symbolic link")
    csv_bytes = _read_bytes(csv_path, MAX_CSV_BYTES)
    if hashlib.sha256(csv_bytes).hexdigest() != manifest["sha256"]:
        raise DatasetError("Dataset CSV SHA-256 mismatch; restore the snapshot or import a new dataset")
    frame = _parse_csv(csv_bytes)
    contract = {key: value for key, value in manifest.items() if key != "csv_file"}
    contract["annualization"] = float(contract["annualization"])
    contract_hash = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()
    provenance = {
        **contract,
        "provider": "local_csv",
        "contract_sha256": contract_hash,
        "rows": len(frame),
        "first_date": str(frame.index[0].date()),
        "last_date": str(frame.index[-1].date()),
        "has_volume": "volume" in frame,
    }
    start_ts = _date_bound(start, "start") if start is not None else frame.index[0]
    end_ts = _date_bound(end, "end") if end is not None else frame.index[-1]
    if end_ts < start_ts:
        raise DatasetError("end must be on or after start")
    if end_ts > frame.index[-1]:
        raise DatasetError(
            "Dataset does not cover the requested end; offline snapshots are never extended automatically"
        )
    if start_ts < frame.index[0]:
        warnings.warn(
            f"Dataset starts at {frame.index[0].date()}, later than requested start {start_ts.date()}; "
            "using the available snapshot without filling earlier dates.",
            RuntimeWarning,
            stacklevel=2,
        )
    selected = frame.loc[start_ts:end_ts].copy()
    if selected.empty:
        raise DatasetError("Dataset contains no observations in the requested range")
    if manifest["price_adjustment"] != "split_and_dividend_adjusted":
        warnings.warn(
            f"Dataset price_adjustment={manifest['price_adjustment']}; splits/dividends are not reconstructed. "
            "Returns may exclude distributions or contain corporate-action jumps.",
            RuntimeWarning,
            stacklevel=2,
        )
    return selected, provenance


def import_dataset(
    csv_path,
    output_dir,
    *,
    ticker,
    source,
    license,
    currency,
    calendar,
    price_adjustment,
    annualization=None,
    dataset_id=None,
):
    """Validate and copy user-supplied CSV bytes into a NEW snapshot directory.

    The caller must have permission to use the data. Input bytes are preserved;
    no prices, rows, dates, volume, adjustments or licenses are inferred/repaired.
    Existing output directories are never reused or overwritten. The manifest is
    written last; an interrupted import without a complete manifest is invalid.
    """
    csv_bytes = _read_bytes(csv_path, MAX_CSV_BYTES)
    digest = hashlib.sha256(csv_bytes).hexdigest()
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_id": dataset_id if dataset_id is not None else f"csv-{digest[:16]}",
        "ticker": ticker.upper() if isinstance(ticker, str) else ticker,
        "source": source,
        "license": license,
        "currency": currency,
        "calendar": calendar,
        "frequency": "1d",
        "price_adjustment": price_adjustment,
        "annualization": CALENDARS.get(calendar)
        if annualization is None and isinstance(calendar, str)
        else annualization,
        "csv_file": "prices.csv",
        "sha256": digest,
    }
    _validate_manifest(manifest)
    _parse_csv(csv_bytes)
    output = Path(output_dir)
    try:
        output.mkdir(parents=True, exist_ok=False)
        with (output / "prices.csv").open("xb") as stream:
            stream.write(csv_bytes)
        # Exclusive creation also ensures that a partial previous import cannot
        # be silently completed using new, unrelated input bytes.
        with (output / "manifest.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    except OSError as exc:
        raise DatasetError(f"Cannot create snapshot in a new directory {output}: {exc}") from exc
    return output / "manifest.json"


def main(argv=None):
    """Dataset import and metadata-only inspection CLI; never uses a network."""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "check":
        from .preflight import main as preflight_main

        return preflight_main(argv[1:])
    parser = argparse.ArgumentParser(prog="momentum-lab data", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("import", help="Validate a CSV and create a new immutable-by-convention snapshot")
    create.add_argument("csv", help="UTF-8 CSV: date,open,high,low,close[,volume]")
    create.add_argument("--output", required=True, help="New directory; existing paths are never overwritten")
    create.add_argument("--ticker", required=True, help="Canonical ticker; use the same ticker across data sources")
    create.add_argument("--source", required=True, help="Provider/export description (user declaration)")
    create.add_argument("--license", required=True, help="Usage terms or license reference (user declaration)")
    create.add_argument("--currency", required=True, help="Uppercase price currency, e.g. USD")
    create.add_argument("--calendar", required=True, choices=CALENDARS)
    create.add_argument("--price-adjustment", required=True, choices=PRICE_ADJUSTMENTS)
    create.add_argument(
        "--annualization", type=float, help="Daily periods/year (default: exchange=252, continuous=365)"
    )
    create.add_argument("--dataset-id", help="Optional human-readable snapshot ID")
    inspect = commands.add_parser("inspect", help="Verify bytes/rows and print declarations, never strategy scores")
    inspect.add_argument("manifest")
    commands.add_parser("check", help="Read-only date/integrity preflight; use data check --help")
    args = parser.parse_args(argv)
    try:
        if args.command == "import":
            manifest = import_dataset(
                args.csv,
                args.output,
                ticker=args.ticker,
                source=args.source,
                license=args.license,
                currency=args.currency,
                calendar=args.calendar,
                price_adjustment=args.price_adjustment,
                annualization=args.annualization,
                dataset_id=args.dataset_id,
            )
            print(f"Created dataset: {manifest}")
        else:
            _, provenance = load_dataset(args.manifest)
            print(json.dumps({**provenance, "notice": PROVENANCE_NOTE}, ensure_ascii=False, indent=2, allow_nan=False))
    except DatasetError as exc:
        parser.error(str(exc))
    return 0
