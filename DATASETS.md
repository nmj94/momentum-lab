# Offline daily datasets (v0.12)

Momentum Lab can research a user-supplied daily OHLCV snapshot without downloading
market data or using the Yahoo cache. This is a data-access and reproducibility
feature, **not a new source of licensed historical investment evidence**. Only
import data you have permission to use. The repository ships synthetic test
observations, not commercial historical prices.

## Import, inspect, research

Prepare a UTF-8 CSV with this header (column order/case may differ):

```csv
date,open,high,low,close,volume
2024-01-02,100,102,99,101,1000
2024-01-03,101,103,100,102,1200
```

These two illustrative rows are insufficient for meaningful research. Use a
longer, independently reviewed history for a real study. Import it into a **new**
directory, declaring the source, usage terms, currency, calendar and adjustment
basis explicitly:

```bash
momentum-lab data import /path/to/your/prices.csv \
  --output datasets/spy-v1 --dataset-id spy-v1 --ticker SPY \
  --source "My provider; export dated 2026-08-28" \
  --license "Private research use under my provider agreement" \
  --currency USD --calendar exchange \
  --price-adjustment split_and_dividend_adjusted

momentum-lab data inspect datasets/spy-v1/manifest.json

# Default end is the last observation in this fixed snapshot, not today.
momentum-lab SPY --dataset datasets/spy-v1/manifest.json --start 2020-01-01 \
  --strategies tsmom,ma_cross --study-id spy-local-v1 --run-id spy-local-v1

# Explicitly reveal the previously frozen selection, with unchanged settings.
momentum-lab SPY --dataset datasets/spy-v1/manifest.json --start 2020-01-01 \
  --strategies tsmom,ma_cross --study-id spy-local-v1 --run-id spy-local-v1 \
  --resume --reveal-test
```

Choose the symbol and dates appropriate for your own file; the example does not
assert that any SPY dataset is bundled or licensed. Without `--dataset`, the
existing Yahoo workflow is unchanged. With it, a missing/invalid file is an
error, **never** a reason to download replacement observations. `--refresh` and
`use_cache` have no effect on this offline path.

Import creates `prices.csv` (the exact input bytes) and `manifest.json`, leaving
the original file untouched. It validates before writing and never overwrites
an existing output directory. The manifest is written last: interruption can
leave an incomplete directory, which must not be treated as a valid snapshot.
Use a new directory to retry; incomplete files are not automatically deleted.
Data commands return exit code `2` for invalid input and `0` on success.

`data inspect` verifies the whole CSV and prints declarations, hashes and
coverage, not price rows, strategy scores or an OOS assessment. It does not
register a study or claim the underlying prices have never been viewed.

## Data contract

| Item | Contract |
|---|---|
| File | UTF-8, optionally BOM-prefixed; comma-separated; at most 64 MiB |
| Columns | `date,open,high,low,close` required; `volume` optional; unknown/duplicate headers rejected |
| Dates | Strict `YYYY-MM-DD` session dates, increasing and unique; no intraday timestamps or timezone conversions |
| Prices | Finite, strictly positive OHLC; high/low must contain open and close |
| Volume | If supplied, every observation must be finite and non-negative; zero is valid |
| Row integrity | No sorting, deduplication, interpolation, backfilling, or dropping invalid/blank observations |
| Frequency | `1d` only; daily OHLC is required even when the selected strategy only reads close |
| Range | Inclusive start/end; an early start warns and uses available history; a later end or empty slice fails |
| Validation scope | All source observations are checked before slicing, including unused rows |

Do not mix raw open/high/low with an adjusted close. The importer records your
adjustment declaration but **does not calculate or validate adjustment factors**.
It rejects `adj_close` as an extra column rather than silently choosing a basis.
Transform a provider export deliberately before importing it.

Volume must be in units of the traded asset, not quote-currency turnover. For
capacity/impact models, its relationship to the price basis must be appropriate;
adjusted historical prices times unadjusted volume need not equal historical
dollar volume. Missing volume is allowed for price-only research; liquidity-aware
search still requires an actual volume column. Missing cells are never replaced
with invented volume.

The supported price-adjustment declarations are:

- `split_and_dividend_adjusted`: OHLC consistently adjusted for splits and
  distributions according to the source's methodology. This is a declaration,
  not an independently verified total-return series or executable-price claim.
- `split_adjusted`: distributions may be excluded from returns.
- `unadjusted`: splits can introduce artificial price jumps; distributions may
  also be excluded.

The latter two produce a warning without changing the observations. The engine
does not reconstruct corporate actions or add dividend cashflows separately.
Even an adjusted dataset can be revised, contain survivorship bias, or expose
information unavailable at the simulated decision time.

## Manifest and calendar declarations

The importer writes strict JSON schema version `1`. Required fields are
`schema_version`, `dataset_id`, `ticker`, `source`, `license`, `currency`,
`calendar`, `frequency`, `price_adjustment`, `annualization`, `csv_file` and
`sha256`. Unknown fields, duplicate JSON keys and unsupported schema versions
fail closed. Manifest size is limited to 64 KiB. The CSV must be a regular file
named directly within the manifest directory, not a URL, parent-relative path
or symlink. No URLs in declarations are fetched.

The importer uppercases the ticker. Use the same canonical symbol for the same
asset across data sources: the research registry still matches uppercase
symbols and dates, not provider-specific identifiers. Renames/aliases, listings
on different venues, strongly correlated assets and separate registries are
**not** automatically reconciled. Selecting a new symbol is not proof of an
independent sample. See [GOVERNANCE.md](GOVERNANCE.md).

`exchange` defaults to 252 return periods/year; `continuous` defaults to 365.
Import-time `--annualization` may specify a different daily convention in
`(0, 366]`. The labels do not verify exchange holidays, missing sessions or
completeness, and `exchange` does not forbid weekend sessions.

Research uses the declared annualization for both features and performance;
ticker suffixes do not override it. An explicit conflicting research-time
`--annualization` fails. If the convention is wrong, correct it by importing a
new snapshot declaration and starting a new study/run, not by altering an
existing research protocol after observing scores.

## Provenance and resume

`run_config.json`, `summary.json`, Python results and both human reports expose
`data_provenance`. For local CSVs it includes the original file SHA-256, a
canonical dataset-contract SHA-256, all declarations, full-file row/date coverage
and whether volume exists. The existing evaluated-frame fingerprint continues
to identify the actual sliced observations used by the engine.

The contract covers every manifest field except the relative CSV filename.
Moving/renaming a byte-identical snapshot or reformatting JSON does not change
the contract. The absolute manifest path is recorded for convenience but is
not itself resume-locked. In contrast, changing **any CSV byte**, even whitespace
or a row outside the selected range, or changing a source/license/adjustment
declaration invalidates the old resume/protocol. Recomputing a file's hash does
not make its new contents compatible with a previously registered study.

The coordinator reads and validates the full file before registration; this is
not preregistration before acquisition or encryption of the holdout. Candidate
workers receive development observations only, without full-snapshot provenance
metadata. Local data uses the same sealed study, explicit reveal, cached replay,
and overlapping-observation rules as online data. Switching data providers or
CSV hashes under the same symbol does not erase observation history.

Checksums provide accidental-change detection, not signatures or tamper-proof
custody. Source and license strings are **user declarations, not an authorization
check**. Momentum Lab neither grants rights to the imported data nor verifies
its accuracy, completeness or point-in-time availability. Keep a permitted copy
of the data, configuration and registry; do not publish proprietary CSVs with
public research reports. The conventional `datasets/` directory is git-ignored.

v0.12 does not change accounting/selection engine schema `5` or registry schema
`1`. New provenance and source/version fingerprints require fresh v0.12 runs;
old checkpoints are not silently upgraded. Known prior exposures remain in the
same shared registry, and old artifacts can be imported explicitly.

## Python and JSON configuration

```python
from momentum_lab import SearchConfig, import_dataset, load_dataset, prepare_data, run_search

manifest = import_dataset(
    "my-prices.csv", "datasets/my-v1", ticker="MY-ASSET", source="My licensed export",
    license="Internal research only", currency="USD", calendar="continuous",
    price_adjustment="split_and_dividend_adjusted",
)
frame, provenance = load_dataset(manifest)  # All rows validated; no network.
data, frame = prepare_data("MY-ASSET", start="2020-01-01", dataset=manifest)
config = SearchConfig(
    ticker="MY-ASSET", dataset=str(manifest), start="2020-01-01",
    strategies=["tsmom"], study_id="my-v1", run_id="my-v1",
)
sealed = run_search(config=config)
revealed = run_search(config=config, resume=True, reveal_test=True)
```

In a JSON config loaded from a file, `dataset` is resolved relative to that
config's directory. With Python objects/mappings or `--dataset`, relative paths
are resolved from the working directory. Other path options retain their prior
semantics. A supplied JSON config remains the complete configuration and takes
precedence over ordinary CLI options; reveal/reuse consent remains invocation-only.

This release adds no network-data dependency or optional ML requirement. A
frozen synthetic CSV parser test and installed-wheel offline lifecycle smoke
complement the unchanged 16-case frozen accounting regression suite.
