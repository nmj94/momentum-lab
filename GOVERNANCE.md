# Research registration and test access (v0.11+, portfolio studies in v0.14)

The research registry prevents accidental reuse from being labelled as a new
test. It is a **local observation ledger, not encrypted test-data custody or a
tamper-proof preregistration service**. It cannot know whether data was inspected
elsewhere, on another machine, before registration, or under another symbol.
"First recorded reveal" never means "proven previously unseen data".

## Portfolio whole-history acknowledgement

The v0.13 `portfolio` workflow is separate from registered single-asset study
selection. It computes a fixed rule over the entire evaluated multi-asset
history, not a sealed portfolio test. `--acknowledge-history` (or Python
`acknowledge_history=True`) is required on each invocation and is forbidden in
the JSON recipe. Missing acknowledgement fails before data or registry access.

After validating aligned offline datasets, **every asset's entire evaluated
date range** is reserved as `development` in the same shared registry, before
any score or portfolio P&L calculation. Those observations overlap later test
claims regardless of different run IDs or data hashes. Already-sealed studies
can consequently become `known_prior_exposure` and require reuse consent.
Acknowledging portfolio history is not a claim that it was previously unseen.

All asset reservations must finish before calculation starts. They are
individual transactions: if a later one fails, earlier records conservatively
remain, and the book is not computed. Calculation/export failures likewise
retain history. There is no implicit resume or overwrite; `summary.json` is
written last, so incomplete runs lack a completion marker. Restore failures
with a new run ID without clearing observation history.

Portfolio low-level numerical APIs, like existing single-asset backtests, do
not manage observation records. A researcher using them is responsible for
recording their own access; no numerical API certifies fresh OOS evidence.
See [PORTFOLIOS.md](PORTFOLIOS.md) for the accounting/data contract. v0.14 adds
the separate registered portfolio workflow below; verified historical data and
complete delisting/settlement support remain future work.

## Registered fixed-rule portfolios (v0.14)

`momentum-lab portfolio study` freezes a fixed rule, all candidate snapshots,
optional membership, development/test boundaries and software identity before
evaluating development. Test data is withheld from the evaluator until a
separate explicit reveal. The coordinator still validates/hashes full input
files; this is not encrypted or custodial concealment.

All candidate test ranges are reserved in **one transaction** before test
computation, including inactive/never-selected assets. Development records are
also atomic. One overlap, from any single-asset/portfolio study or whole-history
run, blocks a fresh group reveal. Interrupted claims remain exposures. A completed
study reopens its frozen summary and logs a replay; it never silently recalculates.
The first completed result is pinned even if an earlier concurrent claim finishes
later. Test books carry holdings, cash and pending instructions across the split.

The base schema/user-version 1, registry identity and old records are preserved.
Four additive tables (`portfolio_registry_info`, `portfolio_studies`,
`portfolio_access_batches`, `portfolio_batch_events`) use portfolio extension
schema 1 and link all asset events to the existing observations table. Study
namespaces are separate; overlap history is shared. Missing/incomplete extensions
or corrupted cached groups are rejected, not recreated. Read-only status/list
never initializes an extension or returns cached metrics.

```bash
momentum-lab portfolio study --config study.json --run-id development
momentum-lab portfolio study status relative-momentum-study-v1
momentum-lab portfolio study --config study.json --run-id test-reveal --reveal-test
```

See [PORTFOLIO_STUDIES.md](PORTFOLIO_STUDIES.md) for consent/reuse flags, cached
summary-only recovery, artifacts and APIs, and [MEMBERSHIP.md](MEMBERSHIP.md)
for declared membership limits. This does not add portfolio parameter selection
or establish that a first recorded reveal is truly unseen evidence.

## Two-step research workflow

Use a study ID and a fixed end date. The first invocation registers the protocol
before candidate evaluation, selects on development data, freezes the winner,
and writes a report **without computing or displaying test metrics**:

```bash
momentum-lab GLD --study-id gld-2026q3 --run-id gld-dev --end 2026-08-28

# Metadata only: these commands do not reveal cached test scores.
momentum-lab study status gld-2026q3
momentum-lab study history --ticker GLD --limit 100

# Explicitly reveal the already-frozen selection in a separate invocation.
momentum-lab GLD --study-id gld-2026q3 --run-id gld-dev --end 2026-08-28 \
  --resume --reveal-test
```

Keep the same search options, source, environment and data snapshot when
revealing. A moving `end=None` or revised provider history can change the
snapshot and will be rejected, not silently substituted. A new protocol needs
a new study ID; changing that ID does not erase observed-date overlap.

Only `study_id` and `registry_path` are added to `SearchConfig`. Consent flags
`--reveal-test`, `--allow-test-reuse` and `--test-reuse-reason` are invocation-only:
they cannot be enabled by a reused JSON configuration. This remains true with
`run_search(config=..., resume=True, reveal_test=True)`.

```python
from momentum_lab import SearchConfig, StudyRegistry, run_search

config = SearchConfig(
    ticker="GLD", strategies=["tsmom", "ma_cross"], end="2026-08-28",
    study_id="gld-2026q3", run_id="gld-dev", robust=False,
)
sealed = run_search(config=config)
assert "test_metrics" not in sealed["best"]  # Provided an eligible winner exists.

# Inspect the validation evidence, then make an explicit decision to reveal.
revealed = run_search(config=config, resume=True, reveal_test=True)
print(revealed["test_access"]["status"])
print(StudyRegistry(create=False).status("gld-2026q3"))  # No test-score payload.
```

## What is fixed, and when

- Registration stores canonical JSON and a SHA-256 protocol digest. It binds
  the indexed full-data hash, actual train/validation/test bounds, ticker,
  candidate strategy grids/universal parameters, quick or halving settings,
  ranking/evidence gates, execution/financing assumptions, annualization,
  bootstrap settings, package/source/lock/environment fingerprints.
- This occurs after the coordinator has prepared and hashed market data, but
  **before candidate evaluation**. It is not a claim of registration before
  downloading or before any human could have seen historical prices.
- Candidate workers receive development observations only. Registered runs
  compute validation bootstrap diagnostics on development-only ledgers.
  Parameter-sensitivity analysis also receives only development data and bounds,
  in both registered and legacy modes.
- The first selected strategy, parameters and validation-selection evidence are
  frozen. Another run cannot replace them inside the same study. Worker count,
  top-N display and report controls remain presentation/execution choices;
  they do not authorize another test selection.
- A new registered run requires an empty run directory; use `--resume` to
  continue it. Registry identity/path and study ID are resume-locked. Missing
  metadata, a missing/replaced registry, or a changed protocol fails closed.

The coordinator necessarily holds the price snapshot to prepare/hash/split it.
"Sealed" refers to this workflow's evaluation/output boundary, **not hiding raw
data in memory or preventing direct filesystem/Python access**.

### Resume and legacy rerun integrity (v0.14.1)

Every `--resume`, including an unregistered legacy run, requires the original
`run_config.json` and matching registry identity. Missing provenance is an error,
not a warning followed by loading unverifiable checkpoint rows. Restore the
original files/environment or use a new run ID; never clear observation history
to make a compatibility check pass.

When intentionally reusing an unregistered run ID **without** `--resume`, the
previous configuration, SQLite/CSV journals, summary, rankings, sensitivity CSV
and reports are moved to matching timestamped `*.bak.*` names before publishing
the new configuration. The SQLite WAL is checkpointed first; an invalid or busy
journal fails rather than being repaired as part of a backup. Unrelated files
are left alone. This prevents an empty/failed replacement from showing an old
winner under the current run's filename. Registered runs still require an empty
directory or a valid resume; they cannot be overwritten this way.

Publication is atomic **per file**, with exclusively created staging files, not
a transaction over an entire run directory. Since v0.15, cooperating search and
portfolio coordinators hold a local OS lock across output validation, research
and publication; a second owner of the same directory fails busy. Older versions
and low-level APIs do not participate. Never delete the stable ownership file to
force a retry. Keep the complete backup set with its original source, data,
environment and registry; do not combine artifacts from different runs.

`momentum-lab runs status PATH` and `runs history PATH` inspect score-free
operational attempts, separately from research observation history. `--verify`
checks published file snapshots; it neither reveals nor recomputes a test. A
failed run may already have durably reserved test access, and an intact completion
receipt does not establish that data was unseen. See [RUNS.md](RUNS.md) for the
separate state journal, interrupted-owner detection and explicit recovery rules.

## Cross-run observation matching

The shared SQLite registry lives in the OS user-data directory for
`momentum-lab`, not in an individual experiment or in `site-packages`.
`run_config.json` records its resolved path and identity. Precedence is:

1. Explicit `registry_path` / `--registry`.
2. Environment variable `MOMENTUM_LAB_REGISTRY_PATH`.
3. Default user-data location, file `research-registry.sqlite3`.

For study subcommands, put the optional registry flag **before** the subcommand:

```bash
momentum-lab study --registry /path/to/research-registry.sqlite3 list
momentum-lab study --registry /path/to/research-registry.sqlite3 status gld-2026q3
```

Observation matching uses the normalized uppercase ticker and inclusive daily
session-date overlap: `old.start <= new.end` and `old.end >= new.start`.
Data hashes, study IDs, run IDs and source versions **are not filters** in this
query. Thus copying results to another directory, changing IDs, revising prices
or partly shifting test boundaries does not clear recorded overlap in the same
registry. Same-day endpoints overlap; the following calendar day does not.

Training/validation ranges are conservatively recorded before search work.
They also count when later proposed as test dates. Interrupted work may thus
cause a warning even if no useful score was ultimately displayed; no inference
is made about whether the human actually read it.

This v1 scheme matches daily canonical tickers from Yahoo or offline CSV snapshots. Symbol aliases,
renames, strongly correlated assets, intraday instant matching, other registries
and external analysis are not reconciled automatically. Those limitations must
not be mistaken for independent investment evidence.

Offline datasets (v0.12) must use the same canonical ticker for the same asset
across providers. Dataset provenance, including CSV bytes and source/usage/
adjustment declarations, is part of the frozen protocol and resume comparison.
Changing those declarations requires a new study/run but does not reset recorded
date overlap. Relocating an unchanged snapshot is supported. No asset-identity
mapping service, price-quality certification or license verification is added.
See [DATASETS.md](DATASETS.md) for the complete offline data contract.

## Reveal, replay and interruption

SQLite `BEGIN IMMEDIATE` transactions serialize overlap checking and reservation.
The possible-exposure reservation is committed with full synchronous writes
**before** a full-data strategy or benchmark is evaluated. The database lock is
not held throughout the backtest. Two ordinary concurrent claims cannot both
receive a first-recorded assessment.

When known observations overlap, a new registered evaluation is blocked unless
the user explicitly acknowledges historical reuse and supplies a reason:

```bash
momentum-lab GLD --study-id gld-2026q3 --run-id gld-dev --end 2026-08-28 \
  --resume --reveal-test --allow-test-reuse \
  --test-reuse-reason "Revised historical comparison, not new out-of-sample evidence"
```

The reason is stored with the event, and the result is marked `repeated_use`.
Acknowledgement does not remove earlier observations or restore independence.

Completed test metrics, benchmark metrics, test-bootstrap diagnostics and their
original evaluation timestamp are stored with a payload digest. A later explicit
reveal of the same fixed study reuses that result without another test backtest.
It is labelled `previously_revealed`, and a separate `reveal_replay` event links
the new access to the original evaluation. It needs no new reuse override because
it is a reread of existing evidence, not a new evaluation.

Since v0.14.1, a single-asset study pins the first completed result in the same
transaction that completes its reservation. An earlier claim finishing later
cannot replace that cached result, even when completion timestamps are equal.
This uses the existing unique observation key; schema/user-version 1 and all
observation history are preserved. Protocols, selections and summaries must be
finite JSON objects no larger than 4 MiB each.

For legacy unpinned records, an existing replay's original source takes
precedence; otherwise recorded completion time and event order select the cache.
The next writable claim/completion pins that choice. Read-only status does not
write to the registry or display scores. Historical ordering cannot be proven
where neither a reliable timestamp nor an earlier replay recorded it.

Reservations survive exceptions, keyboard interruption and failed exports.
`reserved` and `failed` records count as possible exposure. Retrying an incomplete
evaluation requires acknowledgement; a completed evaluation with a failed report
export is recovered from the cached result. Corrupt protocol/selection/payload
digests are errors, not a trigger to recompute or reset history.

## Output states

Final JSON/Python results contain `test_access`; human reports have a matching
audit section. Always read `test_results_visible` as well as `status`.

| Status | Meaning |
|---|---|
| `sealed` | No recorded overlap; this run withholds test computation/results. External history remains unknown. |
| `known_prior_exposure` | Results are withheld, but overlapping observations or incomplete reveals exist. |
| `first_recorded_reveal` | First recorded test evaluation of these dates in this registry, not proof of untouched data. |
| `previously_revealed` | This study already has a completed result; explicit replay reads it, otherwise scores stay hidden. |
| `repeated_use` | A new evaluation overlaps recorded observations; registered mode requires an explicit reason. |
| `history_unknown` | Legacy, unregistered or imported history cannot establish a first use. |
| `no_selection` | No eligible final winner; no test claim was made. |

In registered sealed output, `best` has no `test_metrics` or test-evaluation
timestamp, benchmark test metrics are null, and bootstrap contains validation
only. Candidate CSVs and the SQLite candidate journal still contain development
evidence only. Report renderers also suppress any stale test fields when the
visibility flag is false. Status/list/history commands return metadata, never
the cached score payload. History is bounded to 100 events by default (maximum
1000); overlap checks count **all** matching records, not just displayed rows.

## Legacy compatibility and import

Commands without `--study-id` retain automatic final-selection test evaluation.
They now record development and test access in the shared registry. Reports
explicitly say `history_unknown`, or `repeated_use` where an overlap is known;
they no longer claim that a per-run sealed test proves globally fresh evidence.
Registered mode is therefore recommended for new research.

Older reports are **not automatically scanned or rewritten**. Import each known
old run with visible test results to prevent it being forgotten:

```bash
momentum-lab study import-legacy experiments/old-run
```

Import requires matching `run_config.json`/`summary.json` run IDs, an indexed-data
digest, ordered period bounds and visible test metrics. It records historical
development and test observations with unknown access history. Identical imports
are idempotent even after moving the artifact directory. No old artifact is
modified, and no old test is certified as unused. A sealed report cannot be
imported as already-visible test evidence.

Pre-v0.11 checkpoints do not carry registry identity and cannot be silently
upgraded by resuming; retain the old environment for old runs, import their test
evidence, and use a new run/protocol where necessary. Accounting engine schema 5
and the 16 frozen accounting ledgers are unchanged; new run metadata is stricter.

## Persistence, security and validation

Keep one consistent registry for the research program, on a filesystem suitable
for local SQLite locking. Back it up using SQLite's backup API, or copy it while
all writers are stopped. Restore a complete trusted history, not an older snapshot
that omits later accesses. Registry files contain cached test results; protect
them accordingly and do not commit them to Git. Default `.gitignore` covers
`.sqlite3` registries and sidecars.

There is no deletion/reset command, automatic corruption recovery, or hidden
fallback to a new registry. Nevertheless an administrator can delete/edit files,
restore an old backup, choose a separate registry or inspect cached data directly.
This implementation does not prevent those actions or establish a cryptographic
chain of custody. A first-recorded label remains conditional on the retained
registry, never a global freshness guarantee.

Tests cover protocol/selection immutability, date/data-version matching,
development reuse, concurrent claims and creation, fail-closed interruption,
cached replays, legacy import, output masking, config consent, resume identity,
unchanged ranking and successive-halving integration. `scripts/governance_smoke.py`
exercises the real lifecycle with fixed synthetic data and the installed wheel,
without requesting market data. Licensed historical investment benchmarks and
selection-aware statistical inference remain separate future work.
