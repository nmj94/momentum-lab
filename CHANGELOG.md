# Changelog

All notable changes are documented here. The project follows semantic versioning
for its public Python API and run/checkpoint schema.

## [0.14.1] - 2026-09-03

### Bug and research-integrity audit

- Give symbolic Yahoo tickers collision-free, uppercase cache identities;
  ambiguous legacy symbol caches are not reused or deleted. Round-trip CSV
  floats exactly, reject invalid metadata/volume/session dates, and preserve
  exchange-local daily labels when removing provider timezones.
- Preserve actual constant and single-observation gains/losses and treat
  bankruptcy as absorbing. Reject non-finite/non-real observations and numeric
  overflow; keep finite legacy undefined-ratio sentinels, but exclude undefined
  Sharpe from candidate selection and parameter-sensitivity scores.
- Use the selected execution-price column in sensitivity analysis, including
  next-open studies, without changing signal lag or development-only bounds.
- Fail closed on resume without the original configuration; archive the full
  prior legacy artifact set before a replacement run, retaining configuration,
  journals and reports together and removing stale current-output filenames.
- Atomically pin the first completed single-asset reveal independently of claim
  order, retaining old registry identity/schema/history and read-only behavior.
  Reject malformed, non-finite or oversized protocol/selection/summary objects.
- Use exclusive same-directory temporary files for cache and search exports,
  preserving the previous file on write/replace failure and avoiding same-process
  writer collisions. This does not add a whole-run transaction or coordinator lock.
- Reject ambiguous/oversized JSON configs, non-finite numbers (including exponent
  overflow), invalid search controls, invalid volatility targets and non-real
  schedules. CLI validation errors return actionable usage errors, not tracebacks.
- Add independent reproductions and regression coverage, enforce `uv.lock` in
  CI environment/test commands, and correct the outdated supported-version policy.

Compatibility: engine/checkpoint schema 5, base registry/user-version 1, portfolio
schemas and all 24 frozen accounting references are unchanged. Edge-case metrics
and the corrected next-open sensitivity output intentionally differ. Existing
source/version-locked runs need their original environment; upgrade with new
runs/protocols instead of bypassing checks. See [AUDIT_2026-09-03.md](AUDIT_2026-09-03.md)
for findings, validation evidence, migration cautions and remaining limitations.

## [0.14.0] - 2026-08-30

### Declared historical membership and registered portfolio studies

- Added strict, bounded membership manifests with known/effective dates,
  duplicate/conflict/redundancy checks, source/license/coverage declarations,
  raw and normalized SHA-256 provenance, and causal boolean eligibility.
- Membership changes force next-close signals between normal schedule dates;
  ineligible scores are masked. The matched-clock comparison portfolio also
  respects membership, with capped equal-weight rebalancing and residual cash.
  No-manifest portfolio signals and buy-and-hold behavior remain unchanged.
- Added `PortfolioStudyConfig`, `run_portfolio_study`, `PortfolioStudyRegistry`
  and `momentum-lab portfolio study` development/status/list/reveal commands.
  Freeze rule/data/membership/software before development; pass only copied
  development prefixes and sanitized rules to the evaluator before consent.
- Atomically reserve every candidate asset's test exposure in the existing
  shared registry, including inactive assets. One overlap blocks a fresh group
  claim. Preserve failed/interrupted history; explicit acknowledged reuse is
  labelled, never described as virgin OOS evidence.
- Carry cash, holdings and pending instructions across the test boundary.
  Export a prior-close anchor, include the first test return, exclude development
  fees from test totals and disclose both accounts' separate starting NAVs.
- Pin the first completed summary atomically, including out-of-order concurrent
  completion. Explicit replays validate frozen inputs and cached group integrity,
  log every asset and return summaries/reports without recomputing CSV books.
- Added two independently generated frozen membership/boundary ledgers, strict
  contract/causality/accounting/concurrency/interruption tests, clean core-wheel
  lifecycle CI, synthetic examples and English/Chinese documentation.
- Cross-Python oracle reconstruction permits at most eight ULPs only for
  derived Sharpe/volatility (stdlib rounding differs on Python 3.10); inputs,
  complete books, frozen hashes and production regression tolerances are unchanged.

Compatibility: single-asset engine/checkpoint schema 5, portfolio book schema 1,
base registry/user-version 1 and the 16 + 6 existing frozen cases are unchanged.
Portfolio registry tables and study/membership contracts have separate schema 1.
Older source/version-locked runs still need their original environment; upgrading
requires new runs/protocols, not silent resume. The coordinator reads/hashes raw
input files; sealing is a local workflow boundary, not encrypted custody.

Not included: independently verified historical membership, unbalanced IPO or
delisting histories, missing-price/halts settlement, portfolio parameter search,
selection-adjusted inference or live trading. Complete positive aligned prices
remain required outside membership; no missing quote or recovery value is invented.

## [0.13.0] - 2026-08-30

### Cross-sectional portfolios and shared-cash research

- Added fixed-rule multi-asset `PortfolioConfig` / `run_portfolio` APIs and
  `momentum-lab portfolio`: 2–64 strict offline datasets, identical sessions and
  currency/calendar/annualization/adjustment declarations, bounded work, and no
  implicit alignment, filling or online fallback.
- Added causal relative-momentum scores with optional skip-recent window and
  absolute filter, stable ticker tie-breaking, daily/weekly/monthly signals,
  top-k cash slots and per-rebalance target caps.
- Added a separate self-financing long-only portfolio book with delayed close
  fills, drift between instructions, shared cash, explicit paid two-sided costs
  and effective annual ACT/365 cash interest. No leverage, FX or broker orders.
- Added matched-warm-up equal-weight buy-and-hold comparison, complete CSV
  ledgers, final-vs-signal allocations, version/data/source/environment contracts
  and self-contained HTML/Markdown reports with methodological limitations.
- Require invocation-only whole-history acknowledgement; durably reserve every
  asset/date as development exposure before scoring. Failed/interrupted runs
  retain recorded history, never overwrite outputs and have no completion marker.
- Added six SHA-256-locked complete portfolio reference cases generated by an
  independent Fraction/algebraic oracle, causality/accounting/governance tests,
  Python 3.10–3.13 regression CI, a core-wheel lifecycle smoke, configuration
  example, bilingual quick starts and `PORTFOLIOS.md`.

Comparability: the new portfolio engine uses its own schema 1 and effective
ACT/365 cash compounding; the existing single-asset engine/checkpoint schema 5,
simple ACT/365.25 cash convention, registry schema 1 and sixteen frozen cases
are unchanged. Portfolio results are exploratory full-history simulations,
not fresh OOS evidence. This release does not add portfolio parameter search,
point-in-time constituents, portfolio liquidity/partial fills or live trading.
Existing source/version-locked single-asset resumes still require the original
software environment; use a new run for a changed source/version.

## [0.12.0] - 2026-08-30

### Offline datasets and source provenance

- Added `data import`, `data inspect`, `--dataset`, `import_dataset` and
  `load_dataset` for daily user-supplied CSV research, with no network fallback.
- Preserve original bytes; reject missing/extra columns, malformed rows,
  duplicate/unsorted/intraday dates, invalid prices/volume, changed hashes,
  unsupported manifests, path escapes and CSV symlinks. Imports never overwrite
  existing output directories. All rows are validated before range slicing.
- Record explicit source, usage terms, currency, calendar and price-adjustment
  declarations. Calendar annualization applies consistently to features and
  evaluation; conflicting overrides fail. Unadjusted/split-only series warn
  that corporate actions/distributions are not reconstructed.
- Bind original bytes and declarations into study/resume contracts while
  allowing relocation of unchanged snapshots. Config-file dataset paths resolve
  relative to their JSON file. Reports and returned results expose provenance.
- Preserve development-only candidate isolation, sealed study/reveal semantics
  and ticker/date observation history across local data sources.
- Add frozen synthetic CSV validation, malformed-input/provenance/resume tests,
  bilingual quick starts and an installed-wheel offline lifecycle CI smoke.

No accounting, ranking or uncertainty method changes; engine schema 5, registry
schema 1 and frozen accounting/statistical references remain unchanged. New
source/provenance/version identity requires fresh runs when upgrading. Source
and license declarations are not verified rights, data-quality certification,
point-in-time evidence or tamper-proof custody. See `DATASETS.md`.

## [0.11.0] - 2026-08-30

### Research registration and test-access audit

- Added `--study-id` protocols that bind data, candidate grids, evaluation rules
  and numerical/source identity before candidate work, then freeze the winner.
- Registered searches withhold test evaluation/results until a separate explicit
  reveal. Reveal/reuse acknowledgement is invocation-only, not JSON-configurable.
- Added a shared SQLite observation registry independent of run/result directories;
  overlap checks ignore data hashes/run IDs and include prior development ranges.
- Reserved possible exposure durably before full-data evaluation. Concurrent
  claims are serialized, interrupted attempts remain recorded, and repeated
  evaluations require an explicit acknowledgement and recorded reason.
- Cached completed reveals preserve their original evaluation timestamp; repeat
  views are logged and labelled previously revealed, never another first test.
- Added score-free study list/status/history commands and idempotent legacy
  artifact import. Old files are not rewritten and unknown history is not certified.
- Added report access-audit sections and defensive test-field masking. Legacy
  automatic evaluation stays available, with recorded access and history warnings.
- Restricted parameter-sensitivity data/bounds to development observations.
- Added governance tests and an offline installed-wheel lifecycle smoke in CI.

Registry schema 1 is local accident-prevention/audit infrastructure, not encrypted
or tamper-proof custody. Candidate ranking, engine schema 5, accounting ledgers
and frozen statistical references are unchanged. New registry identity fields
make old checkpoint migration explicit; see `GOVERNANCE.md`.

## [0.10.0] - 2026-08-30

### Conditional return uncertainty

- Added paired circular moving-block percentile intervals for unrounded Sharpe,
  arithmetic annualized mean return, buy-and-hold statistics and annualized
  mean excess return. Both series share every resampling index.
- Final selected-strategy validation and test windows are diagnosed separately
  using existing net-return ledgers, without refitting, replaying trades,
  reselecting candidates or joining the two windows.
- Added recorded PCG64 seeds, block lengths, confidence levels, return-snapshot
  hashes, sample counts, valid-replicate counts and explicit unavailable states.
  Short samples, near-zero variance, degenerate draws and work-budget limits
  never produce fabricated intervals or silently discard observations/draws.
- Added bounded batches and a 50-million-drawn-cell work limit, validated before
  resampling. Bootstrap options are CLI/configurable and resume-locked.
- Added Markdown/HTML uncertainty sections alongside the existing analytic
  Sharpe interval, with explicit post-selection and stationarity limitations.
- Added a frozen correlated-return reference computed by an independent scalar
  oracle, paired/index/causality/configuration tests and a clean-wheel API smoke.

Existing candidate ranking, multiple-testing penalties, backtest accounting,
the 16 frozen software benchmarks and engine/checkpoint schema 5 are unchanged.
These conditional intervals do not correct strategy selection bias or repeated
holdout access and do not establish historical or future profitability.

## [0.9.0] - 2026-08-30

### Frozen regressions and experiment comparison

- Added the offline `momentum-lab benchmark` command and Python benchmark API.
  Four frozen, project-created synthetic OHLCV fixtures exercise weekday/daily
  calendars, trends/reversals, jumps, short insolvency, borrow restrictions,
  zero-volume bars, and partial fills across cash, buy-and-hold, MA cross and
  fixed TSMOM cases. These are software tests, not historical investment evidence.
- Locked input bytes with SHA-256 and bound all parameters, annualization,
  cache settings and execution assumptions to a versioned contract hash.
- Shipped complete reviewed reference ledgers from accounting engine schema 5.
  Fixed numerical tolerances compare every metric and every bar of targets,
  positions, returns, equity, turnover, costs, participation and constraints.
  Better returns are changes too; incomplete or incompatible inputs fail closed.
- Added portable JSON snapshots, machine-readable differences and Markdown
  reports with current/reference source identities and runtime environments.
  New output directories and atomic file writes preserve earlier artifacts.
- Added fresh-cache repeatability checks, wall-time/tracemalloc observations
  and opt-in resource ratio gates. Memory is traced allocation, not process RSS.
- Added frozen-suite gates and downloadable reports to all four CI Python
  versions, plus fixture verification from a clean installed wheel outside the
  source checkout. No strategy, selection or accounting semantics changed;
  the existing search engine/checkpoint schema remains 5.

## [0.8.0] - 2026-08-29

### Scale-aware search

- Added deterministic Successive Halving with configurable per-strategy
  candidate budgets, reduction factor, and increasing validation prefixes.
  Partial-resource rows never enter final selection; only the full-development
  survivors are promoted to the canonical result table.
- Counts every staged evaluation in the Deflated-Sharpe multiple-testing
  hurdle and estimates candidate-score dispersion from the equal-resource
  initial stage, rather than pretending only the final survivors were tried.
- Added transactional SQLite stage journaling, deterministic resume, and a
  `search_stages.csv` audit trail with resource and advancement decisions.
- Candidate processes now receive development observations only. Test prices,
  boundaries, and metrics remain sealed until one final candidate is selected.

### Computation reuse

- Added a bounded per-process indicator DAG shared across strategies, including
  returns, SMA/EMA/WMA/DEMA, volatility, RSI, channels, ATR, ADX, and dependent
  nodes. A zero-sized cache preserves the uncached execution path.
- Added staged-search and indicator-cache diagnostics to JSON, Markdown, and
  HTML reports, plus CLI and `SearchConfig` controls.
- Advanced the research engine schema to 5 and package version to 0.8.0.

## [0.7.0] - 2026-08-29

### Market realism

- Added quoted bid/ask spread, nonlinear participation-based impact, bar-volume
  capacity limits with partial fills, starting NAV, and per-rebalance minimum
  fees. Legacy cost behavior remains the default until these inputs are enabled.
- Added time-varying cash, financing, financing-spread, borrow-fee, and
  collateral-rebate schedules to the Python backtest API. Schedules are
  forward-filled from known observations only.
- Added dated borrow-availability controls. Unavailable borrow blocks new short
  targets and requests an orderly cover, subject to the same liquidity limit.
- Exposed requested versus filled turnover, actual filled positions,
  transaction costs, participation, capacity constraints, and borrow blocks in
  backtest output. Search selection now measures actual filled exposure.

### Research outputs

- Added self-contained `report.md` and `report.html` files with validation,
  sealed-test and benchmark evidence, selection diagnostics, parameter
  sensitivity, and complete execution assumptions.
- Added CLI and `SearchConfig` controls for the new execution model, and locked
  every result-affecting field into resume compatibility.
- Advanced the accounting engine schema to 4 and package version to 0.7.0.

## [0.6.0] - 2026-08-29

### Research integrity

- Withheld the sealed test boundary from all candidate workers and removed test
  columns from checkpoints and validation-ranked outputs.
- Added temporal validation folds, expanding walk-forward selection replay,
  Deflated Sharpe selection, analytic Sharpe intervals, CSCV/PBO diagnostics,
  and minimum evidence gates.
- Replaced lexicographic quick sampling with a deterministic Latin-hypercube
  design and reused base signals across smoothing variants.

### Accounting and data correctness

- Made bankruptcy absorbing and added drift-aware target-weight rebalancing.
- Accrued financing by elapsed calendar days and separated short cash,
  collateral rebates, financing, and borrow fees.
- Added explicit same-close, next-close, next-open, and delayed-close models.
- Preserved one continuous ledger across train/validation/test boundaries.
- Restored ML predictions on the forward-label-unknown tail.
- Added strict OHLC invariants and persistent late-listing cache metadata.

### Reproducibility and operations

- Replaced CSV as the canonical resume checkpoint with transactional SQLite;
  CSV and JSON artifacts are now atomic exports.
- Recorded the dependency lock hash and Python/data-science runtime versions;
  source identity, rather than an incidental Git commit SHA, controls resume.
- Added typed provider-unavailability errors and weekly provider-contract CI.
- Raised the aggregate coverage gate from 75% to 78%.

## [0.5.0] - 2026-08-29

### Changed

- Renamed the distribution to `momentum-research-lab` because the old PyPI name
  belongs to an unrelated project; import and CLI names remain unchanged.
- Made quick, non-ML, streamed searches the safe default.
- Removed risk sizing from the alpha parameter grid and changed the default
  risk-free hurdle to zero.
- Increased ML grid warm-up windows to 252/504/756 observations.
- Moved scikit-learn and XGBoost behind optional extras.
- Reframed the old robustness grade as local parameter sensitivity.

### Fixed

- Enforced the final leverage cap after all strategy-level transformations.
- Applied financing only to exposure above 1x and credited explicit cash returns.
- Included first-bar entry costs in maximum drawdown.
- Converted inclusive public end dates to yfinance's exclusive end convention.
- Inferred 365 periods for common crypto pairs and 252 otherwise.
- Added bounded download retries, atomic cache writes, and a user cache location.
- Made checkpoint columns fixed and resume compatibility source/schema aware,
  including a package-source fingerprint outside clean Git checkouts.
- Added benchmark metrics, error counts, and `summary.json` to run outputs.
- Rejected invalid KNN sample/neighbor configurations before execution.

### Validation

- Added regression tests for leverage, financing, cash, drawdown, inclusive
  dates, annualization, checkpoint schema, source-locked resume, safe defaults,
  dependency separation, and KNN grids.
- Added Python 3.13, wheel-install smoke testing, and a 75% coverage CI floor.

## [0.4.0] - 2026-08-27

- Config-driven resumable runs, data snapshot hashes, streamed checkpoints, and
  the original local parameter-perturbation report.
