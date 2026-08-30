# Changelog

All notable changes are documented here. The project follows semantic versioning
for its public Python API and run/checkpoint schema.

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
