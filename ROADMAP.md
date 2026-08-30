# Momentum Lab Roadmap

Momentum Lab is moving from an “optimal strategy finder” prototype toward a
reproducible research and evidence-evaluation workbench. Configuration count is
not a success metric. The project should optimize for honest out-of-sample
evidence, explicit assumptions, and repeatability.

## v0.5 — Safety and reproducibility

- Safe quick/non-ML/streaming defaults.
- Explicit opt-in for exhaustive and ML searches.
- Final portfolio leverage cap and separation of signal search from risk sizing.
- Cash, excess-leverage financing, borrow, transaction-cost, and slippage inputs.
- Inclusive end dates, crypto annualization inference, retrying downloads, and
  atomic user-cache storage.
- Fixed checkpoint schema and source/data/config-locked resume.
- Lightweight core package with optional ML/XGBoost dependencies.
- Clean-wheel CI, Python 3.10—3.13, and a coverage floor.
- Accurate parameter-sensitivity terminology and documented limitations.

## v0.6 — Research integrity and accounting correctness

- Test boundaries are withheld from candidate workers and test metrics are
  absent from candidate artifacts; only the copied final selection is tested.
- 40/40/20 development split with temporal validation folds and expanding
  walk-forward parameter-selection replay.
- Deflated Sharpe selection, analytic 95% Sharpe intervals, CSCV/PBO estimate,
  and minimum observations/trades/exposure gates.
- Absorbing bankruptcy, calendar-day financing, drift-aware rebalancing, and
  explicit short cash/collateral accounting.
- Same-close, next-close, next-open, and delayed-close execution assumptions.
- Transactional SQLite resume journal, atomic report exports, environment and
  lockfile fingerprints, deterministic Latin-hypercube quick sampling, and
  reusable base-signal calculations.
- Strict OHLC data contracts, reusable late-listing cache metadata, and weekly
  provider-contract CI.

## v0.7 — Market realism

Delivered in v0.7.0:

- Time-varying cash, financing, borrow-fee and rebate schedules; financing
  spreads; and dated borrow availability in the Python API.
- Spread, nonlinear participation impact, bar-volume capacity, partial fills,
  and minimum-fee models, wired through search configuration and CLI defaults.
- Automated self-contained HTML and Markdown research reports.

## v0.8 — Scale-aware search

Delivered in v0.8.0:

- Bounded cross-strategy indicator DAG caching for common price, trend,
  volatility, channel, ATR, and ADX dependencies.
- Deterministic Successive Halving with per-strategy candidate budgets,
  increasing validation prefixes, transactional stage resume, and explicit
  stage-decision exports.
- Candidate isolation now withholds test observations as well as test metrics
  and boundaries until one final selection has been made.

Next scale and evidence work:

- Optional Optuna integration behind the same sealed-development contract.
- Optional Parquet/DuckDB analytics exports on top of the SQLite journal.
- Typed `StrategySpec`, `RunConfig`, `RunResult`, and plugin entry points.
- Cross-asset and cross-regime frozen benchmark suites.
- Block-bootstrap intervals and correlated effective-trial estimates.

## v1.0 — Stable research platform

- Stable public API and migration policy.
- Licensed multi-provider data adapters and point-in-time metadata.
- Portfolio and cross-sectional momentum, including true relative momentum.
- Pre-registered research hypotheses and controlled OOS access.
- Reproducible public benchmarks and third-party result review.
- Signed releases, trusted publishing, SBOM/provenance, and complete documentation.

## Post-1.0 — Paper trading only after validation

Broker adapters, event-driven orders, drift monitoring, and kill switches should
start with paper accounts. Live execution is out of scope until research and
execution assumptions have survived sustained forward testing.

## Release gates

Every release must pass lint, offline tests, the coverage floor, wheel build and
clean installation, and deterministic smoke research. Research-method changes
must include a frozen regression dataset and an explanation of how historical
comparability is affected.
