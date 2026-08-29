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

## v0.7 — Market realism and scale

- Time-varying cash rates, financing spreads, and borrow availability.
- Spread, nonlinear impact, liquidity, capacity, and minimum-fee models.
- Feature and indicator DAG caching beyond the v0.6 base-signal reuse.
- Budgeted staged search, early stopping, successive halving, and optional Optuna.
- Optional Parquet/DuckDB analytics exports on top of the SQLite journal.
- Typed `StrategySpec`, `RunConfig`, `RunResult`, and plugin entry points.
- Automated HTML/Markdown research reports.
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
