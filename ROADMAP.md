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

## v0.6 — Research integrity

- Nested expanding and rolling walk-forward selection.
- Purged/embargoed inner validation and a final outer OOS report.
- Probabilistic and Deflated Sharpe Ratios.
- CSCV / Probability of Backtest Overfitting estimates.
- Effective independent-trial estimates for correlated parameter grids.
- Bootstrap confidence intervals and minimum track-record requirements.
- Cross-asset and cross-regime benchmark suites.
- A sealed test window that is not written for every candidate configuration.

## v0.7 — Execution realism and scale

- Explicit next-close, next-open, and MOC-with-delay execution models.
- Time-varying cash rates, financing spreads, short rebates, and borrow availability.
- Spread, nonlinear impact, liquidity, capacity, and minimum-fee models.
- Feature, indicator, and base-signal caching.
- Budgeted staged search, early stopping, successive halving, and optional Optuna.
- Parquet/DuckDB experiment storage with atomic batch journaling.
- Typed `StrategySpec`, `RunConfig`, `RunResult`, and plugin entry points.
- Automated HTML/Markdown research reports.

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
