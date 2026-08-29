# Changelog

All notable changes are documented here. The project follows semantic versioning
for its public Python API and run/checkpoint schema.

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
