# Conditional return uncertainty (v1)

The paired circular moving-block bootstrap estimates uncertainty **conditional
on a fixed selected strategy and its observed net-return series**. It is not
selection-adjusted inference, proof against overfitting, a probability of future
profit, or a replacement for prospective testing. Validation has already been
used for selection; its interval must not be interpreted as independent evidence.

## Estimands and pairing

For net simple returns `r`, periods per year `A`, and annual risk-free rate `rf`:

- Arithmetic annualized mean: `A * mean(r)`, **not CAGR or compounded return**.
- Unrounded annualized Sharpe: `(A * mean(r) - rf) / (sqrt(A) * std(r, ddof=1))`.
- Paired mean excess: `A * mean(strategy_returns - benchmark_returns)`. This is
  a return difference, not a ratio, CAGR difference or difference in Sharpes.

The Sharpe estimand is consistent with the existing annualization convention,
but does not use `evaluate`'s display rounding, denominator epsilon, or the
search's inactive-strategy sentinel. Zero/near-zero volatility is undefined,
not an artificially large Sharpe or a zero-valued confidence interval.

Every replicate draws `ceil(n / block_length)` uniformly sampled start indices
with replacement. Each start contributes consecutive observations of the fixed
block length; indices wrap modulo `n` within that window. The concatenation is
truncated to exactly `n` observations. The **same indices** sample both strategy
and benchmark. Estimates are recomputed for each replicate; fixed percentile
endpoints use NumPy's `linear` interpolation at `(1-confidence)/2` and
`(1+confidence)/2`. The final incomplete block is truncated, never silently
dropped. The v1 generator is explicitly `Generator(PCG64(seed))`.

Inputs are pandas Series with identical, sorted, unique, non-missing single
indexes and finite real returns. Misalignment, missing values, boolean/complex
returns and duplicate dates are errors, not implicitly joined, dropped or filled.
If no benchmark is supplied to the Python API, its statistics are explicitly
`not_provided`; strategy-only statistics can still be estimated.

## Where it runs

Search first ranks candidates with the existing validation/Deflated-Sharpe
logic, copies the final selection, and evaluates its continuous full ledger
and the buy-and-hold ledger once each. Bootstrap reads slices of those existing
net-return ledgers **after** selection. It does not rerun strategy signals,
refit models, reselect candidates or replay executions on synthetic price paths.

Validation and sealed-test windows are processed separately with the configured
seed. A validation block never draws a test bar or vice versa. Trading costs,
funding, partial fills and the state carried across period boundaries already
exist in the original ledger; sampling their realized return effects is not a
new hypothetical execution simulation. Changing the seed changes intervals,
not candidates, point estimates or accounting.

Diagnostics are written only to final `summary.json`, the final Python return
value (`bootstrap_diagnostics`), and Markdown/HTML reports. Candidate CSVs,
validation-ranked top results and SQLite candidate payloads do not contain them.
The existing analytic validation-Sharpe interval remains separately labelled.

## Configuration and replay

| SearchConfig / run_search | CLI | Default |
|---|---|---:|
| bootstrap | --no-bootstrap to disable | true |
| bootstrap_resamples | --bootstrap-resamples | 2000 |
| bootstrap_block_length | --bootstrap-block-length | 10 bars |
| bootstrap_confidence | --bootstrap-confidence | 0.95 |
| bootstrap_seed | --bootstrap-seed | 42 |
| bootstrap_min_observations | --bootstrap-min-observations | 60 |

All six options are validated before market-data access, recorded in
`run_config.json`, and locked for resume, including disabled configurations.
Changing them requires an explicitly new run; doing so does not make a
previously viewed test set fresh. New parameters are appended to the existing
Python signatures to preserve positional compatibility. Bootstrap does not
affect ranking and can be disabled independently of Markdown/HTML generation.

Each period records the method/schema, original count, required count, nominal
block count, requested/completed replicates, per-statistic valid replicates,
confidence, seed, generator, quantile rule, NumPy version, annualization,
risk-free rate, boundary labels and a hash of the diagnosed indexed return pair.
Full source, dependency and price-data identity remain in the normal run manifest.
Reproducibility requires matching data/settings and compatible numerical versions.

```python
from momentum_lab import paired_block_bootstrap

# strategy_returns and benchmark_returns: identically indexed net daily Series
diagnostic = paired_block_bootstrap(
    strategy_returns, benchmark_returns,
    n_resamples=2000, block_length=10, confidence_level=0.95, seed=42,
    annualization=252, risk_free_rate=0.0, min_observations=60,
)
excess = diagnostic["statistics"]["annualized_mean_excess"]
print(excess["estimate"], excess["ci"], excess["status"], excess["reason"])
```

## Explicit unavailable states and resource bounds

- At least `max(min_observations, 5 * block_length)` observations are required.
  Five blocks is a conservative **nominal gate**, not five independent samples
  or a guarantee of nominal coverage. Short/empty inputs retain any defined
  point estimate but return null bounds and `insufficient_data`.
- Resamples must be integers in 200–20000; seeds are integers in 0–2^32-1.
  Confidence is strictly between 0 and 1 with at least five expected draws in
  each tail. This is a resolution guard, not a Monte Carlo accuracy guarantee.
- Observed standard deviation at or below 1e-12 per bar is `zero_variance`.
  Constant paired differences also get null bounds rather than implying future
  certainty from a zero-width empirical interval.
- If any replicate's statistic is undefined, that statistic has no interval;
  its valid-replicate count and `degenerate_resamples` reason are reported.
  No undefined draws are silently discarded. Other valid statistics can remain
  available. Numerically collapsed distributions also have no interval.
- Arrays are batched with a target of 262144 drawn index cells per batch, plus
  one-batch minimum and linear input/output storage. A total cap of 50 million
  cells, including padded final blocks, is checked before drawing replicates.
  Exceeding it produces `resource_limit`, without changing the requested method.
  Integer options are normalized before arithmetic to prevent NumPy overflow
  bypassing the work limit. No automatic downsampling or block-size tuning occurs.

Period status is `ok` when all requested intervals exist, `partial` when some
exist, and `unavailable` when none do; sample/workload gates have their specific
period statuses. Reports expose per-statistic reasons rather than replacing
missing intervals with zero. Search with no eligible winner or disabled
diagnostics records `no_selection` or `disabled` without running bootstrap.

## Interpretation and validation

Block resampling assumes approximately stationary, weakly dependent returns.
It preserves dependence within blocks, not arbitrarily long memory, changing
regimes, or all path-dependent economic constraints. Circular wrapping joins
the window end to its beginning. Block length 1 reduces to IID resampling and
does not preserve serial dependence. Choose block length and other settings
before viewing results; never pick the interval that looks best on held-out data.

The fixed fixture `tests/fixtures/block_bootstrap_v1.json` contains synthetic
correlated strategy/benchmark returns and an independent scalar oracle using
explicit block indexing, Python `statistics.fmean/stdev`, and manually
interpolated percentiles. Tests also cover pairing, deterministic batches,
null intervals, budget limits, boundaries, unchanged selection and resume locks.
The fixture tests implementation, **not empirical 95% coverage in markets**.
The original 16 accounting benchmark ledgers and search engine schema 5 are
unchanged. Their compatibility does not validate the new statistical assumptions.

Remaining work includes selection-aware inference, correlated effective trial
counts, licensed historical benchmarks and persistent test-reveal governance.
