# Frozen benchmark contract

The `synthetic-core-v1` suite is a **software-compatibility regression suite**.
It is not a historical backtest study, investment-performance leaderboard,
parameter search, statistically independent holdout, or proof of correctness.
Repeated access to these public fixtures conveys no new investment evidence.

## What is frozen

Four 128-bar, project-created JSON OHLCV snapshots ship in both source and wheel
distributions under `momentum_lab/benchmark_data/`. Values were constructed from
integer-cent trends, modular oscillations and explicit jumps, then frozen as
literal observations. No PRNG, data provider, cache or current date changes them.
The dates start on 2020-01-02 only as convenient synthetic calendar labels; they
do not describe events or asset returns on those dates. All data is MIT-licensed.

| Dataset | Calendar / annualization | Deliberately exercised behavior |
|---|---|---|
| equity_trend | Weekdays / 252 | Uptrend, selloff, recovery, drift rebalancing |
| commodity_range | Weekdays / 252 | Choppy reversals, short borrow restrictions |
| crypto_jumps | Every day / 365 | Large discontinuities, leveraged short insolvency |
| illiquid_stress | Weekdays / 252 | Zero volume, partial fills, spread, impact, minimum fee |

Weekdays exclude Saturday/Sunday but do **not** model exchange holidays.
Financing/cash accrual uses actual elapsed calendar time in all four fixtures.
The strategies are fixed cash, buy-and-hold, 5/20 long-only SMA cross, and a
12-bar long/short TSMOM with one skipped recent bar and fixed smoothing/sizing.
No fitting or ranking changes those parameters. All cases use next-close
execution. Exact costs, risk caps, annualization, borrow-unavailability intervals
and indicator-cache settings are in `suite_v1.json` and embedded in every snapshot.

Raw dataset bytes are checked against SHA-256 before execution. `.gitattributes`
preserves LF endings in Windows checkouts. A canonical JSON hash additionally
binds the entire suite contract, so changes to costs, calendars, parameters or
data hashes cannot silently be compared as a new implementation of the same task.

## Comparison semantics

- `snapshot.json`: contract, data hashes, package/source identity, Python and
  dependency environment, every metric, complete ledgers, resource observations.
- `comparison.json`: reviewed-reference provenance, environment-change flag,
  metric deltas, changed ledger fields/first bar/count/maximum absolute delta,
  resource ratios and outcome. Numeric deltas are current minus reference.
- `report.md`: readable scenario/result tables, differences and limitations.

The ledger contains all 128 targets, returns, equity values, actual positions,
actual/requested turnover, transaction costs, participation and constraint flags.
The reference stores full-precision values; comparisons use fixed `rtol=1e-9`,
`atol=1e-11`. Boolean flags compare exactly. Numerical changes fail in **either**
direction: a higher Sharpe or return is not an automatic pass. This can reveal
an intentional method change as well as a bug; a failure is a review signal.

Input-contract differences, missing/extra cases, duplicate IDs, missing metrics,
wrong ledger lengths, invalid schemas and non-finite numbers are incomparable,
not silently skipped. Both files are validated even if equally incomplete.
Package/source/environment changes are recorded but do not make otherwise
identical input contracts incomparable: comparing versions is the purpose.

CLI exit codes are 0 (compatible), 1 (numerical or enabled resource gate changed),
and 2 (incomparable or invalid). `python -m momentum_lab benchmark` propagates the
same exit code. Reports are generated for numerical and compatibility failures
when the current suite completed successfully. Invalid CLI input, a corrupt
packaged reference or execution failure may prevent report generation.

## Runtime and memory

`--repeat N` repeats each case with a fresh indicator cache and checks numerical
identity across repeats. It records median elapsed seconds and the maximum
`tracemalloc` allocation peak over those repeats. The measured region includes
signal creation, backtest execution, metric calculation and ledger extraction;
it excludes fixture loading, metadata gathering and file output. Tracing itself
adds overhead. The peak is not process RSS and may omit native allocations.
Run serially; the tracer is process-global. The Python API can turn measurement
off when embedding in a separately profiled application.

Resources are observational by default. To gate measured, same-machine runs:

```bash
momentum-lab benchmark --repeat 3 --output experiments/benchmarks/before
# Change/install the implementation under review, preserve the previous files.
momentum-lab benchmark --repeat 3 \
  --compare experiments/benchmarks/before/snapshot.json \
  --max-slowdown 1.5 --max-memory-growth 1.5 \
  --output experiments/benchmarks/after
```

Limits are current/reference ratios checked per case. Missing measurements fail
closed when a limit is requested. The packaged reference deliberately has none:
CI runner load, OS and dependency versions are not a stable performance baseline.
Match environments, repeat counts and tracing settings before interpreting ratios.

## Python API

```python
from momentum_lab.benchmarks import (
    compare_benchmarks,
    load_benchmark_reference,
    run_benchmarks,
    write_benchmark_report,
)

current = run_benchmarks(repeats=3)
comparison = compare_benchmarks(current, load_benchmark_reference())
write_benchmark_report(current, comparison, "experiments/benchmarks/python-run")
assert comparison["status"] == "passed"
```

Output directories must be new. Files are written atomically one by one; a
failed export can leave a partial directory, but never overwrites a prior run.
The command has no automatic rebaseline option and cannot replace bundled data.

## Reference-review and release policy

The v1 expected ledgers were captured with the v0.9.0 harness using the unchanged
v0.8.0 accounting engine (schema 5). The reference records the engine's GitHub
base commit as well as the harness source hash. v0.9.0 does not change strategy,
accounting or search-selection semantics. Existing checkpoints still undergo the normal source-fingerprint
compatibility check; this does not promise resume across package upgrades.

Golden files alone can preserve an old bug. Tests therefore also use independent
cash-accrual and buy-and-hold entry-price oracles, prefix-causality checks, cached
versus uncached equality, bankruptcy and capacity invariants, and deliberate
corruption/change injections to check that the harness actually fails.

1. Keep the old reference and run the proposed implementation against it.
2. Explain each changed metric/ledger path in a reviewed commit or pull request;
   include the accounting or methodological reason and independent checks.
3. If **inputs** change, add a new versioned dataset/suite, retain the old one,
   and disclose that the two contracts are not directly comparable.
4. If only implementation semantics intentionally change, update the reviewed
   expected fragments and reference provenance together. Never accept a new
   reference solely because returns improved or to turn CI green.
5. Run offline tests, coverage, all Python CI variants and the installed-wheel
   benchmark. CI uploads `frozen-benchmarks-python-*` artifacts even after a
   comparison failure when reports are available.

Future work remains: licensed historical cross-asset/regime benchmarks, search
selection benchmarks, block-bootstrap uncertainty, correlated-trial estimates,
and persistent out-of-sample reveal governance. This small fixed synthetic suite
is not a substitute for any of those, or for prospective paper trading.
