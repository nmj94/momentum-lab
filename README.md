# Momentum Lab

**Reproducible momentum-strategy research and parameter-sensitivity evaluation.**

[English](README.md) | [中文](README_CN.md)

Momentum Lab is a research workbench for comparing classic momentum rules,
experimental machine-learning models, and regime-conditioned strategies under
explicit data, execution, cost, and risk assumptions.

It does **not** claim to discover a universally optimal strategy. A high
historical score can be caused by selection bias or overfitting, especially
when many configurations are tried. Results are research evidence, not trading
advice.

## Safe quick start

The project is currently installed from source. The old PyPI name
`momentum-lab` belongs to an unrelated package; do not install it.

```bash
git clone https://github.com/nmj94/momentum-lab.git
cd momentum-lab
pip install -e .

# Default: 5 deterministic Latin-hypercube configurations per non-ML strategy,
# streamed to disk without retaining the full result set in RAM.
momentum-lab GLD

# Search a chosen set
momentum-lab SPY --strategies tsmom,ma_cross,rsi,regime_aware

# Full non-ML grid (159,668 experiments; explicitly opt in)
momentum-lab GLD --exhaustive --workers 4

# Budgeted search: 256 candidates/strategy, then deterministic 3x halving
momentum-lab GLD --successive-halving --workers 4
```

Machine-learning strategies are experimental and no longer run or install by
default:

```bash
# scikit-learn strategies
pip install -e ".[ml]"
momentum-lab SPY --strategies ml_logreg,ml_rf

# XGBoost strategy
pip install -e ".[xgb]"
momentum-lab SPY --strategies ml_xgb

# All 26 strategies; exhaustive ML grids can take days
momentum-lab SPY --all-strategies
momentum-lab SPY --all-strategies --exhaustive --workers 8
```

The future PyPI distribution name is `momentum-research-lab`; the Python import
and CLI remain `momentum_lab` and `momentum-lab`.

## v0.15 — Run ownership and recovery visibility

Single-asset searches, exploratory portfolios and registered portfolio studies
now protect each output directory with a whole-run local OS lock. Competing
coordinators fail before writing to the same output; interrupted attempts retain
their history. Completion records include file sizes and SHA-256 receipts.

```bash
# These commands never reveal scores, resume a search or recalculate a test.
momentum-lab runs status experiments/gld-dev
momentum-lab runs history experiments/gld-dev --limit 10
momentum-lab runs status experiments/gld-dev --verify
```

Default inspection reads metadata only; `--verify` hashes the latest completed
artifact set without displaying its contents. See [RUNS.md](RUNS.md) for states,
recovery, backup and filesystem limits. This is not a whole-directory transaction,
distributed lease or tamper-proof signature. Research access history and explicit
reveal/reuse consent are unchanged. Upgrade with new runs/protocols; old
source/version-locked research still needs its original environment.

## v0.14.1 integrity fixes

Fixed symbolic-ticker cache collisions and CSV precision drift, constant-return
and bankruptcy metrics, next-open sensitivity pricing, unsafe resume without
provenance, incomplete rerun backups, and single-asset reveal cache ordering.
Added strict input/config validation and collision-free atomic file writes;
CI now enforces the committed dependency lock. The 24 frozen accounting cases
remain unchanged. See [the audit](AUDIT_2026-09-03.md) for reproductions, tests,
compatibility notes and limitations. Use new runs/protocols after upgrading;
do not bypass the source/version checks on older research.

## What changed in v0.14

- Added fixed-rule **registered portfolio studies**: develop first, explicitly
  reveal later, and record all candidate assets' test access atomically in the
  existing shared registry. Interrupted access and cross-study reuse remain visible.
- Added declared historical membership with known/effective dates. Both momentum
  selection and the comparison portfolio respect eligible membership; changes
  trigger a delayed rebalance even between regular signal dates.
- Test accounting carries holdings, cash and pending instructions across the
  boundary, includes the first test return and reports each account's starting
  NAV. Reopening a completed study uses the frozen summary, never a new backtest.
- Added two independent frozen membership/boundary cases, lifecycle and concurrent
  access tests, and an installed core-wheel check. The old 16 + 6 references remain unchanged.

```bash
# Adapt synthetic examples to aligned snapshots and membership data you may use.
momentum-lab portfolio study --config examples/portfolio_study_config.json --run-id development
momentum-lab portfolio study status relative-momentum-study-v1
momentum-lab portfolio study --config examples/portfolio_study_config.json --run-id test-reveal --reveal-test
momentum-lab portfolio study benchmark
```

See [PORTFOLIO_STUDIES.md](PORTFOLIO_STUDIES.md) and [MEMBERSHIP.md](MEMBERSHIP.md).
This is local workflow/audit control, not proof of untouched OOS data or verified
point-in-time history. Complete positive aligned prices are still required even
outside membership; IPO gaps, missing delisting prices and liquidation payouts
are **not** modeled. No portfolio parameter search or live execution is added.

## What changed in v0.13

- Added multi-asset cross-sectional momentum: rank relative returns, optionally
  filter negative momentum, and choose capped top-k allocations with cash for
  unfilled slots. Daily, weekly and monthly signals use observed history only.
- A shared-cash, long-only portfolio book handles delayed close fills, actual
  weight drift, paid two-sided trading costs and cash interest. It is not an
  average of independently funded single-asset backtests.
- Export account/holding/trade ledgers, signal and realized weights, and offline
  HTML/Markdown reports against a matched-warm-up equal-weight buy-and-hold book.
- This first portfolio workflow is explicitly **exploratory whole-history
  research**, not a sealed OOS study or live trading. Invocation-only consent
  records every asset's evaluated dates as development exposure before scoring.
- Six independently calculated frozen portfolio ledgers supplement the
  unchanged sixteen single-asset regressions.

```bash
# First import aligned, same-currency snapshots you may legally use.
# Adapt the supplied example's tickers, dates, paths and cost assumptions.
momentum-lab portfolio --config examples/portfolio_config.json --acknowledge-history
momentum-lab portfolio benchmark
```

See [PORTFOLIOS.md](PORTFOLIOS.md) for data alignment, formulas, consent,
Python APIs, output files and limitations. The target cap can drift between
rebalances; cash uses effective annual ACT/365, unlike the single-asset engine's
simple ACT/365.25 convention. v0.14 adds declared membership and fixed-rule
registered studies; verified point-in-time data, portfolio parameter selection,
FX and live execution remain out of scope.

## What changed in v0.12

- Added offline CSV snapshots with explicit source, usage terms, currency,
  daily calendar and price-adjustment declarations. Yahoo remains optional at
  runtime: local datasets never fall back to a download.
- Strict OHLCV/date validation and SHA-256 checks reject malformed or changed
  inputs. The exact imported CSV bytes are preserved in a new output directory.
- Dataset provenance is locked into studies/resume and appears in JSON,
  Markdown and HTML. Moving an unchanged snapshot is supported; changing its
  bytes or declarations requires a new run.
- Local datasets follow the existing sealed-test/reveal/audit workflow. Source
  and license strings are declarations, not verified quality or usage rights.

```bash
momentum-lab data import my-prices.csv --output datasets/spy-v1 --ticker SPY \
  --source "My licensed export" --license "Private research use" \
  --currency USD --calendar exchange --price-adjustment split_and_dividend_adjusted
momentum-lab data inspect datasets/spy-v1/manifest.json
momentum-lab SPY --dataset datasets/spy-v1/manifest.json --start 2020-01-01 \
  --study-id spy-local-v1 --run-id spy-local-v1
```

CSV format: `date,open,high,low,close[,volume]`. Only import data you may use;
no licensed historical market dataset is bundled. See [DATASETS.md](DATASETS.md)
for daily-date semantics, adjustment/volume caveats, explicit reveals and APIs.

## What changed in v0.11

- Added registered studies with fixed data, strategy-space and evaluation
  protocols; `--study-id` withholds test evaluation until an explicit reveal.
- Shared SQLite observation history follows overlapping ticker/date ranges
  across run IDs, data revisions and result directories. Previously used
  development periods also count; interrupted reveals are never forgotten.
- Repeated evaluations need acknowledgement and a reason. Reopening the same
  fixed result uses the cached reveal and logs the replay, never a new first use.
- Reports distinguish sealed, first-recorded, previously revealed, repeated-use
  and unknown-history evidence. Existing unregistered commands still run, but
  now record access and explicitly warn about their history limits.

```bash
# Register and select using development data; no test metrics are computed.
momentum-lab GLD --study-id gld-2026q3 --run-id gld-dev --end 2026-08-28

# Inspect metadata, then explicitly reveal the already-frozen selection.
momentum-lab study status gld-2026q3
momentum-lab GLD --study-id gld-2026q3 --run-id gld-dev --end 2026-08-28 \
  --resume --reveal-test

momentum-lab study history --ticker GLD
momentum-lab study import-legacy experiments/old-run
```

Keep one shared registry and fixed search settings. Reveal/reuse consent is
invocation-only, never enabled by a JSON config. First **recorded** reveal is
not proof of untouched data: external history remains unknown and local files
are not tamper-proof custody. See [GOVERNANCE.md](GOVERNANCE.md) for reuse,
interruption recovery, registry backups, legacy migration and Python examples.

## What changed in v0.10

- Final selected strategies now receive paired block-bootstrap confidence
  intervals for arithmetic annualized mean return and Sharpe, plus mean excess
  return against buy-and-hold. Validation and test windows remain separate.
- These are post-selection diagnostics only: existing ranking, analytic Sharpe
  intervals, multiple-testing penalties and backtest accounting are unchanged.
- Reports record seeds, block lengths, sample sizes and unavailable reasons.
  Missing observations are rejected, and undefined resamples are not discarded.
  Bootstrap options are locked when resuming a run.

```bash
# Defaults: 2000 replicates, 10-bar circular blocks, 95% intervals, seed 42.
momentum-lab GLD

# Choose and record settings BEFORE observing the results
momentum-lab GLD --bootstrap-resamples 2000 --bootstrap-block-length 20 \
  --bootstrap-confidence 0.95 --bootstrap-seed 42

# Skip interval estimation; candidate ranking is identical
momentum-lab GLD --no-bootstrap
```

Find `bootstrap_diagnostics` in the final `summary.json`, Python return value,
and uncertainty sections in `report.md` / `report.html`. Intervals require at
least 60 observations and five nominal blocks by default. They assume approximate
stationarity and **do not correct selection bias or repeated test access**.
Arithmetic annualized mean return is not CAGR. Details and Python API:
[UNCERTAINTY.md](UNCERTAINTY.md).

## What changed in v0.9

- Added `momentum-lab benchmark`: 16 fixed cases over four offline synthetic
  market scenarios, including daily/weekday calendars and liquidity stress.
- SHA-256 locks data and assumptions; complete ledgers and metrics are compared
  with a reviewed reference, not just final returns or rounded Sharpe ratios.
- Snapshots and Markdown reports expose version differences, runtime and traced
  allocation peaks. Performance limits are optional; numerical changes in either
  direction require review. No reference is automatically overwritten.
- CI now runs frozen regressions on Python 3.10—3.13, publishes reports, and
  verifies that the fixtures work from a clean wheel outside the source checkout.

### Run frozen regressions (no network needed)

```bash
# Compare against the bundled software-regression reference
momentum-lab benchmark --output experiments/benchmarks/check-090

# Compare two versions on your machine; keep the earlier snapshot
momentum-lab benchmark --repeat 3 --output experiments/benchmarks/before
# Install the version you want to compare, then use a NEW output directory
momentum-lab benchmark --repeat 3 \
  --compare experiments/benchmarks/before/snapshot.json \
  --output experiments/benchmarks/after
```

Each run writes `snapshot.json`, `comparison.json` and `report.md`. Exit codes
are `0` for compatible results, `1` for numerical/resource changes, and `2` for
invalid or incomparable inputs. Optional `--max-slowdown 1.5` and
`--max-memory-growth 1.5` require measured snapshots on both sides; the bundled
reference intentionally contains no machine-specific resource measurements.

These synthetic fixtures check software compatibility, **not historical
profitability or out-of-sample validity**. They do not tune or select strategies.
Memory means `tracemalloc` peak allocation, not total process RSS. See
[BENCHMARKS.md](BENCHMARKS.md) for exact contracts, limitations and reference-update policy.

## What changed in v0.8

- Added deterministic Successive Halving with configurable candidate budget,
  reduction factor, and validation-resource stages. Only candidates evaluated
  on the complete development window enter the canonical ranking; all staged
  evaluations still count toward the multiple-testing hurdle.
- Staged results are transactionally journaled in SQLite and exported to
  `search_stages.csv`; interrupted stages resume without recomputation.
- Candidate processes now receive only development observations, not merely
  train/validation boundary labels. The sealed test snapshot is released once,
  after final selection.
- Added a bounded per-process indicator DAG cache shared across strategies for
  returns, moving averages, volatility, RSI, channels, ATR, ADX, and dependent
  nodes. Reports expose cache and staged-search efficiency diagnostics.

## What changed in v0.7

- Backtests can enforce bar-volume participation limits and gradually fill
  capacity-constrained orders instead of assuming unlimited liquidity.
- The execution model supports quoted bid/ask spread, nonlinear
  participation-based impact, starting NAV, and minimum currency fees.
- Cash, financing, financing-spread, borrow-fee, collateral-rebate, and borrow
  availability inputs can be dated pandas Series. Values are forward-filled
  from information already known; future observations are never backfilled.
- Backtest output exposes requested and filled turnover, actual filled
  positions, transaction costs, participation, capacity constraints, and
  borrow blocks. Search metrics use the actual filled path.
- Every completed search writes portable `report.md` and self-contained
  `report.html` evidence reports in addition to machine-readable artifacts.

## What changed in v0.6

- Candidate workers receive only development observations and train/validation
  boundaries. Test metrics are absent from checkpoints and top-candidate files,
  then evaluated once for a copied final selection.
- Selection uses minimum evidence constraints and Deflated Sharpe probability,
  with temporal folds, a 95% Sharpe interval, expanding walk-forward selection
  replay, and a CSCV/PBO estimate.
- Insolvency is absorbing: equity is clamped at zero and cannot revive.
- Search defaults to explicit next-close execution; same-close, next-open, and
  delayed-close models are available.
- Financing follows elapsed calendar days. Fractional target weights incur
  drift-rebalancing turnover; short cash, collateral rebate, and borrow cost are
  modeled separately.
- SQLite is the transactional resume journal. CSV/JSON reports are exported
  atomically, and the manifest records the lockfile and complete runtime stack.
- ML models can predict the live label-unknown tail, and late-listed symbols can
  reuse a provider-confirmed cache.
- Quick mode uses a seeded Latin-hypercube design, while repeated smoothing
  variants reuse their expensive base signal.

## Research flow

```text
Ticker + versioned run configuration
        -> adjusted daily OHLCV data
        -> 40% train / 40% temporal validation / 20% sealed test
        -> candidate evaluation without access to test observations
        -> optional validation-only staged pruning on increasing resources
        -> constrained Deflated-Sharpe selection + walk-forward/PBO diagnostics
        -> exactly one test-window report for a copied selected result
        -> local parameter-sensitivity analysis
        -> checkpoint, summary, benchmark, and run manifest
```

These diagnostics reduce, but cannot eliminate, data-mining risk. Independent
datasets and prospective paper testing remain release gates for 1.0.

## Strategies

### Classic momentum (15)

| Strategy | Description |
|---|---|
| TSMOM | Time-series momentum |
| MA Cross | SMA/EMA/WMA/DEMA crossover |
| MACD | Crossover, histogram, and zero-filter modes |
| RSI | Momentum and reversal modes |
| ROC | Rate-of-change momentum |
| Bollinger | Breakout and mean-reversion modes |
| Donchian | Persistent channel breakout with exit channel |
| Dual Momentum | Single-asset absolute momentum |
| Triple MA | Three-moving-average alignment |
| Vol Scale | Momentum with volatility-scaled exposure |
| Acceleration | Short-vs-long return acceleration |
| Z-Score | Stateful momentum/reversion |
| Heikin Ashi | Smoothed candle direction |
| Supertrend | ATR-based trend rule |
| Multi Breakout | Multi-horizon breakout vote |

### Experimental ML (8)

Logistic Regression, Random Forest, XGBoost, KNN, SVM, Gaussian Naive Bayes,
AdaBoost, and Extra Trees. Walk-forward fitting purges labels that overlap the
prediction boundary. Grid warm-up windows are at least 252 samples, and invalid
KNN sample/neighbor combinations are rejected before execution.

### Composite / regime-conditioned (3)

Ensemble, Stacked, and Regime Aware.

The complete v0.7 search space is 291,188 configurations: 159,668 non-ML and
131,520 ML. Configuration count is not a measure of research quality.

## Python API

```python
from momentum_lab import SearchConfig, run_search

config = SearchConfig(
    ticker="GLD",
    strategies=["tsmom", "ma_cross", "regime_aware"],
    search_method="successive_halving",
    candidate_budget=256,
    halving_factor=3,
    halving_stages=3,
    indicator_cache_size=256,
    risk_free_rate=0.0,
    cash_rate=0.0,
    financing_rate=0.05,
    financing_spread=0.01,
    spread_bps=4.0,
    impact_bps=2.0,
    max_participation=0.05,
    initial_capital=1_000_000,
    min_fee=0.50,
    max_leverage=1.5,
    execution_model="next_close",
    validation_folds=4,
    min_validation_trades=2,
    result_dir="experiments",
    run_id="gld-research-v1",
    keep_all_results=False,
)

result = run_search(config=config)
print(result["best"])
print(result["benchmark_metrics"])
print(result["parameter_sensitivity"])

# Exact resume refuses to mix different source trees, environments, data,
# strategy spaces, or accounting models. Git SHA itself is informational.
resumed = run_search(config=config, resume=True)
print(resumed["n_skipped"], resumed["n_errors"])
```

Single-strategy use remains available:

```python
import pandas as pd

from momentum_lab import backtest, evaluate, get_strategy, prepare_data

data, frame = prepare_data("BTC-USD")  # annualization inferred as 365
positions = get_strategy("tsmom").run(data, lookback=63, threshold=0.01, long_short=False)
simulation = backtest(
    positions,
    frame["close"],
    volume=frame["volume"],
    cost_bps=2,
    slippage_bps=3,
    spread_bps=4,
    impact_bps=2,
    impact_reference_participation=0.01,
    max_participation=0.05,
    initial_capital=1_000_000,
    min_fee=0.50,
    cash_rate=0.03,
    financing_rate=0.06,
    financing_spread=0.01,
    short_rebate_rate=0.01,
    max_leverage=1.0,
    execution_lag=1,
    annualization=365,
)
print(evaluate(simulation["returns"], risk_free_rate=0.03, annualization=365))
print(simulation["capacity_constrained"].sum(), "capacity-constrained bars")
```

For historical funding or borrow conditions, pass a sorted pandas Series whose
first observation covers the first price bar. Sparse schedules are
forward-filled only:

```python
cash_curve = pd.Series([0.01, 0.05], index=pd.to_datetime(["2020-01-01", "2022-03-17"]))
borrow_available = pd.Series([True, False], index=pd.to_datetime(["2020-01-01", "2021-01-28"]))
simulation = backtest(positions, frame["close"], cash_rate=cash_curve, borrow_available=borrow_available)
```

## CLI options

```text
momentum-lab TICKER [OPTIONS]

  --quick                 Five representative configs per strategy (default)
  --exhaustive            Full parameter grid; explicitly opt in
  --successive-halving    Budgeted deterministic staged search
  --candidate-budget N    Initial candidates per strategy (default: 256)
  --halving-factor N      Candidate reduction factor (default: 3)
  --halving-stages N      Maximum validation-resource stages (default: 3)
  --indicator-cache-size N
                          Reusable indicator nodes/process; 0 disables it
  --strategies NAMES      Comma-separated strategies (default: non-ML)
  --all-strategies        Include experimental ML strategies
  --config PATH           Load a complete JSON SearchConfig
  --resume                Resume an exact run-id checkpoint
  --workers N             Spawned worker processes (default: 1)
  --cost BPS              Linear transaction cost (default: 1)
  --slippage BPS          Additional linear slippage (default: 0)
  --spread-bps BPS        Quoted full bid/ask spread (default: 0)
  --impact-bps BPS        Impact at the reference participation rate
  --impact-exponent X     Nonlinear participation exponent (default: 0.5)
  --impact-reference-participation X
                          Participation where impact is quoted (default: 0.01)
  --max-participation X   Maximum fraction of bar dollar volume traded
  --initial-capital N     Starting NAV for capacity/fees (default: 1,000,000)
  --min-fee N             Minimum currency fee per rebalance (default: 0)
  --cash-rate RATE        Annual cash return (default: 0)
  --financing-rate RATE   Annual rate on exposure above 1x (default: 0)
  --financing-spread R    Annual spread over the financing rate (default: 0)
  --borrow-bps BPS        Annual short borrow fee (default: 0)
  --short-rebate-rate R   Annual short-collateral rebate (default: 0)
  --max-leverage X        Final absolute exposure cap (default: 2)
  --execution-model M     same_close, next_close, next_open, delayed_close
  --execution-lag N       Delay for delayed_close only
  --annualization N       Override inferred 252/365 periods per year
  --risk-free-rate RATE   Metric hurdle rate (default: 0)
  --validation-folds N    Even temporal-fold count (default: 4)
  --min-val-bars N        Minimum validation observations (default: 60)
  --min-val-trades N      Minimum validation trades (default: 1)
  --min-val-exposure X    Minimum mean absolute exposure (default: 0.01)
  --start DATE            Data start date (default: 2004-01-01)
  --end DATE              Inclusive data end date (default: current history)
  --refresh               Ignore cached market data
  --top N                 Top result count (default: 50)
  --result-dir DIR        Run artifact parent (default: ./experiments)
  --run-id ID             Stable run directory name
  --keep-all              Retain all results in RAM (default: stream to disk)
  --no-report             Skip Markdown/HTML report generation
  --robust / --no-robust  Enable/disable local parameter sensitivity
  --robust-frac F         Local perturbation fraction (default: 0.2)
  --list                  List strategies and full-grid counts
  --version               Show version
```

## Outputs

Each run is isolated under `experiments/<run_id>/`:

- `run_config.json`: package/schema/Git metadata, source/data/lock/environment
  hashes, periods, strategies, execution costs, and risk assumptions.
- `results.sqlite3`: canonical transactional checkpoint and resume journal.
- `search_stages.csv`: validation-only stage evidence and advancement decisions
  for Successive Halving runs.
- `all_results.csv`: atomic human-readable export; train/validation only.
- `top_results.csv`: validation-ranked candidates; no test metrics.
- `robustness.csv`: local parameter-sensitivity summary when enabled.
- `summary.json`: selected result, buy-and-hold benchmark, sensitivity output,
  result/error counts, and resume statistics.
- `report.md` and `report.html`: portable human-readable evidence reports with
  assumptions, diagnostics, validation, sealed test, and benchmark results.

## Methodological limitations

- Deflated Sharpe uses a conservative independent-trial approximation; the
  fold-based CSCV/PBO estimate is diagnostic rather than proof against overfit.
- Repeatedly creating new run IDs and observing their final test reports can
  still become human-level test leakage; preregistration is outside the code.
- Local parameter sensitivity remains descriptive and is not itself a
  multiple-testing correction.
- Liquidity uses aggregate bar volume and a parametric impact curve, not order
  book replay. Queue position, intrabar path, venue fragmentation, halts, and
  taxes remain outside the model.
- Search configuration supports scalar market assumptions; dated rate and
  borrow-availability schedules are currently exposed through the direct
  Python backtest API.
- yfinance is suitable for personal research, not an authorized commercial
  market-data redistribution layer.

Do not use a result for capital allocation without independent validation,
realistic execution assumptions, and forward/paper testing.

## Development

```bash
uv sync --all-extras
uv run ruff check .
uv run pytest -m "not network" -q
```

CI tests Python 3.10—3.13 on Linux, checks run ownership and workflow integration
on macOS/Windows, builds and installs the wheel in a clean environment, enforces
a coverage floor, and runs provider-contract tests weekly. Release automation
remains on the roadmap.

## License

MIT
