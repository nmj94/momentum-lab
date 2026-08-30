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

CI tests Python 3.10—3.13, builds and installs the wheel in a clean environment,
enforces a coverage floor, and runs provider-contract tests weekly. Release
automation remains on the roadmap.

## License

MIT
