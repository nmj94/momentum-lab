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

# Default: 5 representative configurations for each non-ML strategy,
# streamed to disk without retaining the full result set in RAM.
momentum-lab GLD

# Search a chosen set
momentum-lab SPY --strategies tsmom,ma_cross,rsi,regime_aware

# Full non-ML grid (159,668 experiments; explicitly opt in)
momentum-lab GLD --exhaustive --workers 4
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

## What changed in v0.5

- Safe defaults: quick, non-ML, streamed results.
- Risk sizing is no longer optimized as an alpha parameter.
- Every backtest enforces a final portfolio leverage cap.
- Cash earns an explicit cash rate; financing applies only to exposure above 1x.
- Crypto annualization is inferred as 365; other daily assets default to 252.
- User-facing end dates are inclusive even though yfinance uses an exclusive end.
- Yahoo downloads retry with bounded exponential backoff.
- Cache files live in the operating system's user cache directory and are
  written atomically.
- Checkpoints use a fixed schema and resume only when data, strategy set,
  search mode, package/schema version, Git revision, costs, and risk settings
  match.
- XGBoost and scikit-learn are optional dependencies.
- The old “robustness grade” is described accurately as local parameter
  sensitivity; it is not a multiple-testing correction.

## Research flow

```text
Ticker + versioned run configuration
        -> adjusted daily OHLCV data
        -> non-overlapping train / validation / test periods
        -> strategy and parameter evaluation
        -> ranking by validation Sharpe
        -> one test-window report for the selected result
        -> local parameter-sensitivity analysis
        -> checkpoint, summary, benchmark, and run manifest
```

The current train/validation/test flow is a baseline, not the final research
methodology. Nested walk-forward validation, Deflated Sharpe Ratio, and
Probability of Backtest Overfitting are planned before a 1.0 release.

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

The complete v0.5 search space is 291,188 configurations: 159,668 non-ML and
131,520 ML. Configuration count is not a measure of research quality.

## Python API

```python
from momentum_lab import SearchConfig, run_search

config = SearchConfig(
    ticker="GLD",
    strategies=["tsmom", "ma_cross", "regime_aware"],
    quick=True,
    risk_free_rate=0.0,
    cash_rate=0.0,
    financing_rate=0.05,
    max_leverage=1.5,
    result_dir="experiments",
    run_id="gld-research-v1",
    keep_all_results=False,
)

result = run_search(config=config)
print(result["best"])
print(result["benchmark_metrics"])
print(result["parameter_sensitivity"])

# Exact resume refuses to mix different source, data, strategy, or cost models.
resumed = run_search(config=config, resume=True)
print(resumed["n_skipped"], resumed["n_errors"])
```

Single-strategy use remains available:

```python
from momentum_lab import backtest, evaluate, get_strategy, prepare_data

data, frame = prepare_data("BTC-USD")  # annualization inferred as 365
positions = get_strategy("tsmom").run(data, lookback=63, threshold=0.01, long_short=False)
simulation = backtest(
    positions,
    frame["close"],
    cost_bps=2,
    slippage_bps=3,
    cash_rate=0.03,
    financing_rate=0.06,
    max_leverage=1.0,
    annualization=365,
)
print(evaluate(simulation["returns"], risk_free_rate=0.03, annualization=365))
```

## CLI options

```text
momentum-lab TICKER [OPTIONS]

  --quick                 Five representative configs per strategy (default)
  --exhaustive            Full parameter grid; explicitly opt in
  --strategies NAMES      Comma-separated strategies (default: non-ML)
  --all-strategies        Include experimental ML strategies
  --config PATH           Load a complete JSON SearchConfig
  --resume                Resume an exact run-id checkpoint
  --workers N             Spawned worker processes (default: 1)
  --cost BPS              Linear transaction cost (default: 1)
  --slippage BPS          Additional linear slippage (default: 0)
  --cash-rate RATE        Annual cash return (default: 0)
  --financing-rate RATE   Annual rate on exposure above 1x (default: 0)
  --borrow-bps BPS        Annual short borrow fee (default: 0)
  --max-leverage X        Final absolute exposure cap (default: 2)
  --annualization N       Override inferred 252/365 periods per year
  --risk-free-rate RATE   Metric hurdle rate (default: 0)
  --start DATE            Data start date (default: 2004-01-01)
  --end DATE              Inclusive data end date (default: current history)
  --refresh               Ignore cached market data
  --top N                 Top result count (default: 50)
  --result-dir DIR        Run artifact parent (default: ./experiments)
  --run-id ID             Stable run directory name
  --keep-all              Retain all results in RAM (default: stream to disk)
  --robust / --no-robust  Enable/disable local parameter sensitivity
  --robust-frac F         Local perturbation fraction (default: 0.2)
  --list                  List strategies and full-grid counts
  --version               Show version
```

## Outputs

Each run is isolated under `experiments/<run_id>/`:

- `run_config.json`: package/schema/Git version, source and data hashes, periods,
  strategies, execution costs, and risk assumptions.
- `all_results.csv`: fixed-schema, incremental checkpoint.
- `top_results.csv`: top validation-ranked configurations.
- `robustness.csv`: local parameter-sensitivity summary when enabled.
- `summary.json`: selected result, buy-and-hold benchmark, sensitivity output,
  result/error counts, and resume statistics.

## Methodological limitations

- The current selector still ranks a large family on one validation window.
- Local parameter sensitivity does not correct for multiple testing.
- Close-derived signals assume execution compatible with the following
  close-to-close return; exact MOC/next-open modeling is not yet available.
- The cost model is linear and does not model spread, market impact, capacity,
  halts, borrow availability, or taxes.
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
and enforces a coverage floor. Scheduled network tests and release automation
remain on the roadmap.

## License

MIT
