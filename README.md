# Momentum Lab

**Find the optimal momentum trading strategy for any asset. Just provide a ticker.**
**只需提供一个股票代码，自动找到最优动量交易策略。**

[English](README.md) | [中文](README_CN.md)

Momentum Lab automatically tests 26 strategies across hundreds of thousands of parameter combinations to find the best-performing strategy for your chosen asset. It includes classic momentum indicators, machine learning models, and an adaptive regime-aware strategy that switches between sub-strategies based on market conditions.

## Quick Start

```bash
# Install
pip install momentum-lab

# Find the best strategy for gold (GLD)
momentum-lab GLD

# Quick search for S&P 500
momentum-lab SPY --quick

# Search Bitcoin with 4 parallel workers
momentum-lab BTC-USD --workers 4

# Search specific strategies only
momentum-lab AAPL --strategies tsmom,ma_cross,rsi,regime_aware
```

## How It Works

```
You provide a ticker (e.g. GLD)
        |
        v
  Download data via Yahoo Finance
        |
        v
  Split into Train / Validation / Test
        |
        v
  Test 26 strategies x thousands of parameters
  (classic momentum, ML models, regime-aware)
        |
        v
  Rank by Validation Sharpe Ratio
        |
        v
  Evaluate best strategy on Test set
        |
        v
  Robustness check: perturb optimal params
  to detect overfitting (isolated peak)
        |
        v
  Report: Best strategy + parameters + metrics
```

## Strategies Included

### Classic Momentum (15)
| Strategy | Description | Key Parameters |
|----------|-------------|----------------|
| TSMOM | Time-series momentum | lookback, threshold, skip_recent |
| MA Cross | Moving average crossover | fast, slow, ma_type (SMA/EMA/WMA/DEMA) |
| MACD | MACD signal crossover | fast, slow, signal, mode |
| RSI | RSI momentum/reversal | period, buy/sell thresholds, smoothing |
| ROC | Rate of change | period, threshold, smoothing |
| Bollinger | Bollinger band breakout | period, num_std, band_width_filter |
| Donchian | Donchian channel breakout | period, exit_period, confirmation |
| Dual Momentum | Absolute momentum | lookback, threshold, smoothing |
| Triple MA | Three MA alignment | fast, medium, slow, ma_type |
| Vol Scale | TSMOM + volatility targeting | lookback, vol_target, vol_lookback |
| Acceleration | Price acceleration | short_lb, long_lb, threshold |
| Z-Score | Z-score momentum/reversion | lookback, entry_z, exit_z |
| Heikin Ashi | HA candlestick | smooth, confirmation |
| Supertrend | ATR-based trend following | atr_period, multiplier |
| Multi Breakout | Multi-period breakout vote | periods, vote_threshold |

### Machine Learning (8)
| Strategy | Description |
|----------|-------------|
| ML LogReg | Logistic Regression with walk-forward training |
| ML RF | Random Forest |
| ML XGB | XGBoost Gradient Boosting |
| ML KNN | K-Nearest Neighbors |
| ML SVM | Support Vector Machine |
| ML NB | Gaussian Naive Bayes |
| ML AdaBoost | AdaBoost with decision stumps |
| ML Extra Trees | Extremely Randomized Trees |

### Adaptive (3)
| Strategy | Description |
|----------|-------------|
| Ensemble | Multi-strategy voting |
| Stacked | Strategy + momentum filter overlay |
| **Regime Aware** | Auto-detects market state (trend/choppy/crisis) and switches sub-strategy |

## Regime-Aware Strategy

The flagship strategy detects market conditions using 4 indicators (ADX, volatility ratio, MA alignment, momentum) and dynamically switches:

| Market State | Strategy Action |
|-------------|----------------|
| Trend + Bullish | Full position with volatility scaling |
| Choppy + Bullish | Full position (catches non-trend up moves) |
| Trend + Bearish | Cash or short (configurable) |
| Crisis | Reduced position with low vol target |
| Neutral choppy | Cash |

Plus a fast-exit circuit breaker that cuts positions when N-day return drops below threshold.

## Python API

```python
from momentum_lab import download_data, backtest, evaluate, get_strategy, run_search

# Run full search
results = run_search("GLD", quick=True)
print(f"Best strategy: {results['best']['strategy']}")
print(f"Best params: {results['best']['params']}")

# Robustness check result (overfitting detection)
rob = results.get("robustness")
print(f"Grade: {rob['grade']} ({rob['verdict']})")
print(f"Baseline val Sharpe: {rob['baseline']:.4f}")
print(f"Neighbor median: {rob['stats']['median']:.4f}")

# Or test a single strategy
from momentum_lab.data import prepare_data
data, df = prepare_data("SPY")
strategy = get_strategy("regime_aware")
positions = strategy.run(data, adx_trend_threshold=15, mom_lookback=63,
                          vol_target_normal=0.12, position_size=2.0)
result = backtest(positions, df["close"])
metrics = evaluate(result["returns"])
print(f"Sharpe: {metrics['sharpe']}")
```

## CLI Options

```
momentum-lab TICKER [OPTIONS]

Arguments:
  TICKER              Yahoo Finance ticker (GLD, SPY, BTC-USD, AAPL, ...)

Options:
  --quick             Quick mode: 5 params per strategy
  --strategies STR    Comma-separated strategy names
  --workers N         Parallel workers (default: 1)
  --cost BPS          Transaction cost in basis points (default: 1.0)
  --start DATE        Data start date (default: 2004-01-01)
  --end DATE          Data end date (default: today)
  --top N             Number of top results (default: 50)
  --robust            Run robustness check on best params (default: True)
  --no-robust         Skip the robustness check
  --robust-frac F     Perturbation fraction (default: 0.2)
  --list              List all strategies and exit
  --version           Show version
```

## Robustness Check

The search can end up on a lucky parameter spike that won't generalize. After finding
the best strategy, momentum-lab perturbs every numeric parameter by +/-20% (integers by
+/-1) and re-evaluates Validation Sharpe for each neighbor:

- **Grade A (Robust)**: neighbors stay close to the optimum — a wide plateau
- **Grade B**: moderately stable
- **Grade C**: fragile — the result depends heavily on exact parameters
- **Grade D / isolated peak**: the optimum is a spike; treat the result as overfit

If you see *ISOLATED PEAK - likely overfit*, lower your expectations for live trading
and consider constraining the search space.

## Installation

### From source
```bash
git clone https://github.com/nmj94/momentum-lab.git
cd momentum-lab
pip install -e .
```

### Requirements
- Python 3.10+
- A Yahoo Finance-accessible ticker (stocks, ETFs, crypto, indices)

## Output

After running, results are saved to `experiments/`:
- `all_results.csv` - Every experiment with train/val/test metrics
- `top_results.csv` - Top N strategies by validation Sharpe
- `robustness.csv` - Robustness check summary for the best strategy
- Console output shows the best strategy with parameters and test set performance

## Example Results (GLD)

Tested 974,000+ parameter combinations across 23 strategies:

| Strategy | Val Sharpe | Test Sharpe | Test CAGR | Test MaxDD |
|----------|-----------|------------|-----------|------------|
| Regime Aware | 0.90 | 1.31 | 36.3% | -19.0% |
| Vol Scale Mom | 0.89 | 1.49 | 54.1% | -20.3% |
| TSMOM | 0.06 | 1.55 | 48.8% | -23.9% |
| Buy & Hold | 0.56 | 1.18 | 24.2% | -21.0% |

## Disclaimer

This is a research tool, not investment advice. Historical backtests do not guarantee future returns. Always do your own research and consider transaction costs, slippage, and taxes before trading.

## License

MIT
