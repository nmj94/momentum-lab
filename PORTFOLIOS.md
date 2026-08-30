# Multi-asset momentum research (v0.14)

The `portfolio --config ... --acknowledge-history` workflow compares assets against one another and simulates a
single long-only, unlevered account. This is **fixed-rule, exploratory
whole-history research**, not parameter selection, a sealed portfolio study,
independent out-of-sample evidence, or an order to trade. Momentum can suffer
large losses and prolonged underperformance; this tool does not establish its
superiority or guarantee investment outcomes.

v0.14 separately adds [registered portfolio studies](PORTFOLIO_STUDIES.md) with
development-first evaluation and explicit test reveals. Both workflows support
optional [declared historical membership](MEMBERSHIP.md). The whole-history
command described here keeps its explicit full-history consent behavior.

## Run an offline portfolio

First import at least two daily CSV snapshots using [DATASETS.md](DATASETS.md).
Use only data you are permitted to use. The project does not provide licensed
market history. Each snapshot needs its own canonical ticker and truthful
source, license, currency, calendar, annualization and adjustment declarations.

For example, save this as `portfolio.json` next to a `datasets` directory
containing your imported snapshots. The symbols are illustrative, not an
investment recommendation:

```json
{
  "datasets": {
    "SPY": "datasets/spy-v1/manifest.json",
    "GLD": "datasets/gld-v1/manifest.json"
  },
  "lookback": 126,
  "skip_recent": 0,
  "top_k": 1,
  "rebalance": "monthly",
  "absolute_threshold": 0.0,
  "max_weight": 0.8,
  "initial_capital": 100000,
  "cost_bps": 5,
  "slippage_bps": 2,
  "spread_bps": 4,
  "cash_rate": 0.0,
  "risk_free_rate": 0.0,
  "start": "2020-01-01",
  "end": "2025-12-31",
  "run_id": "relative-momentum-v1",
  "result_dir": "experiments/portfolios"
}
```

Choose dates and assumptions appropriate to your files. The evaluated bounds
are the first and last actual sessions inside the inclusive requested range.
Dataset paths in JSON resolve relative to the JSON file; `result_dir` and an
optional `registry_path` resolve from the process working directory. Without
`start`/`end`, all supplied history is used, which must align exactly. An omitted
`run_id` generates a unique one. Existing output directories are never reused.

```bash
momentum-lab portfolio --config portfolio.json --acknowledge-history

# Inspect score-free access history afterward.
momentum-lab study history --ticker SPY

# No market data or access acknowledgement needed for this software check.
momentum-lab portfolio benchmark
```

`--acknowledge-history` is mandatory and invocation-only. It cannot be enabled
inside JSON. This explicit acknowledgement permits observation of the entire
evaluated history, including dates used as tests by other studies. Every
asset/date range is durably recorded as development use in the shared
[research registry](GOVERNANCE.md) **before any scores or portfolio returns
are computed**. Existing sealed studies over those dates may consequently
require an explicit reuse acknowledgement at reveal. Missing consent fails
before dataset or registry access.

If a reservation or calculation fails, any recorded history remains. All
asset reservations must succeed before calculations start; they are separate
transactions, so a later failure can leave conservative earlier reservations.
Do not delete history or change registries to manufacture a fresh test label.
The registry is local audit metadata, not encrypted or tamper-proof custody,
and cannot detect previously observed history outside its records.

## Universe and data contract

- The workflow accepts 2–64 distinct canonical tickers; case aliases are
  duplicates. Labels start with a letter, digit or `^` and contain only ticker
  characters. The candidate pool is fixed; all candidates are eligible by default.
  An optional `universe` manifest controls time-varying signal eligibility.
- All assets must declare the **same currency, calendar, annualization and
  price-adjustment convention**, and contain exactly the same session dates
  after explicit slicing. There is no implicit date intersection, filling,
  currency conversion, or fallback to online quotes.
- At least `lookback + 2` aligned observations are required: enough history
  for one signal followed by one delayed fill. Work is bounded to 1,000,000
  asset-session cells; original CSV import limits still apply.
- Prices must be positive, finite, real-valued daily observations. The engine
  uses the supplied `close` series; it does not reconstruct corporate actions
  or dividend cash payments. Adjusted-price holdings are **synthetic units**,
  not literal broker-share quantities. Unadjusted prices can contain splits
  or omit distributions.
- Equal dates/calendar labels do **not** prove synchronized market closes.
  This version assumes a common close/fill clock, chosen by the researcher.
  Do not infer valid cross-market timing from date labels alone.

Source/licensing and adjustment fields are declarations, not independently
verified facts. A current list of surviving assets is not a point-in-time
universe. Optional membership is declared, not independently verified; delisting
settlements, symbol changes and price availability are not reconstructed.
Choosing the candidate pool after seeing returns introduces
selection/survivorship bias even if the numerical signals are causal.

All candidates still need complete positive prices even outside membership.
Pre-IPO gaps or missing delisting/exit prices fail closed; no synthetic quote
or liquidation value is inserted. A membership removal is not a settlement.

## Cross-sectional rule

For each asset `i`, the score at session `t` is:

```text
score[i,t] = close[i,t-skip_recent] / close[i,t-lookback] - 1
```

Both lags count aligned observations, not calendar days. `lookback >= 1`,
`0 <= skip_recent < lookback`. Thus `lookback=252, skip_recent=21` measures
252-to-21-session momentum; it does not add another 252 sessions before the
skip window. No scores are actionable before the full lookback is available.

Assets are sorted by descending score with exact ties broken alphabetically
by uppercase ticker. Only scores **strictly greater than**
`absolute_threshold` are eligible (default `0.0`). JSON `null` / Python `None`
disables the absolute filter. Select at most `top_k` assets, with
`1 <= top_k <= number_of_assets`. Each selected slot has target weight
`min(1 / top_k, max_weight)`; unfilled slots remain cash. For example,
`top_k=2`, cap `0.4`, and one eligible asset means 40% invested and 60% cash,
not a rescaled 100% position.

Signals occur daily, or on the first observed session of each new week
(Monday–Sunday) / calendar month. The first fully warmed-up session always
generates a signal, even mid-period. A month-end resample requiring knowledge
of future data is not used. These are first-session signals, **not** month-end
signals executed at the next month's open.

With `universe`, ineligible scores are masked and never selected. Membership
changes after warm-up force an additional signal, subject to the same next-close
execution delay. An empty member set targets cash. An earlier pending instruction
can still execute on an event day; there is no retroactive cancellation.

## Execution and self-financing accounting

The CLI executes each instruction at the **following observed close**. A
signal computed from today's close cannot also fill at that close. The Python
book supports integer `execution_lag >= 1`; the fixed-recipe workflow uses 1.
There is no fill after the last available bar. The report distinguishes final
realized weights from the last signal, which may still be pending.

All-NaN target rows mean **hold share quantities**, not “restore yesterday's
weights.” On those dates prices change asset values and weights drift. A
complete zero target liquidates to cash. Partial-NaN rows, negative weights,
or target sums greater than one are rejected (only a `1e-12` rounding overshoot
is normalized). `max_weight` caps targets at rebalances, not actual daily
weights; a winning asset can drift above its cap before the next fill.

All assets share one cash account. At each fill, first mark existing holdings
at that session's closes and accrue interest on cash held since the prior
session. Let `V` be this pre-trade NAV, `v[i]` marked asset values, `w[i]`
targets, and the one-way proportional cost rate be:

```text
k = (cost_bps + slippage_bps + spread_bps / 2) / 10000
N + k * sum(abs(w[i] * N - v[i])) = V
```

The engine solves for post-cost NAV `N`, then sets asset values to `w[i]*N`.
For nonnegative weights summing to at most one and `0 <= k < 1`, the root is
unique. Fees are paid from the account, not financed by an invisible loan.
Buying and selling are both charged; rotation can have approximately 200%
two-sided turnover. `turnover` divides gross absolute traded notional by
pre-trade NAV. Quoted spread is full spread; each traded leg pays half.
`slippage_bps` is a linear cost allowance, not a separate fill-price series.

Cash earns the **effective annual** `cash_rate > -1` over actual elapsed days:

```text
cash_interest = previous_cash * ((1 + cash_rate) ** (elapsed_days / 365) - 1)
```

Weekends count; negative rates are permitted. This portfolio convention is
ACT/365 compounding and intentionally differs from the existing single-asset
engine's simple ACT/365.25 convention. Do not compare the two interest ledgers
as if their conventions were identical.

No leverage, shorts, FX, borrow, capacity constraints, nonlinear market impact,
minimum fees, taxes, fractional-share restrictions, asynchronous fills or
broker execution are modeled. Old single-asset execution features are not
implicitly enabled in this separate portfolio engine.

## Benchmark and descriptive statistics

Without membership, the baseline is a single equal-weight **buy-and-hold** allocation after the
same warm-up, using the same prices, delay, costs, cash rate and target cap.
Its per-asset target is `min(1 / number_of_assets, max_weight)`; it never
rebalances afterward. It is not a daily equal-weight index, an investable
external benchmark or a risk-matched comparison.

With membership, the baseline is instead **Membership equal-weight rebalanced**:
on exactly the strategy's signal dates, target every currently eligible asset
at `min(1 / eligible_count, max_weight)`, with residual cash and the same delay
and costs. It does not buy future members early. See [MEMBERSHIP.md](MEMBERSHIP.md).

Both series include warm-up cash intervals. The first structural zero return
is excluded from statistics, leaving `number_of_bars - 1` return intervals.
CAGR uses final equity compounded over those intervals with the dataset's
declared annualization; this is session-count annualization, not a calendar
year fraction. Volatility is sample standard deviation (`ddof=1`) times the
square root of annualization. Sharpe is `(mean_return * annualization -
risk_free_rate) / volatility`. The risk-free input is an annual subtraction
for this statistic and does not set the cash account's interest rate.

Undefined Sharpe/volatility/CAGR is JSON `null`, not zero or Infinity. Cash
growth is still reported when return variance is zero. Drawdown is anchored
to starting equity 1. These are descriptive statistics, with no portfolio
selection adjustment, significance test or portfolio bootstrap interval.

## Python API and output files

```python
from momentum_lab import PortfolioConfig, run_portfolio

config = PortfolioConfig.from_json("portfolio.json")
summary = run_portfolio(config, acknowledge_history=True)
print(summary["metrics"])
print(summary["result_dir"])
```

Low-level `cross_sectional_momentum(prices, ...)`,
`backtest_portfolio(targets, prices, ...)` and `portfolio_metrics(result, ...)`
accept pandas objects and return in-memory research results. They do not read
files or write registry records. As with existing low-level backtest APIs,
callers are responsible for their own observation history; these numerical
functions are not a way to certify unseen data. A one-asset mathematical
book is allowed even though the research workflow requires at least two.

Each run writes:

| File | Meaning |
| --- | --- |
| `run_config.json` | Fixed recipe, full snapshot provenance, evaluated-data hashes, source/environment/lock fingerprints, execution conventions and registry identity |
| `ledger.csv` | NAV, cash, P&L, interest, fees, traded notional, turnover and returns for the shared account |
| `weights.csv`, `holdings.csv`, `asset_values.csv`, `trades.csv` | Realized asset weights, synthetic units, marked values and signed currency-value trades |
| `scores.csv`, `targets.csv`, `executed_targets.csv` | Decision inputs, sparse signal targets and delayed executed instructions; blank rows mean no instruction |
| `benchmark_ledger.csv`, `benchmark_weights.csv` | Matched-warm-up buy-and-hold baseline |
| `eligibility.csv` (when supplied) | Boolean declared membership mask; the benchmark then rebalances eligible members |
| `report.md`, `report.html` | Offline readable performance, allocation, source and assumption reports |
| `summary.json` | Final metrics and allocations; written last as the completion marker |

The SHA-256 research contract binds the recipe, source/version/environment,
execution assumptions, every dataset's content/declarations, and evaluated
snapshots and optional raw/normalized membership hashes. It excludes output/run/registry locations and input paths: moving
unchanged snapshots does not create different evidence. Original source CSVs
are not recopied into each portfolio run; retain your imported snapshots.
There is no overwrite or resume mode in this first portfolio workflow.

## Software evidence and next steps

Portfolio accounting has its own schema 1. The existing single-asset engine
and checkpoint schema 5, registry schema 1, and 16 frozen single-asset cases
are unchanged. Six new SHA-256-locked synthetic cases compare **every** ledger,
holding, value, weight and trade cell with an independent exact-Fraction
oracle that solves buy/sell regions algebraically. Production uses a separate
bisection algorithm. See `scripts/portfolio_reference_oracle.py` and
`momentum_lab/benchmark_data/portfolio_reference_v1.json`.

These cases test software behavior, not investment performance. Changing the
reference is a reviewed accounting-contract change, never an automatic fix
for a failing regression. Tests also exercise causal signal prefixes, cost
reconciliation, audit ordering and interrupted runs. CI checks Python
3.10–3.13 and an installed core wheel without optional ML packages.

v0.14 adds two further independent membership/carried-test cases and a separate
fixed-rule [multi-asset registered workflow](PORTFOLIO_STUDIES.md), bringing the
frozen software suite to 16 + 6 + 2 cases. Existing accounting schemas are unchanged.
Next work is verified historical data, unbalanced IPO/delisting histories and
explicit settlements, then selection-aware portfolio parameter research. These
controls and forward testing should precede paper or live execution.
