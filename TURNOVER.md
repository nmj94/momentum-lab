# Portfolio turnover budgets (v0.19)

`max_turnover` limits the gross value of buys **plus** sells at each scheduled
portfolio execution, divided by that account's pre-trade NAV. This optional
assumption works in fixed-rule research and registered studies. It does not
select a budget or establish that lower turnover improves investment outcomes.

## Use an explicit fixed recipe

Add `"max_turnover": 0.20` inside an existing [portfolio recipe](PORTFOLIOS.md)
or [registered portfolio study](PORTFOLIO_STUDIES.md). Retain existing consent:

```bash
momentum-lab portfolio preflight --config portfolio.json
momentum-lab portfolio --config portfolio.json --acknowledge-history

# Registered workflow: development first, then an explicitly authorized reveal.
momentum-lab portfolio study --config study.json --run-id development-v019
momentum-lab portfolio study --config study.json --run-id reveal-v019 --reveal-test
```

| Value | Execution behavior |
|---|---|
| omitted / JSON `null` / Python `None` | Existing unrestricted fills; default accounting unchanged |
| `0` | Process instructions without trading; the initially cash-only account stays in cash |
| `0.20` | At most 20% of pre-trade NAV in total buys and sells per instruction |
| `2` | Up to 200% gross turnover; nonbinding for this long-only, unlevered linear-cost model |

Numbers must be finite and in `[0, 2]`; booleans and numeric strings are rejected.
The limit excludes fees; those are deducted separately from the same account.
It is not half-turnover, an annual allowance or a volume-participation fraction.
Recalculate the budget after marking held assets and accruing cash interest.

The Python book accepts the same option:

```python
from momentum_lab import backtest_portfolio

book = backtest_portfolio(targets, closes, cost_bps=5, max_turnover=0.20)
print(book["execution"])
```

## Partial execution and funding

For pre-trade NAV `V`, marked holdings `v[i]`, requested weights `w[i]` and
combined one-way proportional cost `k`, first solve the existing full-target book:

```text
N + k * sum(abs(w[i] * N - v[i])) = V
d[i] = w[i] * N - v[i]
G = sum(abs(d[i]))
```

For positive `G`, fill the common fraction
`alpha = min(1, max_turnover / (G / V))`; without a limit, `alpha = 1`.
Fill `alpha*d[i]` on **every** leg and pay `k*alpha*G` in fees. Sells and buys are
scaled together, without ticker-order priority or borrowing to fund one leg.
Resulting cash/holdings are a convex combination of the original marked account
and the fully rebalanced account, so both remain non-negative. Actual post-trade
NAV is `V - k*alpha*G`; requested weights may not be reached.

An instruction is attempted once, at its delayed close. Its unfilled remainder
**expires on that bar**. It does not become a standing order, and HOLD never
retries it. A later signal creates a new instruction from then-current information;
its trade is calculated from holdings and NAV at its execution. No fill is
invented after the final price bar.

For example, with 1,000 cash, no fees and a 25% limit, an instruction to invest
100% fills only 250. If the asset then doubles during HOLD, its marked value is
500 and NAV is 1,250; no further purchase occurs. A cash-exit instruction at
that price can sell only 312.50, leaving 187.50 invested. This is deliberate
partial execution, not a completed liquidation.

## Comparison and membership implications

Both strategy and comparison accounts use the same budget at **their own**
scheduled instruction dates and NAV. The static equal-weight buy-and-hold
comparison attempts entry once; an expired partial entry can leave cash for the
rest of the sample. The membership comparison submits instructions on the existing
signal dates. Report labels explicitly identify a turnover budget. The accounts
can have different exposure/cash allocations; their performance differences do
not isolate strategy skill or imply a like-for-like invested book.

The budget also applies to cash-exit and membership-removal instructions. An
excluded asset can remain held, and actual weights can exceed signal target caps.
This is not a mandatory exit, delisting settlement, volume/liquidity constraint,
impact model or executable broker order. Existing balanced-price, declared
membership and synchronized-close assumptions still apply.

## Execution audit and registered tests

Every book returns three additional frames:

- `execution`: `instruction_processed`, `requested_notional`, `filled_notional`,
  `unfilled_notional`, `fill_fraction`, `turnover_limited`.
- `requested_trades`: signed full-target notional per asset at the execution price.
- `unfilled_trades`: signed requested minus filled notional, immediately expired.

`fill_fraction` is zero on HOLD, one for an already-satisfied zero-notional
instruction, and the common fraction for a nonzero request. Legacy ledger
`rebalance_executed` counts processed instructions, including zero fills; it is
not the count of trades that moved holdings. `executed_targets` remains the
delayed **instruction** weights. Use `weights` for realized allocations.

CSV exports include these frames for the strategy and the `benchmark_` account,
plus `benchmark_trades.csv`. JSON/HTML/Markdown summaries include requested gross
notional, expired unfilled notional and limited-instruction counts. Totals sum
attempted instructions: repeated new requests can count the same economic desired
exposure again. These are not an outstanding-order balance.

Registered studies expose only development diagnostics before explicit reveal.
Holdings, cash and not-yet-due instructions carry across the boundary; expired
remainders do not. The test anchor preserves holdings but zeros all execution
flows/counts, preventing development amounts from being counted twice. The budget
is frozen into the protocol; changing it cannot reuse the same study as if the rule
were unchanged. Cached replays do not recreate execution CSVs.

Preflight validates the value without scoring, reserving observations or checking
whether a useful fill will occur. A zero budget is structurally valid.

## Regression and compatibility

```bash
momentum-lab portfolio benchmark --turnover
```

Eight new SHA-256-locked synthetic ledgers cover partial entries/exits, HOLD,
rotations, high fees, longer delays, zero/nonbinding budgets and cash interest.
The independent standard-library oracle uses exact `Fraction` sign-region algebra
and convex accounts; it never imports production code or captures engine output
as its reference. Every accounting and audit cell is compared.

Portfolio engine schema is now 2; all 24 existing frozen ledgers and default
accounting remain unchanged. Dataset, registry, run-state, preflight and single-
asset engine schemas are unchanged. Use new runs/studies after upgrading;
source/version-locked older research needs its original environment.

## 中文说明

在组合 JSON 配置中添加 `"max_turnover": 0.20`，即可把每次调仓的买卖总额限制在
调仓前净值的 20%。默认不限制。所有买卖腿同比例缩减，未成交部分立即失效，持有日不补单。
报告同时列出请求、实际成交、未成交金额及实际权重。上限也约束退出指令和比较基准，
所以可能继续持有已移出成员名单的资产，也可能使买入持有基准长期保留现金。
这是研究中的换手预算，不是市场流动性证明；原有研究授权、测试封存与揭示流程保持有效。
