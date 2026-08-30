# Registered portfolio studies (v0.14)

`momentum-lab portfolio study` evaluates a **fixed portfolio rule** on development
data first. A separate, explicit invocation reveals its test results. It is not
a parameter search, selection-adjusted significance test or live trading system.
The older `portfolio --acknowledge-history` command remains a distinct,
whole-history exploratory workflow.

The local registry prevents accidental reuse and preserves access history. It
does not encrypt data, verify external observation history or establish that
historical results are genuinely untouched out-of-sample evidence. A first
**recorded** reveal means only that no overlap is recorded in this registry.

## Configure, develop, then reveal

Import aligned snapshots you may legally use; see [DATASETS.md](DATASETS.md).
All 2–64 candidate assets must have complete positive daily prices with matching
dates, currency, calendar, annualization and adjustment declarations. These
requirements also apply when [declared membership](MEMBERSHIP.md) is supplied.
No licensed market history is bundled.

Adapt [examples/portfolio_study_config.json](examples/portfolio_study_config.json)
to your own assets, files, dates and cost assumptions. Its `AAA`/`BBB` membership
events are synthetic examples, not historical constituents or recommendations.
Omit `universe` for a fixed candidate universe.

```json
{
  "datasets": {
    "AAA": "datasets/aaa-v1/manifest.json",
    "BBB": "datasets/bbb-v1/manifest.json"
  },
  "universe": "membership.json",
  "study_id": "relative-momentum-study-v1",
  "test_start": "2025-01-02",
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
  "start": "2020-01-02",
  "end": "2025-12-31",
  "result_dir": "experiments/portfolio-studies"
}
```

`test_start` must be an **actual aligned session**. Development is every selected
session strictly before that date; test starts on that date and ends at the last
selected session. Require at least `lookback + 2` development sessions and two
test sessions. There is no portfolio train/validation split or model selection
in this version: the recipe is fixed before development work.

Dataset and membership paths in JSON resolve relative to the JSON file.
`result_dir` and `registry_path` resolve from the process working directory.
Direct Python mapping paths resolve from the working directory. An omitted
`run_id` generates a new one; every invocation needs a new output directory.

```bash
# 1. Register the fixed protocol and compute development only.
momentum-lab portfolio study --config study.json --run-id development

# These commands never disclose cached test scores.
momentum-lab portfolio study status relative-momentum-study-v1
momentum-lab portfolio study list
momentum-lab study history --ticker AAA

# 2. After inspecting development evidence, explicitly reveal the fixed rule.
momentum-lab portfolio study --config study.json --run-id test-reveal --reveal-test

# 3. Explicitly reopen the same result: cached summary only, no new backtest.
momentum-lab portfolio study --config study.json --run-id test-replay --reveal-test
```

`--registry PATH` may be supplied to the portfolio study run/status/list commands;
the default is the existing shared user-level registry (or
`MOMENTUM_LAB_REGISTRY_PATH`). Keep that registry and your input snapshots.
Do not switch registries or delete records to create a new freshness label.

Consent flags are invocation-only, never stored in JSON. Attempting a reveal
before an earlier completed development run fails before data loading. A known
overlap, including an interrupted reveal, requires explicit reuse consent:

```bash
momentum-lab portfolio study --config study.json --run-id acknowledged-reuse \
  --reveal-test --allow-test-reuse \
  --test-reuse-reason "These dates were already inspected in earlier research"
```

This yields `repeated_use`, not fresh evidence. Reopening the same completed
study uses its cached result and records `previously_revealed` instead of
recalculating it. The **first completed** result is pinned atomically; a slower,
older, explicitly acknowledged concurrent claim cannot replace the cache.

## What the boundary does and does not protect

Before any evaluation, registration freezes the rule, actual period boundaries,
every asset's data/provenance, optional membership hashes, execution convention,
source fingerprint, package version, dependency environment and lockfile.
Moving unchanged input files is supported. Changed source, declarations, raw
membership bytes (including reformatting), dates or software require a new
protocol/study ID. A new ID does not remove known date overlap.

During a sealed development run, the evaluator receives **only copied
development price/membership prefixes and a numeric rule configuration**. Test
boundaries, study IDs, dataset files and membership file paths are not passed
to it. Development exports contain no test rows or test metrics, including
when a non-reveal invocation follows an earlier revealed run.

The coordinator still reads and validates/hashes full input snapshots, and
those raw files remain accessible to their owner. This is a workflow/evaluator
boundary, not physical concealment or custodial security.

Every candidate asset counts toward observation history, including assets that
were inactive or never selected. Development records are inserted atomically.
Before test computation, all candidate test ranges are reserved in **one
SQLite transaction**. One overlapping asset blocks a fresh claim for the whole
portfolio, including overlap with single-asset studies and v0.13 whole-history
portfolio runs. Matching uses canonical tickers and inclusive daily dates,
not data hashes, result directories or study IDs.

The base registry identity, schema/user-version 1 and existing records remain
unchanged. Four additive portfolio tables use their own extension schema 1,
linking portfolio batches to the existing observations table. Single-asset and
portfolio study IDs have separate namespaces but share access history. Missing
or partial extensions, invalid identity, broken cached hashes or incomplete
exposure groups fail closed; there is no reset/delete/rebaseline command.

## Test accounting: carry the account, do not restart it

After the explicit all-asset reservation, the fixed causal rule is evaluated
through the test period with the same shared cash book. Positions, accrued
cash and pending next-close instructions carry across the boundary. There is
no new warm-up or artificial test-day liquidation/re-entry.

Test CSVs include the **last development close as an anchor row**. Its NAV,
holdings, asset values and cash are retained, but its return, fees, P&L,
turnover and trades are zeroed for the period report, with no executed target.
It is not a new fill. Every subsequent return, including the first test-day
return and any first test-day fee, is included.

- `return_intervals` and `n_bars` equal the actual test-session count; the extra
  CSV row is only the anchor.
- Equity is rebased to each account's own boundary NAV. `starting_nav` is shown
  for both strategy and baseline. Absolute final NAVs reflect development
  balances and are **not** an equal-starting-capital or risk-matched comparison.
- Development costs are excluded from test cost/rebalance totals. Sample
  volatility and Sharpe use actual test returns; session-count annualization
  follows the declared dataset convention. Average cash weight includes the
  anchor row, as documented in the boundary policy.
- With membership, the baseline equally weights eligible assets on exactly the
  same signal dates, subject to the target cap. Without membership it is the
  existing matched-warm-up equal-weight buy-and-hold baseline.

The capital, costs, cash convention and other limitations in
[PORTFOLIOS.md](PORTFOLIOS.md) still apply. These metrics are descriptive,
conditional on the chosen candidate set/rule, with no selection correction.

## Artifacts, caching and failure recovery

| Invocation | Exports | `artifact_scope` |
| --- | --- | --- |
| Development | `development_`-prefixed ledger, weights, holdings, values, trades, scores, targets, executed targets and baseline files; optional eligibility | `development_ledgers_only` |
| New explicit reveal | `test_`-prefixed files, including the anchor row; original development summary retained | `test_ledgers` |
| Cached explicit replay | Frozen summaries and new reports; **no regenerated CSVs** | `cached_summary_only` |

Each invocation writes `run_config.json`, `report.md`, `report.html`, and
`summary.json` **last** as its completion marker. A non-reveal summary always
has `test: null` and `test_results_visible: false`. Renderers also defensively
hide stale/injected test fields without an allowed visible access state.

The registry caches bounded JSON summaries (4 MiB maximum per payload), not
complete CSV books. Replay identifies `original_test_output` and keeps the
original test evaluation timestamp. It still revalidates the frozen inputs
and software; keep the original environment and snapshots. Deleted original
ledgers are not recovered or silently recomputed by replay.

An insert failure rolls back the entire group before computation. Once a claim
is durable, calculation/export failure retains all exposures as failed (or
reserved after a hard interruption). Use a new run ID and acknowledge reuse
to retry. Failure **after** a successful cache commit can be recovered through
an explicit cached replay without computing again. Existing directories are
never overwritten. Back up the shared registry as described in [GOVERNANCE.md](GOVERNANCE.md).

## Python and software checks

```python
from momentum_lab import PortfolioStudyConfig, run_portfolio_study

config = PortfolioStudyConfig.from_json("study.json")
config.run_id = "development"
development = run_portfolio_study(config)
# Explicit decision in a later invocation with the same frozen inputs/software:
config.run_id = "test-reveal"
revealed = run_portfolio_study(config, reveal_test=True)
```

```bash
momentum-lab portfolio study benchmark
```

Two packaged SHA-256-locked synthetic cases use an independent Fraction signal
oracle, algebraic fee/holding ledger and scalar period calculations. They cover
entry/exit, monthly changes, all-cash liquidation and carried test boundaries,
comparing complete books and test metrics. The 16 existing single-asset and
6 existing portfolio references are unchanged. CI exercises Python 3.10–3.13
and a clean installed core wheel without optional ML dependencies.

Still out of scope: verified point-in-time data supply, unbalanced IPO histories,
delisting payouts, missing-price/halts settlement, portfolio parameter selection,
selection-aware uncertainty, FX, liquidity/partial fills and live execution.
