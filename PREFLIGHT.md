# Read-only data preflight (v0.18)

Preflight checks offline input structure **before** invoking a research workflow.
It reads and validates prices internally, but exports only integrity, date and
declaration metadata. It never calculates strategy signals, returns, performance,
volume statistics or scores, and never creates a research registry, run record,
cache or exposure reservation. Output files are optional and explicitly requested.

This is not point-in-time certification, a license check, an execution-feasibility
test or proof that a holdout is unseen. Dates, tickers and hashes can themselves
be sensitive. Do not use preflight results to claim research access consent.

## Commands

```bash
# One offline dataset. No --start/--end means its observed first/last dates.
momentum-lab data check datasets/aaa/manifest.json

# Explicit interval and user-declared exchange-session list.
momentum-lab data check datasets/aaa/manifest.json \
  --start 2024-01-01 --end 2024-01-09 --sessions sessions.json \
  --output experiments/preflight/aaa-v1

# Fixed-rule portfolio recipe; no --acknowledge-history and no scoring.
momentum-lab portfolio preflight --config portfolio.json \
  --sessions sessions.json --output experiments/preflight/portfolio-v1
```

`portfolio preflight` accepts the existing `PortfolioConfig` format. Dataset and
membership paths inside JSON are relative to that recipe; command-line session
and output paths are relative to the working directory. Registered-study-only
fields such as `test_start` and `study_id` are not accepted: this does not validate
study boundaries or access history. Use a separate fixed-rule input recipe to
inspect its datasets; do not edit a frozen study recipe in place.

Stdout is one JSON object. Automated callers should inspect the status and exit
code. `data inspect` and other existing commands keep their existing exit codes.

| Exit | Status | Meaning |
|---:|---|---|
| 0 | `passed` | No structural issues under the supplied declarations and checked interval |
| 1 | `warning` | Human review needed, e.g. exchange calendar unknown or possible duplicate files |
| 2 | `error` | Broken input, incomplete dates or incompatible portfolio declarations |

Invalid invocation options, malformed calendars, invalid recipes, exceeded work
limits or output failures use argparse errors with exit 2 instead of a report.
Individual invalid dataset files produce an `invalid_dataset` issue and do not
prevent other assets from being checked. Raw parsing exceptions are omitted to
avoid echoing price cells; use `data inspect` **privately** for detailed debugging.

## Session-calendar contract

No exchange calendar is guessed or downloaded. Supply a UTF-8 JSON object with
exactly these fields. This example is a **synthetic** session declaration, not a
licensed or independently verified exchange calendar:

```json
{
  "schema_version": 1,
  "calendar_id": "synthetic-example-v1",
  "source": "Project-created example; replace with your own verified source",
  "license": "MIT; synthetic dates only",
  "coverage_start": "2024-01-01",
  "coverage_end": "2024-01-09",
  "sessions": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]
}
```

Dates must be ISO `YYYY-MM-DD`, sorted, unique and inside the inclusive coverage
interval. Coverage must enclose the checked interval, including explicit bounds
outside the observed dataset. The list declares **all** sessions in that coverage;
it may omit weekends and holidays. The manifest is limited to 2 MiB and 100,000
sessions; unknown/duplicate keys, malformed declarations and symlinks are rejected.
The report contains raw and canonical calendar hashes, not its source/license text.

Exchange datasets without a calendar get `calendar_unverified`; ordinary weekdays
are not an exchange calendar. Continuous datasets are checked against every
calendar day, with the same 100,000-day work bound. An explicit session calendar
cannot override that continuous declaration by excluding days.

The checked interval defaults to each snapshot's own observed bounds. To detect
missing history at its beginning or end, supply your intended `start`/`end`.
An early start produces `late_start`; a late end produces `truncated_end`.
A supplied session calendar can independently identify missing early observations.
No gaps, dates, prices, volume or settlement proceeds are ever filled or invented.

## Checks and limits

- Whole-snapshot SHA-256 and strict OHLCV validation, including rows outside a slice.
- Selected row counts and date bounds; missing/unexpected sessions with total
  counts and at most ten sample dates per issue, never price-change diagnostics.
- Adjustment warnings for unadjusted or split-only data. An adjusted declaration
  does not establish correct corporate-action treatment or executable prices.
- All portfolio assets are checked in canonical ticker order. Currency, calendar,
  annualization, adjustment basis and exact selected dates must match; no FX
  conversion, implicit intersection or asynchronous-close inference is performed.
- Each asset needs `lookback + 2` observations. This is only a minimum length gate,
  not proof a valid signal, rebalance, liquidity or non-empty test will exist.
- Identical CSV hashes under distinct ticker labels warn about possible duplicated
  inputs; identical observations do not prove common security identity.
- Optional membership declarations are checked against all aligned asset dates.
  Invalid/misaligned datasets mark this check unavailable instead of inventing a pass.
- Existing 64 MiB CSV / 64 KiB manifest limits apply per source; portfolio inputs
  remain 2–64 assets with at most 1,000,000 selected asset-session cells.

## Python API and report handling

```python
from momentum_lab import preflight_dataset, preflight_portfolio, write_preflight_report

report = preflight_dataset("datasets/aaa/manifest.json", sessions="sessions.json")
portfolio = preflight_portfolio("portfolio.json", sessions="sessions.json")
write_preflight_report(portfolio, "experiments/preflight/new-review")
```

`write_preflight_report` accepts a report returned by the preflight APIs and creates
a **new** directory containing `report.md` and `report.json`. JSON is written last.
An interrupted export can leave a partial directory; retry in a different new
directory. Inputs, existing reports and research artifacts are never overwritten.

Report schema 1 includes the package version, dataset/calendar hashes, structural
recipe fingerprint, issue counts and a canonical SHA-256 of the report excluding
its own `report_sha256`. It contains no timestamp or absolute source paths, so
identical bytes/options under the same version produce identical reports after
relocation. The digest is not a signature and proves neither authorship nor freshness.
Invalid membership/dataset inputs are not certified or assigned verified hashes.

Research still revalidates inputs at invocation; this report is not a capability
to bypass source/version checks, study registration, consent or exposure history.
Accounting, dataset schema 1, cache schema 2, engine schema 6 and all 24 frozen
ledgers are unchanged. Upgrading the package/source requires new runs and studies;
old source-locked research remains usable only with its original environment.

## 中文快速说明

预检只检查数据结构与兼容性，不跑策略、不输出收益，也不登记测试观察记录。
没有交易日清单时，交易所数据会明确提示“完整性未验证”；连续交易数据检查每个自然日。
多资产检查会同时列出日期不齐、币种/日历/复权口径冲突、可能重复的文件和成员历史问题。
它不补价格、不更改输入，不证明数据在历史时点可得，也不替代正式研究所需的授权。
