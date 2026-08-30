# Declared historical membership (v0.14)

Portfolio recipes optionally accept `"universe": "membership.json"`. This
controls which assets may receive a **new target** at each signal close. It
supports user-declared announcement/effective dates, not independently verified
point-in-time constituents, exchange trading status or delisting settlement.
The same input is supported by exploratory portfolios and registered
[portfolio studies](PORTFOLIO_STUDIES.md).

## Strict manifest

The following is synthetic software-example data, not market history:

```json
{
  "schema_version": 1,
  "universe_id": "synthetic-example",
  "source": "Project-authored synthetic events",
  "license": "MIT for this example only",
  "coverage_start": "2024-01-01",
  "coverage_end": "2024-12-31",
  "initial_known_on": "2023-12-29",
  "initial_members": ["AAA"],
  "events": [
    {"ticker": "BBB", "known_on": "2024-01-03", "effective_on": "2024-01-08", "action": "add"},
    {"ticker": "AAA", "known_on": "2024-01-05", "effective_on": "2024-01-10", "action": "remove"}
  ]
}
```

Only these fields are accepted. Duplicate/unknown JSON fields, unsupported
versions, invalid dates, conflicting same-asset/effective-date events, duplicate
ticker aliases and redundant add/remove actions are rejected. Schema version
is integer `1`, not boolean. Dates must be exact valid `YYYY-MM-DD` labels.
Source, license and ID must be non-empty trimmed text without control characters
(at most 2048 characters each).

- `initial_members` is the member set at `coverage_start`. It may be empty.
  `initial_known_on <= coverage_start <= coverage_end` is required.
- Each event must become effective **strictly after** `coverage_start` and no
  later than `coverage_end`. `known_on <= effective_on` is required. A later
  announcement cannot retroactively change an earlier portfolio decision;
  late corrections are rejected, not backdated.
- `known_on` is a declaration that the event was available **before that day's
  signal close**. Intraday publication timestamps and cross-market clocks are
  not modeled. If that declaration is not true, do not use the date as given.
- Membership changes on the first observed session on or after `effective_on`.
  A weekend event therefore applies on the next available session. Events
  before a selected price slice establish its initial state; future events do
  not enable early entry.
- Coverage must include every evaluated session. Every asset referenced by the
  initial set or any event needs a supplied dataset, even when an event lies
  outside a selected price slice. A never-active candidate is allowed and is
  still included in research observation records.

Limits: 2 MiB UTF-8 manifest, 10,000 events, 64 candidate assets and the existing
1,000,000 asset-session work limit. No network fetch or implicit membership
inference occurs. Names normalize to uppercase; duplicate case aliases fail.

## Signals, fills and the baseline

Ineligible assets' scores are masked, and they cannot be selected even when
the absolute-momentum filter is disabled. An all-ineligible universe yields a
cash target. Unfilled top-k slots still remain cash; they are not rescaled.

A membership change after warm-up forces a signal even between ordinary
weekly/monthly dates. **Fills still occur at the next observed close**, not on
the event close. Existing holdings, or an earlier pending instruction, may
therefore still be present on a removal day. This timing is deliberate; the
engine does not invent an immediate same-close exit or cancel past instructions
using future information. A final-session exit remains pending if there is no
later quote.

With membership, the comparison portfolio equally weights all currently
eligible assets on the strategy's signal dates (including membership-triggered
dates), with weight `min(1 / eligible_count, max_weight)` per asset and residual
cash. It uses the same costs, delay and cash assumptions. It does not buy a
future member early. Reports label it **Membership equal-weight rebalanced**.
Without a manifest, the original equal-weight buy-and-hold baseline is unchanged.

## Provenance and remaining data limitations

The report/contract includes source, license, declared coverage, event count,
raw manifest SHA-256 and normalized-content SHA-256. Moving unchanged bytes is
supported; even whitespace reformatting changes the raw hash and thus a frozen
study protocol. Preserve the original file used for registration. Paths in a
JSON recipe resolve relative to that JSON; direct API paths use the working
directory. Optional `eligibility.csv` exports the actual boolean signal mask
(with development/test prefixes in registered studies).

This release still requires **complete, positive, aligned prices for every
candidate across the entire selected range**, including inactive periods and
the delayed exit session. It neither forward-fills absent prices nor substitutes
zero proceeds. It does not support a genuine pre-IPO price gap, unquoted delisted
asset, halt or cash/stock liquidation distribution. Narrowing the range changes
the research question and does not solve missing delisting history.

Membership alone does not eliminate survivorship/selection bias: a candidate
superset picked after observing outcomes remains biased. Source/license and
known-on declarations are not independently certified. Stable security IDs,
verified historical data, explicit price availability and corporate-action
settlements are separate next-stage work, not completed by this manifest.

```python
from momentum_lab import load_membership, cross_sectional_momentum

eligibility, provenance = load_membership("membership.json", prices.index, prices.columns)
plan = cross_sectional_momentum(prices, lookback=126, eligibility=eligibility)
```

Low-level numerical APIs do not record observation history. Use the registered
workflow for its audit boundary; neither API establishes fresh investment evidence.
