# Momentum Lab Roadmap

Momentum Lab is moving from an “optimal strategy finder” prototype toward a
reproducible research and evidence-evaluation workbench. Configuration count is
not a success metric. The project should optimize for honest out-of-sample
evidence, explicit assumptions, and repeatability.

## v0.5 — Safety and reproducibility

- Safe quick/non-ML/streaming defaults.
- Explicit opt-in for exhaustive and ML searches.
- Final portfolio leverage cap and separation of signal search from risk sizing.
- Cash, excess-leverage financing, borrow, transaction-cost, and slippage inputs.
- Inclusive end dates, crypto annualization inference, retrying downloads, and
  atomic user-cache storage.
- Fixed checkpoint schema and source/data/config-locked resume.
- Lightweight core package with optional ML/XGBoost dependencies.
- Clean-wheel CI, Python 3.10—3.13, and a coverage floor.
- Accurate parameter-sensitivity terminology and documented limitations.

## v0.6 — Research integrity and accounting correctness

- Test boundaries are withheld from candidate workers and test metrics are
  absent from candidate artifacts; only the copied final selection is tested.
- 40/40/20 development split with temporal validation folds and expanding
  walk-forward parameter-selection replay.
- Deflated Sharpe selection, analytic 95% Sharpe intervals, CSCV/PBO estimate,
  and minimum observations/trades/exposure gates.
- Absorbing bankruptcy, calendar-day financing, drift-aware rebalancing, and
  explicit short cash/collateral accounting.
- Same-close, next-close, next-open, and delayed-close execution assumptions.
- Transactional SQLite resume journal, atomic report exports, environment and
  lockfile fingerprints, deterministic Latin-hypercube quick sampling, and
  reusable base-signal calculations.
- Strict OHLC data contracts, reusable late-listing cache metadata, and weekly
  provider-contract CI.

## v0.7 — Market realism

Delivered in v0.7.0:

- Time-varying cash, financing, borrow-fee and rebate schedules; financing
  spreads; and dated borrow availability in the Python API.
- Spread, nonlinear participation impact, bar-volume capacity, partial fills,
  and minimum-fee models, wired through search configuration and CLI defaults.
- Automated self-contained HTML and Markdown research reports.

## v0.8 — Scale-aware search

Delivered in v0.8.0:

- Bounded cross-strategy indicator DAG caching for common price, trend,
  volatility, channel, ATR, and ADX dependencies.
- Deterministic Successive Halving with per-strategy candidate budgets,
  increasing validation prefixes, transactional stage resume, and explicit
  stage-decision exports.
- Candidate isolation now withholds test observations as well as test metrics
  and boundaries until one final selection has been made.

Remaining scale work:

- Optional Optuna integration behind the same sealed-development contract.
- Optional Parquet/DuckDB analytics exports on top of the SQLite journal.
- Typed `StrategySpec`, `RunConfig`, `RunResult`, and plugin entry points.

## v0.9 — Frozen software-regression evidence

Delivered in v0.9.0:

- Offline, SHA-256-locked synthetic OHLCV fixtures covering four asset styles,
  weekday/daily calendars, trends/reversals, jumps and liquidity stress.
- Sixteen fixed cash/buy-and-hold/MA/TSMOM cases with complete reference ledgers,
  strict input compatibility and symmetric numeric-change gates.
- Version-comparison JSON and Markdown reports; runtime/traced-allocation
  observations and optional resource limits without automatic rebaselining.
- Frozen regression CI across Python 3.10—3.13 and clean-wheel fixture checks.

## v0.10 — Conditional return uncertainty

Delivered in v0.10.0:

- Paired circular block-bootstrap percentile intervals for selected-strategy
  and buy-and-hold annualized means/Sharpes and paired mean excess returns.
- Separate post-selection validation/test diagnostics, fixed recorded seeds,
  return hashes, bounded batches, work limits and explicit unavailable states.
- Resume-locked configuration, CLI controls and Markdown/HTML report sections.
- Independent scalar-oracle frozen tests; existing ranking/accounting baselines
  remain unchanged. These intervals are not selection-adjusted evidence.

## v0.11 — Registered studies and observation history

Delivered in v0.11.0:

- Frozen study protocols and final selections, default-hidden registered test
  evaluation and invocation-only explicit reveal/reuse acknowledgements.
- Shared ticker/date observation history across run IDs, data versions and
  directories, including prior development use and interrupted revelations.
- Transactional pre-evaluation reservations, cached result replays, registry
  identity checks, legacy import and score-free status/history commands.
- Report access states and defensive hiding, development-only sensitivity,
  concurrent/crash-recovery tests and an installed-wheel lifecycle smoke.

Scope: local workflow/audit controls, not encryption, tamper-proof custody or
proof that historical data was never inspected elsewhere.

## v0.12 — Offline research data and provenance

Delivered in v0.12.0:

- User-supplied daily CSV snapshots with strict dates/OHLCV validation, explicit
  calendar/adjustment/currency/source/usage declarations and original-byte hashes.
- Offline import/inspect/search APIs and CLI, no network fallback, portable
  contracts and provenance-locked study/resume integration.
- Provenance in JSON and human reports, synthetic parser regression tests and
  an installed-wheel offline study/reveal/replay lifecycle.

This provides a path for users to bring data they are licensed to use; it does
not supply licensed historical investment benchmarks or certify those rights.

Next evidence work:

- Licensed, frozen historical cross-asset/cross-regime investment benchmarks.
- Correlated effective-trial estimates and selection-aware uncertainty.
- Stable cross-provider asset identity, shared/custodial registration and
  externally verifiable prospective protocols beyond the local v1 registry.

## v0.13 — Cross-sectional portfolios and shared-cash research

Delivered in v0.13.0:

- Offline fixed-universe, top-k relative-momentum portfolios with optional
  absolute-momentum filtering, target caps, cash slots and daily/weekly/monthly
  causal signals; skip-recent observations are explicitly defined.
- One self-financing long-only book, next-close fills, paid buy/sell costs,
  effective ACT/365 cash interest, held-unit weight drift and a matched-warm-up
  equal-weight buy-and-hold baseline.
- Explicit whole-history invocation consent; all asset/date exposures recorded
  in the shared registry before scores or returns. No new sealed/OOS claim.
- Complete CSV books, reproducibility contracts, offline HTML/Markdown reports,
  six frozen exact-rational oracle cases and an installed-wheel lifecycle check.

## v0.14 — Declared membership and registered fixed-rule portfolios

Delivered in v0.14.0:

- Strict known-on/effective-on membership manifests, coverage and provenance
  checks, causal eligibility masks, forced delayed membership rebalances and
  a membership-aware comparison portfolio.
- Fixed-rule portfolio study protocols, development-only evaluator inputs,
  invocation-only test reveals and atomic all-candidate exposure groups sharing
  the existing single-asset registry/history.
- Carried holdings/cash/pending instructions at the test boundary, separately
  rebased period NAVs, first-test-return inclusion and explicit anchor exports.
- Pinned first-completed summary caches, explicit replay and reuse labels,
  interruption/concurrent-claim regression tests, two independent frozen
  membership/boundary cases, bilingual guides and installed core-wheel CI.

Scope: user-declared membership and local audit control, **not** independently
verified point-in-time data, guaranteed unseen tests, or full IPO/delisting handling.
All candidates still require balanced positive price histories; missing quotes
and liquidation proceeds are not invented.

Remaining portfolio work, in order before parameter search or paper execution:

- Verified point-in-time datasets and stable security IDs; explicit listing,
  delisting, price-availability and cash/stock settlement events with independent
  ledger cases. Support genuine unbalanced histories without forward-filling
  absent/illegal fills or silently dropping delisted assets.
- Development-only multi-asset parameter selection and walk-forward replay
  under the registered test boundary, with selection-aware comparisons and
  uncertainty; fixed-rule v0.14 studies do not yet provide this.
- Shared-clock/provider identity validation; explicit FX and trading-calendar
  models before combining asynchronously traded or differently priced assets.
- Portfolio liquidity/impact/partial-fill models validated against independent
  ledgers; the v0.13 linear-cost book does not inherit these single-asset features.

## v0.15 — Local run reliability and recovery visibility

Delivered in v0.15.0:

- Whole-coordinator local OS locks for all three research workflows, native
  filesystem aliases, non-blocking contention and fork-safe descriptor ownership.
- Separate operational attempt history, conservative crash detection and
  score-free status/history commands with explicit recovery guidance.
- Completion-bound file-size/SHA-256 receipts, opt-in verification without
  score display, real-process regressions and macOS/Windows integration CI.

This does not add automatic resume, whole-directory rollback, network-filesystem
leases or trusted remote custody. Existing research exposure and reveal rules
still apply. The point-in-time data and accounting gates above still precede
portfolio search or paper execution.

## v0.16 — Private recovery bundles

Delivered in v0.16.0:

- Idle-run ownership, complete output/operational snapshots and entire shared
  registry backups using SQLite's online API, retaining committed WAL contents.
- Versioned bounded manifests, SHA-256 receipts, explicit sensitive-export
  consent, score-free verification and exclusive archive publication.
- Verified staging and new-directory inactive restoration; no automatic registry
  rollback, history filtering, protocol rebinding or result recomputation.
- Fault, hostile-archive, concurrency, stale-history and cross-platform tests;
  installed-core recovery checks across all three research workflows.

Remaining recovery work: independently held archive digests/signatures, encrypted
storage, licensed input snapshots and compatible source/environment capsules.
Active migration needs a separately designed monotonic history-merge protocol;
an older registry with the same UUID must never replace newer observations.

## v1.0 — Stable research platform

- Stable public API and migration policy.
- Licensed multi-provider data adapters and point-in-time metadata.
- Stable portfolio/cross-sectional APIs and the remaining controls above.
- Pre-registered research hypotheses and controlled OOS access.
- Reproducible public benchmarks and third-party result review.
- Signed releases, trusted publishing, SBOM/provenance, and complete documentation.

## Post-1.0 — Paper trading only after validation

Broker adapters, event-driven orders, drift monitoring, and kill switches should
start with paper accounts. Live execution is out of scope until research and
execution assumptions have survived sustained forward testing.

## Release gates

Every release must pass lint, offline tests, the coverage floor, wheel build and
clean installation, and deterministic smoke research. Research-method changes
must include a frozen regression dataset and an explanation of how historical
comparability is affected.
