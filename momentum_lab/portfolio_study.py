"""Fixed-rule portfolio studies: development first, explicit test reveal later.

The coordinator validates/hashes all input files, but before consent the
evaluator only receives copied development prices/membership and numeric rules.
This is a local audit boundary, not encrypted custody or proof of fresh OOS data.
"""

import argparse
import copy
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np

from .datasets import DatasetError
from .governance import AUDIT_WARNING, RegistryError, _canonical, _hash, _now, validate_study_options
from .portfolio import PortfolioError
from .portfolio_governance import PortfolioStudyRegistry
from .portfolio_reporting import render_portfolio_html, render_portfolio_markdown
from .portfolio_research import (
    PortfolioConfig,
    _book_exports,
    _compute_books,
    _load_universe,
    _membership,
    _research_contract,
    _summarize_books,
    _validate_config,
)
from .search import _git_revision, _write_frame_atomic, _write_text_atomic
from .universe import daily_date

PORTFOLIO_STUDY_SCHEMA = 1
BOUNDARY_POLICY = (
    "One causal account is carried from development into test, including cash, holdings and pending instructions. "
    "Test metrics use the last development close as a zero-return anchor; the first test return is included. "
    "Development costs are excluded from test totals. Each account is rebased to its own boundary NAV, shown as "
    "starting_nav; final absolute NAVs are not an equal-starting-capital comparison. Average cash weight includes the anchor."
)


@dataclass
class PortfolioStudyConfig(PortfolioConfig):
    study_id: str | None = None
    test_start: str | None = None


def load_portfolio_study_config(config):
    if isinstance(config, PortfolioStudyConfig):
        return PortfolioStudyConfig.from_mapping(config.to_dict())
    if isinstance(config, (str, Path)):
        return PortfolioStudyConfig.from_json(config)
    return PortfolioStudyConfig.from_mapping(config)


def _evaluation_recipe(config):
    """No test boundary, study ID, input files or registry paths reach evaluation."""
    names = (
        "lookback",
        "skip_recent",
        "top_k",
        "rebalance",
        "absolute_threshold",
        "max_weight",
        "cost_bps",
        "slippage_bps",
        "spread_bps",
        "cash_rate",
        "risk_free_rate",
        "initial_capital",
    )
    return PortfolioConfig(datasets={}, **{name: getattr(config, name) for name in names})


def _test_books(books, anchor):
    """Rebase both ledgers independently; retain the first test interval/fill."""
    result = {"plan": books["plan"], "eligibility": books["eligibility"]}
    for account in ("result", "benchmark"):
        frames = {name: frame.loc[anchor:].copy(deep=True) for name, frame in books[account].items()}
        ledger = frames["ledger"]
        nav = float(ledger["nav"].iloc[0])
        ledger["equity"] = ledger["nav"] / nav
        for name in (
            "asset_pnl",
            "cash_interest",
            "transaction_cost",
            "traded_notional",
            "turnover",
            "gross_return",
            "return",
        ):
            ledger.loc[anchor, name] = 0.0
        ledger.loc[anchor, "pre_trade_nav"] = nav
        ledger.loc[anchor, "rebalance_executed"] = False
        frames["trades"].loc[anchor] = 0.0
        frames["executed_targets"].loc[anchor] = np.nan
        result[account] = frames
    return result


def _phase_files(output, phase, books, start=None):
    for name, frame in _book_exports(books).items():
        visible = frame.loc[start:] if start is not None else frame
        _write_frame_atomic(visible.reset_index(), output / f"{phase}_{name}")


def _visible_phase(summary):
    access = summary.get("test_access", {})
    visible = (
        access.get("test_results_visible") is True
        and access.get("status") in {"first_recorded_reveal", "repeated_use", "previously_revealed"}
        and isinstance(summary.get("test"), dict)
    )
    phase = copy.deepcopy(summary["test"] if visible else summary["development"])
    phase["run_id"] = summary["run_id"]
    phase["performance_scope"] = "revealed test" if visible else "development only; test hidden"
    phase["history_notice"] = (
        f"Registered fixed-rule portfolio study {summary['study_id']}. "
        + (
            f"Test access: {access['status']}. "
            if visible
            else "Test results are hidden; no test metrics or ledgers exported. "
        )
        + (
            "Cached summary replay only; original ledgers are not recreated. "
            if access.get("cached") and visible
            else ""
        )
        + AUDIT_WARNING
        + " "
        + BOUNDARY_POLICY
    )
    return phase


def render_portfolio_study_markdown(summary, metadata):
    return render_portfolio_markdown(_visible_phase(summary), metadata)


def render_portfolio_study_html(summary, metadata):
    return render_portfolio_html(_visible_phase(summary), metadata)


def run_portfolio_study(config, *, reveal_test=False, allow_test_reuse=False, test_reuse_reason=None):
    """Register and evaluate development, or reveal the previously frozen rule.

    Consent is invocation-only. Every invocation requires a new output directory.
    Cached replays revalidate frozen inputs/software but never recalculate books.
    """
    config = load_portfolio_study_config(config)
    validate_study_options(config.study_id, reveal_test, allow_test_reuse, test_reuse_reason)
    if config.study_id is None:
        raise RegistryError("Portfolio study requires study_id")
    test_date = daily_date(config.test_start, "test_start").isoformat()
    _validate_config(config)
    registry = PortfolioStudyRegistry(config.registry_path, create=not reveal_test)
    if reveal_test:
        registry.require_reveal_ready(config.study_id)
    prices, provenance, snapshots = _load_universe(config)
    eligibility, membership_source = _membership(config, prices)
    sessions = list(prices.index.strftime("%Y-%m-%d"))
    if test_date not in sessions:
        raise PortfolioError("test_start must be an actual aligned session date")
    split = sessions.index(test_date)
    if split < config.lookback + 2 or len(prices) - split < 2:
        raise PortfolioError("Require at least lookback+2 development sessions and two test sessions")
    periods = {"development": [sessions[0], sessions[split - 1]], "test": [test_date, sessions[-1]]}
    contract = _research_contract(config, provenance, snapshots, membership_source)
    contract.update(
        portfolio_study_schema=PORTFOLIO_STUDY_SCHEMA,
        observation_scope="development_only_until_explicit_test_reveal",
        boundary_policy=BOUNDARY_POLICY,
    )
    protocol = {**contract, "kind": "fixed_rule_portfolio_v1", "assets": snapshots, "periods": periods}
    run_id = config.run_id or f"portfolio_study_{uuid4().hex[:16]}"
    output = Path(config.result_dir).resolve() / run_id
    if output.exists():
        raise PortfolioError("Portfolio output already exists; use a new run_id (no overwrite or implicit resume)")
    if reveal_test:
        registry.require_protocol(config.study_id, protocol)
    else:
        registry.register(config.study_id, protocol)
    recipe = config.to_dict()
    recipe.update(
        datasets={ticker.upper(): str(Path(path).resolve()) for ticker, path in config.datasets.items()},
        result_dir=str(Path(config.result_dir).resolve()),
        registry_path=str(registry.path),
        run_id=run_id,
    )
    first_source = next(iter(provenance.values()))
    metadata = {
        **contract,
        "config": recipe,
        "run_id": run_id,
        "study_id": config.study_id,
        "created_at": _now(),
        "git_sha": _git_revision(),
        "contract_sha256": _hash(_canonical(protocol)),
        "registry_id": registry.registry_id,
        "registry_path": str(registry.path),
        "periods": periods,
        "annualization": first_source["annualization"],
        "currency": first_source["currency"],
    }
    output.mkdir(parents=True, exist_ok=False)
    _write_text_atomic(output / "run_config.json", json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))
    numeric_recipe = _evaluation_recipe(config)
    test, original_output = None, None
    if not reveal_test:
        registry.record_portfolio_development(config.study_id, run_id, output)
        development_prices = prices.iloc[:split].copy(deep=True)
        development_mask = None if eligibility is None else eligibility.iloc[:split].copy(deep=True)
        books = _compute_books(numeric_recipe, development_prices, development_mask)
        development = _summarize_books(numeric_recipe, development_prices, books, metadata)
        development.update(
            research_status="registered_fixed_rule_development",
            history_acknowledged=False,
            history_notice="Registered fixed-rule development only; test hidden. " + AUDIT_WARNING,
            performance_scope="development only; test hidden",
            evaluated_at=_now(),
        )
        _phase_files(output, "development", books)
        registry.complete_development(config.study_id, {"recipe": contract["recipe"]}, development)
        access = registry.status(config.study_id)
        artifact_scope = "development_ledgers_only"
    else:
        development = registry.development_payload(config.study_id)
        claim = registry.claim_test(
            config.study_id, run_id, output, allow_reuse=allow_test_reuse, reason=test_reuse_reason
        )
        access = claim["access"]
        if claim["cached"]:
            test = claim["payload"]["test"]
            original_output = claim["payload"]["original_test_output"]
            artifact_scope = "cached_summary_only"
        else:
            try:
                books = _compute_books(
                    numeric_recipe, prices.copy(deep=True), None if eligibility is None else eligibility.copy(deep=True)
                )
                anchor = prices.index[split - 1]
                test_books = _test_books(books, anchor)
                test = _summarize_books(numeric_recipe, prices.iloc[split:], test_books, metadata)
                test.update(
                    research_status="registered_fixed_rule_test",
                    history_acknowledged=False,
                    history_notice="Registered fixed-rule test; explicitly revealed. " + AUDIT_WARNING,
                    performance_scope="revealed test",
                    warmup_bars=0,
                    anchor_date=str(anchor.date()),
                    evaluated_at=_now(),
                )
                for metric_key, account in (("metrics", "result"), ("benchmark_metrics", "benchmark")):
                    test[metric_key]["starting_nav"] = float(test_books[account]["ledger"]["nav"].iloc[0])
                _phase_files(output, "test", test_books, start=anchor)
                original_output = str(output)
                registry.complete_test(access["batch_id"], {"test": test, "original_test_output": original_output})
            except Exception as exc:
                registry.fail_test(access["batch_id"], exc)
                raise
            access = {**access, "test_results_visible": True}
            artifact_scope = "test_ledgers"
    summary = {
        "run_id": run_id,
        "study_id": config.study_id,
        "status": "completed",
        "research_status": "registered_fixed_rule_portfolio",
        "periods": periods,
        "contract_sha256": metadata["contract_sha256"],
        "test_access": access,
        "development": development,
        "test": test,
        "boundary_policy": BOUNDARY_POLICY,
        "original_test_output": original_output,
        "artifact_scope": artifact_scope,
        "reports": {"markdown": "report.md", "html": "report.html"},
    }
    # A failure after cache commit is recoverable through an explicit cached replay.
    _write_text_atomic(output / "report.md", render_portfolio_study_markdown(summary, metadata))
    _write_text_atomic(output / "report.html", render_portfolio_study_html(summary, metadata))
    _write_text_atomic(output / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return {**summary, "result_dir": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="momentum-lab portfolio study",
        description="Fixed-rule portfolio development and explicit sealed-test reveals. " + AUDIT_WARNING,
        epilog="Inspect without test scores: momentum-lab portfolio study status ID --registry PATH (or list).",
    )
    if argv and argv[0] in {"status", "list", "benchmark"}:
        command = argv[0]
        if command == "status":
            parser.add_argument("study_id")
        if command != "benchmark":
            parser.add_argument("--registry")
        args = parser.parse_args(argv[1:])
        try:
            if command == "benchmark":
                from .portfolio_benchmarks import check_portfolio_study_reference

                result = check_portfolio_study_reference()
                print(f"Frozen portfolio study: {result['status']} ({result['cases']} cases)")
                return 0
            registry = PortfolioStudyRegistry(args.registry, create=False)
            result = (
                registry.status(args.study_id)
                if command == "status"
                else {
                    "registry_id": registry.registry_id,
                    "studies": registry.list_studies(),
                    "test_results_visible": False,
                    "warning": AUDIT_WARNING,
                }
            )
        except (RegistryError, PortfolioError, OSError, sqlite3.DatabaseError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    parser.add_argument("--config", required=True, help="JSON fixed-rule portfolio study recipe")
    parser.add_argument("--run-id", help="New output directory name for this invocation")
    parser.add_argument("--registry", help="Shared observation registry path")
    parser.add_argument("--reveal-test", action="store_true", help="Explicitly reveal the previously frozen rule")
    parser.add_argument("--allow-test-reuse", action="store_true", help="Acknowledge recorded overlapping observations")
    parser.add_argument("--test-reuse-reason", help="Required reason for acknowledged reuse")
    args = parser.parse_args(argv)
    try:
        config = load_portfolio_study_config(args.config)
        if args.run_id is not None:
            config.run_id = args.run_id
        if args.registry is not None:
            config.registry_path = args.registry
        result = run_portfolio_study(
            config,
            reveal_test=args.reveal_test,
            allow_test_reuse=args.allow_test_reuse,
            test_reuse_reason=args.test_reuse_reason,
        )
    except (PortfolioError, RegistryError, DatasetError, OSError, sqlite3.DatabaseError) as exc:
        parser.error(str(exc))
    print(f"Portfolio study completed: {result['study_id']} / {result['run_id']}")
    print(f"Test access: {result['test_access']['status']}; visible: {result['test_access']['test_results_visible']}")
    print(AUDIT_WARNING)
    print(f"Report: {Path(result['result_dir']) / 'report.html'}")
    return 0
