"""Fixed-rule, explicitly acknowledged full-history portfolio research.

This workflow is not registered OOS selection. All asset/date ranges are
recorded as development observations BEFORE scores or portfolio P&L are computed.
"""

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from ._version import __version__
from .datasets import DatasetError, load_dataset
from .governance import RegistryError, StudyRegistry
from .portfolio import (
    MAX_PORTFOLIO_ASSETS,
    MAX_PORTFOLIO_CELLS,
    PORTFOLIO_ENGINE_SCHEMA,
    PortfolioError,
    _number,
    _symbols,
    backtest_portfolio,
    check_portfolio_reference,
    cross_sectional_momentum,
    portfolio_metrics,
    validate_execution_options,
    validate_momentum_options,
)
from .portfolio_reporting import render_portfolio_html, render_portfolio_markdown
from .search import (
    _data_snapshot,
    _environment_manifest,
    _file_fingerprint,
    _git_revision,
    _source_fingerprint,
    _write_frame_atomic,
    _write_text_atomic,
)

HISTORY_NOTICE = (
    "Exploratory fixed-rule full-history simulation, not a sealed test or independent out-of-sample evidence. "
    "Every asset's entire evaluated date range is recorded as development exposure before calculation. "
    "Existing sealed studies over these observations may require reuse acknowledgement afterward. "
    "The local registry is not tamper-proof custody and cannot know external history."
)


@dataclass
class PortfolioConfig:
    """JSON-compatible fixed portfolio recipe; consent is invocation-only."""

    datasets: dict[str, str]
    lookback: int = 126
    skip_recent: int = 0
    top_k: int = 1
    rebalance: str = "monthly"
    absolute_threshold: float | None = 0.0
    max_weight: float = 1.0
    cost_bps: float = 1.0
    slippage_bps: float = 0.0
    spread_bps: float = 0.0
    cash_rate: float = 0.0
    risk_free_rate: float = 0.0
    initial_capital: float = 1_000_000.0
    start: str | None = None
    end: str | None = None
    result_dir: str = "experiments/portfolios"
    run_id: str | None = None
    registry_path: str | None = None

    @classmethod
    def from_mapping(cls, values):
        if not isinstance(values, Mapping):
            raise PortfolioError("Portfolio configuration must be a JSON object")
        if not all(isinstance(key, str) for key in values):
            raise PortfolioError("Portfolio configuration keys must be strings")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise PortfolioError(f"Unknown portfolio config fields: {', '.join(sorted(unknown))}")
        if "datasets" not in values:
            raise PortfolioError("Portfolio config requires a datasets mapping")
        return cls(**values)

    @classmethod
    def from_json(cls, path):
        path = Path(path)

        def unique_fields(pairs):
            values = {}
            for key, value in pairs:
                if key in values:
                    raise PortfolioError(f"Duplicate portfolio config field: {key}")
                values[key] = value
            return values

        try:
            values = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_fields)
        except (OSError, ValueError) as exc:
            raise PortfolioError(f"Cannot read portfolio config {path}: {exc}") from exc
        config = cls.from_mapping(values)
        _validate_dataset_mapping(config.datasets)
        config.datasets = {
            ticker: str((path.parent / manifest).resolve()) for ticker, manifest in config.datasets.items()
        }
        return config

    def to_dict(self):
        return asdict(self)


def _validate_dataset_mapping(datasets):
    if not isinstance(datasets, Mapping) or not 2 <= len(datasets) <= MAX_PORTFOLIO_ASSETS:
        raise PortfolioError(f"datasets must map 2-{MAX_PORTFOLIO_ASSETS} canonical tickers to offline manifests")
    _symbols(datasets.keys())
    if any(not isinstance(path, (str, Path)) or not str(path).strip() for path in datasets.values()):
        raise PortfolioError("Every dataset requires a non-empty local manifest path")


def load_portfolio_config(config):
    if isinstance(config, PortfolioConfig):
        return PortfolioConfig.from_mapping(config.to_dict())
    if isinstance(config, (str, Path)):
        return PortfolioConfig.from_json(config)
    return PortfolioConfig.from_mapping(config)


def _validate_config(config):
    _validate_dataset_mapping(config.datasets)
    validate_momentum_options(
        config.lookback,
        config.skip_recent,
        config.top_k,
        config.rebalance,
        config.absolute_threshold,
        config.max_weight,
    )
    if config.top_k > len(config.datasets):
        raise PortfolioError("top_k cannot exceed the number of datasets")
    validate_execution_options(
        config.initial_capital,
        config.cost_bps,
        config.slippage_bps,
        config.spread_bps,
        config.cash_rate,
        1,
    )
    _number(config.risk_free_rate, "risk_free_rate", exclusive_minimum=-1)
    if not isinstance(config.result_dir, (str, Path)) or not str(config.result_dir).strip():
        raise PortfolioError("result_dir must be a non-empty path")
    if config.run_id is not None and (
        not isinstance(config.run_id, str)
        or not config.run_id
        or config.run_id in {".", ".."}
        or "/" in config.run_id
        or "\\" in config.run_id
        or Path(config.run_id).name != config.run_id
    ):
        raise PortfolioError("run_id must be a single directory name")


def _load_universe(config):
    closes, provenance, snapshots = {}, {}, {}
    index = None
    convention = None
    cells = 0
    for symbol, path in sorted(config.datasets.items(), key=lambda pair: pair[0].upper()):
        ticker = symbol.upper()
        frame, source = load_dataset(path, ticker=ticker, start=config.start, end=config.end)
        cells += len(frame)
        if cells > MAX_PORTFOLIO_CELLS:
            raise PortfolioError(f"Portfolio exceeds the {MAX_PORTFOLIO_CELLS}-cell work limit")
        identity = tuple(source[key] for key in ("currency", "calendar", "annualization", "price_adjustment"))
        if convention is not None and identity != convention:
            raise PortfolioError(
                "Assets must share currency, calendar, annualization and price-adjustment declarations; no FX inference"
            )
        if index is not None and not index.equals(frame.index):
            raise PortfolioError(
                "Asset session dates must match exactly; choose an explicit common start/end, never implicit filling/intersection"
            )
        index, convention = frame.index, identity
        closes[ticker] = frame["close"]
        provenance[ticker] = source
        snapshots[ticker] = _data_snapshot(frame)
    prices = pd.DataFrame(closes, index=index)
    prices.index.name = "date"
    if len(prices) < config.lookback + 2:
        raise PortfolioError("Need at least lookback+2 aligned observations for one delayed execution")
    return prices, provenance, snapshots


def run_portfolio(config, *, acknowledge_history=False):
    """Run a fixed, long-only full-history recipe and export an auditable book.

    Consent cannot be stored in PortfolioConfig/JSON. The program must receive
    acknowledge_history=True explicitly. It reserves all asset/date exposures
    before scoring/backtesting; partial failures retain any recorded history.
    New output directories only: no implicit overwrite, resume or OOS claim.
    """
    if acknowledge_history is not True:
        raise PortfolioError(
            "Full-history portfolio research requires explicit acknowledge_history=True / --acknowledge-history"
        )
    config = load_portfolio_config(config)
    _validate_config(config)
    prices, provenance, snapshots = _load_universe(config)
    run_id = config.run_id or f"portfolio_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    output = Path(config.result_dir) / run_id
    if output.exists():
        raise PortfolioError("Portfolio output already exists; use a new run_id (no overwrite or implicit resume)")
    registry = StudyRegistry(config.registry_path)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise PortfolioError(f"Cannot create a new portfolio output directory: {exc}") from exc
    first_source = next(iter(provenance.values()))
    recipe = config.to_dict()
    recipe["datasets"] = {ticker.upper(): str(Path(path).resolve()) for ticker, path in config.datasets.items()}
    recipe["result_dir"] = str(Path(config.result_dir).resolve())
    recipe["run_id"] = run_id
    recipe["registry_path"] = str(registry.path)
    source_root = Path(__file__).resolve().parent.parent
    contract = {
        "recipe": {
            key: value
            for key, value in recipe.items()
            if key not in {"datasets", "result_dir", "run_id", "registry_path"}
        },
        "data_provenance": provenance,
        "evaluated_snapshots": snapshots,
        "portfolio_engine_schema": PORTFOLIO_ENGINE_SCHEMA,
        "source_fingerprint": _source_fingerprint(),
        "package_version": __version__,
        "environment": _environment_manifest(),
        "lock_fingerprint": _file_fingerprint(source_root / "uv.lock"),
        "execution_model": "next_close",
        "execution_lag": 1,
        "cash_convention": "effective annual rate, ACT/365",
        "benchmark": "equal_weight_buy_and_hold_after_same_warmup_with_same_target_cap",
        "observation_scope": "entire_evaluated_history_is_development",
    }
    contract_hash = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()
    metadata = {
        **contract,
        "config": recipe,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_revision(),
        "contract_sha256": contract_hash,
        "registry_id": registry.registry_id,
        "registry_path": str(registry.path),
        "history_acknowledged": True,
        "history_notice": HISTORY_NOTICE,
        "data_start": str(prices.index[0].date()),
        "data_end": str(prices.index[-1].date()),
        "annualization": first_source["annualization"],
        "currency": first_source["currency"],
    }
    _write_text_atomic(output / "run_config.json", json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))
    # Do not start even signal calculation until ALL reservations are durable.
    # If a later reservation fails, earlier records conservatively remain;
    # retrying must not erase them or manufacture fresh evidence.
    for ticker in prices.columns:
        registry.record_development(
            ticker=ticker,
            start=prices.index[0],
            end=prices.index[-1],
            data_snapshot=snapshots[ticker],
            run_id=run_id,
            run_path=output,
            study_id=None,
        )
    plan = cross_sectional_momentum(
        prices,
        lookback=config.lookback,
        skip_recent=config.skip_recent,
        top_k=config.top_k,
        rebalance=config.rebalance,
        absolute_threshold=config.absolute_threshold,
        max_weight=config.max_weight,
    )
    execution = {
        "initial_capital": config.initial_capital,
        "cost_bps": config.cost_bps,
        "slippage_bps": config.slippage_bps,
        "spread_bps": config.spread_bps,
        "cash_rate": config.cash_rate,
        "execution_lag": 1,
    }
    result = backtest_portfolio(plan["targets"], prices, **execution)
    benchmark_targets = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    first_signal = plan["rebalance"].loc[lambda values: values].index[0]
    benchmark_targets.loc[first_signal] = min(1.0 / len(prices.columns), config.max_weight)
    benchmark = backtest_portfolio(benchmark_targets, prices, **execution)
    annualization = first_source["annualization"]
    metrics = portfolio_metrics(result, annualization=annualization, risk_free_rate=config.risk_free_rate)
    benchmark_metrics = portfolio_metrics(benchmark, annualization=annualization, risk_free_rate=config.risk_free_rate)
    last_signal = plan["rebalance"].loc[lambda values: values].index[-1]
    execution_dates = result["ledger"].loc[lambda frame: frame["rebalance_executed"]].index
    summary = {
        "run_id": run_id,
        "status": "completed",
        "research_status": "exploratory_full_history",
        "history_notice": HISTORY_NOTICE,
        "history_acknowledged": True,
        "assets": list(prices.columns),
        "data_start": metadata["data_start"],
        "data_end": metadata["data_end"],
        "n_bars": len(prices),
        "warmup_bars": config.lookback,
        "currency": first_source["currency"],
        "contract_sha256": contract_hash,
        "data_provenance": provenance,
        "metrics": metrics,
        "benchmark_metrics": benchmark_metrics,
        "latest_weights": result["weights"].iloc[-1].to_dict(),
        "latest_cash_weight": float(result["ledger"]["cash_weight"].iloc[-1]),
        "last_signal_date": str(last_signal.date()),
        "last_execution_date": str(execution_dates[-1].date()) if len(execution_dates) else None,
        "last_signal_targets": plan["targets"].loc[last_signal].to_dict(),
        "last_signal_scores": plan["scores"].loc[last_signal].to_dict(),
    }
    exports = {
        "ledger.csv": result["ledger"],
        "weights.csv": result["weights"],
        "holdings.csv": result["holdings"],
        "asset_values.csv": result["asset_values"],
        "trades.csv": result["trades"],
        "targets.csv": plan["targets"],
        "scores.csv": plan["scores"],
        "executed_targets.csv": result["executed_targets"],
        "benchmark_ledger.csv": benchmark["ledger"],
        "benchmark_weights.csv": benchmark["weights"],
    }
    for name, frame in exports.items():
        _write_frame_atomic(frame.reset_index(), output / name)
    _write_text_atomic(output / "report.md", render_portfolio_markdown(summary, metadata))
    _write_text_atomic(output / "report.html", render_portfolio_html(summary, metadata))
    summary["reports"] = {"markdown": "report.md", "html": "report.html"}
    # Completion marker comes last; a failed export never looks completed.
    _write_text_atomic(output / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return {**summary, "result_dir": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="momentum-lab portfolio",
        description=HISTORY_NOTICE,
        epilog="Frozen software ledger check: momentum-lab portfolio benchmark",
    )
    if argv and argv[0] == "benchmark":
        parser.parse_args(argv[1:])
        try:
            result = check_portfolio_reference()
        except PortfolioError as exc:
            parser.error(str(exc))
        print(f"Frozen portfolio ledger: {result['status']} ({result['cases']} cases)")
        return 0
    parser.add_argument("--config", required=True, help="JSON fixed-rule portfolio recipe with 2+ offline datasets")
    parser.add_argument(
        "--acknowledge-history",
        action="store_true",
        help="Explicitly allow all-history scoring and record every asset/date as development exposure",
    )
    args = parser.parse_args(argv)
    try:
        result = run_portfolio(args.config, acknowledge_history=args.acknowledge_history)
    except (PortfolioError, DatasetError, RegistryError) as exc:
        parser.error(str(exc))
    print(f"Portfolio completed: {result['run_id']} ({len(result['assets'])} assets, {result['n_bars']} bars)")
    print(HISTORY_NOTICE)
    print(f"Report: {Path(result['result_dir']) / 'report.html'}")
    return 0
