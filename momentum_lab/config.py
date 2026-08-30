"""Configuration helpers for reproducible strategy searches.

The search engine intentionally accepts a small, JSON-serializable config
object.  Keeping the format dependency-free makes a run easy to reproduce in
CI, a notebook, or a later resume operation.
"""

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class SearchConfig:
    """JSON-compatible configuration for :func:`momentum_lab.run_search`."""

    ticker: str = "GLD"
    strategies: list[str] | None = None
    cost_bps: float = 1.0
    slippage_bps: float = 0.0
    financing_rate: float = 0.0
    financing_spread: float = 0.0
    borrow_bps: float = 0.0
    cash_rate: float = 0.0
    short_rebate_rate: float = 0.0
    spread_bps: float = 0.0
    impact_bps: float = 0.0
    impact_exponent: float = 0.5
    impact_reference_participation: float = 0.01
    max_participation: float | None = None
    initial_capital: float = 1_000_000.0
    min_fee: float = 0.0
    max_leverage: float = 2.0
    execution_model: str = "next_close"
    execution_lag: int = 1
    annualization: float | None = None
    risk_free_rate: float = 0.0
    validation_folds: int = 4
    min_validation_bars: int = 60
    min_validation_trades: int = 1
    min_validation_exposure: float = 0.01
    workers: int = 1
    quick: bool = True
    top_n: int = 50
    start: str = "2004-01-01"
    end: str | None = None
    robust: bool = True
    robust_frac: float = 0.2
    use_cache: bool = True
    result_dir: str | None = None
    run_id: str | None = None
    keep_all_results: bool = False
    generate_report: bool = True
    search_method: str = "grid"
    candidate_budget: int = 256
    halving_factor: int = 3
    halving_stages: int = 3
    indicator_cache_size: int = 256

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SearchConfig":
        """Build a config and reject misspelled keys early."""
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            names = ", ".join(unknown)
            raise ValueError(f"unknown search config field(s): {names}")
        normalized = dict(values)
        if "strategies" in normalized and normalized["strategies"] is not None:
            if isinstance(normalized["strategies"], str):
                normalized["strategies"] = [
                    name.strip() for name in normalized["strategies"].split(",") if name.strip()
                ]
            else:
                normalized["strategies"] = list(normalized["strategies"])
        return cls(**normalized)

    @classmethod
    def from_json(cls, path: str | Path) -> "SearchConfig":
        """Load a config from a UTF-8 JSON file."""
        config_path = Path(path)
        try:
            values = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"search config not found: {config_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON search config {config_path}: {exc.msg}") from exc
        if not isinstance(values, Mapping):
            raise TypeError("search config must contain a JSON object")
        return cls.from_mapping(values)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping suitable for run metadata."""
        return asdict(self)

    def to_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments accepted by ``run_search``."""
        return self.to_dict()


def load_search_config(config: SearchConfig | Mapping[str, Any] | str | Path) -> SearchConfig:
    """Normalize a config supplied as an object, mapping, or JSON path."""
    if isinstance(config, SearchConfig):
        return config
    if isinstance(config, (str, Path)):
        return SearchConfig.from_json(config)
    if isinstance(config, Mapping):
        return SearchConfig.from_mapping(config)
    raise TypeError("config must be SearchConfig, a mapping, or a JSON path")
