"""momentum-lab: Autonomous momentum strategy research."""

__version__ = "0.4.0"

from .backtest import backtest, evaluate, evaluate_strategy
from .config import SearchConfig, load_search_config
from .data import download_data, prepare_data
from .robustness import robustness_check
from .search import run_search
from .strategies import (
    STRATEGY_REGISTRY,
    RegimeAware,
    get_strategy,
    list_strategies,
)

__all__ = [
    "STRATEGY_REGISTRY",
    "RegimeAware",
    "SearchConfig",
    "backtest",
    "download_data",
    "evaluate",
    "evaluate_strategy",
    "get_strategy",
    "list_strategies",
    "load_search_config",
    "prepare_data",
    "robustness_check",
    "run_search",
]
