"""momentum-lab: Autonomous momentum strategy research."""

__version__ = "0.2.0"

from .backtest import backtest, evaluate, evaluate_strategy
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
    "backtest",
    "download_data",
    "evaluate",
    "evaluate_strategy",
    "get_strategy",
    "list_strategies",
    "prepare_data",
    "robustness_check",
    "run_search",
]
