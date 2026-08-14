"""momentum-lab: Autonomous momentum strategy research."""

__version__ = "0.2.0"

from .data import download_data, prepare_data
from .backtest import backtest, evaluate, evaluate_strategy
from .strategies import (
    STRATEGY_REGISTRY,
    get_strategy,
    list_strategies,
    RegimeAware,
)
from .search import run_search
from .robustness import robustness_check

__all__ = [
    "download_data",
    "prepare_data",
    "backtest",
    "evaluate",
    "evaluate_strategy",
    "STRATEGY_REGISTRY",
    "get_strategy",
    "list_strategies",
    "RegimeAware",
    "run_search",
    "robustness_check",
]
