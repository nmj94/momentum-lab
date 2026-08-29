"""momentum-lab: Reproducible momentum strategy research."""

from ._version import __version__
from .backtest import backtest, evaluate, evaluate_strategy
from .config import SearchConfig, load_search_config
from .data import MarketDataUnavailableError, download_data, prepare_data
from .reporting import render_html_report, render_markdown_report
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
    "MarketDataUnavailableError",
    "RegimeAware",
    "SearchConfig",
    "__version__",
    "backtest",
    "download_data",
    "evaluate",
    "evaluate_strategy",
    "get_strategy",
    "list_strategies",
    "load_search_config",
    "prepare_data",
    "render_html_report",
    "render_markdown_report",
    "robustness_check",
    "run_search",
]
