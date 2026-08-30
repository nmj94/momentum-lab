"""momentum-lab: Reproducible momentum strategy research."""

from ._version import __version__
from .backtest import backtest, evaluate, evaluate_strategy
from .config import SearchConfig, load_search_config
from .data import MarketDataUnavailableError, download_data, prepare_data
from .indicators import IndicatorDAG
from .reporting import render_html_report, render_markdown_report
from .robustness import robustness_check
from .search import run_search
from .strategies import (
    STRATEGY_REGISTRY,
    RegimeAware,
    get_strategy,
    list_strategies,
)
from .uncertainty import paired_block_bootstrap

__all__ = [
    "STRATEGY_REGISTRY",
    "IndicatorDAG",
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
    "paired_block_bootstrap",
    "prepare_data",
    "render_html_report",
    "render_markdown_report",
    "robustness_check",
    "run_search",
]
