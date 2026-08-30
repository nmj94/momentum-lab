"""momentum-lab: Reproducible momentum strategy research."""

from ._version import __version__
from .backtest import backtest, evaluate, evaluate_strategy
from .config import SearchConfig, load_search_config
from .data import MarketDataUnavailableError, download_data, prepare_data
from .datasets import DatasetError, import_dataset, load_dataset
from .governance import RegistryError, StudyRegistry, TestReuseError
from .indicators import IndicatorDAG
from .portfolio import PortfolioError, backtest_portfolio, cross_sectional_momentum, portfolio_metrics
from .portfolio_research import PortfolioConfig, run_portfolio
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
    "DatasetError",
    "IndicatorDAG",
    "MarketDataUnavailableError",
    "PortfolioConfig",
    "PortfolioError",
    "RegimeAware",
    "RegistryError",
    "SearchConfig",
    "StudyRegistry",
    "TestReuseError",
    "__version__",
    "backtest",
    "backtest_portfolio",
    "cross_sectional_momentum",
    "download_data",
    "evaluate",
    "evaluate_strategy",
    "get_strategy",
    "import_dataset",
    "list_strategies",
    "load_dataset",
    "load_search_config",
    "paired_block_bootstrap",
    "portfolio_metrics",
    "prepare_data",
    "render_html_report",
    "render_markdown_report",
    "robustness_check",
    "run_portfolio",
    "run_search",
]
