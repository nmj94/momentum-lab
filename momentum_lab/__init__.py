"""momentum-lab: Reproducible momentum strategy research."""

from ._version import __version__
from .backtest import backtest, evaluate, evaluate_strategy
from .backups import BackupError, create_backup, inspect_backup, restore_backup
from .config import SearchConfig, load_search_config
from .data import MarketDataUnavailableError, download_data, prepare_data
from .datasets import DatasetError, import_dataset, load_dataset
from .governance import RegistryError, StudyRegistry, TestReuseError
from .indicators import IndicatorDAG
from .portfolio import PortfolioError, backtest_portfolio, cross_sectional_momentum, portfolio_metrics
from .portfolio_governance import PortfolioStudyRegistry
from .portfolio_research import PortfolioConfig, run_portfolio
from .portfolio_study import PortfolioStudyConfig, run_portfolio_study
from .preflight import preflight_dataset, preflight_portfolio, write_preflight_report
from .reporting import render_html_report, render_markdown_report
from .robustness import robustness_check
from .run_control import RunBusyError, RunStateError, inspect_run
from .search import run_search
from .strategies import (
    STRATEGY_REGISTRY,
    RegimeAware,
    get_strategy,
    list_strategies,
)
from .uncertainty import paired_block_bootstrap
from .universe import load_membership

__all__ = [
    "STRATEGY_REGISTRY",
    "BackupError",
    "DatasetError",
    "IndicatorDAG",
    "MarketDataUnavailableError",
    "PortfolioConfig",
    "PortfolioError",
    "PortfolioStudyConfig",
    "PortfolioStudyRegistry",
    "RegimeAware",
    "RegistryError",
    "RunBusyError",
    "RunStateError",
    "SearchConfig",
    "StudyRegistry",
    "TestReuseError",
    "__version__",
    "backtest",
    "backtest_portfolio",
    "create_backup",
    "cross_sectional_momentum",
    "download_data",
    "evaluate",
    "evaluate_strategy",
    "get_strategy",
    "import_dataset",
    "inspect_backup",
    "inspect_run",
    "list_strategies",
    "load_dataset",
    "load_membership",
    "load_search_config",
    "paired_block_bootstrap",
    "portfolio_metrics",
    "preflight_dataset",
    "preflight_portfolio",
    "prepare_data",
    "render_html_report",
    "render_markdown_report",
    "restore_backup",
    "robustness_check",
    "run_portfolio",
    "run_portfolio_study",
    "run_search",
    "write_preflight_report",
]
