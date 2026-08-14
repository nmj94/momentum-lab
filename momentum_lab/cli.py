"""cli.py - Command-line interface for momentum-lab."""

import argparse
import sys

from . import __version__
from .search import run_search
from .strategies import list_strategies


def main():
    parser = argparse.ArgumentParser(
        prog="momentum-lab",
        description="Find the optimal momentum strategy for any asset. Just provide a ticker.",
        epilog="Examples:\n"
               "  momentum-lab GLD                    # Search gold ETF\n"
               "  momentum-lab SPY --quick            # Quick search S&P 500\n"
               "  momentum-lab BTC-USD --workers 4    # Search Bitcoin with 4 cores\n"
               "  momentum-lab AAPL --strategies tsmom,ma_cross,rsi\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("ticker", nargs="?", default=None,
                        help="Yahoo Finance ticker (e.g. GLD, SPY, BTC-USD, AAPL). "
                             "Omit when using --list.")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 5 params per strategy")
    parser.add_argument("--strategies", type=str, default=None,
                        help="Comma-separated strategy names (default: all)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--cost", type=float, default=1.0, help="Transaction cost in bps")
    parser.add_argument("--start", type=str, default="2004-01-01", help="Data start date")
    parser.add_argument("--end", type=str, default=None, help="Data end date (default: today)")
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore cached data and re-download from Yahoo")
    parser.add_argument("--top", type=int, default=50, help="Number of top results to keep")
    parser.add_argument("--robust", dest="robust", action="store_true", default=True,
                        help="Run robustness check on best params (default: True)")
    parser.add_argument("--no-robust", dest="robust", action="store_false",
                        help="Skip robustness check")
    parser.add_argument("--robust-frac", type=float, default=0.2,
                        help="Perturbation fraction for robustness check (default: 0.2)")
    parser.add_argument("--list", action="store_true", help="List all strategies and exit")
    parser.add_argument("--version", action="version", version=f"momentum-lab {__version__}")

    args = parser.parse_args()

    if args.list:
        list_strategies()
        return

    if not args.ticker:
        parser.error("ticker is required (e.g. momentum-lab GLD). Use --list to see strategies.")

    strategies = None
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",")]

    run_search(
        ticker=args.ticker,
        strategies=strategies,
        cost_bps=args.cost,
        workers=args.workers,
        quick=args.quick,
        top_n=args.top,
        start=args.start,
        end=args.end,
        robust=args.robust,
        robust_frac=args.robust_frac,
        use_cache=not args.refresh,
    )


if __name__ == "__main__":
    main()
