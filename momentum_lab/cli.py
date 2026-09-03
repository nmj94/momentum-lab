"""cli.py - Command-line interface for momentum-lab."""

import argparse
import sqlite3
import sys

from . import __version__
from .data import MarketDataUnavailableError
from .search import run_search
from .strategies import STRATEGY_REGISTRY, list_strategies


def _run_checked(parser, **kwargs):
    try:
        return run_search(**kwargs)
    except (ValueError, TypeError, OSError, sqlite3.DatabaseError, MarketDataUnavailableError) as exc:
        parser.error(str(exc))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "portfolio":
        from .portfolio_research import main as portfolio_main

        return portfolio_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "data":
        from .datasets import main as data_main

        return data_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "study":
        from .governance import main as study_main

        return study_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        from .benchmarks import main as benchmark_main

        return benchmark_main(sys.argv[2:])
    parser = argparse.ArgumentParser(
        prog="momentum-lab",
        description="Compare momentum strategies with reproducible out-of-sample evaluation.",
        epilog="Examples:\n"
        "  momentum-lab GLD                    # Search gold ETF\n"
        "  momentum-lab SPY --quick            # Quick search S&P 500\n"
        "  momentum-lab BTC-USD --workers 4    # Search Bitcoin with 4 cores\n"
        "  momentum-lab AAPL --strategies tsmom,ma_cross,rsi\n"
        "  momentum-lab benchmark              # Offline frozen regression suite\n"
        "  momentum-lab portfolio --help       # Fixed-rule multi-asset research\n"
        "  momentum-lab portfolio study --help # Registered portfolio study\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        default=None,
        help="Yahoo Finance ticker (e.g. GLD, SPY, BTC-USD, AAPL). Optional with --config.",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="JSON search config; its values take precedence over CLI defaults"
    )
    parser.add_argument(
        "--dataset", help="Offline daily CSV manifest; validates bytes/declarations and never downloads data"
    )
    parser.add_argument("--resume", action="store_true", help="Resume an existing --run-id from its SQLite journal")
    parser.add_argument(
        "--study-id", help="Register a fixed research protocol; hide test results until explicitly revealed"
    )
    parser.add_argument(
        "--registry", dest="registry_path", help="Shared observation registry (independent of result-dir)"
    )
    parser.add_argument("--reveal-test", action="store_true", help="Reveal the previously frozen study selection")
    parser.add_argument(
        "--allow-test-reuse", action="store_true", help="Acknowledge known prior test-period observations"
    )
    parser.add_argument(
        "--test-reuse-reason", help="Required reason for --allow-test-reuse; remains in the audit record"
    )
    search_mode = parser.add_mutually_exclusive_group()
    search_mode.add_argument(
        "--quick", dest="quick", action="store_true", default=True, help="Quick mode: 5 params per strategy (default)"
    )
    search_mode.add_argument(
        "--exhaustive", dest="quick", action="store_false", help="Run the full grid; can take days for ML strategies"
    )
    search_mode.add_argument(
        "--successive-halving",
        dest="search_method",
        action="store_const",
        const="successive_halving",
        help="Budgeted deterministic search with staged validation resources",
    )
    parser.set_defaults(search_method="grid")
    strategy_mode = parser.add_mutually_exclusive_group()
    strategy_mode.add_argument(
        "--strategies", type=str, default=None, help="Comma-separated strategy names (default: non-ML strategies)"
    )
    strategy_mode.add_argument(
        "--all-strategies", action="store_true", help="Include experimental ML strategies as well"
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument(
        "--candidate-budget",
        type=int,
        default=256,
        help="Initial deterministic candidates per strategy for successive halving",
    )
    parser.add_argument("--halving-factor", type=int, default=3, help="Candidate reduction factor per stage")
    parser.add_argument("--halving-stages", type=int, default=3, help="Maximum validation-resource stages")
    parser.add_argument(
        "--indicator-cache-size",
        type=int,
        default=256,
        help="Maximum reusable indicator nodes per process; 0 disables caching",
    )
    parser.add_argument("--cost", type=float, default=1.0, help="Transaction cost in bps")
    parser.add_argument("--slippage", type=float, default=0.0, help="Additional slippage in bps")
    parser.add_argument("--financing-rate", type=float, default=0.0, help="Annual financing rate (decimal)")
    parser.add_argument("--financing-spread", type=float, default=0.0, help="Annual spread over the financing rate")
    parser.add_argument("--borrow-bps", type=float, default=0.0, help="Annual short borrow fee in bps")
    parser.add_argument("--cash-rate", type=float, default=0.0, help="Annual return earned by uninvested cash")
    parser.add_argument(
        "--short-rebate-rate", type=float, default=0.0, help="Annual rebate earned on short-sale collateral"
    )
    parser.add_argument("--spread-bps", type=float, default=0.0, help="Quoted full bid/ask spread in bps")
    parser.add_argument(
        "--impact-bps", type=float, default=0.0, help="Market impact in bps at the reference participation rate"
    )
    parser.add_argument("--impact-exponent", type=float, default=0.5, help="Positive nonlinear impact exponent")
    parser.add_argument(
        "--impact-reference-participation",
        type=float,
        default=0.01,
        help="Participation rate where --impact-bps is quoted (default: 0.01)",
    )
    parser.add_argument(
        "--max-participation",
        type=float,
        default=None,
        help="Maximum fraction of bar dollar volume traded; enables capacity limits",
    )
    parser.add_argument(
        "--initial-capital", type=float, default=1_000_000.0, help="Starting NAV for capacity and fee calculations"
    )
    parser.add_argument("--min-fee", type=float, default=0.0, help="Minimum currency fee per non-zero rebalance")
    parser.add_argument("--max-leverage", type=float, default=2.0, help="Final absolute exposure cap")
    parser.add_argument(
        "--execution-model",
        choices=["same_close", "next_close", "next_open", "delayed_close"],
        default="next_close",
        help="Fill model for close-derived signals (default: next_close)",
    )
    parser.add_argument(
        "--execution-lag",
        type=int,
        default=1,
        help="Close-bar delay used only by --execution-model delayed_close",
    )
    parser.add_argument(
        "--annualization", type=float, default=None, help="Return periods per year (default: infer 252/365)"
    )
    parser.add_argument(
        "--risk-free-rate", type=float, default=0.0, help="Annual risk-free rate as a decimal (default: 0)"
    )
    parser.add_argument(
        "--validation-folds", type=int, default=4, help="Even number of temporal validation folds (default: 4)"
    )
    parser.add_argument("--min-val-bars", type=int, default=60, help="Minimum validation observations")
    parser.add_argument("--min-val-trades", type=int, default=1, help="Minimum validation trades")
    parser.add_argument("--min-val-exposure", type=float, default=0.01, help="Minimum mean validation exposure")
    parser.add_argument(
        "--no-bootstrap",
        dest="bootstrap",
        action="store_false",
        default=True,
        help="Disable post-selection uncertainty diagnostics (ranking is unaffected)",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2000, help="Fixed bootstrap repetitions (200-20000)")
    parser.add_argument("--bootstrap-block-length", type=int, default=10, help="Circular block length in bars")
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95, help="Bootstrap confidence level")
    parser.add_argument(
        "--bootstrap-seed", type=int, default=42, help="Fixed bootstrap seed, chosen before seeing results"
    )
    parser.add_argument(
        "--bootstrap-min-observations", type=int, default=60, help="Minimum observations for an interval"
    )
    parser.add_argument("--start", type=str, default="2004-01-01", help="Data start date")
    parser.add_argument("--end", type=str, default=None, help="Data end date (default: today)")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached data and re-download from Yahoo")
    parser.add_argument("--top", type=int, default=50, help="Number of top results to keep")
    parser.add_argument(
        "--result-dir", type=str, default=None, help="Parent directory for run artifacts (default: ./experiments)"
    )
    parser.add_argument("--run-id", type=str, default=None, help="Custom run directory name (default: auto-generated)")
    parser.add_argument(
        "--keep-all",
        dest="keep_all_results",
        action="store_true",
        default=False,
        help="Retain every result in memory (default: stream to disk and keep top-N)",
    )
    parser.add_argument("--no-keep-all", dest="keep_all_results", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-report", dest="generate_report", action="store_false", default=True, help="Skip Markdown/HTML reports"
    )
    parser.add_argument(
        "--robust",
        dest="robust",
        action="store_true",
        default=True,
        help="Run local parameter-sensitivity analysis on the selected params (default: True)",
    )
    parser.add_argument("--no-robust", dest="robust", action="store_false", help="Skip parameter-sensitivity analysis")
    parser.add_argument(
        "--robust-frac", type=float, default=0.2, help="Perturbation fraction for sensitivity analysis (default: 0.2)"
    )
    parser.add_argument("--list", action="store_true", help="List all strategies and exit")
    parser.add_argument("--version", action="version", version=f"momentum-lab {__version__}")

    args = parser.parse_args()

    if args.list:
        list_strategies()
        return

    if not args.ticker and not args.config:
        parser.error("ticker is required unless --config is provided. Use --list to see strategies.")

    strategies = None
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",")]
    elif args.all_strategies:
        strategies = list(STRATEGY_REGISTRY)

    _run_checked(
        parser,
        ticker=args.ticker or "GLD",
        strategies=strategies,
        cost_bps=args.cost,
        slippage_bps=args.slippage,
        financing_rate=args.financing_rate,
        financing_spread=args.financing_spread,
        borrow_bps=args.borrow_bps,
        cash_rate=args.cash_rate,
        short_rebate_rate=args.short_rebate_rate,
        spread_bps=args.spread_bps,
        impact_bps=args.impact_bps,
        impact_exponent=args.impact_exponent,
        impact_reference_participation=args.impact_reference_participation,
        max_participation=args.max_participation,
        initial_capital=args.initial_capital,
        min_fee=args.min_fee,
        max_leverage=args.max_leverage,
        execution_model=args.execution_model,
        execution_lag=args.execution_lag,
        annualization=args.annualization,
        risk_free_rate=args.risk_free_rate,
        validation_folds=args.validation_folds,
        min_validation_bars=args.min_val_bars,
        min_validation_trades=args.min_val_trades,
        min_validation_exposure=args.min_val_exposure,
        bootstrap=args.bootstrap,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_block_length=args.bootstrap_block_length,
        bootstrap_confidence=args.bootstrap_confidence,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_min_observations=args.bootstrap_min_observations,
        workers=args.workers,
        quick=args.quick,
        search_method=args.search_method,
        candidate_budget=args.candidate_budget,
        halving_factor=args.halving_factor,
        halving_stages=args.halving_stages,
        indicator_cache_size=args.indicator_cache_size,
        top_n=args.top,
        start=args.start,
        end=args.end,
        robust=args.robust,
        robust_frac=args.robust_frac,
        use_cache=not args.refresh,
        result_dir=args.result_dir,
        run_id=args.run_id,
        keep_all_results=args.keep_all_results,
        generate_report=args.generate_report,
        config=args.config,
        dataset=args.dataset,
        resume=args.resume,
        study_id=args.study_id,
        registry_path=args.registry_path,
        reveal_test=args.reveal_test,
        allow_test_reuse=args.allow_test_reuse,
        test_reuse_reason=args.test_reuse_reason,
    )


if __name__ == "__main__":
    raise SystemExit(main())
