"""Full search: Test all strategies with all parameters."""

from momentum_lab import STRATEGY_REGISTRY, run_search


def main():
    # Full exhaustive search, including experimental ML (may take days).
    results = run_search(
        ticker="SPY",
        strategies=list(STRATEGY_REGISTRY),
        cost_bps=1.0,  # 1 bps transaction cost
        workers=8,  # 8 parallel workers
        quick=False,  # Full parameter grid
        top_n=50,  # Keep top 50
        start="2004-01-01",
        keep_all_results=False,  # Stream the fixed-schema checkpoint to disk
    )

    # Access results
    print(f"\n{'=' * 60}")
    print("  Search complete!")
    print(f"  Total experiments: {results['n_results']}")
    best = results.get("best")
    if best is not None:
        print(f"  Best strategy: {best['strategy']}")
    print(f"{'=' * 60}")

    # Or test a specific strategy
    from momentum_lab import backtest, evaluate, get_strategy, prepare_data

    data, df = prepare_data("SPY")
    strategy = get_strategy("regime_aware")
    positions = strategy.run(
        data,
        adx_trend_threshold=15,
        adx_smooth=0,
        regime_confirm=1,
        vol_fast=5,
        mom_lookback=63,
        mom_threshold=0.0,
        vol_target_normal=0.12,
        vol_target_crisis=0.05,
        choppy_bull_mode="full_vol",
        fast_exit_days=10,
        fast_exit_threshold=-0.05,
        bearish_mode="cash",
        position_size=2.0,
        signal_smooth=5,
    )

    result = backtest(positions, df["close"], cost_bps=1.0)
    metrics = evaluate(result["returns"])
    print("\nRegime Aware Strategy:")
    print(f"  Sharpe:  {metrics['sharpe']}")
    print(f"  CAGR:    {metrics['cagr']:.2%}")
    print(f"  MaxDD:   {metrics['max_drawdown']:.2%}")
    print(f"  Sortino: {metrics['sortino']}")


# Required under spawn-based multiprocessing (macOS/Windows): child
# processes re-import this module and must not re-launch the search.
if __name__ == "__main__":
    main()
