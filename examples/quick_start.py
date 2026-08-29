"""Quick start: compare momentum strategies for a ticker."""

from momentum_lab import run_search

# Just provide a ticker - that's it!
results = run_search("GLD", quick=True)

best = results.get("best")
if best is None:
    print("\nNo strategy produced valid results.")
else:
    print(f"\nBest strategy: {best['strategy']}")
    print(f"Best params: {best['params']}")

# Local parameter sensitivity; this is not an overfitting test.
rob = results.get("parameter_sensitivity") or {}
if rob.get("grade"):
    print(f"Sensitivity grade: {rob['grade']} ({rob.get('verdict', 'n/a')})")
