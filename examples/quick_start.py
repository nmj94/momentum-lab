"""Quick start: Find the best momentum strategy for any ticker in 3 lines."""

from momentum_lab import run_search

# Just provide a ticker - that's it!
results = run_search("GLD", quick=True)

best = results.get("best")
if best is None:
    print("\nNo strategy produced valid results.")
else:
    print(f"\nBest strategy: {best['strategy']}")
    print(f"Best params: {best['params']}")

# Robustness check result (overfitting detection)
rob = results.get("robustness") or {}
if rob.get("grade"):
    print(f"Robustness grade: {rob['grade']} ({rob.get('verdict', 'n/a')})")
