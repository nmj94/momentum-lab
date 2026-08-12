"""Quick start: Find the best momentum strategy for any ticker in 3 lines."""

from momentum_lab import run_search

# Just provide a ticker - that's it!
results = run_search("GLD", quick=True)

print(f"\nBest strategy: {results['best']['strategy']}")
print(f"Best params: {results['best']['params']}")
