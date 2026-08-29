# Contributing

Contributions are welcome when they improve correctness, research integrity,
reproducibility, execution realism, or performance without weakening defaults.

## Local setup

```bash
git clone https://github.com/nmj94/momentum-lab.git
cd momentum-lab
uv sync --all-extras
uv run ruff check .
uv run pytest -m "not network" -q
```

## Pull requests

- Keep changes focused and include regression tests.
- Explain market-data, execution-timing, cost, and look-ahead assumptions.
- Strategy additions need a written hypothesis and a simple baseline, not only a
  larger parameter grid.
- Research-method changes must state whether old checkpoints remain comparable;
  bump `ENGINE_SCHEMA_VERSION` when they do not.
- Do not use the final test window to tune a change.
- Update README, CHANGELOG, and ROADMAP when behavior or scope changes.

## Test categories

Offline tests must be deterministic and are required in CI. Tests requiring
Yahoo or another external provider must use the `network` marker and should not
silently pass when a request fails.

## Style

Ruff is the source of truth for formatting and lint. Prefer small typed public
interfaces, fixed schemas, explicit validation, and actionable error messages.
