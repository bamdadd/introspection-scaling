# Contributing to introspection-scaling

Thanks for your interest. This is a research repo; correctness and
reproducibility come before features.

## Setup
```bash
uv sync
uv run pre-commit install
```

## Before you open a PR
- `uv run ruff check . && uv run ruff format --check .`
- `uv run mypy src`
- `uv run pytest -q`
- New behaviour needs a test. Stochastic results need a fixed seed and a
  reported mean ± std over 3+ seeds.

## Good first issues
See the `good first issue` label. If the tracker is empty, open an issue
describing what you'd like to add and we'll scope it together.

## Reproducibility rules
- Pin versions (the `uv.lock` is committed).
- Any results table states seeds, hardware, and wall-clock.
