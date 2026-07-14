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
Browse the
[good first issues](https://github.com/bamdadd/introspection-scaling/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
— each has file:line pointers, acceptance criteria, and the test to add. If the
tracker is empty, open an issue describing what you'd like to add and we'll
scope it together.

## Reproducibility rules
- Pin versions (the `uv.lock` is committed).
- Any results table states seeds, hardware, and wall-clock.
