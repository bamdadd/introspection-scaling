"""Smoke test: the package imports and exposes its version."""

import introspection_scaling


def test_import() -> None:
    assert introspection_scaling.__version__
