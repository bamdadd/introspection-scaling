"""Shared pytest config."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: loads model weights / runs a real extraction pass")
