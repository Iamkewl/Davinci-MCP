"""Conftest: shared pytest fixtures for the resolve-mcp test suite.

Keeps individual test files focused on behaviour rather than glue code.
"""

from __future__ import annotations

import pytest
from resolve_mcp.fake_backend import FakeResolveBackend


@pytest.fixture
def fake() -> FakeResolveBackend:
    return FakeResolveBackend()
