"""conftest fixtures for director tests.

A common ``tmp_run`` fixture gives every test a fresh SQLite + JSONL run store
under a tempdir.
"""

from __future__ import annotations

import pathlib

import pytest
from director.mcp_client import StubResolveClient
from director.store import EventLog, RunStore
from resolve_mcp.fake_backend import FakeResolveBackend


@pytest.fixture
def tmp_run(tmp_path: pathlib.Path) -> tuple[RunStore, EventLog, pathlib.Path]:
    sqlite = tmp_path / "runs.sqlite"
    jsonl = tmp_path / "events.jsonl"
    return RunStore(sqlite), EventLog(jsonl), tmp_path


@pytest.fixture
def fake_client() -> StubResolveClient:
    return StubResolveClient(FakeResolveBackend(allow_destructive=True))
