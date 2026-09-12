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


@pytest.fixture
def clips(tmp_path: pathlib.Path) -> list[str]:
    """Three real (empty) files: import_media checks existence, like Resolve does.

    They are not decodable, so ffprobe reports no duration — the "length unknown"
    path the planner and editor must both tolerate.
    """
    directory = tmp_path / "clips"
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        target = directory / name
        target.write_bytes(b"")
        paths.append(str(target))
    return paths


@pytest.fixture
def music(tmp_path: pathlib.Path) -> str:
    """A real (empty) audio file for tests that pass a music track."""
    target = tmp_path / "track.wav"
    target.write_bytes(b"")
    return str(target)
