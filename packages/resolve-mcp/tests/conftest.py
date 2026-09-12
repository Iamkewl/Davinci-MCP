"""Conftest: shared pytest fixtures for the resolve-mcp test suite.

Keeps individual test files focused on behaviour rather than glue code.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from resolve_mcp.fake_backend import FakeResolveBackend


@pytest.fixture
def fake() -> FakeResolveBackend:
    return FakeResolveBackend()


MediaFactory = Callable[..., list[str]]


@pytest.fixture
def make_media(tmp_path: Path) -> MediaFactory:
    """Factory creating real (empty) files so import_media's existence check passes.

    They are not decodable, so ffprobe (when installed) reports no duration —
    exactly the "duration unknown" path the backend must tolerate.
    """

    def _make(*names: str, subdir: str = "") -> list[str]:
        base = tmp_path / subdir if subdir else tmp_path
        paths: list[str] = []
        for name in names:
            target = base / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"")
            paths.append(str(target))
        return paths

    return _make


@pytest.fixture
def media(make_media: MediaFactory) -> list[str]:
    """One video, one audio and one image file on disk."""
    return make_media("a.mp4", "b.wav", "c.png")


@pytest.fixture
def seeded_timeline(
    fake: FakeResolveBackend, make_media: MediaFactory
) -> tuple[FakeResolveBackend, str]:
    """A project with one imported video clip and a current 24fps timeline."""
    fake.create_project("reel", 24.0, 1920, 1080)
    clips = fake.import_media(make_media("a.mp4"))
    fake.create_timeline("main", 24.0)
    return fake, clips[0].id
