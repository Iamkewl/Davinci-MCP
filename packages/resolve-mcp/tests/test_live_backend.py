"""End-to-end record/replay test: builds a test harness for DaVinciResolveScript,
imports the guarded :class:`DaVinciResolveBackend`, exercises Phase-1+2 LIVE
methods, and asserts the call log matches expectations:

* Phase 1 surface: project open/save, media pool, timeline create/append.
* Phase 2 surface: set_transform, add_fade, add_marker, add_render_job, etc.
* Destructive ops require explicit ``confirm=True``.

This catches argument shape and method-name drift between the Phase-2 LIVE
implementations and the DaVinci scripting API, without a real Resolve install.
"""

from __future__ import annotations

from typing import Any

import pytest
from resolve_mcp.davinci_backend import DaVinciResolveBackend

# Imports the harness helpers
from .fake_resolve import (
    CallLog,
    install_fake_resolve,
)


@pytest.fixture
def live_backend(monkeypatch: pytest.MonkeyPatch) -> tuple[DaVinciResolveBackend, Any, CallLog]:
    fake, log = install_fake_resolve(monkeypatch)
    backend = DaVinciResolveBackend()
    assert backend.is_connected
    return backend, fake, log


def test_live_create_project_calls_resolve_save(live_backend: tuple) -> None:
    backend, _fake, log = live_backend
    info = backend.create_project("reel", 24.0, 1920, 1080)
    assert info.name == "reel"
    methods = log.methods_called()
    assert "ProjectManager.CreateProject" in methods


def test_live_save_project_calls_resolve_save(live_backend: tuple) -> None:
    backend, _fake, log = live_backend
    backend.create_project("reel", 24.0, 1920, 1080)
    log.reset()
    backend.save_project()
    assert "Project.SaveProject" in log.methods_called()


def test_live_open_project_calls_load(live_backend: tuple) -> None:
    backend, _fake, log = live_backend
    info = backend.open_project("reel")
    assert info.name == "reel"
    assert "ProjectManager.LoadProject" in log.methods_called()


def test_live_import_media_calls_resolve(live_backend: tuple) -> None:
    backend, _fake, log = live_backend
    imports = backend.import_media(["/foo/a.mp4", "/foo/b.wav"])
    assert len(imports) == 2
    assert "MediaPool.ImportMedia" in log.methods_called()


def test_live_create_bin_calls_addsubfolder(live_backend: tuple) -> None:
    backend, _fake, log = live_backend
    backend.create_bin("B-roll")
    assert "MediaPool.AddSubFolder" in log.methods_called()


def test_live_create_timeline_calls_resolve(live_backend: tuple) -> None:
    backend, _fake, log = live_backend
    tl = backend.create_timeline("main", 24.0)
    assert tl.name == "main"
    assert "MediaPool.CreateEmptyTimeline" in log.methods_called()
    assert "Timeline.SetSetting" in log.methods_called()


def test_live_get_timeline_state_calls_resolve(live_backend: tuple) -> None:
    backend, _fake, log = live_backend
    state = backend.get_timeline_state()
    assert state.name == "Timeline 1"
    assert "Project.GetCurrentTimeline" in log.methods_called()


def test_live_append_clip_calls_resolve(live_backend: tuple) -> None:
    backend, _fake, log = live_backend
    media = backend.import_media(["/foo/x.mp4"])[0]
    # `id` is `cv_<hex>`; backend resolves by enumerating the pool.
    delta = backend.append_clip(
        media_clip_id=media.id,
        timeline_track_index=1,
        start_seconds=0.0,
        duration_seconds=2.0,
    )
    assert delta.changed_paths
    assert "Timeline.AppendItemsInTimeline" in log.methods_called()


def test_live_item_set_transform_calls_resolve(live_backend: tuple) -> None:
    backend, _fake, log = live_backend
    media = backend.import_media(["/foo/x.mp4"])[0]
    backend.append_clip(media.id, 1, 0.0, 2.0)
    log.reset()
    state = backend.get_timeline_state()
    item_id = state.tracks[0].items[0].id
    backend.set_transform(
        item_id, pan_x=0.1, pan_y=-0.2, zoom_x=1.0, zoom_y=1.0, rotation=0.0
    )
    assert "TimelineItem.SetProperty" in log.methods_called()


def test_live_item_add_fade_calls_resolve(live_backend: tuple) -> None:
    backend, _fake, log = live_backend
    media = backend.import_media(["/foo/x.mp4"])[0]
    backend.append_clip(media.id, 1, 0.0, 2.0)
    state = backend.get_timeline_state()
    item_id = state.tracks[0].items[0].id
    log.reset()
    backend.add_fade(item_id, fade_in_seconds=0.2, fade_out_seconds=0.3)
    assert any(
        e["method"] == "TimelineItem.SetProperty" and e["args"][0] == "Ease"
        for e in log.entries
    )


def test_live_item_set_opacity_calls_resolve(live_backend: tuple) -> None:
    backend, _fake, log = live_backend
    media = backend.import_media(["/foo/x.mp4"])[0]
    backend.append_clip(media.id, 1, 0.0, 2.0)
    state = backend.get_timeline_state()
    item_id = state.tracks[0].items[0].id
    log.reset()
    backend.set_opacity(item_id, 0.5)
    assert any(
        e["method"] == "TimelineItem.SetProperty" and e["args"][0] == "Opacity"
        for e in log.entries
    )


def test_live_item_add_marker_calls_resolve(live_backend: tuple) -> None:
    backend, _fake, log = live_backend
    media = backend.import_media(["/foo/x.mp4"])[0]
    backend.append_clip(media.id, 1, 0.0, 2.0)
    state = backend.get_timeline_state()
    item_id = state.tracks[0].items[0].id
    log.reset()
    backend.add_marker(item_id, position_seconds=0.5, label="cut here", color="blue")
    assert any(
        e["method"] == "TimelineItem.AddMarker" for e in log.entries
    )


def test_live_add_transition_calls_resolve(live_backend: tuple) -> None:
    backend, _fake, log = live_backend
    media = backend.import_media(["/foo/x.mp4"])[0]
    backend.append_clip(media.id, 1, 0.0, 2.0)
    state = backend.get_timeline_state()
    item_id = state.tracks[0].items[0].id
    log.reset()
    backend.add_transition(item_id, 1, "cross_dissolve", 0.5, "mid")
    assert any(
        e["method"] == "Timeline.AddTransition" for e in log.entries
    )


def test_live_render_job_lifecycle_calls_resolve(live_backend: tuple) -> None:
    backend, _fake, log = live_backend
    backend.create_timeline("Timeline 1", 24.0)
    job = backend.add_render_job("Timeline 1", "mp4", "/tmp/out.mp4")
    assert job.id
    log.reset()
    backend.start_render(job.id)
    assert any(
        e["method"] == "Project.StartRendering" for e in log.entries
    )
    log.reset()
    status = backend.get_render_status(job.id)
    assert status.id == job.id


def test_live_quit_requires_confirm(live_backend: tuple) -> None:
    backend, fake, log = live_backend
    with pytest.raises(Exception):  # noqa: B017 — broad is intentional; we just need refused-without-confirm
        backend.quit_app(confirm=False)
    log.reset()
    backend.quit_app(confirm=True)
    assert fake._quit_called is True
    assert "Resolve.Quit" in log.methods_called()
