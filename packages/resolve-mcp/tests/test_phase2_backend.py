"""Tests for Phase 2 fake-backend operations: items, effects, render, and the
destructive gate.

The destructive gate has two layers; we test both:

* The **server-side flag** --allow-destructive controls registration of the
  destructive tools. (Covered via the server wiring test in test_server.py.)
* The **per-call confirm=true** argument is enforced regardless of the flag.

The fake backend exposes the same constructor argument `allow_destructive=True`
that mirrors the server flag, so we exercise both behaviours from one suite.
"""

from __future__ import annotations

import pytest
from resolve_mcp.backend import (
    DestructiveDisabledError,
    InvalidStateError,
    NotFoundError,
)
from resolve_mcp.fake_backend import FakeResolveBackend
from resolve_mcp.schemas import (
    MarkerColor,
    RenderJobStatus,
    TrackKind,
    TransitionAlignment,
    TransitionStyle,
)

# --- per-item: transforms / crop / composite / opacity / fade / speed / marker


@pytest.fixture
def with_item() -> tuple[FakeResolveBackend, str]:
    be = FakeResolveBackend()
    be.create_project("reel", 24.0, 1920, 1080)
    be.import_media(["/tmp/a.mp4"])
    be.create_timeline("main", 24.0)
    media = be.list_media_pool().clips[0].id
    be.append_clip(media, 0, 0.0, 4.0)
    item_id = be.get_timeline_state().tracks[0].items[0].id
    return be, item_id


def test_set_transform_returns_delta(with_item: tuple[FakeResolveBackend, str]) -> None:
    be, item_id = with_item
    delta = be.set_transform(item_id, pan_x=2.0, pan_y=-1.0,
                              zoom_x=1.5, zoom_y=1.5, rotation=10.0)
    assert delta.changed_paths
    state = be.get_timeline_state()
    saved = state.tracks[0].items[0]
    assert saved.transform.pan_x == 2.0
    assert saved.transform.rotation == 10.0


def test_set_crop(with_item: tuple[FakeResolveBackend, str]) -> None:
    be, item_id = with_item
    be.set_crop(item_id, left=0.1, right=0.2, top=0.05, bottom=0.06)
    saved = be.get_timeline_state().tracks[0].items[0]
    assert saved.crop.left == 0.1


def test_set_composite_mode_str_input(with_item: tuple[FakeResolveBackend, str]) -> None:
    be, item_id = with_item
    be.set_composite_mode(item_id, "multiply")
    saved = be.get_timeline_state().tracks[0].items[0]
    assert saved.composite_mode.value == "multiply"


def test_set_opacity_validates_bounds(with_item: tuple[FakeResolveBackend, str]) -> None:
    be, item_id = with_item
    be.set_opacity(item_id, 0.4)
    state = be.get_timeline_state()
    assert state.tracks[0].items[0].opacity == 0.4
    with pytest.raises(ValueError):
        be.set_opacity(item_id, 1.5)


def test_add_fade(with_item: tuple[FakeResolveBackend, str]) -> None:
    be, item_id = with_item
    be.add_fade(item_id, fade_in_seconds=0.5, fade_out_seconds=1.0)
    saved = be.get_timeline_state().tracks[0].items[0]
    assert saved.fade_in_seconds == 0.5
    assert saved.fade_out_seconds == 1.0


def test_add_fade_validates_total(with_item: tuple[FakeResolveBackend, str]) -> None:
    be, item_id = with_item
    with pytest.raises(ValueError):
        be.add_fade(item_id, fade_in_seconds=3.0, fade_out_seconds=3.0)  # exceeds item duration


def test_set_speed(with_item: tuple[FakeResolveBackend, str]) -> None:
    be, item_id = with_item
    be.set_speed(item_id, 2.0)
    saved = be.get_timeline_state().tracks[0].items[0]
    assert saved.speed == 2.0
    with pytest.raises(ValueError):
        be.set_speed(item_id, -1.0)


def test_add_marker_appends(with_item: tuple[FakeResolveBackend, str]) -> None:
    be, item_id = with_item
    delta = be.add_marker(item_id, position_seconds=1.0, label="hi", color=MarkerColor.RED, note="n")
    assert delta.changed_paths
    saved = be.get_timeline_state().tracks[0].items[0]
    assert len(saved.markers) == 1
    assert saved.markers[0].color is MarkerColor.RED


def test_add_marker_position_bound(with_item: tuple[FakeResolveBackend, str]) -> None:
    be, item_id = with_item
    with pytest.raises(ValueError):
        be.add_marker(item_id, position_seconds=999.0, label="x", color="red")


def test_set_transform_unknown_item_raises() -> None:
    be = FakeResolveBackend()
    be.create_project("reel", 24.0, 1920, 1080)
    be.create_timeline("main", 24.0)
    with pytest.raises(NotFoundError):
        be.set_transform("nope", pan_x=0, pan_y=0, zoom_x=1.0, zoom_y=1.0, rotation=0.0)


@pytest.fixture
def be_fixture() -> FakeResolveBackend:
    return FakeResolveBackend()


# --- transitions ------------------------------------------------------------


def test_add_transition_attaches_to_item(with_item: tuple[FakeResolveBackend, str]) -> None:
    be, item_id = with_item
    be.add_transition(
        timeline_item_id=item_id,
        track_index=0,
        style=TransitionStyle.CROSS_DISSOLVE,
        duration_seconds=1.0,
        alignment=TransitionAlignment.MID,
    )
    state = be.get_timeline_state()
    assert len(state.transitions) == 1
    t = state.transitions[0]
    assert t.style is TransitionStyle.CROSS_DISSOLVE
    assert t.alignment is TransitionAlignment.MID
    assert t.timeline_item_id == item_id


def test_add_transition_wrong_track(with_item: tuple[FakeResolveBackend, str]) -> None:
    be, item_id = with_item
    with pytest.raises(NotFoundError):
        be.add_transition(item_id, track_index=99, style="cross_dissolve",
                           duration_seconds=1.0, alignment="mid")


# --- render ------------------------------------------------------------------


def test_render_job_lifecycle() -> None:
    be = FakeResolveBackend()
    be.create_project("reel", 24.0, 1920, 1080)
    be.create_timeline("main", 24.0)
    job = be.add_render_job("main", "mp4", "/tmp/out.mp4")
    assert job.status is RenderJobStatus.QUEUED
    started = be.start_render(job.id)
    assert started.status is RenderJobStatus.RUNNING
    fetched = be.get_render_status(job.id)
    assert fetched.status is RenderJobStatus.RUNNING


def test_start_render_invalid_state() -> None:
    be = FakeResolveBackend()
    be.create_project("reel", 24.0, 1920, 1080)
    be.create_timeline("main", 24.0)
    job = be.add_render_job("main", "mp4", "/tmp/out.mp4")
    be.start_render(job.id)
    with pytest.raises(InvalidStateError):
        be.start_render(job.id)


def test_get_render_status_unknown() -> None:
    be = FakeResolveBackend()
    with pytest.raises(NotFoundError):
        be.get_render_status("job_nope")


def test_add_render_job_unknown_timeline() -> None:
    be = FakeResolveBackend()
    be.create_project("reel", 24.0, 1920, 1080)
    with pytest.raises(NotFoundError):
        be.add_render_job("absent", "mp4", "/x.mp4")


# --- destructive gate --------------------------------------------------------


def test_destructive_blocked_when_flag_off() -> None:
    be = FakeResolveBackend(allow_destructive=False)
    be.create_project("reel", 24.0, 1920, 1080)
    be.import_media(["/tmp/a.mp4"])
    be.create_timeline("main", 24.0)
    with pytest.raises(DestructiveDisabledError):
        be.quit_app(confirm=True)
    with pytest.raises(DestructiveDisabledError):
        be.restart_app(confirm=True)
    with pytest.raises(DestructiveDisabledError):
        be.delete_timeline("main", confirm=True)
    with pytest.raises(DestructiveDisabledError):
        be.delete_media("x", confirm=True)


def test_destructive_blocked_when_flag_on_but_no_confirm() -> None:
    be = FakeResolveBackend(allow_destructive=True)
    be.create_project("reel", 24.0, 1920, 1080)
    be.import_media(["/tmp/a.mp4"])
    be.create_timeline("main", 24.0)
    # confirm=False is still refused even when the flag is on.
    with pytest.raises(DestructiveDisabledError):
        be.quit_app(confirm=False)
    with pytest.raises(DestructiveDisabledError):
        be.delete_timeline("main", confirm=False)


def test_destructive_allowed_with_flag_and_confirm() -> None:
    be = FakeResolveBackend(allow_destructive=True)
    be.create_project("reel", 24.0, 1920, 1080)
    be.import_media(["/tmp/a.mp4"])
    be.create_timeline("main", 24.0)
    res = be.quit_app(confirm=True)
    assert res["quit"] is True
    # Project is now un-set; saving should raise.
    from resolve_mcp.backend import InvalidStateError
    with pytest.raises(InvalidStateError):
        be.current_project()


def test_delete_timeline_with_flag_and_confirm() -> None:
    be = FakeResolveBackend(allow_destructive=True)
    be.create_project("reel", 24.0, 1920, 1080)
    be.create_timeline("main", 24.0)
    delta = be.delete_timeline("main", confirm=True)
    assert "timelines.main" in delta.changed_paths[0]
    assert "main" not in be._timelines


def test_delete_media_with_flag_and_confirm() -> None:
    be = FakeResolveBackend(allow_destructive=True)
    be.create_project("reel", 24.0, 1920, 1080)
    be.import_media(["/tmp/x.mp4"])
    media_id = be.list_media_pool().clips[0].id
    res = be.delete_media(media_id, confirm=True)
    assert res["deleted"]["id"] == media_id
    # Bin no longer references this clip.
    pool = be.list_media_pool()
    assert media_id not in pool.bins[0].clip_ids


def test_track_kind_detect() -> None:
    """Smoke test: ensure TrackKind enum is integrated with backend."""
    be = FakeResolveBackend()
    be.create_project("reel", 24.0, 1920, 1080)
    be.create_timeline("main", 24.0)
    kinds = [t.kind for t in be.get_timeline_state().tracks]
    assert TrackKind.VIDEO in kinds
    assert TrackKind.AUDIO in kinds
