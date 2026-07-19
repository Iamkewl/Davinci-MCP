"""Tests for FakeResolveBackend: project / media pool / timeline mutations.

These tests exercise the in-memory model that the FastMCP server talks to. They do
not spin up the MCP layer at all — the unit boundary is the backend itself, so we
catch regressions independent of FastMCP wiring.
"""

from __future__ import annotations

import pytest
from resolve_mcp.backend import AlreadyExistsError, InvalidStateError, NotFoundError
from resolve_mcp.fake_backend import FakeResolveBackend
from resolve_mcp.schemas import MediaKind


def test_create_project_and_open(fake: FakeResolveBackend) -> None:
    info = fake.create_project("reel", 24.0, 1920, 1080)
    assert info.name == "reel"
    assert fake.list_projects() == ["reel"]
    assert fake.current_project().name == "reel"

    fake.open_project("reel")
    assert fake.current_project().name == "reel"


def test_create_duplicate_project_raises(fake: FakeResolveBackend) -> None:
    fake.create_project("reel", 24.0, 1920, 1080)
    with pytest.raises(AlreadyExistsError):
        fake.create_project("reel", 24.0, 1920, 1080)


def test_open_unknown_project(fake: FakeResolveBackend) -> None:
    with pytest.raises(NotFoundError):
        fake.open_project("absent")


def test_create_project_requires_frame_rate_dict(fake: FakeResolveBackend) -> None:
    info = fake.create_project("reel", {"fps": 29.97, "drop_frame": True}, 1280, 720)
    assert info.frame_rate.fps == 29.97
    assert info.frame_rate.drop_frame is True


def test_current_project_requires_open(fake: FakeResolveBackend) -> None:
    with pytest.raises(InvalidStateError):
        fake.current_project()


# --- imports / bins -----------------------------------------------------------


def test_import_media_creates_clips(fake: FakeResolveBackend) -> None:
    fake.create_project("reel", 24.0, 1920, 1080)
    clips = fake.import_media(["/tmp/a.mp4", "/tmp/b.wav", "/tmp/c.png"])
    assert [c.name for c in clips] == ["a", "b", "c"]
    kinds = {c.name: c.kind for c in clips}
    assert kinds["a"] == MediaKind.VIDEO
    assert kinds["b"] == MediaKind.AUDIO
    assert kinds["c"] == MediaKind.IMAGE
    state = fake.list_media_pool()
    assert {b.name for b in state.bins} == {"Master"}
    assert {c.id for c in state.clips} == {clips[0].id, clips[1].id, clips[2].id}


def test_import_into_new_bin(fake: FakeResolveBackend) -> None:
    fake.create_project("reel", 24.0, 1920, 1080)
    fake.create_bin("b-roll")
    clips = fake.import_media(["/tmp/a.mp4"], bin="b-roll")
    assert clips[0].bin == "b-roll"
    state = fake.list_media_pool()
    assert any(b.name == "b-roll" for b in state.bins)


def test_import_into_unknown_bin_fails(fake: FakeResolveBackend) -> None:
    fake.create_project("reel", 24.0, 1920, 1080)
    with pytest.raises(NotFoundError):
        fake.import_media(["/tmp/a.mp4"], bin="not-a-bin")


def test_create_duplicate_bin_fails(fake: FakeResolveBackend) -> None:
    fake.create_project("reel", 24.0, 1920, 1080)
    fake.create_bin("b-roll")
    with pytest.raises(AlreadyExistsError):
        fake.create_bin("b-roll")


def test_import_dedupes_names(fake: FakeResolveBackend) -> None:
    fake.create_project("reel", 24.0, 1920, 1080)
    a = fake.import_media(["/x/a.mp4"])[0]
    b = fake.import_media(["/y/a.mp4"])[0]
    assert a.name == "a"
    assert b.name != "a"
    assert a.id != b.id


# --- timelines ----------------------------------------------------------------


def _seed(fake: FakeResolveBackend) -> tuple[str, str]:
    fake.create_project("reel", 24.0, 1920, 1080)
    clips = fake.import_media(["/tmp/a.mp4"])
    fake.create_timeline("main", {"fps": 24.0})
    return clips[0].id, "main"


def test_create_timeline_and_get_state(fake: FakeResolveBackend) -> None:
    _seed(fake)
    state = fake.get_timeline_state()
    assert state.name == "main"
    assert state.duration_seconds == 0.0
    assert len(state.tracks) == 2
    assert state.frame_rate.fps == 24.0


def test_append_clip_extends_duration(fake: FakeResolveBackend) -> None:
    media_id, _ = _seed(fake)
    delta = fake.append_clip(
        media_clip_id=media_id,
        timeline_track_index=0,
        start_seconds=0.0,
        duration_seconds=4.0,
    )
    state = fake.get_timeline_state()
    assert state.duration_seconds == 4.0
    assert len(state.tracks[0].items) == 1
    item = state.tracks[0].items[0]
    assert item.start_seconds == 0.0
    assert item.duration_seconds == 4.0
    assert delta.after["tracks"][0]["items"][0]["id"] == item.id


def test_append_sorts_and_appends(fake: FakeResolveBackend) -> None:
    media_id, _ = _seed(fake)
    fake.append_clip(media_id, 0, 0.0, 2.0)
    fake.append_clip(media_id, 0, 2.0, 2.0)
    state = fake.get_timeline_state()
    items = state.tracks[0].items
    assert [i.start_seconds for i in items] == [0.0, 2.0]
    assert state.duration_seconds == 4.0


def test_insert_clip_shifts_existing(fake: FakeResolveBackend) -> None:
    media_id, _ = _seed(fake)
    fake.append_clip(media_id, 0, 0.0, 2.0)
    delta = fake.insert_clip(media_id, 0, 0.0, 1.5)
    state = fake.get_timeline_state()
    starts = [i.start_seconds for i in state.tracks[0].items]
    # The originally-first item should have moved 1.5 seconds to the right.
    assert starts == [0.0, 1.5]
    assert state.duration_seconds == 3.5
    assert delta.changed_paths  # paths populated


def test_delete_clip_returns_path(fake: FakeResolveBackend) -> None:
    media_id, _ = _seed(fake)
    delta = fake.append_clip(media_id, 0, 0.0, 2.0)
    item_id = delta.after["tracks"][0]["items"][0]["id"]
    delta = fake.delete_clip(item_id)
    assert fake.get_timeline_state().duration_seconds == 0.0
    assert delta.changed_paths


def test_delete_unknown_item_raises(fake: FakeResolveBackend) -> None:
    _seed(fake)
    with pytest.raises(NotFoundError):
        fake.delete_clip("item_nope")


def test_move_clip(fake: FakeResolveBackend) -> None:
    media_id, _ = _seed(fake)
    fake.append_clip(media_id, 0, 0.0, 2.0)
    fake.append_clip(media_id, 0, 2.0, 2.0)
    original_first_id = fake.get_timeline_state().tracks[0].items[0].id
    delta = fake.move_clip(original_first_id, 10.0)
    items = fake.get_timeline_state().tracks[0].items
    moved = next(i for i in items if i.id == original_first_id)
    assert moved.start_seconds == 10.0
    # Items must be re-sorted by start_seconds after a move.
    assert [i.start_seconds for i in items] == sorted(i.start_seconds for i in items)
    assert delta.changed_paths


def test_append_unknown_media_raises(fake: FakeResolveBackend) -> None:
    _seed(fake)
    with pytest.raises(NotFoundError):
        fake.append_clip("clip_nope", 0, 0.0, 1.0)


def test_append_video_to_audio_track_rejected(fake: FakeResolveBackend) -> None:
    media_id, _ = _seed(fake)  # media_id is a video
    with pytest.raises(InvalidStateError):
        fake.append_clip(media_id, 1, 0.0, 1.0)  # audio track


def test_save_clears_modified_flag(fake: FakeResolveBackend) -> None:
    info = fake.create_project("reel", 24.0, 1920, 1080)
    assert info.is_modified is False
    mutated = info.model_copy(update={"is_modified": True})
    fake._projects["reel"] = mutated  # simulate a modification
    saved = fake.save_project()
    assert saved.is_modified is False
