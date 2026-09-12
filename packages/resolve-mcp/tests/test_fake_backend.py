"""Tests for FakeResolveBackend: project / media pool / timeline mutations.

These tests exercise the in-memory model that the FastMCP server talks to. They do
not spin up the MCP layer at all — the unit boundary is the backend itself, so we
catch regressions independent of FastMCP wiring.

The fake is meant to behave like an NLE, so the suite asserts the guarantees a
caller relies on: real files only, per-project scoping, frame quantization, and no
silently overlapping clips.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from resolve_mcp.backend import AlreadyExistsError, InvalidStateError, NotFoundError
from resolve_mcp.fake_backend import FakeResolveBackend
from resolve_mcp.schemas import MediaKind

MediaFactory = Callable[..., list[str]]


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


def test_drop_frame_rejected_for_non_ntsc_rate(fake: FakeResolveBackend) -> None:
    # 24fps has no drop-frame form; the timecode rules must be enforced here, not
    # silently accepted and then blow up somewhere downstream.
    with pytest.raises(InvalidStateError):
        fake.create_project("reel", {"fps": 24.0, "drop_frame": True}, 1920, 1080)


def test_current_project_requires_open(fake: FakeResolveBackend) -> None:
    with pytest.raises(InvalidStateError):
        fake.current_project()


def test_state_is_scoped_per_project(fake: FakeResolveBackend, make_media: MediaFactory) -> None:
    fake.create_project("one", 24.0, 1920, 1080)
    fake.import_media(make_media("a.mp4", subdir="one"))
    fake.create_timeline("t1", 24.0)

    fake.create_project("two", 24.0, 1920, 1080)
    assert fake.list_media_pool().clips == []
    assert fake.list_timelines() == []
    with pytest.raises(InvalidStateError):
        fake.get_timeline_state()

    fake.open_project("one")
    assert len(fake.list_media_pool().clips) == 1
    assert [t.name for t in fake.list_timelines()] == ["t1"]


# --- imports / bins -----------------------------------------------------------


def test_import_media_creates_clips(fake: FakeResolveBackend, media: list[str]) -> None:
    fake.create_project("reel", 24.0, 1920, 1080)
    clips = fake.import_media(media)
    assert [c.name for c in clips] == ["a", "b", "c"]
    kinds = {c.name: c.kind for c in clips}
    assert kinds["a"] == MediaKind.VIDEO
    assert kinds["b"] == MediaKind.AUDIO
    assert kinds["c"] == MediaKind.IMAGE
    state = fake.list_media_pool()
    assert {b.name for b in state.bins} == {"Master"}
    assert {c.id for c in state.clips} == {clips[0].id, clips[1].id, clips[2].id}


def test_import_missing_file_raises(fake: FakeResolveBackend, tmp_path: Path) -> None:
    fake.create_project("reel", 24.0, 1920, 1080)
    with pytest.raises(NotFoundError, match="not found"):
        fake.import_media([str(tmp_path / "does_not_exist.mp4")])
    assert fake.list_media_pool().clips == []


def test_import_into_new_bin(fake: FakeResolveBackend, make_media: MediaFactory) -> None:
    fake.create_project("reel", 24.0, 1920, 1080)
    fake.create_bin("b-roll")
    clips = fake.import_media(make_media("a.mp4"), bin="b-roll")
    assert clips[0].bin == "b-roll"
    state = fake.list_media_pool()
    assert any(b.name == "b-roll" for b in state.bins)


def test_import_into_unknown_bin_fails(fake: FakeResolveBackend, make_media: MediaFactory) -> None:
    fake.create_project("reel", 24.0, 1920, 1080)
    with pytest.raises(NotFoundError):
        fake.import_media(make_media("a.mp4"), bin="not-a-bin")


def test_create_duplicate_bin_fails(fake: FakeResolveBackend) -> None:
    fake.create_project("reel", 24.0, 1920, 1080)
    fake.create_bin("b-roll")
    with pytest.raises(AlreadyExistsError):
        fake.create_bin("b-roll")


def test_import_dedupes_names(fake: FakeResolveBackend, make_media: MediaFactory) -> None:
    fake.create_project("reel", 24.0, 1920, 1080)
    a = fake.import_media(make_media("a.mp4", subdir="x"))[0]
    b = fake.import_media(make_media("a.mp4", subdir="y"))[0]
    assert a.name == "a"
    assert b.name == "a (2)"
    assert a.id != b.id


# --- timelines ----------------------------------------------------------------


def test_create_timeline_and_get_state(seeded_timeline: tuple[FakeResolveBackend, str]) -> None:
    fake, _ = seeded_timeline
    state = fake.get_timeline_state()
    assert state.name == "main"
    assert state.duration_seconds == 0.0
    assert [(t.index, t.kind.value) for t in state.tracks] == [(1, "video"), (2, "audio")]
    assert state.frame_rate.fps == 24.0


def test_list_and_switch_timelines(seeded_timeline: tuple[FakeResolveBackend, str]) -> None:
    fake, _ = seeded_timeline
    fake.create_timeline("second", 24.0)
    assert [(t.name, t.is_current) for t in fake.list_timelines()] == [
        ("main", False),
        ("second", True),
    ]
    switched = fake.set_current_timeline("main")
    assert switched.name == "main"
    assert fake.get_timeline_state().name == "main"
    with pytest.raises(NotFoundError):
        fake.set_current_timeline("nope")


def test_append_clip_extends_duration(seeded_timeline: tuple[FakeResolveBackend, str]) -> None:
    fake, media_id = seeded_timeline
    delta = fake.append_clip(
        media_clip_id=media_id,
        timeline_track_index=1,
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
    assert delta.id_remap == {}


def test_append_quantizes_to_frames(seeded_timeline: tuple[FakeResolveBackend, str]) -> None:
    fake, media_id = seeded_timeline
    # 1.01s at 24fps is 24.24 frames -> snaps to frame 24 -> exactly 1.0s.
    fake.append_clip(media_id, 1, 1.01, 2.0)
    item = fake.get_timeline_state().tracks[0].items[0]
    assert item.start_seconds == 1.0


def test_append_sorts_and_appends(seeded_timeline: tuple[FakeResolveBackend, str]) -> None:
    fake, media_id = seeded_timeline
    fake.append_clip(media_id, 1, 2.0, 2.0)
    fake.append_clip(media_id, 1, 0.0, 2.0)
    state = fake.get_timeline_state()
    items = state.tracks[0].items
    assert [i.start_seconds for i in items] == [0.0, 2.0]
    assert state.duration_seconds == 4.0


def test_append_rejects_overlap(seeded_timeline: tuple[FakeResolveBackend, str]) -> None:
    fake, media_id = seeded_timeline
    fake.append_clip(media_id, 1, 0.0, 5.0)
    with pytest.raises(InvalidStateError, match="overlap"):
        fake.append_clip(media_id, 1, 2.0, 3.0)
    assert len(fake.get_timeline_state().tracks[0].items) == 1


def test_append_rejects_zero_duration(seeded_timeline: tuple[FakeResolveBackend, str]) -> None:
    fake, media_id = seeded_timeline
    with pytest.raises(InvalidStateError):
        fake.append_clip(media_id, 1, 0.0, 0.0)


def test_insert_clip_shifts_existing(seeded_timeline: tuple[FakeResolveBackend, str]) -> None:
    fake, media_id = seeded_timeline
    fake.append_clip(media_id, 1, 0.0, 2.0)
    delta = fake.insert_clip(media_id, 1, 0.0, 1.5)
    state = fake.get_timeline_state()
    starts = [i.start_seconds for i in state.tracks[0].items]
    # The originally-first item should have moved 1.5 seconds to the right.
    assert starts == [0.0, 1.5]
    assert state.duration_seconds == 3.5
    assert delta.changed_paths


def test_insert_inside_a_clip_is_rejected(seeded_timeline: tuple[FakeResolveBackend, str]) -> None:
    fake, media_id = seeded_timeline
    fake.append_clip(media_id, 1, 0.0, 4.0)
    with pytest.raises(InvalidStateError, match="falls inside"):
        fake.insert_clip(media_id, 1, 2.0, 1.0)


def test_delete_clip_returns_path(seeded_timeline: tuple[FakeResolveBackend, str]) -> None:
    fake, media_id = seeded_timeline
    delta = fake.append_clip(media_id, 1, 0.0, 2.0)
    item_id = delta.after["tracks"][0]["items"][0]["id"]
    delta = fake.delete_clip(item_id)
    assert fake.get_timeline_state().duration_seconds == 0.0
    assert delta.changed_paths


def test_delete_unknown_item_raises(seeded_timeline: tuple[FakeResolveBackend, str]) -> None:
    fake, _ = seeded_timeline
    with pytest.raises(NotFoundError):
        fake.delete_clip("item_nope")


def test_move_clip(seeded_timeline: tuple[FakeResolveBackend, str]) -> None:
    fake, media_id = seeded_timeline
    fake.append_clip(media_id, 1, 0.0, 2.0)
    fake.append_clip(media_id, 1, 2.0, 2.0)
    original_first_id = fake.get_timeline_state().tracks[0].items[0].id
    delta = fake.move_clip(original_first_id, 10.0)
    items = fake.get_timeline_state().tracks[0].items
    moved = next(i for i in items if i.id == original_first_id)
    assert moved.start_seconds == 10.0
    # Items must be re-sorted by start_seconds after a move.
    assert [i.start_seconds for i in items] == sorted(i.start_seconds for i in items)
    assert delta.changed_paths
    # The fake keeps ids stable, unlike live Resolve.
    assert delta.id_remap == {}


def test_move_onto_another_clip_is_rejected(seeded_timeline: tuple[FakeResolveBackend, str]) -> None:
    fake, media_id = seeded_timeline
    fake.append_clip(media_id, 1, 0.0, 2.0)
    fake.append_clip(media_id, 1, 2.0, 2.0)
    first_id = fake.get_timeline_state().tracks[0].items[0].id
    with pytest.raises(InvalidStateError, match="overlap"):
        fake.move_clip(first_id, 3.0)


def test_append_unknown_media_raises(seeded_timeline: tuple[FakeResolveBackend, str]) -> None:
    fake, _ = seeded_timeline
    with pytest.raises(NotFoundError):
        fake.append_clip("clip_nope", 1, 0.0, 1.0)


def test_append_video_to_audio_track_rejected(
    seeded_timeline: tuple[FakeResolveBackend, str],
) -> None:
    fake, media_id = seeded_timeline  # media_id is a video
    with pytest.raises(InvalidStateError):
        fake.append_clip(media_id, 2, 0.0, 1.0)  # audio track


def test_unknown_track_index_rejected(seeded_timeline: tuple[FakeResolveBackend, str]) -> None:
    fake, media_id = seeded_timeline
    with pytest.raises(NotFoundError, match="track index"):
        fake.append_clip(media_id, 9, 0.0, 1.0)


def test_mutation_marks_project_modified_and_save_clears_it(
    fake: FakeResolveBackend, make_media: MediaFactory
) -> None:
    info = fake.create_project("reel", 24.0, 1920, 1080)
    assert info.is_modified is False
    fake.import_media(make_media("a.mp4"))
    assert fake.current_project().is_modified is True
    assert fake.save_project().is_modified is False


def test_unsupported_tools_is_empty(fake: FakeResolveBackend) -> None:
    assert fake.unsupported_tools() == frozenset()
