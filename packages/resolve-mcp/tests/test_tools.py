"""Tests for the tools module: each tool is exercised through the FakeResolveBackend
to verify the JSON-shaped output FastMCP will see.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from resolve_mcp.fake_backend import FakeResolveBackend
from resolve_mcp.tools import (
    append_clip,
    create_bin,
    create_project,
    create_timeline,
    delete_clip,
    get_project_info,
    get_timeline_state,
    import_media,
    insert_clip,
    list_media_pool,
    list_projects,
    list_timelines,
    open_project,
    save_project,
    set_current_timeline,
)

MediaFactory = Callable[..., list[str]]


@pytest.fixture
def seeded(make_media: MediaFactory) -> FakeResolveBackend:
    be = FakeResolveBackend()
    be.create_project("reel", 24.0, 1920, 1080)
    be.import_media(make_media("a.mp4", "b.mp4"))
    be.create_timeline("main", 24.0)
    return be


def test_create_project_dump(seeded: FakeResolveBackend) -> None:
    info = create_project(seeded, "show", 23.976, False, 1280, 720)
    assert info["name"] == "show"
    assert info["resolution_width"] == 1280


def test_open_project(seeded: FakeResolveBackend) -> None:
    dump = open_project(seeded, "reel")
    assert dump["name"] == "reel"


def test_save_get_info(seeded: FakeResolveBackend) -> None:
    save_project(seeded)
    info = get_project_info(seeded)
    assert info["is_modified"] is False


def test_import_into_default_master_bin(
    seeded: FakeResolveBackend, make_media: MediaFactory
) -> None:
    out = import_media(seeded, make_media("c.mp4"))
    assert out[0]["bin"] == "Master"


def test_list_media_pool_returns_dict(seeded: FakeResolveBackend) -> None:
    state = list_media_pool(seeded)
    assert state["bins"]
    assert all("clip_ids" in b for b in state["bins"])


def test_create_bin(seeded: FakeResolveBackend) -> None:
    bin_dump = create_bin(seeded, "B-roll")
    assert bin_dump["name"] == "B-roll" and bin_dump["clip_ids"] == []


def test_create_timeline_keys(seeded: FakeResolveBackend) -> None:
    tl = create_timeline(seeded, "secondary", 30.0)
    assert tl["name"] == "secondary"
    assert len(tl["tracks"]) >= 2


def test_append_clip_delta_shape(seeded: FakeResolveBackend) -> None:
    pool = seeded.list_media_pool()
    media_id = pool.clips[0].id
    delta = append_clip(
        seeded,
        media_clip_id=media_id,
        timeline_track_index=1,
        start_seconds=0.0,
        duration_seconds=2.5,
    )
    assert delta["before"]["duration_seconds"] == 0.0
    assert delta["after"]["duration_seconds"] == 2.5
    assert delta["changed_paths"]
    # State must now reflect the clip we just added.
    state = get_timeline_state(seeded)
    assert state["duration_seconds"] == 2.5


def test_full_round_trip(seeded: FakeResolveBackend) -> None:
    pool = seeded.list_media_pool()
    media_id = pool.clips[0].id
    # Append, then insert, then delete one — verify duration state after each step.
    append_clip(seeded, media_clip_id=media_id, timeline_track_index=1, start_seconds=0.0, duration_seconds=2.0)
    assert get_timeline_state(seeded)["duration_seconds"] == 2.0
    insert_clip(seeded, media_clip_id=media_id, timeline_track_index=1, timeline_position_seconds=2.0, duration_seconds=1.5)
    assert get_timeline_state(seeded)["duration_seconds"] == 3.5
    # Delete the first item (start_seconds == 0.0). The 1.5-second item at 2.0 survives -> duration = 3.5.
    state = get_timeline_state(seeded)
    first_id = state["tracks"][0]["items"][0]["id"]
    delete_clip(seeded, timeline_item_id=first_id)
    final = get_timeline_state(seeded)
    assert final["duration_seconds"] == 3.5
    assert len(final["tracks"][0]["items"]) == 1


def test_list_projects_tool(seeded: FakeResolveBackend) -> None:
    assert list_projects(seeded) == {"projects": ["reel"]}


def test_list_and_set_current_timeline_tools(seeded: FakeResolveBackend) -> None:
    create_timeline(seeded, "second", 24.0)
    listed = list_timelines(seeded)["timelines"]
    assert {t["name"]: t["is_current"] for t in listed} == {"main": False, "second": True}
    switched = set_current_timeline(seeded, "main")
    assert switched["name"] == "main"
    assert get_timeline_state(seeded)["name"] == "main"
