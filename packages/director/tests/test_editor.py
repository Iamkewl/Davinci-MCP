"""Offline tests for the Editor agent.

Covers the plan.md Phase 1 fixes: media import before appends, honest
(frozen-crash-free) tool-call records, deep symbolic-id substitution, and
auto-injection of ``<item:N>`` symbols for appends. Everything runs against
FakeResolveBackend via StubResolveClient — no Resolve, no network.
"""

from __future__ import annotations

import pathlib

import pytest
from director.agents import Editor
from director.mcp_client import StubResolveClient
from director.schemas import PerClipMap, Plan, PlanOp, PlanOpKind
from director.settings import DirectorSettings
from director.store import EventLog, RunStore
from resolve_mcp.fake_backend import FakeResolveBackend


def _make_editor(
    store: RunStore,
    log: EventLog,
    client: StubResolveClient,
) -> Editor:
    return Editor(
        gemini=None,
        settings=DirectorSettings(gemini_api_key=None),
        client=client,
        run_store=store,
        event_log=log,
    )


def _seed_project(backend: FakeResolveBackend) -> None:
    backend.create_project("auto-reel", 24.0, 1920, 1080)
    backend.create_timeline("Timeline 1", 24.0)


def _make_media_files(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    file_a = tmp_path / "shot_a.mp4"
    file_b = tmp_path / "shot_b.mp4"
    file_a.write_bytes(b"fake-video-a")
    file_b.write_bytes(b"fake-video-b")
    return file_a, file_b


async def test_success_path_imports_binds_and_fades(
    tmp_run: tuple[RunStore, EventLog, pathlib.Path],
    tmp_path: pathlib.Path,
) -> None:
    store, log, _tmp = tmp_run
    backend = FakeResolveBackend(allow_destructive=True)
    client = StubResolveClient(backend)
    _seed_project(backend)
    file_a, file_b = _make_media_files(tmp_path)
    backend.import_media([str(file_a), str(file_b)])

    contextualizer_id = "clip_c0ffee42"
    per_clip = [PerClipMap(clip_id=contextualizer_id, source_path=str(file_a))]
    plan = Plan(
        plan_id="plan_ok",
        target_project="auto-reel",
        target_timeline="Timeline 1",
        ops=[
            PlanOp(
                id="op_append",
                kind=PlanOpKind.APPEND_CLIP,
                args={
                    "media_clip_id": contextualizer_id,
                    "timeline_track_index": 1,
                    "start_seconds": 0.0,
                    "duration_seconds": 1.0,
                },
                rationale="place first shot",
            ),
            PlanOp(
                id="op_fade",
                kind=PlanOpKind.ADD_FADE,
                args={
                    "timeline_item_id": "<item:0>",
                    "fade_in_seconds": 0.15,
                    "fade_out_seconds": 0.2,
                },
                rationale="soften cut",
            ),
        ],
        summary="success path",
    )
    editor = _make_editor(store, log, client)

    result = await editor.run(
        run_id="run_editor_success", plan=plan, iteration=1, per_clip=per_clip
    )

    assert result.errors == [], f"unexpected errors: {result.errors}"
    state = backend.get_timeline_state()
    video_items = state.tracks[0].items
    assert len(video_items) == 1, "append did not land on the video track"
    landed = video_items[0]
    assert landed.fade_in_seconds == pytest.approx(0.15)
    assert landed.fade_out_seconds == pytest.approx(0.2)
    # The append targeted a fake contextualizer id but must address the real pool clip.
    pool_a = next(c for c in backend.list_media_pool().clips if c.path == str(file_a))
    assert landed.media_clip_id == pool_a.id
    assert result.item_id_map == {"<item:0>": landed.id}
    assert [c.tool_name for c in result.tool_calls] == ["append_clip", "add_fade"]
    assert all(c.ok for c in result.tool_calls)
    stored = store.list_tool_calls("run_editor_success")
    assert [c.tool_name for c in stored] == ["append_clip", "add_fade"]
    assert all(c.ok for c in stored)
    # Idempotent import: pre-seeded pool still holds exactly the two clips.
    assert len(backend.list_media_pool().clips) == 2


async def test_missing_media_surfaces_real_backend_error(
    tmp_run: tuple[RunStore, EventLog, pathlib.Path],
    tmp_path: pathlib.Path,
) -> None:
    store, log, _tmp = tmp_run
    backend = FakeResolveBackend(allow_destructive=True)
    client = StubResolveClient(backend)
    _seed_project(backend)
    _file_a, _file_b = _make_media_files(tmp_path)

    # The clip exists on disk, but the op references an id nothing maps to.
    ghost = PerClipMap(clip_id="clip_ghost01", source_path=str(_file_a))
    plan = Plan(
        plan_id="plan_missing",
        target_project="auto-reel",
        target_timeline="Timeline 1",
        ops=[
            PlanOp(
                id="op_append",
                kind=PlanOpKind.APPEND_CLIP,
                args={
                    "media_clip_id": "clip_never_seen",
                    "timeline_track_index": 1,
                    "start_seconds": 0.0,
                    "duration_seconds": 1.0,
                },
                rationale="targets an id that exists nowhere",
            ),
        ],
        summary="missing media",
    )
    editor = _make_editor(store, log, client)

    result = await editor.run(run_id="run_editor_missing", plan=plan, iteration=1, per_clip=[ghost])

    assert result.errors, "expected the unknown clip id to surface as an error"
    failed = [c for c in result.tool_calls if not c.ok]
    assert failed, "failure was not recorded as an ok=False tool-call row"
    for record in failed:
        assert record.error is not None
        assert "frozen" not in record.error.lower()
        assert "clip_never_seen" in record.error, record.error
    for message in result.errors:
        assert "frozen" not in message.lower()
    assert any(c.path == str(_file_a) for c in backend.list_media_pool().clips), (
        "the clip's real source should still have been imported"
    )


async def test_deep_nested_symbol_substitution(
    tmp_run: tuple[RunStore, EventLog, pathlib.Path],
    tmp_path: pathlib.Path,
) -> None:
    store, log, _tmp = tmp_run
    backend = FakeResolveBackend(allow_destructive=True)
    client = StubResolveClient(backend)
    _seed_project(backend)
    file_a, _file_b = _make_media_files(tmp_path)
    backend.import_media([str(file_a)])

    per_clip = [PerClipMap(clip_id="clip_beef00d", source_path=str(file_a))]
    nested_symbol_args = {
        "opacity": 0.5,
        "timeline_item_id": "<item:0>",
        "extras": {"targets": [{"timeline_item_id": "<item:0>", "note": "keep me"}]},
    }
    plan = Plan(
        plan_id="plan_deep",
        target_project="auto-reel",
        target_timeline="Timeline 1",
        ops=[
            PlanOp(
                id="op_append",
                kind=PlanOpKind.APPEND_CLIP,
                args={
                    "media_clip_id": "clip_beef00d",
                    "timeline_track_index": 1,
                    "start_seconds": 0.0,
                    "duration_seconds": 1.0,
                },
                rationale="anchor item",
            ),
            PlanOp(
                id="op_opacity",
                kind=PlanOpKind.SET_OPACITY,
                args=dict(nested_symbol_args),
                rationale="symbols at several nesting depths",
            ),
        ],
        summary="deep substitution",
    )
    editor = _make_editor(store, log, client)

    result = await editor.run(run_id="run_editor_deep", plan=plan, iteration=1, per_clip=per_clip)

    assert result.errors == [], f"unexpected errors: {result.errors}"
    real_id = result.item_id_map.get("<item:0>")
    assert real_id, f"symbol never bound: {result.item_id_map}"
    sent = result.tool_calls[1]
    assert sent.tool_name == "set_opacity"
    assert sent.arguments["timeline_item_id"] == real_id
    assert sent.arguments["extras"]["targets"][0]["timeline_item_id"] == real_id
    # The frozen source op must be untouched by substitution.
    original = plan.ops[1].args
    assert original["timeline_item_id"] == "<item:0>"
    assert original["extras"]["targets"][0]["timeline_item_id"] == "<item:0>"
    # Outcome: opacity actually applied to the real timeline item.
    items = [it for tr in backend.get_timeline_state().tracks for it in tr.items]
    assert len(items) == 1
    assert items[0].opacity == pytest.approx(0.5)


async def test_symbol_auto_injected_when_planner_omits_it(
    tmp_run: tuple[RunStore, EventLog, pathlib.Path],
    tmp_path: pathlib.Path,
) -> None:
    store, log, _tmp = tmp_run
    backend = FakeResolveBackend(allow_destructive=True)
    client = StubResolveClient(backend)
    _seed_project(backend)
    file_a, _file_b = _make_media_files(tmp_path)
    backend.import_media([str(file_a)])

    per_clip = [PerClipMap(clip_id="clip_aut0b1nd", source_path=str(file_a))]
    append_args = {
        "media_clip_id": "clip_aut0b1nd",
        "timeline_track_index": 1,
        "start_seconds": 0.0,
        "duration_seconds": 1.0,
    }
    assert "__symbolic_id__" not in append_args
    plan = Plan(
        plan_id="plan_auto",
        target_project="auto-reel",
        target_timeline="Timeline 1",
        ops=[
            PlanOp(
                id="op_append",
                kind=PlanOpKind.APPEND_CLIP,
                args=dict(append_args),
                rationale="planner forgot the symbolic id",
            ),
            PlanOp(
                id="op_fade",
                kind=PlanOpKind.ADD_FADE,
                args={"timeline_item_id": "<item:0>", "fade_in_seconds": 0.1, "fade_out_seconds": 0.1},
                rationale="must bind to the auto-bound symbol",
            ),
        ],
        summary="auto-injection",
    )
    editor = _make_editor(store, log, client)

    result = await editor.run(run_id="run_editor_auto", plan=plan, iteration=1, per_clip=per_clip)

    assert result.errors == [], f"unexpected errors: {result.errors}"
    items = [it for tr in backend.get_timeline_state().tracks for it in tr.items]
    assert len(items) == 1
    assert result.item_id_map == {"<item:0>": items[0].id}
    assert items[0].fade_in_seconds > 0 and items[0].fade_out_seconds > 0
    # The internal binding marker must never leak onto the wire payload.
    assert "__symbolic_id__" not in result.tool_calls[0].arguments


async def test_missing_source_file_aborts_before_touching_the_timeline(
    tmp_run: tuple[RunStore, EventLog, pathlib.Path],
) -> None:
    """A source that does not exist is a setup error, not a half-built timeline."""
    store, log, _tmp = tmp_run
    backend = FakeResolveBackend(allow_destructive=True)
    client = StubResolveClient(backend)
    _seed_project(backend)
    missing = PerClipMap(clip_id="clip_missing", source_path="/nowhere/ghost.mp4")
    plan = Plan(
        plan_id="plan_no_file",
        target_project="auto-reel",
        target_timeline="Timeline 1",
        ops=[
            PlanOp(
                id="op_append",
                kind=PlanOpKind.APPEND_CLIP,
                args={
                    "media_clip_id": "clip_missing",
                    "timeline_track_index": 1,
                    "start_seconds": 0.0,
                    "duration_seconds": 1.0,
                },
                rationale="source file is absent",
            )
        ],
        summary="missing file",
    )
    editor = _make_editor(store, log, client)

    result = await editor.run(run_id="run_missing_file", plan=plan, iteration=1, per_clip=[missing])

    assert result.errors, "an unimportable source must surface as an error"
    assert result.tool_calls == [], "nothing should have been dispatched"
    assert not any(tr.items for tr in backend.get_timeline_state().tracks)
