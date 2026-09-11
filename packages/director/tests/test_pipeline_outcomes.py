"""Outcome-level regression tests (plan.md Phase 0).

These encode the contract that the offline pipeline actually APPLIES plans:
items land on the timeline, fades bind to real ids, and execution errors
surface in the run status. They were written failing (xfail) before the fixes.
"""

from __future__ import annotations

import pathlib

import pytest
from director.agents import PlannerRequest
from director.mcp_client import StubResolveClient
from director.pipeline import Orchestrator
from director.schemas import Plan, PlanOp, PlanOpKind, RunStatus
from director.settings import DirectorSettings
from director.store import EventLog, RunStore
from resolve_mcp.fake_backend import FakeResolveBackend


@pytest.fixture
def orchestrator_setup(
    tmp_run: tuple[RunStore, EventLog, pathlib.Path],
) -> tuple[Orchestrator, RunStore, EventLog, FakeResolveBackend]:
    store, log, _tmp = tmp_run
    backend = FakeResolveBackend(allow_destructive=True)
    client = StubResolveClient(backend)
    settings = DirectorSettings(gemini_api_key=None, max_planner_iterations=5)
    orchestrator = Orchestrator(
        settings=settings,
        gemini=None,
        client=client,
        run_store=store,
        event_log=log,
    )
    return orchestrator, store, log, backend


async def test_auto_builds_nonempty_timeline_without_errors(
    orchestrator_setup: tuple[Orchestrator, RunStore, EventLog, FakeResolveBackend],
) -> None:
    orch, _store, _log, backend = orchestrator_setup
    result = await orch.run_auto(
        clip_paths=["/clips/a.mp4", "/clips/b.mp4"],
        music_path=None,
        user_prompt="tight reel",
    )
    assert result.status is RunStatus.COMPLETED_APPROVED
    assert result.edit_result is not None, "approved plan was never executed"
    assert result.edit_result.errors == [], f"execution errors: {result.edit_result.errors}"
    state = backend.get_timeline_state()
    items = [it for tr in state.tracks for it in tr.items]
    assert len(items) >= 1, "timeline is empty after an approved run"


async def test_symbolic_fade_binding_resolves(
    orchestrator_setup: tuple[Orchestrator, RunStore, EventLog, FakeResolveBackend],
) -> None:
    orch, _store, _log, backend = orchestrator_setup
    result = await orch.run_auto(
        clip_paths=["/clips/a.mp4"],
        music_path=None,
        user_prompt="faded reel",
    )
    assert result.edit_result is not None
    assert all("<item:" not in (e or "") for e in result.edit_result.errors)
    state = backend.get_timeline_state()
    faded = [it for tr in state.tracks for it in tr.items if it.fade_in_seconds > 0]
    assert faded, "add_fade never landed on a real timeline item"


async def test_all_ops_failing_marks_run_failed(
    orchestrator_setup: tuple[Orchestrator, RunStore, EventLog, FakeResolveBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If every executed op errored, status must be FAILED even though the plan
    itself got APPROVED.

    Sabotage: force a plan whose ops target a nonexistent track / unresolved
    symbolic item, so every op fails regardless of media handling. The plan is
    shaped to score APPROVED under the deterministic offline director (4 ops,
    non-empty summary), guaranteeing it actually reaches the Editor.
    """
    orch, _store, _log, _backend = orchestrator_setup
    failing_plan = Plan(
        plan_id="plan_sabotaged",
        version=1,
        target_project="auto-reel",
        target_timeline="Timeline 1",
        ops=[
            PlanOp(
                id="op_fail_0",
                kind=PlanOpKind.APPEND_CLIP,
                args={
                    "media_clip_id": "no-such-clip",
                    "timeline_track_index": 99,
                    "start_seconds": 0.0,
                    "duration_seconds": 1.0,
                },
                rationale="sabotaged append",
            ),
            *[
                PlanOp(
                    id=f"op_fail_{i}",
                    kind=PlanOpKind.ADD_FADE,
                    args={
                        "timeline_item_id": "<item:0>",
                        "fade_in_seconds": 0.15,
                        "fade_out_seconds": 0.2,
                    },
                    rationale="sabotaged fade",
                )
                for i in range(1, 4)
            ],
        ],
        summary="Sabotaged plan: every op must fail.",
    )

    async def _always_failing_plan(request: PlannerRequest) -> Plan:
        return failing_plan

    monkeypatch.setattr(orch._planner, "run", _always_failing_plan)
    result = await orch.run_auto(
        clip_paths=["/clips/a.mp4"],
        music_path=None,
        user_prompt="nothing",
    )
    assert result.edit_result is not None, "plan was never executed"
    assert result.edit_result.errors, f"expected every op to fail; got {result.edit_result.errors}"
    assert result.status is RunStatus.FAILED
