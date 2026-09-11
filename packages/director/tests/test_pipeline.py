"""End-to-end pipeline test: auto mode against the StubResolveClient + fake
backend. Gemini is OFFLINE (gemini=None) so the deterministic planner + offline
director are exercised.

Execution contract under test:
* Only APPROVED plans reach the Editor (warned plans iterate, never execute).
* Execution outcomes feed the terminal status honestly.
"""

from __future__ import annotations

import pathlib

import pytest
from director.agents import EditResult, PlannerRequest
from director.mcp_client import StubResolveClient
from director.pipeline import Orchestrator
from director.schemas import (
    DirectorEvaluation,
    DirectorVerdict,
    EventKind,
    PerClipMap,
    Plan,
    PlanOp,
    PlanOpKind,
    RunStatus,
    ToolCallRecord,
)
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
    settings = DirectorSettings(
        gemini_api_key=None,
        max_planner_iterations=5,
        director_min_overall=0.4,
        director_min_per_axis=0.3,
    )
    orchestrator = Orchestrator(
        settings=settings,
        gemini=None,
        client=client,
        run_store=store,
        event_log=log,
    )
    return orchestrator, store, log, backend


def _plan_with_ops(ops: list[PlanOp], *, summary: str = "deterministic test plan") -> Plan:
    return Plan(
        plan_id="plan_test",
        version=1,
        target_project="auto-reel",
        target_timeline="Timeline 1",
        ops=ops,
        summary=summary,
    )


def _single_op_plan() -> Plan:
    """One append op → narrative 0.25 (< per-axis floor) with music-less beats,
    so the offline director scores ACCEPTED_WITH_WARNINGS deterministically."""
    return _plan_with_ops(
        [
            PlanOp(
                id="op_warn_0",
                kind=PlanOpKind.APPEND_CLIP,
                args={
                    "media_clip_id": "clip_x",
                    "timeline_track_index": 1,
                    "start_seconds": 0.0,
                    "duration_seconds": 2.0,
                },
            )
        ]
    )


def _approved_shaped_plan() -> Plan:
    """Four ops → narrative 1.0, beat_sync 0.5, all axes above the floors, so
    the offline director scores APPROVED. The ops themselves are sabotaged to
    fail against the fake backend (nonexistent clip/track, unresolved symbol)."""
    return _plan_with_ops(
        [
            PlanOp(
                id="op_sab_0",
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
                    id=f"op_sab_{i}",
                    kind=PlanOpKind.ADD_FADE,
                    args={
                        "timeline_item_id": "<item:9>",
                        "fade_in_seconds": 0.15,
                        "fade_out_seconds": 0.2,
                    },
                    rationale="sabotaged fade",
                )
                for i in range(1, 4)
            ],
        ],
        summary="Sabotaged plan shaped to score APPROVED.",
    )


def _patch_planner(orch: Orchestrator, monkeypatch: pytest.MonkeyPatch, plan: Plan) -> None:
    async def _fixed_plan(_request: PlannerRequest) -> Plan:
        return plan

    monkeypatch.setattr(orch._planner, "run", _fixed_plan)


async def test_auto_run_produces_plan_and_state(
    orchestrator_setup: tuple[Orchestrator, RunStore, EventLog, FakeResolveBackend],
) -> None:
    orch, store, log, _backend = orchestrator_setup
    result = await orch.run_auto(
        clip_paths=["/clips/a.mp4", "/clips/b.mp4"],
        music_path=None,
        user_prompt="tight 4-second reel",
    )
    assert result.status in (
        RunStatus.COMPLETED_APPROVED,
        RunStatus.COMPLETED_WITH_WARNINGS,
    )
    assert result.plan is not None
    assert result.verdict is not None
    kinds = [op.kind.value for op in result.plan.ops]
    assert "append_clip" in kinds
    fetched = store.get_run(result.run_id)
    assert fetched is not None
    assert fetched.iterations >= 1
    assert len(store.list_verdicts(result.run_id)) >= 1
    assert len(store.list_tool_calls(result.run_id)) >= 1
    assert len(log.read_all(run_id=result.run_id)) >= 3


async def test_approved_plan_executes_exactly_once(
    orchestrator_setup: tuple[Orchestrator, RunStore, EventLog, FakeResolveBackend],
) -> None:
    """The offline 2-clip plan scores APPROVED (all axes >= floors), so the
    Editor must run exactly once and hand back an EditResult."""
    orch, _store, log, _backend = orchestrator_setup
    result = await orch.run_auto(
        clip_paths=["/clips/a.mp4", "/clips/b.mp4"],
        music_path=None,
        user_prompt="tight 4-second reel",
    )
    assert result.verdict is not None
    assert result.verdict.verdict == DirectorVerdict.APPROVED
    assert result.edit_result is not None, "approved plan was never executed"
    applied = [
        e
        for e in log.read_all(run_id=result.run_id)
        if e.kind in (EventKind.PLAN_APPLIED, EventKind.ERROR)
    ]
    assert len(applied) == 1, f"expected exactly one execution pass, got {len(applied)}"


async def test_editor_applies_plan_with_state_observed(
    orchestrator_setup: tuple[Orchestrator, RunStore, EventLog, FakeResolveBackend],
) -> None:
    orch, _store, _log, backend = orchestrator_setup
    await orch.run_auto(
        clip_paths=["/clips/a.mp4", "/clips/b.mp4"],
        music_path=None,
        user_prompt="make a tiny reel",
    )
    projects = backend.list_projects()
    assert projects
    assert backend.get_timeline_state().name is not None


async def test_event_log_records_checkpoint_and_verdict(
    orchestrator_setup: tuple[Orchestrator, RunStore, EventLog, FakeResolveBackend],
) -> None:
    orch, _store, log, _backend = orchestrator_setup
    result = await orch.run_auto(
        clip_paths=["/clips/a.mp4"],
        music_path=None,
        user_prompt="solo clip",
    )
    events = log.read_all(run_id=result.run_id)
    kinds = [e.kind for e in events]
    assert EventKind.CHECKPOINT in kinds
    assert EventKind.PLAN_COMPILED in kinds
    assert EventKind.DIRECTOR_VERDICT in kinds


async def test_pipeline_records_failure_when_input_invalid(
    orchestrator_setup: tuple[Orchestrator, RunStore, EventLog, FakeResolveBackend],
) -> None:
    orch, store, _log, _backend = orchestrator_setup
    result = await orch.run_auto(
        clip_paths=[],
        music_path=None,
        user_prompt="empty",
    )
    # Empty input → empty plan → narrative 0 warns forever; warned plans never
    # execute, so budget exhaustion lands on completed_with_warnings.
    assert result.status is RunStatus.COMPLETED_WITH_WARNINGS
    assert result.edit_result is None
    fetched = store.get_run(result.run_id)
    assert fetched is not None
    assert fetched.status is RunStatus.COMPLETED_WITH_WARNINGS


async def test_warnings_verdict_never_executes(
    orchestrator_setup: tuple[Orchestrator, RunStore, EventLog, FakeResolveBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ACCEPTED_WITH_WARNINGS runs must iterate WITHOUT executing: no editor
    call, no tool-call records, and the budget exhausts into a warnings exit."""
    orch, store, _log, _backend = orchestrator_setup
    _patch_planner(orch, monkeypatch, _single_op_plan())
    result = await orch.run_auto(
        clip_paths=["/clips/a.mp4"],
        music_path=None,
        user_prompt="nothing usable",
    )
    assert result.verdict is not None
    assert result.verdict.verdict == DirectorVerdict.ACCEPTED_WITH_WARNINGS
    assert result.status is RunStatus.COMPLETED_WITH_WARNINGS
    assert result.edit_result is None, "a warned plan was executed"
    assert result.iterations == 5, "warnings should consume the whole budget"
    assert store.list_tool_calls(result.run_id) == []
    fetched = store.get_run(result.run_id)
    assert fetched is not None
    assert fetched.status is RunStatus.COMPLETED_WITH_WARNINGS


async def test_approved_all_ops_fail_maps_failed(
    orchestrator_setup: tuple[Orchestrator, RunStore, EventLog, FakeResolveBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """APPROVED + every executed op erroring must end FAILED, not approved."""
    orch, store, _log, _backend = orchestrator_setup
    _patch_planner(orch, monkeypatch, _approved_shaped_plan())
    result = await orch.run_auto(
        clip_paths=["/clips/a.mp4"],
        music_path=None,
        user_prompt="nothing",
    )
    assert result.edit_result is not None, "approved plan was never executed"
    assert result.edit_result.errors, f"expected every op to fail; got {result.edit_result.errors}"
    assert result.status is RunStatus.FAILED
    assert result.iterations == 1, "approved plan must execute once, not per iteration"
    fetched = store.get_run(result.run_id)
    assert fetched is not None
    assert fetched.status is RunStatus.FAILED


async def test_approved_partial_success_maps_completed_with_warnings(
    orchestrator_setup: tuple[Orchestrator, RunStore, EventLog, FakeResolveBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """APPROVED + some ops errored but at least one tool call ok → warnings."""
    orch, _store, _log, _backend = orchestrator_setup
    _patch_planner(orch, monkeypatch, _approved_shaped_plan())
    executions: list[int] = []

    async def _partial_editor_run(
        *,
        run_id: str,
        plan: Plan,
        iteration: int,
        per_clip: list[PerClipMap],
    ) -> EditResult:
        executions.append(iteration)
        return EditResult(
            iteration=iteration,
            errors=["synthetic partial failure"],
            tool_calls=[
                ToolCallRecord(
                    run_id=run_id,
                    iteration=iteration,
                    tool_name="append_clip",
                    arguments={},
                    ok=True,
                )
            ],
        )

    monkeypatch.setattr(orch._editor, "run", _partial_editor_run)
    result = await orch.run_auto(
        clip_paths=["/clips/a.mp4"],
        music_path=None,
        user_prompt="nothing",
    )
    assert executions == [1]
    assert result.edit_result is not None
    assert result.edit_result.errors == ["synthetic partial failure"]
    assert result.status is RunStatus.COMPLETED_WITH_WARNINGS


async def test_approved_clean_execution_maps_completed_approved(
    orchestrator_setup: tuple[Orchestrator, RunStore, EventLog, FakeResolveBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """APPROVED + clean execution (no errors) → completed_approved."""
    orch, _store, _log, _backend = orchestrator_setup
    _patch_planner(orch, monkeypatch, _approved_shaped_plan())

    async def _clean_editor_run(
        *,
        run_id: str,
        plan: Plan,
        iteration: int,
        per_clip: list[PerClipMap],
    ) -> EditResult:
        return EditResult(
            iteration=iteration,
            errors=[],
            tool_calls=[
                ToolCallRecord(
                    run_id=run_id,
                    iteration=iteration,
                    tool_name="append_clip",
                    arguments={},
                    ok=True,
                )
            ],
        )

    monkeypatch.setattr(orch._editor, "run", _clean_editor_run)
    result = await orch.run_auto(
        clip_paths=["/clips/a.mp4"],
        music_path=None,
        user_prompt="nothing",
    )
    assert result.status is RunStatus.COMPLETED_APPROVED
    assert result.edit_result is not None
    assert result.edit_result.errors == []


def test_terminal_status_honest_mapping() -> None:
    approved = DirectorEvaluation(verdict=DirectorVerdict.APPROVED, overall=0.9)
    warnings = DirectorEvaluation(verdict=DirectorVerdict.ACCEPTED_WITH_WARNINGS, overall=0.7)
    failed = DirectorEvaluation(verdict=DirectorVerdict.FAILED, overall=0.2)

    def _edit_result(*, errors: list[str], ok_calls: int) -> EditResult:
        return EditResult(
            iteration=1,
            errors=errors,
            tool_calls=[
                ToolCallRecord(run_id="r", iteration=1, tool_name="append_clip", arguments={}, ok=True)
                for _ in range(ok_calls)
            ],
        )

    map_status = Orchestrator._terminal_status
    assert map_status(None, None) is RunStatus.FAILED
    assert map_status(failed, None) is RunStatus.FAILED
    assert map_status(warnings, None) is RunStatus.COMPLETED_WITH_WARNINGS
    assert map_status(approved, None) is RunStatus.COMPLETED_APPROVED
    assert map_status(approved, _edit_result(errors=[], ok_calls=2)) is RunStatus.COMPLETED_APPROVED
    assert map_status(approved, _edit_result(errors=["x"], ok_calls=0)) is RunStatus.FAILED
    assert map_status(approved, _edit_result(errors=["x"], ok_calls=1)) is RunStatus.COMPLETED_WITH_WARNINGS


def test_stub_resolve_client_lists_tools() -> None:
    from director.mcp_client import StubResolveClient
    from resolve_mcp.fake_backend import FakeResolveBackend

    backend = FakeResolveBackend()
    client = StubResolveClient(backend)

    async def _check() -> None:
        tools = await client.list_tools()
        assert "create_project" in tools
        await client.close()

    import asyncio

    asyncio.run(_check())
