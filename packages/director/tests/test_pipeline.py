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


def _valid_plan_for(clip_path: str) -> Plan:
    """A plan with no structural problems: real source, sane track, tiled shots."""
    return _plan_with_ops(
        [
            PlanOp(
                id=f"op_ok_{index}",
                kind=PlanOpKind.APPEND_CLIP,
                args={
                    "media_clip_id": clip_path,
                    "timeline_track_index": 1,
                    "start_seconds": float(index) * 2.0,
                    "duration_seconds": 2.0,
                    "__symbolic_id__": f"<item:{index}>",
                },
                rationale="valid shot",
            )
            for index in range(4)
        ],
        summary="Four tiled shots from a real source.",
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
    clips: list[str],
) -> None:
    orch, store, log, _backend = orchestrator_setup
    result = await orch.run_auto(
        clip_paths=clips[:2],
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
    clips: list[str],
) -> None:
    """The offline 2-clip plan scores APPROVED (all axes >= floors), so the
    Editor must run exactly once and hand back an EditResult."""
    orch, _store, log, _backend = orchestrator_setup
    result = await orch.run_auto(
        clip_paths=clips[:2],
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
    clips: list[str],
) -> None:
    orch, _store, _log, backend = orchestrator_setup
    await orch.run_auto(
        clip_paths=clips[:2],
        music_path=None,
        user_prompt="make a tiny reel",
    )
    projects = backend.list_projects()
    assert projects
    assert backend.get_timeline_state().name is not None


async def test_event_log_records_checkpoint_and_verdict(
    orchestrator_setup: tuple[Orchestrator, RunStore, EventLog, FakeResolveBackend],
    clips: list[str],
) -> None:
    orch, _store, log, _backend = orchestrator_setup
    result = await orch.run_auto(
        clip_paths=clips[:1],
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
    # No clips means no plan and therefore no timeline. A run that built nothing
    # reports failed; calling it "completed" would be a lie.
    assert result.status is RunStatus.FAILED
    fetched = store.get_run(result.run_id)
    assert fetched is not None
    assert fetched.status is RunStatus.FAILED


async def test_warned_plan_is_attempted_but_status_stays_honest(
    orchestrator_setup: tuple[Orchestrator, RunStore, EventLog, FakeResolveBackend],
    monkeypatch: pytest.MonkeyPatch,
    clips: list[str],
) -> None:
    """ACCEPTED_WITH_WARNINGS means "acceptable", so the plan is built once the
    planner stops changing it — but the status still reports what landed. This
    plan references a clip that does not exist, so nothing lands and the run is
    failed rather than a cheerful "completed_with_warnings"."""
    orch, store, _log, _backend = orchestrator_setup
    _patch_planner(orch, monkeypatch, _single_op_plan())
    result = await orch.run_auto(
        clip_paths=clips[:1],
        music_path=None,
        user_prompt="nothing usable",
    )
    assert result.verdict is not None
    assert result.verdict.verdict == DirectorVerdict.ACCEPTED_WITH_WARNINGS
    # The plan is attempted, but its clip does not exist, so nothing lands.
    assert result.edit_result is not None, "a warned plan should still be attempted"
    assert result.edit_result.applied_count == 0
    assert result.status is RunStatus.FAILED
    # Re-planning stops as soon as the planner repeats itself instead of burning
    # the whole budget on an identical plan.
    assert result.iterations == 2, "an unchanged plan should end the loop"
    fetched = store.get_run(result.run_id)
    assert fetched is not None
    assert fetched.status is RunStatus.FAILED


async def test_approved_all_ops_fail_maps_failed(
    orchestrator_setup: tuple[Orchestrator, RunStore, EventLog, FakeResolveBackend],
    monkeypatch: pytest.MonkeyPatch,
    clips: list[str],
) -> None:
    """APPROVED + every executed op erroring must end FAILED, not approved."""
    orch, store, _log, _backend = orchestrator_setup
    _patch_planner(orch, monkeypatch, _approved_shaped_plan())
    result = await orch.run_auto(
        clip_paths=clips[:1],
        music_path=None,
        user_prompt="nothing",
    )
    assert result.edit_result is not None, "approved plan was never executed"
    assert result.edit_result.errors, f"expected every op to fail; got {result.edit_result.errors}"
    assert result.status is RunStatus.FAILED
    executed = store.list_tool_calls(result.run_id)
    assert len(executed) == 4, "the plan must be executed exactly once"
    fetched = store.get_run(result.run_id)
    assert fetched is not None
    assert fetched.status is RunStatus.FAILED


async def test_approved_partial_success_maps_completed_with_warnings(
    orchestrator_setup: tuple[Orchestrator, RunStore, EventLog, FakeResolveBackend],
    monkeypatch: pytest.MonkeyPatch,
    clips: list[str],
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
        **_: object,
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
        clip_paths=clips[:1],
        music_path=None,
        user_prompt="nothing",
    )
    assert len(executions) == 1, 'the plan must be executed exactly once'
    assert result.edit_result is not None
    assert result.edit_result.errors == ["synthetic partial failure"]
    assert result.status is RunStatus.COMPLETED_WITH_WARNINGS


async def test_approved_clean_execution_maps_completed_approved(
    orchestrator_setup: tuple[Orchestrator, RunStore, EventLog, FakeResolveBackend],
    monkeypatch: pytest.MonkeyPatch,
    clips: list[str],
) -> None:
    """APPROVED + clean execution (no errors) → completed_approved."""
    orch, _store, _log, _backend = orchestrator_setup
    _patch_planner(orch, monkeypatch, _valid_plan_for(clips[0]))

    async def _clean_editor_run(
        *,
        run_id: str,
        plan: Plan,
        iteration: int,
        per_clip: list[PerClipMap],
        **_: object,
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
        clip_paths=clips[:1],
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
    # Nothing executed is a failure, whatever the verdict said.
    assert map_status(warnings, None) is RunStatus.FAILED
    # An approved plan that was never executed still produced no timeline.
    assert map_status(approved, None) is RunStatus.FAILED
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
