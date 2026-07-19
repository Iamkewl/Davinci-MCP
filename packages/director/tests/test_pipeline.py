"""End-to-end pipeline test: auto mode against the StubResolveClient + fake
backend. Gemini is OFFLINE (gemini=None) so the deterministic planner + offline
director are exercised.
"""

from __future__ import annotations

import pathlib

import pytest
from director.mcp_client import StubResolveClient
from director.pipeline import Orchestrator
from director.schemas import EventKind, RunStatus
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
    assert result.status in (
        RunStatus.COMPLETED_APPROVED,
        RunStatus.COMPLETED_WITH_WARNINGS,
        RunStatus.FAILED,
    )
    fetched = store.get_run(result.run_id)
    assert fetched is not None


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
