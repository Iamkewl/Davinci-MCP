"""Tests for the Phase-4 interactive REPL.

We drive the REPL with a fake input stream and StubResolveClient + fake
backend so the suite stays offline and deterministic.
"""

from __future__ import annotations

import pathlib

from director.agents import Director, Editor, Planner
from director.interactive import InteractiveSession, run_repl
from director.mcp_client import StubResolveClient
from director.settings import DirectorSettings
from director.store import EventLog, RunStore
from resolve_mcp.fake_backend import FakeResolveBackend


def _drive(input_lines: list[str]):
    """Return (lines_iter, recorded_lines, echo) helpers"""
    it = iter(input_lines)
    out: list[str] = []

    def reader() -> str | None:
        return next(it, None)

    def printer(s: str) -> None:
        out.append(s)

    return reader, printer, out


async def test_repl_quits_cleanly(
    tmp_run: tuple[RunStore, EventLog, pathlib.Path], clips: list[str]
) -> None:
    store, log, _tmp = tmp_run
    backend = FakeResolveBackend(allow_destructive=True)
    client = StubResolveClient(backend)
    backend.create_project("interactive-reel", 24.0, 1920, 1080)
    backend.create_timeline("Timeline 1", 24.0)
    backend.import_media(clips[:1])
    backend.append_clip(backend.list_media_pool().clips[0].id, 1, 0.0, 2.0)

    settings = DirectorSettings(gemini_api_key=None)
    planner = Planner(gemini=None, settings=settings)
    director = Director(gemini=None, settings=settings)
    editor = Editor(
        gemini=None,
        settings=settings,
        client=client,
        run_store=store,
        event_log=log,
        allow_destructive=False,
    )
    session = InteractiveSession(
        settings=settings,
        client=client,
        run_store=store,
        event_log=log,
        planner=planner,
        director=director,
        editor=editor,
        target_project="interactive-reel",
        target_timeline="Timeline 1",
    )
    reader, printer, out = _drive(["state", "tools", "quit"])
    await run_repl(session=session, input_reader=reader, printer=printer)
    joined = "\n".join(out)
    assert "director interactive" in joined
    assert "resolve_mcp_state" not in joined or "tracks" in joined  # state printed
    assert "append_clip" in joined  # tools printed
    assert "bye." in joined


async def test_repl_interprets_instruction(
    tmp_run: tuple[RunStore, EventLog, pathlib.Path], clips: list[str]
) -> None:
    store, log, _tmp = tmp_run
    backend = FakeResolveBackend(allow_destructive=True)
    client = StubResolveClient(backend)
    backend.create_project("interactive-reel", 24.0, 1920, 1080)
    backend.create_timeline("Timeline 1", 24.0)
    pool = backend.import_media(clips[:1])
    backend.append_clip(pool[0].id, 1, 0.0, 2.0)

    settings = DirectorSettings(gemini_api_key=None)
    planner = Planner(gemini=None, settings=settings)
    director = Director(gemini=None, settings=settings)
    editor = Editor(
        gemini=None,
        settings=settings,
        client=client,
        run_store=store,
        event_log=log,
        allow_destructive=False,
    )
    session = InteractiveSession(
        settings=settings,
        client=client,
        run_store=store,
        event_log=log,
        planner=planner,
        director=director,
        editor=editor,
        target_project="interactive-reel",
        target_timeline="Timeline 1",
    )
    reader, printer, out = _drive([
        "mark clip 1 'hook' at 0.5s",
        "fade in the first clip 1s",
        "tighten the intro cut to the first downbeat",  # not parseable offline
        "quit",
    ])
    await run_repl(session=session, input_reader=reader, printer=printer)
    # The orchestrator should have:
    # - recorded a CHECKPOINT event for the instruction.
    # - recorded a Director verdict for the instruction.
    # - modified the timeline state of the backend.
    state = backend.get_timeline_state()
    item = state.tracks[0].items[0]
    assert [m.label for m in item.markers] == ["hook"]
    assert item.fade_in_seconds == 1.0
    # The instruction it could not parse is reported, not silently "applied".
    joined = "\n".join(out)
    assert "could not interpret offline" in joined
    events = log.read_all(run_id=session.run_id)
    kinds = [e.kind for e in events]
    from director.schemas import EventKind
    assert EventKind.CHECKPOINT in kinds
    assert EventKind.DIRECTOR_VERDICT in kinds


async def test_interactive_session_creates_run_id(tmp_run: tuple[RunStore, EventLog, pathlib.Path]) -> None:
    store, log, _tmp = tmp_run
    backend = FakeResolveBackend(allow_destructive=True)
    client = StubResolveClient(backend)
    settings = DirectorSettings(gemini_api_key=None)
    session = InteractiveSession(
        settings=settings,
        client=client,
        run_store=store,
        event_log=log,
        planner=Planner(gemini=None, settings=settings),
        director=Director(gemini=None, settings=settings),
        editor=Editor(
            gemini=None, settings=settings, client=client,
            run_store=store, event_log=log,
        ),
        target_project="p",
        target_timeline="t",
    )
    assert session.run_id
    fetched = store.get_run(session.run_id)
    assert fetched is not None
    assert fetched.mode.value == "interactive"


# --- regressions: the interactive path obeys the same invariants as auto ----------


def _session(
    store: RunStore,
    log: EventLog,
    backend: FakeResolveBackend,
    client: StubResolveClient,
) -> InteractiveSession:
    settings = DirectorSettings(gemini_api_key=None)
    return InteractiveSession(
        settings=settings,
        client=client,
        run_store=store,
        event_log=log,
        planner=Planner(gemini=None, settings=settings),
        director=Director(gemini=None, settings=settings),
        editor=Editor(
            gemini=None,
            settings=settings,
            client=client,
            run_store=store,
            event_log=log,
            allow_destructive=False,
        ),
        target_project="interactive-reel",
        target_timeline="Timeline 1",
    )


def _loaded_backend(clips: list[str]) -> tuple[FakeResolveBackend, StubResolveClient]:
    backend = FakeResolveBackend(allow_destructive=True)
    client = StubResolveClient(backend)
    backend.create_project("interactive-reel", 24.0, 1920, 1080)
    backend.create_timeline("Timeline 1", 24.0)
    pool = backend.import_media(clips[:1])
    backend.append_clip(pool[0].id, 1, 0.0, 2.0)
    return backend, client


async def test_a_structurally_invalid_delta_plan_is_not_applied(
    tmp_run: tuple[RunStore, EventLog, pathlib.Path], clips: list[str], monkeypatch
) -> None:
    """An LLM that drifts on argument names used to sail through: the interactive
    path never called validate_plan, so validity scored 1.0 and the bad op went
    straight to the tool."""
    from director.schemas import DirectorVerdict, Plan, PlanOp, PlanOpKind

    store, log, _tmp = tmp_run
    backend, client = _loaded_backend(clips)
    session = _session(store, log, backend, client)

    bad = Plan(
        plan_id="p", version=1, target_project="interactive-reel", target_timeline="Timeline 1",
        ops=[PlanOp(id="o", kind=PlanOpKind.SET_OPACITY, args={"item_id": "x", "level": 0.6})],
        summary="drifted argument names",
    )

    async def _interpret(**_kwargs: object) -> Plan:
        return bad

    monkeypatch.setattr(session._planner, "interpret", _interpret)
    before = backend.get_timeline_state()
    result = await session.interpret("set opacity of clip 1 to 60%")

    assert result.executed is False
    assert "would not execute" in result.note
    assert result.verdict.evaluation.verdict != DirectorVerdict.APPROVED
    assert backend.get_timeline_state() == before


async def test_an_invented_item_id_is_caught_against_the_live_timeline(
    tmp_run: tuple[RunStore, EventLog, pathlib.Path], clips: list[str], monkeypatch
) -> None:
    from director.schemas import Plan, PlanOp, PlanOpKind

    store, log, _tmp = tmp_run
    backend, client = _loaded_backend(clips)
    session = _session(store, log, backend, client)

    async def _interpret(**_kwargs: object) -> Plan:
        return Plan(
            plan_id="p", version=1, target_project="interactive-reel", target_timeline="Timeline 1",
            ops=[
                PlanOp(
                    id="o",
                    kind=PlanOpKind.SET_OPACITY,
                    args={"timeline_item_id": "item_that_does_not_exist", "opacity": 0.6},
                )
            ],
            summary="hallucinated id",
        )

    monkeypatch.setattr(session._planner, "interpret", _interpret)
    result = await session.interpret("dim clip 1")
    assert result.executed is False
    assert "not on this timeline" in result.note


async def test_a_session_that_applied_nothing_is_not_completed(
    tmp_run: tuple[RunStore, EventLog, pathlib.Path], clips: list[str], monkeypatch
) -> None:
    """Same invariant as auto mode: attempted an edit, applied nothing -> failed."""
    from director.schemas import Plan, PlanOp, PlanOpKind, RunStatus

    store, log, _tmp = tmp_run
    backend, client = _loaded_backend(clips)
    session = _session(store, log, backend, client)

    async def _interpret(**_kwargs: object) -> Plan:
        return Plan(
            plan_id="p", version=1, target_project="interactive-reel", target_timeline="Timeline 1",
            ops=[PlanOp(id="o", kind=PlanOpKind.SET_OPACITY, args={"nope": 1})],
            summary="broken",
        )

    monkeypatch.setattr(session._planner, "interpret", _interpret)
    await session.interpret("dim clip 1")
    session.finalize()
    record = store.get_run(session.run_id)
    assert record is not None
    assert record.status == RunStatus.FAILED


async def test_a_browse_only_session_is_not_a_failure(
    tmp_run: tuple[RunStore, EventLog, pathlib.Path], clips: list[str]
) -> None:
    from director.schemas import RunStatus

    store, log, _tmp = tmp_run
    backend, client = _loaded_backend(clips)
    session = _session(store, log, backend, client)
    reader, printer, _out = _drive(["state", "quit"])
    await run_repl(session=session, input_reader=reader, printer=printer)
    record = store.get_run(session.run_id)
    assert record is not None
    assert record.status != RunStatus.FAILED
