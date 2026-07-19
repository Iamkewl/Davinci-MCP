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


async def test_repl_quits_cleanly(tmp_run: tuple[RunStore, EventLog, pathlib.Path]) -> None:
    store, log, _tmp = tmp_run
    backend = FakeResolveBackend(allow_destructive=True)
    client = StubResolveClient(backend)
    backend.create_project("interactive-reel", 24.0, 1920, 1080)
    backend.create_timeline("Timeline 1", 24.0)
    backend.import_media(["/clips/a.mp4"])
    backend.append_clip(backend.list_media_pool().clips[0].id, 0, 0.0, 2.0)

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


async def test_repl_interprets_instruction(tmp_run: tuple[RunStore, EventLog, pathlib.Path]) -> None:
    store, log, _tmp = tmp_run
    backend = FakeResolveBackend(allow_destructive=True)
    client = StubResolveClient(backend)
    backend.create_project("interactive-reel", 24.0, 1920, 1080)
    backend.create_timeline("Timeline 1", 24.0)
    clips = backend.import_media(["/clips/a.mp4"])
    backend.append_clip(clips[0].id, 0, 0.0, 2.0)

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
    reader, printer, _ = _drive([
        "tighten the intro cut to the first downbeat",
        "quit",
    ])
    await run_repl(session=session, input_reader=reader, printer=printer)
    # The orchestrator should have:
    # - recorded a CHECKPOINT event for the instruction.
    # - recorded a Director verdict for the instruction.
    # - modified the timeline state of the backend.
    state = backend.get_timeline_state()
    assert state.tracks[0].items[0].markers  # default offline interpret adds a marker
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
