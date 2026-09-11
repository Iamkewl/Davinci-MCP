"""Director CLI: ``auto`` / ``interactive`` / ``inspect`` / ``resume``.

We back the CLI with Typer, but the orchestrator itself has no Typer
dependency (instantiated in tests directly).

Bootstrap walk
--------------

* Resolve the ``resolve-mcp`` project path so the subprocess launched in
  :class:`StdioResolveClient` can find its package without PYTHONPATH hacks.
* Build the Gemini client from env, build the run store under
  ``DIRECTOR_RUN_STORE`` (default ``./runs``).
* For ``auto``: pure one-shot. For ``interactive``: a textual REPL via stdin.
  For ``resume``: Surface the latest verdict + checklist of tools.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys

import typer

from .agents import Director, Editor, Planner
from .agents.logging_setup import configure_logging, get_logger
from .interactive import InteractiveSession, run_repl
from .llm import LLMClient, get_llm_client
from .mcp_client import StdioResolveClient
from .pipeline import Orchestrator, ResumeError
from .settings import DirectorSettings
from .store import EventLog, RunStore

app = typer.Typer(no_args_is_help=True, help="Director orchestrator commands.")
inspect_app = typer.Typer(help="Operations on past runs (list, show).")
app.add_typer(inspect_app, name="run")

logger = get_logger("director.cli")


def _director_settings() -> DirectorSettings:
    return DirectorSettings()


def _store_paths(settings: DirectorSettings) -> tuple[pathlib.Path, pathlib.Path]:
    base = pathlib.Path(settings.run_store_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base / "runs.sqlite", base / "events.jsonl"


async def _spawn_client(backend: str, uv_project: str | None) -> StdioResolveClient:
    """Spawn a resolve-mcp server subprocess for the requested backend."""
    if backend not in {"fake", "davinci"}:
        typer.echo(f"unsupported backend {backend!r}", err=True)
        raise typer.Exit(code=2)
    client = StdioResolveClient.default(
        backend=backend,
        allow_destructive=False,
        log_level="WARNING",
        uv_project=uv_project,
    )
    await client.start()
    return client


async def _auto_async(
    *,
    clips_dir: pathlib.Path,
    music_path: pathlib.Path | None,
    prompt: str,
    backend: str,
    uv_project: str | None,
    fast: bool,
    llm_choice: str | None = None,
) -> None:
    settings = _director_settings()
    configure_logging("INFO")
    if not clips_dir.is_dir():
        typer.echo(f"clips dir not found: {clips_dir}", err=True)
        raise typer.Exit(code=1)
    clip_paths = sorted(str(p) for p in clips_dir.iterdir() if p.is_file())
    if not clip_paths:
        typer.echo(f"no clips found in {clips_dir}", err=True)
        raise typer.Exit(code=1)
    if music_path is not None and not music_path.exists():
        typer.echo(f"music track not found: {music_path}", err=True)
        raise typer.Exit(code=1)

    llm: LLMClient | None
    if fast:
        llm = None
    else:
        provider = llm_choice or settings.llm_provider
        llm = get_llm_client(settings.model_copy(update={"llm_provider": provider}))

    sqlite_path, event_path = _store_paths(settings)
    store = RunStore(sqlite_path)
    log = EventLog(event_path)

    client = await _spawn_client(backend, uv_project)

    orchestrator = Orchestrator(
        settings=settings,
        llm=llm,
        client=client,
        run_store=store,
        event_log=log,
    )

    try:
        result = await orchestrator.run_auto(
            clip_paths=clip_paths,
            music_path=str(music_path) if music_path else None,
            user_prompt=prompt,
        )
    finally:
        await client.close()
    typer.echo(
        json.dumps(
            {
                "run_id": result.run_id,
                "status": result.status.value,
                "iterations": result.iterations,
                "plan_id": result.plan.plan_id if result.plan else None,
                "verdict": result.verdict.model_dump(mode="json") if result.verdict else None,
            },
            indent=2,
        )
    )


@app.command()
def auto(
    clips_dir: pathlib.Path = typer.Argument(..., help="Directory with source clips."),
    music: pathlib.Path | None = typer.Option(None, "--music", "-m", help="Music track path."),
    prompt: str = typer.Option(
        "high-energy 30s reel",
        "--prompt",
        "-p",
        help="User brief.",
    ),
    backend: str = typer.Option("fake", "--backend", help="Backend for resolve-mcp (fake|davinci)."),
    uv_project: str | None = typer.Option(None, "--uv-project", help="uv project dir for resolve-mcp."),
    fast: bool = typer.Option(
        False,
        "--fast",
        help="Skip Gemini integration (use offline planner + director).",
    ),
    llm_choice: str | None = typer.Option(
        None,
        "--llm",
        help="LLM provider override: gemini | openai_compatible | none.",
    ),
) -> None:
    """Make a beat-synced timeline from clips + music + prompt."""
    asyncio.run(
        _auto_async(
            clips_dir=clips_dir,
            music_path=music,
            prompt=prompt,
            backend=backend,
            uv_project=uv_project,
            fast=fast,
            llm_choice=llm_choice,
        )
    )


async def _interactive_async(
    *,
    backend: str,
    uv_project: str | None,
    fast: bool,
    llm_choice: str | None = None,
) -> None:
    settings = _director_settings()
    configure_logging("INFO")
    llm: LLMClient | None
    if fast:
        llm = None
    else:
        provider = llm_choice or settings.llm_provider
        llm = get_llm_client(settings.model_copy(update={"llm_provider": provider}))
    sqlite_path, event_path = _store_paths(settings)
    store = RunStore(sqlite_path)
    log = EventLog(event_path)

    client = await _spawn_client(backend, uv_project)

    planner = Planner(llm=llm, settings=settings)
    director = Director(llm=llm, settings=settings)
    editor = Editor(
        llm=None,
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

    def printer(s: str) -> None:
        typer.echo(s)

    def stdin_reader() -> str:
        return input("> ")

    try:
        await run_repl(session=session, input_reader=stdin_reader, printer=printer)
    finally:
        await client.close()


@app.command()
def interactive(
    backend: str = typer.Option("fake", "--backend", help="resolve-mcp backend (fake)."),
    uv_project: str | None = typer.Option(None, "--uv-project", help="uv project dir for resolve-mcp."),
    fast: bool = typer.Option(False, "--fast", help="Offline REPL with deterministic planner/director."),
    llm_choice: str | None = typer.Option(
        None,
        "--llm",
        help="LLM provider override: gemini | openai_compatible | none.",
    ),
) -> None:
    """Start an interactive textual REPL against the resolve-mcp server."""
    asyncio.run(
        _interactive_async(backend=backend, uv_project=uv_project, fast=fast, llm_choice=llm_choice)
    )


@inspect_app.command("list")
def run_list() -> None:
    """List known runs in the run store."""
    settings = _director_settings()
    sqlite_path, _ = _store_paths(settings)
    store = RunStore(sqlite_path)
    runs = []
    with store._cur() as cur:
        cur.execute("SELECT run_id, status, mode, iterations, last_verdict, created_at FROM runs ORDER BY created_at DESC LIMIT 50")
        for r in cur.fetchall():
            runs.append(dict(r))
    typer.echo(json.dumps(runs, indent=2))
    store.close()


@inspect_app.command("show")
def run_show(run_id: str = typer.Argument(...)) -> None:
    """Show a run's record + verdicts + tool calls."""
    settings = _director_settings()
    sqlite_path, _ = _store_paths(settings)
    store = RunStore(sqlite_path)
    rec = store.get_run(run_id)
    if rec is None:
        typer.echo(f"run {run_id} not found", err=True)
        raise typer.Exit(code=1)
    out: dict[str, object] = {
        "run": rec.model_dump(mode="json"),
        "verdicts": [v.model_dump(mode="json") for v in store.list_verdicts(run_id)],
        "tool_calls": [tc.model_dump(mode="json") for tc in store.list_tool_calls(run_id)],
    }
    typer.echo(json.dumps(out, indent=2))
    store.close()



@app.command()
def resume(
    run_id: str = typer.Argument(..., help="Run id from `director run list`."),
    backend: str = typer.Option("fake", "--backend", help="Backend for resolve-mcp (fake)."),
    uv_project: str | None = typer.Option(None, "--uv-project", help="uv project dir for resolve-mcp."),
    fast: bool = typer.Option(False, "--fast", help="Offline fallback if the run must fully restart."),
    llm_choice: str | None = typer.Option(None, "--llm", help="LLM provider override (restart path only)."),
) -> None:
    """Resume a stored run at its last agreed state (re-executes the agreed plan)."""
    asyncio.run(_resume_async(run_id=run_id, backend=backend, uv_project=uv_project))


async def _resume_async(*, run_id: str, backend: str, uv_project: str | None) -> None:
    settings = _director_settings()
    configure_logging("INFO")
    sqlite_path, event_path = _store_paths(settings)
    store = RunStore(sqlite_path)
    log = EventLog(event_path)
    record = store.get_run(run_id)
    if record is None:
        typer.echo(f"run {run_id} not found in {sqlite_path}", err=True)
        raise typer.Exit(code=1)

    client = await _spawn_client(backend, uv_project)

    orchestrator = Orchestrator(
        settings=settings,
        llm=None,
        client=client,
        run_store=store,
        event_log=log,
    )
    try:
        result = await orchestrator.resume_run(run_id=run_id)
    except ResumeError as err:
        typer.echo(str(err), err=True)
        raise typer.Exit(code=1) from err
    finally:
        await client.close()
    typer.echo(
        json.dumps(
            {
                "run_id": result.run_id,
                "status": result.status.value,
                "iterations": result.iterations,
                "plan_id": result.plan.plan_id if result.plan else None,
            },
            indent=2,
        )
    )


main = app


if __name__ == "__main__":
    sys.exit(app())
