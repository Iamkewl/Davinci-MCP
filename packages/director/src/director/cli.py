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
from .ingestion.gemini_client import GeminiClient, get_gemini_client
from .interactive import InteractiveSession, run_repl
from .mcp_client import StdioResolveClient
from .pipeline import Orchestrator
from .settings import DirectorSettings
from .store import EventLog, RunStore

app = typer.Typer(no_args_is_help=True, help="Director orchestrator commands.")
inspect_app = typer.Typer(help="Operations on past runs (resume, list, show).")
app.add_typer(inspect_app, name="run")

logger = get_logger("director.cli")


def _director_settings() -> DirectorSettings:
    return DirectorSettings()


def _store_paths(settings: DirectorSettings) -> tuple[pathlib.Path, pathlib.Path]:
    base = pathlib.Path(settings.run_store_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base / "runs.sqlite", base / "events.jsonl"


async def _auto_async(
    *,
    clips_dir: pathlib.Path,
    music_path: pathlib.Path | None,
    prompt: str,
    backend: str,
    uv_project: str | None,
    fast: bool,
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

    gemini: GeminiClient | None = None if fast else get_gemini_client(settings)

    sqlite_path, event_path = _store_paths(settings)
    store = RunStore(sqlite_path)
    log = EventLog(event_path)

    if backend == "fake":
        client = StdioResolveClient.default(
            backend="fake",
            allow_destructive=False,
            log_level="WARNING",
            uv_project=uv_project,
        )
        await client.start()
    else:
        typer.echo(f"unsupported backend {backend!r}", err=True)
        raise typer.Exit(code=2)

    orchestrator = Orchestrator(
        settings=settings,
        gemini=gemini,
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
        )
    )


async def _interactive_async(
    *,
    backend: str,
    uv_project: str | None,
    fast: bool,
) -> None:
    settings = _director_settings()
    configure_logging("INFO")
    gemini: GeminiClient | None = None if fast else get_gemini_client(settings)
    sqlite_path, event_path = _store_paths(settings)
    store = RunStore(sqlite_path)
    log = EventLog(event_path)

    if backend == "fake":
        client = StdioResolveClient.default(
            backend="fake",
            allow_destructive=False,
            log_level="WARNING",
            uv_project=uv_project,
        )
        await client.start()
    else:
        typer.echo(f"unsupported backend {backend!r}", err=True)
        raise typer.Exit(code=2)

    planner = Planner(gemini=gemini, settings=settings)
    director = Director(gemini=gemini, settings=settings)
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
) -> None:
    """Start an interactive textual REPL against the resolve-mcp server."""
    asyncio.run(
        _interactive_async(backend=backend, uv_project=uv_project, fast=fast)
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


main = app


if __name__ == "__main__":
    sys.exit(app())
