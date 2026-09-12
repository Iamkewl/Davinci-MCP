"""Director CLI: ``auto`` / ``interactive`` / ``run list|show`` / ``resume``.

We back the CLI with Typer, but the orchestrator itself has no Typer dependency
(tests instantiate it directly).

Bootstrap walk
--------------

* Spawn the ``resolve-mcp`` server over stdio (``--backend fake`` by default, or
  ``davinci`` to drive a running Resolve Studio).
* Build the LLM client from settings/flags — ``--fast`` forces the fully offline
  deterministic path.
* Put the run store under ``DIRECTOR_RUN_STORE_DIR`` (default ``./runs``).
* ``auto`` is one-shot; ``interactive`` is a REPL; ``resume`` re-executes a
  stored run's agreed plan.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys

import typer

from .agents import Director, Editor, Planner
from .agents.logging_setup import configure_logging, get_logger
from .ingestion.media_probe import is_media_file
from .interactive import InteractiveSession, run_repl
from .llm import LLMClient, get_llm_client
from .mcp_client import StdioResolveClient
from .pipeline import DEFAULT_PROJECT, AutoResult, Orchestrator, ResumeError
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


def _build_llm(
    settings: DirectorSettings, *, fast: bool, llm_choice: str | None
) -> LLMClient | None:
    """``--fast`` beats everything; otherwise flag > settings."""
    if fast:
        return None
    provider = llm_choice or settings.llm_provider
    if provider not in {"gemini", "openai_compatible", "none"}:
        typer.echo(
            f"unknown --llm {provider!r}; expected gemini, openai_compatible or none", err=True
        )
        raise typer.Exit(code=2)
    try:
        return get_llm_client(settings.model_copy(update={"llm_provider": provider}))
    except Exception as exc:  # missing key, bad base url, ...
        typer.echo(f"LLM provider {provider!r} unavailable: {exc}", err=True)
        typer.echo("re-run with --fast for the offline deterministic path.", err=True)
        raise typer.Exit(code=2) from exc


async def _spawn_client(backend: str, uv_project: str | None) -> StdioResolveClient:
    """Spawn a resolve-mcp server subprocess for the requested backend."""
    if backend not in {"fake", "davinci"}:
        typer.echo(f"unsupported backend {backend!r}; expected fake or davinci", err=True)
        raise typer.Exit(code=2)
    client = StdioResolveClient.default(
        backend=backend,
        allow_destructive=False,
        log_level="WARNING",
        uv_project=uv_project,
    )
    await client.start()
    return client


def _collect_clips(clips_dir: pathlib.Path) -> list[str]:
    """Media files only — a stray .txt or .DS_Store is not a clip."""
    if not clips_dir.is_dir():
        typer.echo(f"clips dir not found: {clips_dir}", err=True)
        raise typer.Exit(code=1)
    everything = sorted(p for p in clips_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    clips = [str(p) for p in everything if is_media_file(str(p))]
    skipped = [p.name for p in everything if not is_media_file(str(p))]
    if skipped:
        typer.echo(f"skipping {len(skipped)} non-media file(s): {', '.join(skipped[:5])}", err=True)
    if not clips:
        typer.echo(f"no media files found in {clips_dir}", err=True)
        raise typer.Exit(code=1)
    return clips


async def _current_project_name(client: StdioResolveClient) -> str | None:
    try:
        info = await client.call_tool("get_project_info", {})
    except Exception:
        return None
    name = info.get("name") if isinstance(info, dict) else None
    return name if isinstance(name, str) else None


def _auto_report(result: AutoResult) -> dict[str, object]:
    timeline = result.timeline_summary
    payload: dict[str, object] = {
        "run_id": result.run_id,
        "status": result.status.value,
        "iterations": result.iterations,
        "plan_id": result.plan.plan_id if result.plan else None,
        "timeline": {
            "name": result.target_timeline,
            "items": timeline["items"],
            "duration_seconds": timeline["duration_seconds"],
        },
        "verdict": result.verdict.model_dump(mode="json") if result.verdict else None,
    }
    if result.edit_result is not None:
        payload["applied_ops"] = result.edit_result.applied_count
        payload["errors"] = result.edit_result.errors
        payload["warnings"] = result.edit_result.warnings
    if result.issues:
        payload["issues"] = result.issues
    return payload


async def _auto_async(
    *,
    clips_dir: pathlib.Path,
    music_path: pathlib.Path | None,
    prompt: str,
    backend: str,
    uv_project: str | None,
    fast: bool,
    llm_choice: str | None,
    project: str | None,
    timeline: str | None,
    fps: float | None,
) -> None:
    settings = _director_settings()
    configure_logging("INFO")
    clip_paths = _collect_clips(clips_dir)
    if music_path is not None and not music_path.exists():
        typer.echo(f"music track not found: {music_path}", err=True)
        raise typer.Exit(code=1)

    llm = _build_llm(settings, fast=fast, llm_choice=llm_choice)
    sqlite_path, event_path = _store_paths(settings)
    store = RunStore(sqlite_path)
    log = EventLog(event_path)
    client = await _spawn_client(backend, uv_project)

    orchestrator = Orchestrator(
        settings=settings, llm=llm, client=client, run_store=store, event_log=log
    )
    target_project = project or await _current_project_name(client) or DEFAULT_PROJECT
    try:
        result = await orchestrator.run_auto(
            clip_paths=clip_paths,
            music_path=str(music_path) if music_path else None,
            user_prompt=prompt,
            target_project=target_project,
            target_timeline=timeline,
            target_fps=fps,
        )
    finally:
        await client.close()
        store.close()
    typer.echo(json.dumps(_auto_report(result), indent=2))
    if result.status.value == "failed":
        raise typer.Exit(code=1)


@app.command()
def auto(
    clips_dir: pathlib.Path = typer.Argument(..., help="Directory with source clips."),
    music: pathlib.Path | None = typer.Option(None, "--music", "-m", help="Music track path."),
    prompt: str = typer.Option("high-energy 30s reel", "--prompt", "-p", help="User brief."),
    backend: str = typer.Option("fake", "--backend", help="resolve-mcp backend: fake | davinci."),
    uv_project: str | None = typer.Option(
        None, "--uv-project", help="Launch the server via `uv run --project DIR` instead of this interpreter."
    ),
    fast: bool = typer.Option(
        False, "--fast", help="Skip the LLM entirely (deterministic planner + reviewer)."
    ),
    llm_choice: str | None = typer.Option(
        None, "--llm", help="LLM provider override: gemini | openai_compatible | none."
    ),
    project: str | None = typer.Option(
        None, "--project", help="Resolve project to build in (default: the open one, else auto-reel)."
    ),
    timeline: str | None = typer.Option(
        None, "--timeline", help="Timeline name to create (default: a fresh 'Reel <run id>')."
    ),
    fps: float | None = typer.Option(
        None, "--fps", help="Timeline frame rate (default: inferred from the clips)."
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
            project=project,
            timeline=timeline,
            fps=fps,
        )
    )


async def _interactive_async(
    *,
    backend: str,
    uv_project: str | None,
    fast: bool,
    llm_choice: str | None,
    from_run: str | None,
    project: str | None,
    timeline: str | None,
) -> None:
    settings = _director_settings()
    configure_logging("INFO")
    llm = _build_llm(settings, fast=fast, llm_choice=llm_choice)
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

    try:
        target_project = project or await _current_project_name(client) or "interactive-reel"
        target_timeline = timeline
        if from_run:
            # Rebuild a previous run's cut here first, so there is something to refine.
            rebuild = Orchestrator(
                settings=settings, llm=None, client=client, run_store=store, event_log=log
            )
            try:
                rebuilt = await rebuild.resume_run(run_id=from_run)
            except ResumeError as err:
                typer.echo(str(err), err=True)
                raise typer.Exit(code=1) from err
            target_timeline = target_timeline or rebuilt.target_timeline
            typer.echo(
                f"rebuilt run {from_run} onto {rebuilt.target_timeline!r} "
                f"({rebuilt.timeline_summary['items']} items)"
            )
        if target_timeline is None:
            state = await _current_timeline_name(client)
            target_timeline = state or "Interactive"

        session = InteractiveSession(
            settings=settings,
            client=client,
            run_store=store,
            event_log=log,
            planner=planner,
            director=director,
            editor=editor,
            target_project=target_project,
            target_timeline=target_timeline,
        )

        def printer(s: str) -> None:
            typer.echo(s)

        def stdin_reader() -> str:
            return input("> ")

        await run_repl(session=session, input_reader=stdin_reader, printer=printer)
    finally:
        await client.close()
        store.close()


async def _current_timeline_name(client: StdioResolveClient) -> str | None:
    try:
        state = await client.call_tool("get_timeline_state", {})
    except Exception:
        return None
    name = state.get("name") if isinstance(state, dict) else None
    return name if isinstance(name, str) else None


@app.command()
def interactive(
    backend: str = typer.Option("fake", "--backend", help="resolve-mcp backend: fake | davinci."),
    uv_project: str | None = typer.Option(None, "--uv-project", help="uv project dir for resolve-mcp."),
    fast: bool = typer.Option(False, "--fast", help="Offline REPL with the deterministic planner."),
    llm_choice: str | None = typer.Option(
        None, "--llm", help="LLM provider override: gemini | openai_compatible | none."
    ),
    from_run: str | None = typer.Option(
        None, "--from-run", help="Rebuild this run's cut first, then refine it (see `run list`)."
    ),
    project: str | None = typer.Option(None, "--project", help="Resolve project to work in."),
    timeline: str | None = typer.Option(None, "--timeline", help="Timeline to refine."),
) -> None:
    """Start an interactive REPL against the resolve-mcp server."""
    asyncio.run(
        _interactive_async(
            backend=backend,
            uv_project=uv_project,
            fast=fast,
            llm_choice=llm_choice,
            from_run=from_run,
            project=project,
            timeline=timeline,
        )
    )


@inspect_app.command("list")
def run_list() -> None:
    """List known runs in the run store."""
    settings = _director_settings()
    sqlite_path, _ = _store_paths(settings)
    store = RunStore(sqlite_path)
    runs = []
    with store._cur() as cur:
        cur.execute(
            "SELECT run_id, status, mode, iterations, last_verdict, created_at "
            "FROM runs ORDER BY created_at DESC LIMIT 50"
        )
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
        store.close()
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
    backend: str = typer.Option("fake", "--backend", help="resolve-mcp backend: fake | davinci."),
    uv_project: str | None = typer.Option(None, "--uv-project", help="uv project dir for resolve-mcp."),
    fast: bool = typer.Option(
        False, "--fast", help="Force the offline path if the run has to restart from scratch."
    ),
    llm_choice: str | None = typer.Option(
        None, "--llm", help="LLM provider override (used only if the run restarts)."
    ),
) -> None:
    """Resume a stored run at its last agreed state (re-executes the agreed plan)."""
    asyncio.run(
        _resume_async(
            run_id=run_id,
            backend=backend,
            uv_project=uv_project,
            fast=fast,
            llm_choice=llm_choice,
        )
    )


async def _resume_async(
    *,
    run_id: str,
    backend: str,
    uv_project: str | None,
    fast: bool,
    llm_choice: str | None,
) -> None:
    settings = _director_settings()
    configure_logging("INFO")
    sqlite_path, event_path = _store_paths(settings)
    store = RunStore(sqlite_path)
    log = EventLog(event_path)
    record = store.get_run(run_id)
    if record is None:
        typer.echo(f"run {run_id} not found in {sqlite_path}", err=True)
        store.close()
        raise typer.Exit(code=1)

    # A resume normally replays a stored plan (no model needed); the LLM is only
    # built for the restart path, where the run never produced an executable plan.
    llm = None if record.final_plan_id else _build_llm(settings, fast=fast, llm_choice=llm_choice)
    client = await _spawn_client(backend, uv_project)
    orchestrator = Orchestrator(
        settings=settings, llm=llm, client=client, run_store=store, event_log=log
    )
    try:
        result = await orchestrator.resume_run(run_id=run_id)
    except ResumeError as err:
        typer.echo(str(err), err=True)
        raise typer.Exit(code=1) from err
    finally:
        await client.close()
    typer.echo(json.dumps(_auto_report(result), indent=2))
    store.close()
    if result.status.value == "failed":
        raise typer.Exit(code=1)


main = app


if __name__ == "__main__":
    sys.exit(app())
