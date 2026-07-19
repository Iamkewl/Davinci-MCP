"""resolve-mcp: FastMCP server with a swappable Resolve backend.

Public entry points:

* :func:`build_server` — returns a configured ``FastMCP`` instance.
* :func:`main` — CLI. Parses flags (``--backend``, ``--allow-destructive``) and runs
  the server over stdio (HTTP/SSE to follow in a later phase).

Design notes
------------

The tool/resource module functions take a backend as their first argument. FastMCP
doesn't support partial application in its schemas, so we use a closure layer in
:func:`build_server` to create per-server wrappers that escape the binding scope.

Every tool here is generated from the corresponding function in ``tools/`` and
``resources/``. The dual registration is intentional: tools/resources carry their
type hints and docstrings as JSON schemas for MCP clients.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from .backend import ResolveBackend
from .davinci_backend import DaVinciResolveBackend
from .fake_backend import FakeResolveBackend
from .logging_setup import configure_logging, get_logger
from .resources import media_pool_resource, project_resource, timeline_resource
from .settings import ResolveMCPSettings
from .tools import (
    add_fade,
    add_marker,
    add_render_job,
    add_transition,
    append_clip,
    create_bin,
    create_project,
    create_timeline,
    delete_clip,
    delete_media,
    delete_timeline,
    get_project_info,
    get_render_status,
    get_timeline_state,
    import_media,
    insert_clip,
    list_media_pool,
    move_clip,
    open_project,
    quit_app,
    restart_app,
    save_project,
    set_composite_mode,
    set_crop,
    set_opacity,
    set_speed,
    set_transform,
    start_render,
)

_LOG = get_logger("resolve_mcp.server")


# --- backend selection --------------------------------------------------------


def select_backend(name: str, *, allow_destructive: bool = False) -> ResolveBackend:
    """Construct a backend impl by name. Raises on unknown name."""
    if name == "fake":
        return FakeResolveBackend(allow_destructive=allow_destructive)
    if name == "davinci":
        return DaVinciResolveBackend()
    msg = f"unknown backend {name!r}; expected 'fake' or 'davinci'"
    raise ValueError(msg)


# --- server builder -----------------------------------------------------------


def build_server(backend: ResolveBackend, *, allow_destructive: bool = False) -> FastMCP:
    """Wire tools + resources around ``backend`` into a fresh ``FastMCP``."""
    server: FastMCP = FastMCP("davinci-resolve", stateless_http=False)

    # Bind to keep tool function objects small and well-named.
    be: ResolveBackend = backend

    # --- project tools (registered with FastMCP) ---

    @server.tool(name="open_project", description="Open an existing DaVinci Resolve project by name.")
    def _open_project(name: str) -> dict[str, Any]:
        return open_project(be, name)

    @server.tool(name="create_project", description="Create a new project; errors if the name already exists.")
    def _create_project(name: str, fps: float, drop_frame: bool = False, width: int = 1920, height: int = 1080) -> dict[str, Any]:
        return create_project(be, name, fps, drop_frame, width, height)

    @server.tool(name="save_project", description="Save the currently-open project.")
    def _save_project() -> dict[str, Any]:
        return save_project(be)

    @server.tool(name="get_project_info", description="Return the currently-open project's metadata.")
    def _get_project_info() -> dict[str, Any]:
        return get_project_info(be)

    # --- media pool tools ---

    @server.tool(name="import_media", description="Import media files into the project's media pool.")
    def _import_media(paths: list[str], bin: str | None = None) -> list[dict[str, Any]]:
        return import_media(be, paths, bin)

    @server.tool(name="list_media_pool", description="Return the full state of the media pool.")
    def _list_media_pool() -> dict[str, Any]:
        return list_media_pool(be)

    @server.tool(name="create_bin", description="Create a new bin in the media pool.")
    def _create_bin(name: str) -> dict[str, Any]:
        return create_bin(be, name)

    # --- timeline tools ---

    @server.tool(name="create_timeline", description="Create a new timeline and make it current.")
    def _create_timeline(name: str, fps: float, drop_frame: bool = False) -> dict[str, Any]:
        return create_timeline(be, name, fps, drop_frame)

    @server.tool(name="get_timeline_state", description="Return full state of the current timeline.")
    def _get_timeline_state() -> dict[str, Any]:
        return get_timeline_state(be)

    @server.tool(name="append_clip", description="Append a media clip onto a timeline track.")
    def _append_clip(
        media_clip_id: str,
        timeline_track_index: int = 0,
        start_seconds: float = 0.0,
        duration_seconds: float = 1.0,
        source_in_seconds: float = 0.0,
    ) -> dict[str, Any]:
        return append_clip(
            be,
            media_clip_id=media_clip_id,
            timeline_track_index=timeline_track_index,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            source_in_seconds=source_in_seconds,
        )

    @server.tool(name="insert_clip", description="Insert a clip at a position; later clips shift right.")
    def _insert_clip(
        media_clip_id: str,
        timeline_track_index: int,
        timeline_position_seconds: float,
        duration_seconds: float,
        source_in_seconds: float = 0.0,
    ) -> dict[str, Any]:
        return insert_clip(
            be,
            media_clip_id=media_clip_id,
            timeline_track_index=timeline_track_index,
            timeline_position_seconds=timeline_position_seconds,
            duration_seconds=duration_seconds,
            source_in_seconds=source_in_seconds,
        )

    @server.tool(name="delete_clip", description="Delete a timeline item by id. (No destructive gate in Phase 1.)")
    def _delete_clip(timeline_item_id: str) -> dict[str, Any]:
        return delete_clip(be, timeline_item_id)

    @server.tool(name="move_clip", description="Reposition a timeline item.")
    def _move_clip(timeline_item_id: str, new_position_seconds: float) -> dict[str, Any]:
        return move_clip(be, timeline_item_id=timeline_item_id, new_position_seconds=new_position_seconds)

    # --- Phase 2: per-item tools ---

    @server.tool(name="set_transform", description="Set the transform on a timeline item.")
    def _set_transform(
        timeline_item_id: str,
        pan_x: float,
        pan_y: float,
        zoom_x: float,
        zoom_y: float,
        rotation: float,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> dict[str, Any]:
        return set_transform(
            be,
            timeline_item_id=timeline_item_id,
            pan_x=pan_x,
            pan_y=pan_y,
            zoom_x=zoom_x,
            zoom_y=zoom_y,
            rotation=rotation,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
        )

    @server.tool(name="set_crop", description="Set crop on a timeline item.")
    def _set_crop(
        timeline_item_id: str,
        left: float,
        right: float,
        top: float,
        bottom: float,
    ) -> dict[str, Any]:
        return set_crop(be, timeline_item_id=timeline_item_id, left=left, right=right, top=top, bottom=bottom)

    @server.tool(name="set_composite_mode", description="Set composite/blending mode on a timeline item.")
    def _set_composite_mode(timeline_item_id: str, mode: str) -> dict[str, Any]:
        return set_composite_mode(be, timeline_item_id=timeline_item_id, mode=mode)

    @server.tool(name="set_opacity", description="Set opacity on a timeline item (0.0..1.0).")
    def _set_opacity(timeline_item_id: str, opacity: float) -> dict[str, Any]:
        return set_opacity(be, timeline_item_id=timeline_item_id, opacity=opacity)

    @server.tool(name="add_fade", description="Set fade-in/out durations on a timeline item.")
    def _add_fade(timeline_item_id: str, fade_in_seconds: float, fade_out_seconds: float) -> dict[str, Any]:
        return add_fade(be, timeline_item_id=timeline_item_id,
                        fade_in_seconds=fade_in_seconds,
                        fade_out_seconds=fade_out_seconds)

    @server.tool(name="set_speed", description="Set playback speed multiplier on a timeline item (>0).")
    def _set_speed(timeline_item_id: str, speed: float) -> dict[str, Any]:
        return set_speed(be, timeline_item_id=timeline_item_id, speed=speed)

    @server.tool(name="add_marker", description="Add a point marker on a timeline item.")
    def _add_marker(
        timeline_item_id: str,
        position_seconds: float,
        label: str,
        color: str,
        note: str = "",
    ) -> dict[str, Any]:
        return add_marker(
            be,
            timeline_item_id=timeline_item_id,
            position_seconds=position_seconds,
            label=label,
            color=color,
            note=note,
        )

    @server.tool(name="add_transition", description="Attach a transition to a timeline item.")
    def _add_transition(
        timeline_item_id: str,
        track_index: int,
        style: str,
        duration_seconds: float,
        alignment: str,
    ) -> dict[str, Any]:
        return add_transition(
            be,
            timeline_item_id=timeline_item_id,
            track_index=track_index,
            style=style,
            duration_seconds=duration_seconds,
            alignment=alignment,
        )

    # --- Phase 2: render tools ---

    @server.tool(name="add_render_job", description="Queue a render job for a timeline.")
    def _add_render_job(timeline_name: str, format: str, output_path: str) -> dict[str, Any]:
        return add_render_job(be, timeline_name=timeline_name, format=format, output_path=output_path)

    @server.tool(name="start_render", description="Kick off a queued render job.")
    def _start_render(job_id: str) -> dict[str, Any]:
        return start_render(be, job_id=job_id)

    @server.tool(name="get_render_status", description="Return the render-job record for a given id.")
    def _get_render_status(job_id: str) -> dict[str, Any]:
        return get_render_status(be, job_id=job_id)

    # --- Phase 2: destructive (gated) ---
    # Only registered when ``allow_destructive=True`` AND we still require
    # ``confirm=true`` at call time. The wrappers below ALWAYS require confirm=True
    # from the caller; the gate flag controls whether the tool is registered at all.

    if allow_destructive:
        @server.tool(name="quit_app", description="Quit DaVinci Resolve. Requires confirm=true.")
        def _quit_app(confirm: bool) -> dict[str, Any]:
            return quit_app(be, confirm=confirm)

        @server.tool(name="restart_app", description="Restart DaVinci Resolve. Requires confirm=true.")
        def _restart_app(confirm: bool) -> dict[str, Any]:
            return restart_app(be, confirm=confirm)

        @server.tool(name="delete_timeline", description="Delete a timeline. Requires confirm=true.")
        def _delete_timeline(name: str, confirm: bool) -> dict[str, Any]:
            return delete_timeline(be, name=name, confirm=confirm)

        @server.tool(name="delete_media", description="Delete a media clip. Requires confirm=true.")
        def _delete_media(media_clip_id: str, confirm: bool) -> dict[str, Any]:
            return delete_media(be, media_clip_id=media_clip_id, confirm=confirm)

    # --- resources ---

    @server.resource("resolve://project")
    def _resource_project() -> str:
        return project_resource(be)

    @server.resource("resolve://media-pool")
    def _resource_media_pool() -> str:
        return media_pool_resource(be)

    @server.resource("resolve://timeline/current")
    def _resource_timeline() -> str:
        return timeline_resource(be)

    _LOG.info("server.built", backend=type(be).__name__, allow_destructive=allow_destructive)
    return server


# --- CLI ----------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="resolve-mcp",
        description="MCP server for DaVinci Resolve. Default backend is 'fake' (no Resolve needed).",
    )
    parser.add_argument(
        "--backend",
        choices=("fake", "davinci"),
        default="fake",
        help="Backend implementation. 'fake' is in-memory, 'davinci' connects to a running Resolve.",
    )
    parser.add_argument(
        "--allow-destructive",
        action="store_true",
        help="Enable destructive tools (quit_app, restart_app, delete_timeline, delete_media).",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio",),
        default="stdio",
        help="Transport. (HTTP/SSE in a later phase.)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    settings = ResolveMCPSettings(
        transport=args.transport,
        allow_destructive=args.allow_destructive,
        backend=args.backend,
        log_level=args.log_level,
    )
    configure_logging(settings.log_level)
    try:
        backend = select_backend(settings.backend, allow_destructive=settings.allow_destructive)
    except ValueError as exc:
        _LOG.error("backend.selection_failed", error=str(exc))
        return 2
    server = build_server(backend, allow_destructive=settings.allow_destructive)
    _LOG.info("server.start", transport=settings.transport, allow_destructive=settings.allow_destructive)
    if settings.transport == "stdio":
        server.run(transport="stdio")
    else:
        _LOG.error("transport.unsupported", transport=settings.transport)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
