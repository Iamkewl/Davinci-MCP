"""resolve-mcp: FastMCP server with a swappable Resolve backend.

Public entry points:

* :func:`build_server` — returns a configured ``FastMCP`` instance.
* :func:`main` — CLI. Parses flags (``--backend``, ``--allow-destructive``, …) and
  runs the server over stdio.

Design notes
------------

The tool/resource module functions take a backend as their first argument. FastMCP
doesn't support partial application in its schemas, so we use a closure layer in
:func:`build_server` to create per-server wrappers that escape the binding scope.

Wrapper signatures are the MCP contract: they use the real enums and bounded
``Annotated`` types so the generated JSON schema advertises legal values
(``"enum": [...]``, ``minimum``, …) instead of a bare ``string``/``number``.

Tools a backend cannot honor are **not registered at all** — a client's
``list_tools`` therefore reflects what will actually work. The live DaVinci
backend hides ``add_fade``, ``set_speed``, ``add_transition`` and ``restart_app``
because Resolve's documented scripting API has no entry point for them.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Annotated, Any, TypeVar

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from .backend import ResolveBackend
from .davinci_backend import DaVinciResolveBackend
from .fake_backend import FakeResolveBackend
from .logging_setup import configure_logging, get_logger
from .resources import media_pool_resource, project_resource, timeline_resource
from .schemas import (
    CompositeMode,
    MarkerColor,
    RenderJobFormat,
    TransitionAlignment,
    TransitionStyle,
)
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
    list_projects,
    list_timelines,
    move_clip,
    open_project,
    quit_app,
    restart_app,
    save_project,
    set_composite_mode,
    set_crop,
    set_current_timeline,
    set_opacity,
    set_speed,
    set_transform,
    start_render,
)

_LOG = get_logger("resolve_mcp.server")

SERVER_INSTRUCTIONS = """\
Drive DaVinci Resolve: media pool, timelines, per-item transforms, markers and renders.

Conventions:
* Times are seconds from the start of the timeline (floats); the backend snaps them
  to whole frames and reports the snapped value back.
* Track indexes are 1-based across video tracks first, then audio; on a fresh
  timeline 1 = V1 and 2 = A1. get_timeline_state().tracks lists the real mapping.
* Transform/crop values use Resolve's Inspector units: pan/anchor in pixels from
  centre, zoom as a multiplier (1.0 = 100%), rotation in degrees, crop in pixels.
* Marker positions are relative to the start of their item.
* Every mutating tool returns a before/after state delta so you can verify the edit
  landed rather than assuming it did. Ids in the delta's id_remap replaced ids that
  the operation had to recreate.

Typical flow: create_project (or open_project) -> import_media -> create_timeline ->
append_clip per cut -> per-item tweaks -> add_render_job -> start_render.\
"""

# --- MCP parameter types (these become the advertised JSON schema) --------------

TrackIndexArg = Annotated[
    int,
    Field(ge=1, description="1-based track index: video tracks first, then audio (1 = V1, 2 = A1)."),
]
Seconds = Annotated[float, Field(ge=0.0, description="Seconds from the start of the timeline.")]
PositiveSeconds = Annotated[float, Field(gt=0.0, description="Duration in seconds (> 0).")]
SourceSeconds = Annotated[float, Field(ge=0.0, description="Offset inside the source clip, in seconds.")]
FpsArg = Annotated[float, Field(gt=0.0, le=240.0, description="Frames per second, e.g. 24, 25, 29.97.")]
PixelsArg = Annotated[float, Field(description="Pixels, offset from the frame centre.")]
CropPixels = Annotated[float, Field(ge=0.0, description="Pixels cropped from this edge.")]
ZoomArg = Annotated[float, Field(gt=0.0, le=100.0, description="Scale multiplier; 1.0 = 100%.")]
RotationArg = Annotated[float, Field(ge=-360.0, le=360.0, description="Degrees.")]
OpacityArg = Annotated[float, Field(ge=0.0, le=1.0, description="0.0 transparent .. 1.0 opaque.")]
SpeedArg = Annotated[float, Field(gt=0.0, description="Playback multiplier; 1.0 = normal.")]
ResolutionArg = Annotated[int, Field(gt=0, le=16384, description="Pixels.")]
ConfirmArg = Annotated[bool, Field(description="Must be true; destructive operations never assume consent.")]

_F = TypeVar("_F", bound=Callable[..., Any])


# --- backend selection --------------------------------------------------------


def select_backend(name: str, *, allow_destructive: bool = False) -> ResolveBackend:
    """Construct a backend impl by name. Raises on unknown name."""
    if name == "fake":
        return FakeResolveBackend(allow_destructive=allow_destructive)
    if name == "davinci":
        return DaVinciResolveBackend(allow_destructive=allow_destructive)
    msg = f"unknown backend {name!r}; expected 'fake' or 'davinci'"
    raise ValueError(msg)


# --- server builder -----------------------------------------------------------


def build_server(backend: ResolveBackend, *, allow_destructive: bool = False) -> FastMCP:
    """Wire tools + resources around ``backend`` into a fresh ``FastMCP``."""
    server: FastMCP = FastMCP(
        "davinci-resolve", instructions=SERVER_INSTRUCTIONS, stateless_http=False
    )

    # Bind to keep tool function objects small and well-named.
    be: ResolveBackend = backend
    unsupported = frozenset(be.unsupported_tools())
    if unsupported:
        _LOG.info(
            "server.tools_unsupported",
            backend=type(be).__name__,
            tools=sorted(unsupported),
        )

    def tool(name: str, description: str) -> Callable[[_F], _F]:
        """Register a tool unless this backend cannot honor it."""

        def decorate(fn: _F) -> _F:
            if name in unsupported:
                return fn
            server.tool(name=name, description=description)(fn)
            return fn

        return decorate

    # --- project tools ---

    @tool("open_project", "Open an existing DaVinci Resolve project by name.")
    def _open_project(name: str) -> dict[str, Any]:
        return open_project(be, name)

    @tool("create_project", "Create a new project; errors if the name already exists.")
    def _create_project(
        name: str,
        fps: FpsArg,
        drop_frame: bool = False,
        width: ResolutionArg = 1920,
        height: ResolutionArg = 1080,
    ) -> dict[str, Any]:
        return create_project(be, name, fps, drop_frame, width, height)

    @tool("list_projects", "List projects in the current project-manager folder.")
    def _list_projects() -> dict[str, Any]:
        return list_projects(be)

    @tool("save_project", "Save the currently-open project.")
    def _save_project() -> dict[str, Any]:
        return save_project(be)

    @tool("get_project_info", "Return the currently-open project's metadata.")
    def _get_project_info() -> dict[str, Any]:
        return get_project_info(be)

    # --- media pool tools ---

    @tool(
        "import_media",
        "Import media files into the media pool. Returns one clip record per file, "
        "including the id later calls use and the real duration where known.",
    )
    def _import_media(paths: list[str], bin: str | None = None) -> list[dict[str, Any]]:
        return import_media(be, paths, bin)

    @tool("list_media_pool", "Return the media pool: bins and clips with ids, paths and durations.")
    def _list_media_pool() -> dict[str, Any]:
        return list_media_pool(be)

    @tool("create_bin", "Create a new bin in the media pool.")
    def _create_bin(name: str) -> dict[str, Any]:
        return create_bin(be, name)

    # --- timeline tools ---

    @tool("create_timeline", "Create an empty timeline and make it current.")
    def _create_timeline(name: str, fps: FpsArg, drop_frame: bool = False) -> dict[str, Any]:
        return create_timeline(be, name, fps, drop_frame)

    @tool("list_timelines", "List the project's timelines and which one is current.")
    def _list_timelines() -> dict[str, Any]:
        return list_timelines(be)

    @tool("set_current_timeline", "Make an existing timeline current; later edits apply to it.")
    def _set_current_timeline(name: str) -> dict[str, Any]:
        return set_current_timeline(be, name)

    @tool(
        "get_timeline_state",
        "Full state of the current timeline: tracks, items with ids/positions/"
        "transforms/markers, and the track index mapping.",
    )
    def _get_timeline_state() -> dict[str, Any]:
        return get_timeline_state(be)

    @tool(
        "append_clip",
        "Place a media-pool clip on a timeline track at a given position. "
        "Fails rather than overlapping an existing clip.",
    )
    def _append_clip(
        media_clip_id: str,
        duration_seconds: PositiveSeconds,
        timeline_track_index: TrackIndexArg = 1,
        start_seconds: Seconds = 0.0,
        source_in_seconds: SourceSeconds = 0.0,
    ) -> dict[str, Any]:
        return append_clip(
            be,
            media_clip_id=media_clip_id,
            timeline_track_index=timeline_track_index,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            source_in_seconds=source_in_seconds,
        )

    @tool(
        "insert_clip",
        "Ripple-insert a clip: items at or after the position on that track shift "
        "right by the inserted duration. The position must be a clip boundary.",
    )
    def _insert_clip(
        media_clip_id: str,
        timeline_track_index: TrackIndexArg,
        timeline_position_seconds: Seconds,
        duration_seconds: PositiveSeconds,
        source_in_seconds: SourceSeconds = 0.0,
    ) -> dict[str, Any]:
        return insert_clip(
            be,
            media_clip_id=media_clip_id,
            timeline_track_index=timeline_track_index,
            timeline_position_seconds=timeline_position_seconds,
            duration_seconds=duration_seconds,
            source_in_seconds=source_in_seconds,
        )

    @tool("delete_clip", "Remove one item from the timeline; the media stays in the pool.")
    def _delete_clip(timeline_item_id: str) -> dict[str, Any]:
        return delete_clip(be, timeline_item_id)

    @tool(
        "move_clip",
        "Move a timeline item to a new position. On live Resolve the item is "
        "recreated, so its id changes (see id_remap in the returned delta).",
    )
    def _move_clip(timeline_item_id: str, new_position_seconds: Seconds) -> dict[str, Any]:
        return move_clip(be, timeline_item_id=timeline_item_id, new_position_seconds=new_position_seconds)

    # --- per-item tools ---

    @tool(
        "set_transform",
        "Set pan/zoom/rotation on a timeline item, in Resolve Inspector units "
        "(pan and anchor in pixels from centre, zoom 1.0 = 100%, rotation degrees).",
    )
    def _set_transform(
        timeline_item_id: str,
        pan_x: PixelsArg = 0.0,
        pan_y: PixelsArg = 0.0,
        zoom_x: ZoomArg = 1.0,
        zoom_y: ZoomArg = 1.0,
        rotation: RotationArg = 0.0,
        anchor_x: PixelsArg = 0.0,
        anchor_y: PixelsArg = 0.0,
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

    @tool("set_crop", "Crop a timeline item by pixels removed from each edge.")
    def _set_crop(
        timeline_item_id: str,
        left: CropPixels = 0.0,
        right: CropPixels = 0.0,
        top: CropPixels = 0.0,
        bottom: CropPixels = 0.0,
    ) -> dict[str, Any]:
        return set_crop(be, timeline_item_id=timeline_item_id, left=left, right=right, top=top, bottom=bottom)

    @tool("set_composite_mode", "Set the composite (blend) mode of a timeline item.")
    def _set_composite_mode(timeline_item_id: str, mode: CompositeMode) -> dict[str, Any]:
        return set_composite_mode(be, timeline_item_id=timeline_item_id, mode=mode.value)

    @tool("set_opacity", "Set a timeline item's opacity (0.0 transparent .. 1.0 opaque).")
    def _set_opacity(timeline_item_id: str, opacity: OpacityArg) -> dict[str, Any]:
        return set_opacity(be, timeline_item_id=timeline_item_id, opacity=opacity)

    @tool(
        "add_fade",
        "Set fade-in/out durations on a timeline item. Fake backend only: Resolve's "
        "scripting API has no fade handles.",
    )
    def _add_fade(
        timeline_item_id: str,
        fade_in_seconds: Seconds = 0.0,
        fade_out_seconds: Seconds = 0.0,
    ) -> dict[str, Any]:
        return add_fade(
            be,
            timeline_item_id=timeline_item_id,
            fade_in_seconds=fade_in_seconds,
            fade_out_seconds=fade_out_seconds,
        )

    @tool(
        "set_speed",
        "Set a timeline item's playback speed. Fake backend only: retiming is not in "
        "Resolve's documented scripting API.",
    )
    def _set_speed(timeline_item_id: str, speed: SpeedArg) -> dict[str, Any]:
        return set_speed(be, timeline_item_id=timeline_item_id, speed=speed)

    @tool("add_marker", "Add a point marker on a timeline item, positioned from the item's start.")
    def _add_marker(
        timeline_item_id: str,
        position_seconds: Seconds,
        label: str,
        color: MarkerColor = MarkerColor.BLUE,
        note: str = "",
    ) -> dict[str, Any]:
        return add_marker(
            be,
            timeline_item_id=timeline_item_id,
            position_seconds=position_seconds,
            label=label,
            color=color.value,
            note=note,
        )

    @tool(
        "add_transition",
        "Attach a transition to a timeline item. Fake backend only: Resolve's "
        "documented scripting API cannot create transitions.",
    )
    def _add_transition(
        timeline_item_id: str,
        track_index: TrackIndexArg,
        duration_seconds: PositiveSeconds,
        style: TransitionStyle = TransitionStyle.CROSS_DISSOLVE,
        alignment: TransitionAlignment = TransitionAlignment.MID,
    ) -> dict[str, Any]:
        return add_transition(
            be,
            timeline_item_id=timeline_item_id,
            track_index=track_index,
            style=style.value,
            duration_seconds=duration_seconds,
            alignment=alignment.value,
        )

    # --- render tools ---

    @tool("add_render_job", "Queue a render job for a timeline and return its job record.")
    def _add_render_job(
        timeline_name: str,
        output_path: str,
        format: RenderJobFormat = RenderJobFormat.MP4,
    ) -> dict[str, Any]:
        return add_render_job(be, timeline_name=timeline_name, format=format.value, output_path=output_path)

    @tool("start_render", "Start a queued render job.")
    def _start_render(job_id: str) -> dict[str, Any]:
        return start_render(be, job_id=job_id)

    @tool("get_render_status", "Return a render job's status and progress.")
    def _get_render_status(job_id: str) -> dict[str, Any]:
        return get_render_status(be, job_id=job_id)

    # --- destructive (double-gated: server flag AND confirm=true) ---

    if allow_destructive:

        @tool("quit_app", "Quit DaVinci Resolve. Requires confirm=true.")
        def _quit_app(confirm: ConfirmArg) -> dict[str, Any]:
            return quit_app(be, confirm=confirm)

        @tool("restart_app", "Restart the app (fake backend only). Requires confirm=true.")
        def _restart_app(confirm: ConfirmArg) -> dict[str, Any]:
            return restart_app(be, confirm=confirm)

        @tool("delete_timeline", "Delete a timeline. Requires confirm=true.")
        def _delete_timeline(name: str, confirm: ConfirmArg) -> dict[str, Any]:
            return delete_timeline(be, name=name, confirm=confirm)

        @tool("delete_media", "Delete a clip from the media pool. Requires confirm=true.")
        def _delete_media(media_clip_id: str, confirm: ConfirmArg) -> dict[str, Any]:
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
    """Flags default to ``None`` so unset flags fall through to RESOLVE_MCP_* env/.env."""
    parser = argparse.ArgumentParser(
        prog="resolve-mcp",
        description=(
            "MCP server for DaVinci Resolve. Precedence: CLI flag > RESOLVE_MCP_* env "
            "or .env > default. Default backend is 'fake' (no Resolve needed)."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("fake", "davinci"),
        default=None,
        help="Backend implementation. 'fake' is in-memory, 'davinci' drives a running Resolve. [default: fake]",
    )
    parser.add_argument(
        "--allow-destructive",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Register destructive tools (quit_app, restart_app, delete_timeline, delete_media). [default: off]",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio",),
        default=None,
        help="Transport (stdio only). [default: stdio]",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level: DEBUG, INFO, WARNING, ERROR. Logs go to stderr. [default: INFO]",
    )
    return parser


def settings_from_argv(argv: list[str] | None = None) -> ResolveMCPSettings:
    """Resolve settings with precedence: CLI flag > RESOLVE_MCP_* env/.env > default.

    Only flags the user actually passed become overrides; anything left as ``None``
    falls through to pydantic-settings, which is what makes the env vars work.
    """
    args = _build_arg_parser().parse_args(argv)
    overrides = {
        key: value
        for key, value in (
            ("transport", args.transport),
            ("allow_destructive", args.allow_destructive),
            ("backend", args.backend),
            ("log_level", args.log_level),
        )
        if value is not None
    }
    return ResolveMCPSettings(**overrides)


def main(argv: list[str] | None = None) -> int:
    settings = settings_from_argv(argv)
    configure_logging(settings.log_level)
    try:
        backend = select_backend(settings.backend, allow_destructive=settings.allow_destructive)
    except ValueError as exc:
        _LOG.error("backend.selection_failed", error=str(exc))
        return 2
    if settings.transport != "stdio":
        _LOG.error("transport.unsupported", transport=settings.transport)
        return 3
    server = build_server(backend, allow_destructive=settings.allow_destructive)
    _LOG.info("server.start", transport=settings.transport, allow_destructive=settings.allow_destructive)
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
