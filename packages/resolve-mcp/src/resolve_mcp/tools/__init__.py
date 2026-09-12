"""MCP tools for projects, media pool, and timelines.

Each function below is registered by :mod:`resolve_mcp.server` as an MCP tool. They
take their backend as their first arg (``backend``); the server wires the chosen
impl in. We expose **one tool per operation** (no action-dispatch strings) and every
mutating tool returns the resulting :class:`StateDelta` so callers can verify the
edit landed — directly addressing the silent-success failure mode of the prior build.

Notes on argument style
-----------------------
We use TypeDoc-style docstrings + plain Python defaults. FastMCP's schema
generation reads from type hints + docstrings; we deliberately avoid Pydantic
``Field(...)`` defaults because they shadow real Python defaults when tools are
called in-process by tests.
"""

from __future__ import annotations

from typing import Any, cast

from ..backend import ResolveBackend

# --- project tools -------------------------------------------------------------


def open_project(backend: ResolveBackend, name: str) -> dict[str, Any]:
    """Open an existing project.

    Args:
        name: Project name to open.
    """
    info = backend.open_project(name)
    return cast("dict[str, Any]", info.model_dump(mode="json"))


def create_project(
    backend: ResolveBackend,
    name: str,
    fps: float,
    drop_frame: bool = False,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Create a new project. Errors if the name already exists.

    Args:
        name: Unique project name.
        fps: Frame rate as a float (e.g., 24, 25, 29.97).
        drop_frame: True for SMPTE drop-frame timecode (29.97/59.94 only).
        width: Resolution width in pixels.
        height: Resolution height in pixels.
    """
    from ..schemas import FrameRate

    info = backend.create_project(name, FrameRate(fps=fps, drop_frame=drop_frame), width, height)
    return cast("dict[str, Any]", info.model_dump(mode="json"))


def list_projects(backend: ResolveBackend) -> dict[str, Any]:
    """List the projects in the current project-manager folder."""
    return {"projects": list(backend.list_projects())}


def save_project(backend: ResolveBackend) -> dict[str, Any]:
    """Persist the current project."""
    return cast("dict[str, Any]", backend.save_project().model_dump(mode="json"))


def get_project_info(backend: ResolveBackend) -> dict[str, Any]:
    """Return a snapshot of the currently-open project."""
    return cast("dict[str, Any]", backend.current_project().model_dump(mode="json"))


# --- media tools ---------------------------------------------------------------


def import_media(
    backend: ResolveBackend,
    paths: list[str],
    bin: str | None = None,
) -> list[dict[str, Any]]:
    """Import media files into the media pool.

    Args:
        paths: Filesystem paths to import.
        bin: Target bin name; defaults to 'Master'.
    """
    clips = backend.import_media(paths, bin)
    return [cast("dict[str, Any]", c.model_dump(mode="json")) for c in clips]


def list_media_pool(backend: ResolveBackend) -> dict[str, Any]:
    """Return the full media-pool state: bins + clips."""
    return cast("dict[str, Any]", backend.list_media_pool().model_dump(mode="json"))


def create_bin(backend: ResolveBackend, name: str) -> dict[str, Any]:
    """Create a bin in the media pool.

    Args:
        name: Unique bin name within the project.
    """
    return cast("dict[str, Any]", backend.create_bin(name).model_dump(mode="json"))


# --- timeline tools ------------------------------------------------------------


def create_timeline(
    backend: ResolveBackend,
    name: str,
    fps: float,
    drop_frame: bool = False,
) -> dict[str, Any]:
    """Create a new timeline and make it current.

    Args:
        name: Unique timeline name within the project.
        fps: Timeline frame rate as a float.
        drop_frame: Use SMPTE drop-frame if 29.97 / 59.94.
    """
    from ..schemas import FrameRate

    tl = backend.create_timeline(name, FrameRate(fps=fps, drop_frame=drop_frame))
    return cast("dict[str, Any]", tl.model_dump(mode="json"))


def list_timelines(backend: ResolveBackend) -> dict[str, Any]:
    """List the timelines in the current project and which one is current."""
    return {"timelines": [t.model_dump(mode="json") for t in backend.list_timelines()]}


def set_current_timeline(backend: ResolveBackend, name: str) -> dict[str, Any]:
    """Make an existing timeline current; later edits apply to it.

    Args:
        name: Timeline name as reported by list_timelines.
    """
    state = backend.set_current_timeline(name)
    return cast("dict[str, Any]", state.model_dump(mode="json"))


def get_timeline_state(backend: ResolveBackend) -> dict[str, Any]:
    """Return the full state of the current timeline."""
    return cast("dict[str, Any]", backend.get_timeline_state().model_dump(mode="json"))


def append_clip(
    backend: ResolveBackend,
    media_clip_id: str,
    timeline_track_index: int = 1,
    start_seconds: float = 0.0,
    duration_seconds: float = 1.0,
    source_in_seconds: float = 0.0,
) -> dict[str, Any]:
    """Append a media clip onto a timeline track. Returns the state delta.

    Args:
        media_clip_id: Media-clip id from the media pool.
        timeline_track_index: 1-based; 1 for video track 1, 2 for audio track 1 (Resolve V1/A1 semantics).
        start_seconds: Where in the timeline to place the clip.
        duration_seconds: Length of the clip on the timeline (seconds).
        source_in_seconds: In-point inside the source clip.
    """
    delta = backend.append_clip(
        media_clip_id=media_clip_id,
        timeline_track_index=timeline_track_index,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        source_in_seconds=source_in_seconds,
    )
    return cast("dict[str, Any]", delta.model_dump(mode="json"))


def insert_clip(
    backend: ResolveBackend,
    media_clip_id: str,
    timeline_track_index: int,
    timeline_position_seconds: float,
    duration_seconds: float,
    source_in_seconds: float = 0.0,
) -> dict[str, Any]:
    """Ripple-insert a clip: items at/after the position on that track shift right.

    The position must fall on a clip boundary or empty space; inserting into the
    middle of an existing clip is rejected rather than silently overlapping.

    Args:
        media_clip_id: Media-clip id from the media pool.
        timeline_track_index: 1-based track index (see get_timeline_state.tracks).
        timeline_position_seconds: Where to insert (seconds from timeline start).
        duration_seconds: Length of the new clip (seconds).
        source_in_seconds: In-point inside the source clip (seconds).
    """
    delta = backend.insert_clip(
        media_clip_id=media_clip_id,
        timeline_track_index=timeline_track_index,
        timeline_position_seconds=timeline_position_seconds,
        duration_seconds=duration_seconds,
        source_in_seconds=source_in_seconds,
    )
    return cast("dict[str, Any]", delta.model_dump(mode="json"))


def delete_clip(backend: ResolveBackend, timeline_item_id: str) -> dict[str, Any]:
    """Remove one item from the timeline (the media stays in the pool).

    Args:
        timeline_item_id: Item id to delete, from get_timeline_state.
    """
    delta = backend.delete_clip(timeline_item_id)
    return cast("dict[str, Any]", delta.model_dump(mode="json"))


def move_clip(
    backend: ResolveBackend,
    timeline_item_id: str,
    new_position_seconds: float,
) -> dict[str, Any]:
    """Reposition a timeline item; returns the state delta.

    Resolve has no reposition API, so the live DaVinci backend recreates the
    item at the new position: its id changes and the delta's ``id_remap`` maps
    the old id to the new one. Colour grades and Fusion comps do not survive
    that round-trip.

    Args:
        timeline_item_id: Item id to move.
        new_position_seconds: New start position (seconds from timeline start).
    """
    delta = backend.move_clip(timeline_item_id, new_position_seconds)
    return cast("dict[str, Any]", delta.model_dump(mode="json"))


# --- Phase 2: per-item tool surface -------------------------------------------


def set_transform(
    backend: ResolveBackend,
    timeline_item_id: str,
    pan_x: float,
    pan_y: float,
    zoom_x: float,
    zoom_y: float,
    rotation: float,
    anchor_x: float = 0.5,
    anchor_y: float = 0.5,
) -> dict[str, Any]:
    """Set the transform on a timeline item, in Resolve Inspector units.

    Args:
        timeline_item_id: Item to transform.
        pan_x: Horizontal offset in pixels from centre (Resolve "Pan").
        pan_y: Vertical offset in pixels from centre (Resolve "Tilt").
        zoom_x: Horizontal scale multiplier, 1.0 = 100%.
        zoom_y: Vertical scale multiplier, 1.0 = 100%.
        rotation: Rotation in degrees, -360..360.
        anchor_x: Anchor point offset in pixels from centre.
        anchor_y: Anchor point offset in pixels from centre.
    """
    delta = backend.set_transform(
        timeline_item_id=timeline_item_id,
        pan_x=pan_x,
        pan_y=pan_y,
        zoom_x=zoom_x,
        zoom_y=zoom_y,
        rotation=rotation,
        anchor_x=anchor_x,
        anchor_y=anchor_y,
    )
    return cast("dict[str, Any]", delta.model_dump(mode="json"))


def set_crop(
    backend: ResolveBackend,
    timeline_item_id: str,
    left: float,
    right: float,
    top: float,
    bottom: float,
) -> dict[str, Any]:
    """Crop a timeline item by pixels removed from each edge.

    Args:
        timeline_item_id: Item to crop.
        left: Pixels cropped from the left edge.
        right: Pixels cropped from the right edge.
        top: Pixels cropped from the top edge.
        bottom: Pixels cropped from the bottom edge.
    """
    delta = backend.set_crop(timeline_item_id, left, right, top, bottom)
    return cast("dict[str, Any]", delta.model_dump(mode="json"))


def set_composite_mode(
    backend: ResolveBackend,
    timeline_item_id: str,
    mode: str,
) -> dict[str, Any]:
    """Set the composite (blend) mode of a timeline item.

    Args:
        timeline_item_id: Item to change.
        mode: Blend mode, e.g. "normal", "screen", "multiply", "add".
    """
    delta = backend.set_composite_mode(timeline_item_id, mode)
    return cast("dict[str, Any]", delta.model_dump(mode="json"))


def set_opacity(
    backend: ResolveBackend,
    timeline_item_id: str,
    opacity: float,
) -> dict[str, Any]:
    """Set item opacity.

    Args:
        timeline_item_id: Item to change.
        opacity: 0.0 (transparent) .. 1.0 (opaque).
    """
    delta = backend.set_opacity(timeline_item_id, opacity)
    return cast("dict[str, Any]", delta.model_dump(mode="json"))


def add_fade(
    backend: ResolveBackend,
    timeline_item_id: str,
    fade_in_seconds: float,
    fade_out_seconds: float,
) -> dict[str, Any]:
    """Set fade-in/fade-out durations on a timeline item (fake backend only).

    Resolve's scripting API exposes no fade handles, so the live DaVinci backend
    rejects this call with an "unsupported" error instead of pretending.

    Args:
        timeline_item_id: Item to fade.
        fade_in_seconds: Fade-in length in seconds (0 = none).
        fade_out_seconds: Fade-out length in seconds (0 = none).
    """
    delta = backend.add_fade(timeline_item_id, fade_in_seconds, fade_out_seconds)
    return cast("dict[str, Any]", delta.model_dump(mode="json"))


def set_speed(
    backend: ResolveBackend,
    timeline_item_id: str,
    speed: float,
) -> dict[str, Any]:
    """Set the playback speed multiplier of a timeline item (fake backend only).

    Retiming is not in Resolve's documented scripting API, so the live DaVinci
    backend rejects this call with an "unsupported" error.

    Args:
        timeline_item_id: Item to retime.
        speed: Multiplier > 0 (0.5 = half speed, 2.0 = double speed).
    """
    delta = backend.set_speed(timeline_item_id, speed)
    return cast("dict[str, Any]", delta.model_dump(mode="json"))


def add_marker(
    backend: ResolveBackend,
    timeline_item_id: str,
    position_seconds: float,
    label: str,
    color: str,
    note: str = "",
) -> dict[str, Any]:
    """Add a point marker on a timeline item.

    Args:
        timeline_item_id: Item to mark.
        position_seconds: Offset from the START OF THE ITEM, not the timeline.
        label: Short marker name.
        color: Resolve marker colour, e.g. "blue", "red", "cream".
        note: Optional longer note.
    """
    delta = backend.add_marker(timeline_item_id, position_seconds, label, color, note)
    return cast("dict[str, Any]", delta.model_dump(mode="json"))


# --- Phase 2: effects --------------------------------------------------------


def add_transition(
    backend: ResolveBackend,
    timeline_item_id: str,
    track_index: int,
    style: str,
    duration_seconds: float,
    alignment: str,
) -> dict[str, Any]:
    """Attach a transition to a timeline item (fake backend only).

    Resolve's documented scripting API cannot create transitions, so the live
    DaVinci backend rejects this call with an "unsupported" error.

    Args:
        timeline_item_id: Item the transition attaches to.
        track_index: 1-based index of the track holding that item.
        style: e.g. "cross_dissolve", "dip_to_black".
        duration_seconds: Transition length in seconds.
        alignment: "start", "end" or "mid" of the item boundary.
    """
    delta = backend.add_transition(timeline_item_id, track_index, style, duration_seconds, alignment)
    return cast("dict[str, Any]", delta.model_dump(mode="json"))


# --- Phase 2: render ---------------------------------------------------------


def add_render_job(
    backend: ResolveBackend,
    timeline_name: str,
    format: str,
    output_path: str,
) -> dict[str, Any]:
    """Queue a render job for a timeline.

    Args:
        timeline_name: Timeline to render (see list_timelines).
        format: Output preset: "mp4", "mov", "prores" or "dnxhr".
        output_path: Full output file path; its directory is created if needed.
    """
    job = backend.add_render_job(timeline_name, format, output_path)
    return cast("dict[str, Any]", job.model_dump(mode="json"))


def start_render(backend: ResolveBackend, job_id: str) -> dict[str, Any]:
    """Kick off a queued render job. Returns the updated job record."""
    job = backend.start_render(job_id)
    return cast("dict[str, Any]", job.model_dump(mode="json"))


def get_render_status(backend: ResolveBackend, job_id: str) -> dict[str, Any]:
    """Return the render-job record for ``job_id``."""
    job = backend.get_render_status(job_id)
    return cast("dict[str, Any]", job.model_dump(mode="json"))


# --- Phase 2: destructive (gated) -------------------------------------------


def quit_app(backend: ResolveBackend, confirm: bool) -> dict[str, Any]:
    """Quit DaVinci Resolve. Requires ``confirm=True`` AND --allow-destructive."""
    return cast("dict[str, Any]", backend.quit_app(confirm=confirm))


def restart_app(backend: ResolveBackend, confirm: bool) -> dict[str, Any]:
    """Restart the app — fake backend only.

    Resolve can quit but cannot relaunch itself, so the live DaVinci backend does
    not expose this tool at all; use quit_app and start Resolve yourself.
    """
    return cast("dict[str, Any]", backend.restart_app(confirm=confirm))


def delete_timeline(backend: ResolveBackend, name: str, confirm: bool) -> dict[str, Any]:
    """Delete a timeline. Requires ``confirm=True`` AND --allow-destructive."""
    delta = backend.delete_timeline(name, confirm=confirm)
    return cast("dict[str, Any]", delta.model_dump(mode="json"))


def delete_media(backend: ResolveBackend, media_clip_id: str, confirm: bool) -> dict[str, Any]:
    """Delete a media clip. Requires ``confirm=True`` AND --allow-destructive."""
    return cast("dict[str, Any]", backend.delete_media(media_clip_id, confirm=confirm))


__all__ = [
    "add_fade",
    "add_marker",
    "add_render_job",
    "add_transition",
    "append_clip",
    "create_bin",
    "create_project",
    "create_timeline",
    "delete_clip",
    "delete_media",
    "delete_timeline",
    "get_project_info",
    "get_render_status",
    "get_timeline_state",
    "import_media",
    "insert_clip",
    "list_media_pool",
    "list_projects",
    "list_timelines",
    "move_clip",
    "open_project",
    "quit_app",
    "restart_app",
    "save_project",
    "set_composite_mode",
    "set_crop",
    "set_current_timeline",
    "set_opacity",
    "set_speed",
    "set_transform",
    "start_render",
]
