"""Backend protocol + shared exceptions.

Every tool handler ultimately calls into a ``ResolveBackend`` instance. There are two
implementations:

* :class:`FakeResolveBackend` — in-memory, deterministic, used by every unit test.
* :class:`DaVinciResolveBackend` — real, using the DaVinci Resolve scripting API,
  optional-imported so the server boots even when Resolve is absent.

Both impls are constructed by ``server.py`` based on the CLI flag (``--backend fake`` vs
the default). They live in separate modules to keep the import-cost and dependency
weight of each explicit.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# --- Exception hierarchy -------------------------------------------------------


class ResolveMCPError(Exception):
    """Base for everything the backend or tools raise. Carries a stable code."""

    code: str = "resolve_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class NotFoundError(ResolveMCPError):
    code = "not_found"


class AlreadyExistsError(ResolveMCPError):
    code = "already_exists"


class InvalidStateError(ResolveMCPError):
    code = "invalid_state"


class DestructiveDisabledError(ResolveMCPError):
    code = "destructive_disabled"


class ResolveUnavailableError(ResolveMCPError):
    code = "resolve_unavailable"


# --- Protocol ------------------------------------------------------------------


@runtime_checkable
class ResolveBackend(Protocol):
    """The contract every backend implementation must satisfy.

    Methods are deliberately small and named after the underlying Resolve concept so
    the impls stay close to the scripting API.

    Type widths are intentionally loose (``Any`` / broad unions) here so Phase 2 can
    evolve impl signatures (and accept ``FrameRate | dict | float``) without breaking
    the protocol check. We validate the *returned* shapes, which are strict.
    """

    # ---- project ----

    def list_projects(self) -> Any: ...
    def open_project(self, name: str) -> Any: ...
    def create_project(self, name: str, frame_rate: Any, width: int, height: int) -> Any: ...
    def save_project(self) -> Any: ...
    def current_project(self) -> Any: ...

    # ---- media pool ----

    def list_media_pool(self) -> Any: ...
    def import_media(self, paths: list[str], bin: str | None = None) -> Any: ...
    def create_bin(self, name: str) -> Any: ...

    # ---- timeline ----

    def create_timeline(self, name: str, frame_rate: Any) -> Any: ...
    def get_timeline_state(self) -> Any: ...
    def append_clip(
        self,
        media_clip_id: str,
        timeline_track_index: int,
        start_seconds: float,
        duration_seconds: float,
        source_in_seconds: float = 0.0,
    ) -> Any: ...
    def insert_clip(
        self,
        media_clip_id: str,
        timeline_track_index: int,
        timeline_position_seconds: float,
        duration_seconds: float,
        source_in_seconds: float = 0.0,
    ) -> Any: ...
    def delete_clip(self, timeline_item_id: str) -> Any: ...
    def move_clip(self, timeline_item_id: str, new_position_seconds: float) -> Any: ...

    # ---- item (per-item mutations) ----

    def set_transform(
        self,
        timeline_item_id: str,
        pan_x: float,
        pan_y: float,
        zoom_x: float,
        zoom_y: float,
        rotation: float,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> Any: ...
    def set_crop(
        self,
        timeline_item_id: str,
        left: float,
        right: float,
        top: float,
        bottom: float,
    ) -> Any: ...
    def set_composite_mode(self, timeline_item_id: str, mode: Any) -> Any: ...
    def set_opacity(self, timeline_item_id: str, opacity: float) -> Any: ...
    def add_fade(
        self,
        timeline_item_id: str,
        fade_in_seconds: float,
        fade_out_seconds: float,
    ) -> Any: ...
    def set_speed(self, timeline_item_id: str, speed: float) -> Any: ...
    def add_marker(
        self,
        timeline_item_id: str,
        position_seconds: float,
        label: str,
        color: Any,
        note: str = "",
    ) -> Any: ...

    # ---- effects ----

    def add_transition(
        self,
        timeline_item_id: str,
        track_index: int,
        style: Any,
        duration_seconds: float,
        alignment: Any,
    ) -> Any: ...

    # ---- render ----

    def add_render_job(
        self,
        timeline_name: str,
        format: Any,
        output_path: str,
    ) -> Any: ...
    def start_render(self, job_id: str) -> Any: ...
    def get_render_status(self, job_id: str) -> Any: ...

    # ---- destructive (gated) ----

    def quit_app(self, confirm: bool = False) -> Any: ...
    def restart_app(self, confirm: bool = False) -> Any: ...
    def delete_timeline(self, name: str, confirm: bool = False) -> Any: ...
    def delete_media(self, media_clip_id: str, confirm: bool = False) -> Any: ...
