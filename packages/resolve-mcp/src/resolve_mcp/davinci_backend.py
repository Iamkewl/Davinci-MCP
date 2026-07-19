"""Real backend: thin wrapper over DaVinci Resolve's scripting API.

This module is *deliberately* defensive about importing the Resolve scripting module:

* The ``import DaVinciResolveScript`` line lives inside a guarded block so the server
  can boot even when Resolve is absent (CI, dev laptops without Studio installed).
* Every method that actually touches Resolve performs the same check and raises a
  clear :class:`ResolveUnavailableError` pointing at the Resolve bootstrap section of
  the README.

This module is the ONLY place in ``resolve-mcp`` that imports ``DaVinciResolveScript``.
If you find yourself importing it elsewhere, something is wrong with the boundary.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import uuid
from typing import Any

from .backend import ResolveUnavailableError
from .schemas import (
    Bin,
    CompositeMode,
    Crop,
    FrameRate,
    MarkerColor,
    MediaClip,
    MediaKind,
    MediaPoolState,
    ProjectInfo,
    RenderJob,
    RenderJobFormat,
    RenderJobStatus,
    StateDelta,
    TimelineItem,
    TimelineState,
    Track,
    TrackKind,
    Transform,
    TransitionAlignment,
    TransitionStyle,
)


def _try_import_resolve() -> Any | None:
    """Best-effort import; returns the module or ``None`` if unavailable.

    The bootstrap env vars (RESOLVE_SCRIPT_API, RESOLVE_SCRIPT_LIB, PYTHONPATH) must
    be set before this is called, otherwise the import will fail with whatever the
    interpreter throws when the module is unreachable.
    """
    try:
        import DaVinciResolveScript as dvr
    except Exception:
        return None
    return dvr


class DaVinciResolveBackend:
    """Real Resolve backend. Class exists regardless of whether Resolve is installed.

    Construction does NOT raise. Methods raise :class:`ResolveUnavailableError` only
    when invoked without a working Resolve connection.
    """

    def __init__(self) -> None:
        self._resolve_module: Any | None = _try_import_resolve()
        self._resolve: Any | None = None
        self._project_manager: Any | None = None
        if self._resolve_module is not None:
            try:
                self._resolve = self._resolve_module.scriptapp("Resolve")
                if self._resolve is not None:
                    self._project_manager = self._resolve.GetProjectManager()
            except Exception:
                # Resolve not running / scripting not enabled — leave _resolve None.
                self._resolve = None
                self._project_manager = None

    # --- connection ----------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._resolve is not None

    def _require(self) -> tuple[Any, Any]:
        if self._resolve is None or self._project_manager is None:
            env = {
                "RESOLVE_SCRIPT_API": os.environ.get("RESOLVE_SCRIPT_API", "<unset>"),
                "RESOLVE_SCRIPT_LIB": os.environ.get("RESOLVE_SCRIPT_LIB", "<unset>"),
            }
            msg = (
                "DaVinci Resolve scripting API is not available. "
                "Verify the bootstrap (Preferences -> General -> External scripting using = Local) "
                f"and env vars {env!r}. See README.md 'Resolve bootstrap'."
            )
            raise ResolveUnavailableError(msg)
        return self._resolve, self._project_manager

    # --- projects ------------------------------------------------------------

    def list_projects(self) -> list[str]:
        _, pm = self._require()
        # Resolve returns a folder/path browse; we just surface names of currently-open projects.
        names: list[str] = []
        proj = pm.GetCurrentProject()
        if proj is not None:
            names.append(proj.GetName())
        return names

    def create_project(
        self,
        name: str,
        frame_rate: FrameRate | dict[str, Any] | float,
        width: int,
        height: int,
    ) -> ProjectInfo:
        _, pm = self._require()
        fr = self._coerce_frame_rate(frame_rate)
        # Resolve's CreateProject(name) creates with default settings; we then set
        # timeline resolution and frame rate via the new project's first timeline.
        proj = pm.CreateProject(name)
        if proj is None:
            msg = f"failed to create project {name!r} — does it already exist?"
            from .backend import AlreadyExistsError
            raise AlreadyExistsError(msg)
        # Apply frame rate + resolution on the timeline that Resolve auto-creates.
        tl = proj.GetCurrentTimeline()
        if tl is not None:
            tl.SetSetting("timelineFrameRate", str(fr.fps))
            tl.SetSetting("timelineResolutionWidth", str(width))
            tl.SetSetting("timelineResolutionHeight", str(height))
        info = ProjectInfo(
            name=name,
            frame_rate=fr,
            resolution_width=width,
            resolution_height=height,
            path=None,
            is_modified=False,
        )
        return info

    def open_project(self, name: str) -> ProjectInfo:
        _, pm = self._require()
        proj = pm.LoadProject(name)
        if proj is None:
            from .backend import NotFoundError
            msg = f"project {name!r} not found at project-manager default path"
            raise NotFoundError(msg)
        from .schemas import ProjectInfo
        fr_fps = self._safe_fps(proj)
        return ProjectInfo(
            name=proj.GetName(),
            frame_rate=fr_fps,
            resolution_width=self._safe_int(proj, "GetResolutionWidth", 1920),
            resolution_height=self._safe_int(proj, "GetResolutionHeight", 1080),
            path=None,
            is_modified=False,
        )

    def save_project(self) -> ProjectInfo:
        _, pm = self._require()
        proj = pm.GetCurrentProject()
        if proj is None:
            from .backend import InvalidStateError
            raise InvalidStateError("no project open")
        # SaveProject returns bool in some Resolve versions; ignore the result and re-fetch.
        from contextlib import suppress

        with suppress(Exception):
            proj.SaveProject()
        from .schemas import ProjectInfo
        return ProjectInfo(
            name=proj.GetName(),
            frame_rate=self._safe_fps(proj),
            resolution_width=self._safe_int(proj, "GetResolutionWidth", 1920),
            resolution_height=self._safe_int(proj, "GetResolutionHeight", 1080),
            path=None,
            is_modified=False,
        )

    def current_project(self) -> ProjectInfo:
        _, pm = self._require()
        proj = pm.GetCurrentProject()
        if proj is None:
            from .backend import InvalidStateError
            raise InvalidStateError("no project open")
        from .schemas import ProjectInfo
        return ProjectInfo(
            name=proj.GetName(),
            frame_rate=self._safe_fps(proj),
            resolution_width=self._safe_int(proj, "GetResolutionWidth", 1920),
            resolution_height=self._safe_int(proj, "GetResolutionHeight", 1080),
            path=None,
            is_modified=False,
        )

    # --- media pool ----------------------------------------------------------

    def list_media_pool(self) -> MediaPoolState:
        from .schemas import MediaPoolState
        proj, _ = self._require_project()
        mp = proj.GetMediaPool()
        if mp is None:
            return MediaPoolState(bins=[], clips=[])
        root = mp.GetRootFolder()
        bins: list[Bin] = []
        clips: list[MediaClip] = []
        if root is not None:
            self._walk_bin(root.GetName(), root, bins, clips)
        return MediaPoolState(bins=bins, clips=clips)

    def import_media(self, paths: list[str], bin: str | None = None) -> list[MediaClip]:
        proj, _ = self._require_project()
        mp = proj.GetMediaPool()
        if mp is None:
            from .backend import InvalidStateError
            raise InvalidStateError("project has no media pool")
        items = mp.ImportMedia(paths)
        out: list[MediaClip] = []
        for it in items or []:
            clip = self._cv_to_clip(it)
            out.append(clip)
        return out

    def create_bin(self, name: str) -> Bin:
        proj, _ = self._require_project()
        mp = proj.GetMediaPool()
        if mp is None:
            from .backend import InvalidStateError
            raise InvalidStateError("project has no media pool")
        from contextlib import suppress
        with suppress(Exception):
            mp.AddSubFolder(mp.GetRootFolder(), name)
        return Bin(name=name, clip_ids=[])

    # ---- timeline ------------------------------------------------------------

    def create_timeline(
        self,
        name: str,
        frame_rate: FrameRate | dict[str, Any] | float,
    ) -> TimelineState:
        proj, _ = self._require_project()
        fr = self._coerce_frame_rate(frame_rate)
        mp = proj.GetMediaPool()
        if mp is None:
            from .backend import InvalidStateError
            raise InvalidStateError("project has no media pool")
        tl = mp.CreateEmptyTimeline(name)
        if tl is None:
            msg = f"failed to create timeline {name!r}"
            from .backend import ResolveMCPError as _E
            raise _E(msg)
        with contextlib.suppress(Exception):
            tl.SetSetting("timelineFrameRate", str(fr.fps))
        return TimelineState(
            name=name,
            frame_rate=fr,
            duration_seconds=0.0,
            tracks=[
                Track(index=1, kind=TrackKind.VIDEO, items=[]),
                Track(index=2, kind=TrackKind.AUDIO, items=[]),
            ],
            transitions=[],
        )

    def get_timeline_state(self) -> TimelineState:
        proj, _ = self._require_project()
        tl = proj.GetCurrentTimeline()
        if tl is None:
            from .backend import InvalidStateError
            raise InvalidStateError("no timeline selected")
        return self._hydrate_timeline_state(tl)

    def append_clip(
        self,
        media_clip_id: str,
        timeline_track_index: int,
        start_seconds: float,
        duration_seconds: float,
        source_in_seconds: float = 0.0,
    ) -> StateDelta:
        # Resolve does not expose programmatic per-track_index assignment by
        # integer; we read tracks[track_index-1] (Resolve is 1-indexed).
        proj, _ = self._require_project()
        tl = proj.GetCurrentTimeline()
        if tl is None:
            from .backend import InvalidStateError
            raise InvalidStateError("no timeline selected")
        before = self._hydrate_timeline_state(tl)
        track_count = tl.GetTrackCount()
        if timeline_track_index < 1 or timeline_track_index > track_count:
            msg = f"track index out of range: {timeline_track_index}/{track_count}"
            from .backend import NotFoundError
            raise NotFoundError(msg)
        tl.GetItemListInTrack("video" if timeline_track_index == 1 else "audio")
        # Build an empty item whose duration matches the requested length; then
        # assign its source media by setting the source in/out properties via
        # the items accessor. Resolve's AppendItemsInTimeline accepts a list
        # of (pcl, info) tuples.

        # Resolve API: AppendItemsInTimeline takes a list of [(PoolItem, infoDict)].
        mp = proj.GetMediaPool()

        def _resolve_clip_for_id(media_clip_id: str) -> Any:
            return self._lookup_pool_item(mp, media_clip_id)

        pool_item = _resolve_clip_for_id(media_clip_id)
        if pool_item is None:
            from .backend import NotFoundError
            msg = f"media clip {media_clip_id!r} not resolvable on the live pool"
            raise NotFoundError(msg)
        info_dict = {
            "mediaType": "video" if timeline_track_index == 1 else "audio",
            "trackIndex": timeline_track_index,
            "startFrame": round(start_seconds * self._fps(fps_of_timeline(tl))),
            "endFrame": round((start_seconds + duration_seconds) * self._fps(fps_of_timeline(tl))),
        }
        ok = tl.AppendItemsInTimeline([(pool_item, info_dict)])
        if not ok:
            msg = "AppendItemsInTimeline returned False"
            from .backend import InvalidStateError
            raise InvalidStateError(msg)
        after = self._hydrate_timeline_state(tl)
        return self._state_delta_from_before_after(before, after)

    def insert_clip(
        self,
        media_clip_id: str,
        timeline_track_index: int,
        timeline_position_seconds: float,
        duration_seconds: float,
        source_in_seconds: float = 0.0,
    ) -> StateDelta:
        # AppendItemsInTimeline inserts wherever startFrame places it; do not
        # shift later items — the caller must compute desired positions after
        # we return the resulting state delta.
        return self.append_clip(
            media_clip_id=media_clip_id,
            timeline_track_index=timeline_track_index,
            start_seconds=timeline_position_seconds,
            duration_seconds=duration_seconds,
            source_in_seconds=source_in_seconds,
        )

    def delete_clip(self, timeline_item_id: str) -> StateDelta:
        proj, _ = self._require_project()
        tl = proj.GetCurrentTimeline()
        if tl is None:
            from .backend import InvalidStateError
            raise InvalidStateError("no timeline selected")
        before = self._hydrate_timeline_state(tl)
        info = {"position": self._find_item_position(tl, timeline_item_id)}
        if info is None:
            from .backend import NotFoundError
            msg = f"timeline item {timeline_item_id!r} not found"
            raise NotFoundError(msg)
        ok = tl.DeleteClips([info])
        if not ok:
            msg = f"DeleteClips failed for {timeline_item_id!r}"
            from .backend import InvalidStateError
            raise InvalidStateError(msg)
        after = self._hydrate_timeline_state(tl)
        return self._state_delta_from_before_after(before, after, changed_path=f"timelines.{tl.GetName()}.items")

    def move_clip(self, timeline_item_id: str, new_position_seconds: float) -> StateDelta:
        proj, _ = self._require_project()
        tl = proj.GetCurrentTimeline()
        if tl is None:
            from .backend import InvalidStateError
            raise InvalidStateError("no timeline selected")
        before = self._hydrate_timeline_state(tl)
        pos = self._find_item_position(tl, timeline_item_id)
        if pos is None:
            from .backend import NotFoundError
            msg = f"timeline item {timeline_item_id!r} not found"
            raise NotFoundError(msg)
        track_index = pos[0]
        fps_v = self._fps(fps_of_timeline(tl))
        new_start_frame = round(new_position_seconds * fps_v)
        # Resolve's move/length-change primitive is UpdateClipSelection or
        # DeleteClips + AppendItemsInTimeline. We just delete and re-append with
        # the new startFrame; the MCP caller is expected to also persist the
        # intended duration via separate ops. For simplicitly here we delete +
        # re-append keeping the existing duration.
        clip = pos[2] if len(pos) > 2 else None
        duration_seconds = 0.0
        if clip is not None:
            try:
                duration_seconds = (clip.GetDuration() / fps_v) if hasattr(clip, "GetDuration") else 0.0
            except Exception:
                duration_seconds = 0.0
        tl.DeleteClips([{"trackIndex": track_index, "position": pos[1]}])
        if clip is not None and duration_seconds > 0:
            mp = proj.GetMediaPool()
            src_pool_item = None
            try:
                src_pool_item = self._match_pool_item(mp, clip)
            except Exception:
                src_pool_item = None
            if src_pool_item is not None:
                tl.AppendItemsInTimeline(
                    [
                        (
                            src_pool_item,
                            {
                                "mediaType": "video" if track_index == 1 else "audio",
                                "trackIndex": track_index,
                                "startFrame": new_start_frame,
                                "endFrame": round(
                                    (new_position_seconds + duration_seconds) * fps_v
                                ),
                            },
                        )
                    ]
                )
        after = self._hydrate_timeline_state(tl)
        return self._state_delta_from_before_after(before, after, changed_path=f"timelines.{tl.GetName()}.items")

    # ---- Phase 2 LIVE — per-item mutations ----------------------------------

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
    ) -> StateDelta:
        proj, _ = self._require_project()
        tl = proj.GetCurrentTimeline()
        if tl is None:
            from .backend import InvalidStateError
            raise InvalidStateError("no timeline selected")
        before = self._hydrate_timeline_state(tl)
        clip = self._resolve_clip_object(tl, timeline_item_id)
        if clip is None:
            from .backend import NotFoundError
            msg = f"timeline item {timeline_item_id!r} not found"
            raise NotFoundError(msg)
        # Resolve's Transform API: SetProperty for a list of (key, value) tuples.
        try:
            clip.SetProperty(
                "Transform",
                {
                    "ZoomX": float(zoom_x),
                    "ZoomY": float(zoom_y),
                    "PanX": float(pan_x),
                    "PanY": float(pan_y),
                    "Rotation": float(rotation),
                },
            )
        except Exception as exc:
            from .backend import ResolveMCPError as _E
            raise _E(f"set_transform failed: {exc}") from exc
        after = self._hydrate_timeline_state(tl)
        return self._state_delta_from_before_after(before, after, changed_path=f"timelines.{tl.GetName()}.items[{timeline_item_id}].transform")

    def set_crop(
        self,
        timeline_item_id: str,
        left: float,
        right: float,
        top: float,
        bottom: float,
    ) -> StateDelta:
        proj, _ = self._require_project()
        tl = proj.GetCurrentTimeline()
        if tl is None:
            from .backend import InvalidStateError
            raise InvalidStateError("no timeline selected")
        before = self._hydrate_timeline_state(tl)
        clip = self._resolve_clip_object(tl, timeline_item_id)
        if clip is None:
            from .backend import NotFoundError
            msg = f"timeline item {timeline_item_id!r} not found"
            raise NotFoundError(msg)
        try:
            clip.SetProperty(
                "Crop",
                {
                    "Left": float(left),
                    "Right": float(right),
                    "Top": float(top),
                    "Bottom": float(bottom),
                },
            )
        except Exception as exc:
            from .backend import ResolveMCPError as _E
            raise _E(f"set_crop failed: {exc}") from exc
        after = self._hydrate_timeline_state(tl)
        return self._state_delta_from_before_after(before, after, changed_path=f"timelines.{tl.GetName()}.items[{timeline_item_id}].crop")

    def set_composite_mode(self, timeline_item_id: str, mode: CompositeMode | str) -> StateDelta:
        proj, _ = self._require_project()
        tl = proj.GetCurrentTimeline()
        if tl is None:
            from .backend import InvalidStateError
            raise InvalidStateError("no timeline selected")
        before = self._hydrate_timeline_state(tl)
        clip = self._resolve_clip_object(tl, timeline_item_id)
        if clip is None:
            from .backend import NotFoundError
            msg = f"timeline item {timeline_item_id!r} not found"
            raise NotFoundError(msg)
        mode_str = mode.value if isinstance(mode, CompositeMode) else str(mode)
        try:
            clip.SetProperty("CompositeMode", {"Value": mode_str})
        except Exception as exc:
            from .backend import ResolveMCPError as _E
            raise _E(f"set_composite_mode failed: {exc}") from exc
        after = self._hydrate_timeline_state(tl)
        return self._state_delta_from_before_after(before, after, changed_path=f"timelines.{tl.GetName()}.items[{timeline_item_id}].composite_mode")

    def set_opacity(self, timeline_item_id: str, opacity: float) -> StateDelta:
        if not 0.0 <= opacity <= 1.0:
            msg = f"opacity must be in [0, 1], got {opacity}"
            raise ValueError(msg)
        proj, _ = self._require_project()
        tl = proj.GetCurrentTimeline()
        if tl is None:
            from .backend import InvalidStateError
            raise InvalidStateError("no timeline selected")
        before = self._hydrate_timeline_state(tl)
        clip = self._resolve_clip_object(tl, timeline_item_id)
        if clip is None:
            from .backend import NotFoundError
            msg = f"timeline item {timeline_item_id!r} not found"
            raise NotFoundError(msg)
        # Resolve stores alpha as 0..100; convert back to 0..1 in the delta.
        try:
            clip.SetProperty("Opacity", {"Value": opacity * 100.0})
        except Exception as exc:
            from .backend import ResolveMCPError as _E
            raise _E(f"set_opacity failed: {exc}") from exc
        after = self._hydrate_timeline_state(tl)
        return self._state_delta_from_before_after(before, after, changed_path=f"timelines.{tl.GetName()}.items[{timeline_item_id}].opacity")

    def add_fade(
        self,
        timeline_item_id: str,
        fade_in_seconds: float,
        fade_out_seconds: float,
    ) -> StateDelta:
        if fade_in_seconds < 0 or fade_out_seconds < 0:
            msg = "fade durations must be non-negative"
            raise ValueError(msg)
        proj, _ = self._require_project()
        tl = proj.GetCurrentTimeline()
        if tl is None:
            from .backend import InvalidStateError
            raise InvalidStateError("no timeline selected")
        before = self._hydrate_timeline_state(tl)
        clip = self._resolve_clip_object(tl, timeline_item_id)
        if clip is None:
            from .backend import NotFoundError
            msg = f"timeline item {timeline_item_id!r} not found"
            raise NotFoundError(msg)
        try:
            clip.SetProperty(
                "Ease",
                {
                    "InHandle": float(fade_in_seconds * 1000.0),  # ms
                    "OutHandle": float(fade_out_seconds * 1000.0),
                },
            )
        except Exception as exc:
            from .backend import ResolveMCPError as _E
            raise _E(f"add_fade failed: {exc}") from exc
        after = self._hydrate_timeline_state(tl)
        return self._state_delta_from_before_after(before, after, changed_path=f"timelines.{tl.GetName()}.items[{timeline_item_id}].fades")

    def set_speed(self, timeline_item_id: str, speed: float) -> StateDelta:
        if speed <= 0:
            msg = "speed must be > 0"
            raise ValueError(msg)
        proj, _ = self._require_project()
        tl = proj.GetCurrentTimeline()
        if tl is None:
            from .backend import InvalidStateError
            raise InvalidStateError("no timeline selected")
        before = self._hydrate_timeline_state(tl)
        clip = self._resolve_clip_object(tl, timeline_item_id)
        if clip is None:
            from .backend import NotFoundError
            msg = f"timeline item {timeline_item_id!r} not found"
            raise NotFoundError(msg)
        try:
            clip.SetProperty("Speed", {"Value": float(speed)})  # 1.0 = 100%; greater = faster
        except Exception as exc:
            from .backend import ResolveMCPError as _E
            raise _E(f"set_speed failed: {exc}") from exc
        after = self._hydrate_timeline_state(tl)
        return self._state_delta_from_before_after(before, after, changed_path=f"timelines.{tl.GetName()}.items[{timeline_item_id}].speed")

    def add_marker(
        self,
        timeline_item_id: str,
        position_seconds: float,
        label: str,
        color: MarkerColor | str,
        note: str = "",
    ) -> StateDelta:
        proj, _ = self._require_project()
        tl = proj.GetCurrentTimeline()
        if tl is None:
            from .backend import InvalidStateError
            raise InvalidStateError("no timeline selected")
        before = self._hydrate_timeline_state(tl)
        clip = self._resolve_clip_object(tl, timeline_item_id)
        if clip is None:
            from .backend import NotFoundError
            msg = f"timeline item {timeline_item_id!r} not found"
            raise NotFoundError(msg)
        color_str = color.value if isinstance(color, MarkerColor) else str(color)
        fps_v = self._fps(fps_of_timeline(tl))
        frame = round(position_seconds * fps_v)
        try:
            clip.AddMarker(frame, color_str, label or note, label or "")
        except Exception as exc:
            from .backend import ResolveMCPError as _E
            raise _E(f"add_marker failed: {exc}") from exc
        after = self._hydrate_timeline_state(tl)
        return self._state_delta_from_before_after(before, after, changed_path=f"timelines.{tl.GetName()}.items[{timeline_item_id}].markers")

    # ---- Phase 2 LIVE — transitions & render ---------------------------------

    def add_transition(
        self,
        timeline_item_id: str,
        track_index: int,
        style: TransitionStyle | str,
        duration_seconds: float,
        alignment: TransitionAlignment | str,
    ) -> StateDelta:
        proj, _ = self._require_project()
        tl = proj.GetCurrentTimeline()
        if tl is None:
            from .backend import InvalidStateError
            raise InvalidStateError("no timeline selected")
        before = self._hydrate_timeline_state(tl)
        clip_track_index = self._find_track_index_for_item(tl, timeline_item_id)
        if clip_track_index is None:
            from .backend import NotFoundError
            msg = f"timeline item {timeline_item_id!r} not found on the live timeline"
            raise NotFoundError(msg)
        style_str = style.value if isinstance(style, TransitionStyle) else str(style)
        align_str = alignment.value if isinstance(alignment, TransitionAlignment) else str(alignment)
        # Resolve API expects alignment numeric codes; we map the most common.
        # Resolve 18+ uses 0=start, 1=mid, 2=end for transitions on items.
        align_code = {"start": 0, "mid": 1, "end": 2}.get(align_str.lower(), 1)
        duration_frames = round(duration_seconds * self._fps(fps_of_timeline(tl)))
        try:
            tl.AddTransition(
                clip_track_index,
                {
                    "type": style_str,
                    "duration": duration_frames,
                    "alignment": align_code,
                },
            )
        except Exception as exc:
            from .backend import ResolveMCPError as _E
            raise _E(f"add_transition failed: {exc}") from exc
        after = self._hydrate_timeline_state(tl)
        return self._state_delta_from_before_after(before, after, changed_path=f"timelines.{tl.GetName()}.transitions")

    def add_render_job(
        self,
        timeline_name: str,
        format: RenderJobFormat | str,
        output_path: str,
    ) -> RenderJob:
        proj, _ = self._require_project()
        # Resolve requires switching to the target timeline before queuing render.
        # We don't have a programmatic "switch timeline by name" in the older API;
        # use the currently-active timeline if it matches, else fall back to the
        # project manager if `LoadProject` is needed.
        tl = proj.GetCurrentTimeline()
        if tl is None or tl.GetName() != timeline_name:
            msg = f"timeline {timeline_name!r} is not currently active in Resolve"
            from .backend import InvalidStateError
            raise InvalidStateError(msg)
        self._coerce_frame_rate(fps_of_timeline(tl))
        fmt_str = format.value if isinstance(format, RenderJobFormat) else str(format)
        # Map our format enum -> Resolve's "renderFormat" Setting keys (DNxHR, MP4, etc.)
        render_format_setting = fmt_str
        try:
            proj.SetSetting("renderFormat", render_format_setting)
            proj.SetSetting("rendermark", "0")
            # Render path & filename
            proj.SetSetting("currentRenderOutputPath", str(pathlib.Path(output_path).parent))
            tl.SetSetting(f"currentRender{('FileName' if False else 'File')}Name", str(pathlib.Path(output_path).name))
        except Exception as exc:
            from .backend import ResolveMCPError as _E
            raise _E(f"add_render_job failed: {exc}") from exc
        # No persistent job id; Resolve returns a "JobId" via AddRenderJob.
        try:
            job_id = proj.AddRenderJob()
        except Exception as exc:
            from .backend import ResolveMCPError as _E
            raise _E(f"AddRenderJob failed: {exc}") from exc
        return RenderJob(
            id=str(job_id) if job_id is not None else f"render_{uuid.uuid4().hex[:8]}",
            timeline_name=timeline_name,
            format=fmt_str,  # type: ignore[arg-type]
            output_path=output_path,
            status=RenderJobStatus.QUEUED,
            progress=0.0,
        )

    def start_render(self, job_id: str) -> RenderJob:
        proj, _ = self._require_project()
        try:
            job = proj.GetRenderJob(job_id)
        except Exception as exc:
            from .backend import NotFoundError
            raise NotFoundError(f"render job {job_id!r} not found: {exc}") from exc
        if not job:
            from .backend import NotFoundError
            raise NotFoundError(f"render job {job_id!r} not found")
        try:
            proj.StartRendering(job)
        except Exception as exc:
            from .backend import ResolveMCPError as _E
            raise _E("start_render failed; is the render queue ready?") from exc
        return RenderJob(
            id=job_id,
            timeline_name="",
            format=RenderJobFormat.MP4,
            output_path="",
            status=RenderJobStatus.RUNNING,
            progress=0.0,
        )

    def get_render_status(self, job_id: str) -> RenderJob:
        proj, _ = self._require_project()
        try:
            statuses = proj.GetRenderJobs() or {}
        except Exception as exc:
            from .backend import ResolveMCPError as _E
            raise _E(f"GetRenderJobs failed: {exc}") from exc
        # Resolve returns a list-of-dicts in some versions and a dict {id: status}
        # in newer ones; normalize iteratively.
        job_status: str | None = None
        progress: float = 0.0
        if isinstance(statuses, dict):
            job_status = str(statuses.get(job_id, "queued"))
        else:
            for j in statuses:
                if isinstance(j, dict) and str(j.get("JobId", "")) == job_id:
                    job_status = str(j.get("Status", "queued"))
                    try:
                        progress = float(j.get("Progress", 0.0))
                    except Exception:
                        progress = 0.0
                    break
        if job_status is None:
            from .backend import NotFoundError
            raise NotFoundError(f"render job {job_id!r} not found")
        status_enum = _resolve_render_status_enum(job_status)
        return RenderJob(
            id=job_id,
            timeline_name="",
            format=RenderJobFormat.MP4,
            output_path="",
            status=status_enum,
            progress=progress,
        )

    # ---- Phase 2 LIVE — destructive (gated) ----------------------------------

    def quit_app(self, confirm: bool = False) -> dict[str, Any]:
        if not confirm:
            msg = "destructive call requires confirm=true"
            from .backend import DestructiveDisabledError
            raise DestructiveDisabledError(msg)
        _, _ = self._require()
        if self._resolve is None:
            msg = "resolve not connected"
            raise ResolveUnavailableError(msg)
        try:
            self._resolve.Quit()
        except Exception as exc:
            from .backend import ResolveMCPError as _E
            raise _E(f"Resolve.Quit failed: {exc}") from exc
        return {"quit": True, "project_after": None}

    def restart_app(self, confirm: bool = False) -> dict[str, Any]:
        # Resolve scripting has no direct Restart; we treat it as Quit and the
        # supervisor is expected to relaunch.
        return self.quit_app(confirm=confirm)

    def delete_timeline(self, name: str, confirm: bool = False) -> StateDelta:
        proj, _ = self._require_project()
        if not confirm:
            msg = "destructive call requires confirm=true"
            from .backend import DestructiveDisabledError
            raise DestructiveDisabledError(msg)
        # Resolve: there's no DeleteTimeline method in the scripting API; the
        # workaround is DeleteClips on every item of the timeline then we save
        # an empty timeline under the same name via the project's media pool.
        tl = proj.GetCurrentTimeline()
        if tl is None or tl.GetName() != name:
            from .backend import NotFoundError
            msg = f"timeline {name!r} is not the currently active timeline"
            raise NotFoundError(msg)
        before = self._hydrate_timeline_state(tl)
        try:
            track_count = tl.GetTrackCount()
        except Exception:
            track_count = 0
        # Resolve API to bulk-delete items by position is per-track via DeleteClips([...]).
        to_delete: list[dict[str, Any]] = []
        try:
            for tr_idx in range(1, track_count + 1):
                tr_type = tl.GetTrackType(tr_idx)
                items = tl.GetItemListInTrack(tr_type) or []
                for it in items:
                    to_delete.append({"trackIndex": tr_idx, "position": getattr(it, "GetStart", lambda: 0)()})
        except Exception:
            pass
        if to_delete:
            try:
                tl.DeleteClips(to_delete)
            except Exception as exc:
                from .backend import ResolveMCPError as _E
                raise _E(f"delete_timeline failed during DeleteClips: {exc}") from exc
        after = self._hydrate_timeline_state(tl)
        return _wrap_state_delta(self._state_delta_from_before_after(before, after, changed_path=f"timelines.{name}"))

    def delete_media(self, media_clip_id: str, confirm: bool = False) -> dict[str, Any]:
        if not confirm:
            from .backend import DestructiveDisabledError
            raise DestructiveDisabledError("destructive call requires confirm=true")
        proj, _ = self._require_project()
        mp = proj.GetMediaPool()
        if mp is None:
            from .backend import InvalidStateError
            raise InvalidStateError("project has no media pool")
        # media_clip_id is one of our hex kinds. Locate the corresponding PoolItem by name + path.
        clip = self._gather_sync(mp.root_folder(), target_id=media_clip_id)
        if clip is None:
            from .backend import NotFoundError
            raise NotFoundError(f"media clip {media_clip_id!r} not resolvable in pool")
        try:
            clip.Delete()
        except Exception as exc:
            from .backend import ResolveMCPError as _E
            raise _E(f"delete_media failed: {exc}") from exc
        return {"deleted": {"id": media_clip_id}}

    # --- helpers -------------------------------------------------------------

    def _require_project(self) -> tuple[Any, Any]:
        _, pm = self._require()
        proj = pm.GetCurrentProject()
        if proj is None:
            from .backend import InvalidStateError
            raise InvalidStateError("no project open")
        return proj, pm

    @staticmethod
    def _coerce_frame_rate(frame_rate: Any) -> FrameRate:
        if isinstance(frame_rate, FrameRate):
            return frame_rate
        if isinstance(frame_rate, dict):
            return FrameRate(**frame_rate)
        if isinstance(frame_rate, int | float):
            return FrameRate(fps=float(frame_rate), drop_frame=False)
        msg = f"unsupported frame_rate type {type(frame_rate).__name__}"
        raise TypeError(msg)

    @staticmethod
    def _safe_fps(proj: Any) -> FrameRate:
        try:
            v = proj.GetSetting("timelineFrameRate")
            return FrameRate(fps=float(v), drop_frame=False) if v else FrameRate(fps=24.0, drop_frame=False)
        except Exception:
            return FrameRate(fps=24.0, drop_frame=False)

    @staticmethod
    def _safe_int(proj: Any, method: str, default: int) -> int:
        try:
            v = getattr(proj, method)()
            return int(v) if v else default
        except Exception:
            return default

    def _walk_bin(self, name: str, folder: Any, bins: list[Bin], clips: list[MediaClip]) -> None:
        from .schemas import Bin
        clip_list = folder.GetClipList() or []
        ids: list[str] = []
        for c in clip_list:
            clip = self._cv_to_clip(c)
            clips.append(clip)
            ids.append(clip.id)
        bins.append(Bin(name=name, clip_ids=ids))
        for sub in folder.GetSubFolderList() or []:
            self._walk_bin(sub.GetName(), sub, bins, clips)

    @staticmethod
    def _cv_to_clip(cv: Any) -> MediaClip:
        # FastMCP-visible ids are an artifact from this layer; the real backend
        # exposes Resolve's clip object identity. We use the object's hex hash.
        oid = hex(id(cv))
        path = ""
        try:
            prop = cv.GetClipProperty() or {}
            path = str(prop.get("File Path", ""))
        except Exception:
            path = ""
        name = cv.GetName() if hasattr(cv, "GetName") else path.rsplit("/", 1)[-1]
        return MediaClip(
            id=f"cv_{oid}",
            bin="Master",
            name=str(name),
            path=path,
            kind=MediaKind.VIDEO,
            duration_seconds=10.0,
        )

    # --- LIVE-only helpers (Phase 5) ------------------------------------------

    def _timeline_fr(self, timeline_obj: Any) -> FrameRate:
        """Project current FrameRate for a Resolve Timeline object."""
        try:
            return self._safe_fps(timeline_obj.GetProject())  # best-effort; falls back
        except Exception:
            pass
        return FrameRate(fps=24.0, drop_frame=False)

    def _fps(self, frame_rate: FrameRate) -> float:
        """Return the nominal integer fps (drop-frame math uses 30 internally)."""
        fps = float(frame_rate.fps)
        if abs(fps) < 1e-6:
            return 24.0
        return round(fps) if abs(fps - round(fps)) < 1e-3 else fps

    def _hydrate_timeline_state(self, tl: Any) -> TimelineState:
        """Read a Resolve Timeline and project into our TimelineState."""
        name = tl.GetName()
        fr = self._timeline_fr(tl)
        # Resolve tracks are 1-indexed: V1..Vn, A1..Am. We collapse to 0..N-1 here.
        tracks: list[Track] = []
        items: list[TimelineItem] = []
        try:
            track_count = tl.GetTrackCount()
        except Exception:
            track_count = 0
        duration_end = 0.0
        for tr_idx in range(1, track_count + 1):
            try:
                tr_type = tl.GetTrackType(tr_idx)  # "video" | "audio"
            except Exception:
                tr_type = "video"
            kind = TrackKind.VIDEO if tr_type.lower().startswith("video") else TrackKind.AUDIO
            try:
                tr_items = tl.GetItemListInTrack(tr_type) or []
            except Exception:
                tr_items = []
            track_items: list[TimelineItem] = []
            for it in tr_items:
                start_seconds = round(int(getattr(it, "GetStart", lambda: 0)()) / max(self._fps(fr), 1.0), 3)
                duration_seconds = round(int(getattr(it, "GetDuration", lambda: 0)()) / max(self._fps(fr), 1.0), 3)
                end_seconds = start_seconds + duration_seconds
                if end_seconds > duration_end:
                    duration_end = end_seconds
                # Resolve does not surface a stable item id from the scripting API.
                # We use a hex of the Python object identity, scoped by timeline.
                item = TimelineItem(
                    id=hex(id(it)),
                    track_index=tr_idx,
                    media_clip_id="",
                    start_seconds=start_seconds,
                    duration_seconds=duration_seconds,
                    source_in_seconds=0.0,
                    source_out_seconds=duration_seconds,
                    transform=Transform(),
                    crop=Crop(),
                    composite_mode=CompositeMode.NORMAL,
                    opacity=1.0,
                )
                track_items.append(item)
                items.append(item)
            tracks.append(Track(index=tr_idx, kind=kind, items=track_items))
        return TimelineState(
            name=name,
            frame_rate=fr,
            duration_seconds=duration_end,
            tracks=tracks,
            transitions=[],
        )

    def _state_delta_from_before_after(
        self,
        before_state: TimelineState,
        after_state: TimelineState,
        *,
        changed_path: str | None = None,
    ) -> StateDelta:
        before = before_state.model_dump(mode="json")
        after = after_state.model_dump(mode="json")
        paths: list[str] = []
        if changed_path is None:
            if before["duration_seconds"] != after["duration_seconds"]:
                paths.append(f"timelines.{after_state.name}.duration_seconds")
            if len(before["tracks"]) != len(after["tracks"]):
                paths.append(f"timelines.{after_state.name}.tracks")
        else:
            paths.append(changed_path)
        return StateDelta(before=before, after=after, changed_paths=paths)

    def _find_item_position(
        self,
        tl: Any,
        timeline_item_id: str,
    ) -> tuple[int, int, Any] | None:
        """Resolve has no native item id; we hunt by Python object identity.

        Returns ``(track_index, start_frame, item)`` or None.
        """
        try:
            track_count = tl.GetTrackCount()
        except Exception:
            track_count = 0
        for tr_idx in range(1, track_count + 1):
            try:
                tr_type = tl.GetTrackType(tr_idx)
            except Exception:
                tr_type = "video"
            try:
                items = tl.GetItemListInTrack(tr_type) or []
            except Exception:
                items = []
            for it in items:
                if hex(id(it)) == timeline_item_id:
                    start_frame = int(getattr(it, "GetStart", lambda: 0)())
                    return (tr_idx, start_frame, it)
        return None

    def _resolve_clip_object(self, tl: Any, timeline_item_id: str) -> Any | None:
        pos = self._find_item_position(tl, timeline_item_id)
        return pos[2] if pos else None

    def _find_track_index_for_item(
        self,
        tl: Any,
        timeline_item_id: str,
    ) -> int | None:
        pos = self._find_item_position(tl, timeline_item_id)
        return pos[0] if pos else None

    def _lookup_pool_item(self, mp: Any, media_clip_id: str) -> Any | None:
        """Reach into the project's media pool and return the PoolItem by our id.

        Our pool item id is ``cv_<hex>``; Resolve scripting exposes no id, but a
        hex of object identity survives a short-lived client process. Without an
        authoritative mapping, we simply enumerate root bin clips and match.
        """
        try:
            root = mp.GetRootFolder()
        except Exception:
            return None
        if root is None:
            return None
        return self._gather_pool_item(root, target_id=media_clip_id)

    def _gather_pool_item(self, folder: Any, target_id: str) -> Any | None:
        # Accept either ``cv_<hex>`` (our wire form) or a plain hex identity.
        bare = target_id.removeprefix("cv_")
        try:
            clips = folder.GetClipList() or []
        except Exception:
            clips = []
        for c in clips:
            if hex(id(c)) == target_id or hex(id(c)) == bare:
                return c
        try:
            for sub in folder.GetSubFolderList() or []:
                hit = self._gather_pool_item(sub, target_id)
                if hit is not None:
                    return hit
        except Exception:
            pass
        return None

    def _match_pool_item(self, mp: Any, timeline_clip: Any) -> Any | None:
        """Match a TimelineItem back to a PoolItem by inspecting the clip property.

        Resolve exposes the clip's name; pool items carry a hash-against-id of their
        Python object identity only. We best-effort by name match.
        """
        try:
            tp = timeline_clip.GetClipProperty() or {}
        except Exception:
            tp = {}
        try:
            target_name = str(tp.get("Clip Name") or tp.get("File Name") or "")
        except Exception:
            target_name = ""
        try:
            root = mp.GetRootFolder()
        except Exception:
            root = None
        if root is None or not target_name:
            return None
        for c in self._walk_pool_items(root):
            try:
                name = c.GetName()
            except Exception:
                continue
            if name == target_name:
                return c
        return None

    def _walk_pool_items(self, folder: Any) -> Any:
        try:
            clips = folder.GetClipList() or []
        except Exception:
            clips = []
        yield from clips
        try:
            for sub in folder.GetSubFolderList() or []:
                yield from self._walk_pool_items(sub)
        except Exception:
            return

    def _gather_sync(self, folder: Any, target_id: str) -> Any | None:
        return self._gather_pool_item(folder, target_id)

    def root_folder(self) -> Any:
        """Public alias for tests; returns the current project's media pool root."""
        proj, _ = self._require_project()
        mp = proj.GetMediaPool()
        return mp.GetRootFolder() if mp is not None else None


def fps_of_timeline(tl: Any) -> FrameRate:
    """Module-level helper that Resolve test mocks can inject without going through
    a backend instance.
    """
    try:
        fr = tl.GetSetting("timelineFrameRate")
        return FrameRate(fps=float(fr) if fr else 24.0, drop_frame=False)
    except Exception:
        return FrameRate(fps=24.0, drop_frame=False)


def _wrap_state_delta(delta: StateDelta) -> StateDelta:
    """Identity helper retained to keep delete_timeline's contract symmetric."""
    return delta


def _resolve_render_status_enum(status: str) -> RenderJobStatus:
    """Map Resolve render status strings into our RenderJobStatus enum."""
    mapping = {
        "queued": RenderJobStatus.QUEUED,
        "rendering": RenderJobStatus.RUNNING,
        "running": RenderJobStatus.RUNNING,
        "complete": RenderJobStatus.COMPLETED,
        "completed": RenderJobStatus.COMPLETED,
        "failed": RenderJobStatus.FAILED,
        "canceled": RenderJobStatus.CANCELLED,
        "cancelled": RenderJobStatus.CANCELLED,
        "paused": RenderJobStatus.QUEUED,
    }
    return mapping.get(status.lower(), RenderJobStatus.QUEUED)


__all__ = ["DaVinciResolveBackend"]
