"""Real backend: thin wrapper over DaVinci Resolve's scripting API.

Every call in this module uses ONLY the documented Resolve scripting surface
(Blackmagic's "DaVinci Resolve Scripting API" README, v18+). Anything not in
that surface raises an honest error rather than fabricating success — see
:meth:`add_transition`. Timeline item ids are opaque session handles
(``tl_item_<n>``) that resolve via a registry rebuilt on every state read,
keyed by ``(clip name, media_type, intra-type track index, start frame)``.

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

# Documented (Resolve Scripting API) value spellings.
_COMPOSITE_MAP: dict[str, str] = {
    "normal": "Normal",
    "add": "Add",
    "subtract": "Subtract",
    "difference": "Difference",
    "multiply": "Multiply",
    "screen": "Screen",
    "overlay": "Overlay",
    "soft_light": "SoftLight",
    "hard_light": "HardLight",
    "color_dodge": "ColorDodge",
    "color_burn": "ColorBurn",
    "darken": "Darken",
    "lighten": "Lighten",
}
_MARKER_COLOR_MAP: dict[str, str] = {
    "blue": "Blue", "cyan": "Cyan", "green": "Green", "yellow": "Yellow",
    "red": "Red", "pink": "Pink", "purple": "Purple", "fuchsia": "Fuchsia",
    "rose": "Rose", "lavender": "Lavender", "sky": "Sky", "mint": "Mint",
    "lemon": "Lemon", "sand": "Sand", "cocoa": "Cocoa", "cream": "Cream",
}


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
        # Session-scoped stable-ID registry: stable id -> identity key
        # (pool_item_name, media_type, within_type_track_index, start_frame), plus
        # the live object for this process. Ids stay valid for the server-process
        # lifetime; they are NOT stable across restarts (Resolve gives us no ids).
        self._item_registry: dict[str, tuple[str, str, int, int]] = {}
        self._item_by_id: dict[str, Any] = {}
        self._next_item_num = 1
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
        names = pm.GetProjectListInCurrentFolder()
        if not isinstance(names, list) or not names:
            proj = pm.GetCurrentProject()
            return [proj.GetName()] if proj is not None else []
        return [str(n) for n in names]

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
        # SaveProject lives on the ProjectManager, not the Project (documented).
        pm.SaveProject()
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
        if bin is not None:
            root = mp.GetRootFolder()
            folder = self._find_folder(root, bin) if root is not None else None
            if folder is None and root is not None:
                folder = mp.AddSubFolder(root, bin)
            if folder is None:
                from .backend import InvalidStateError
                raise InvalidStateError(f"could not find or create bin {bin!r}")
            if not mp.SetCurrentFolder(folder):
                from .backend import InvalidStateError
                raise InvalidStateError(f"could not select bin {bin!r}")
        items = mp.ImportMedia(paths)
        if not items:
            from .backend import ResolveMCPError as _E
            raise _E("ImportMedia returned no clips (empty paths or unsupported files)")
        return [self._cv_to_clip(it) for it in items]

    def create_bin(self, name: str) -> Bin:
        proj, _ = self._require_project()
        mp = proj.GetMediaPool()
        if mp is None:
            from .backend import InvalidStateError
            raise InvalidStateError("project has no media pool")
        root = mp.GetRootFolder()
        created = mp.AddSubFolder(root, name) if root is not None else None
        if not created:
            from .backend import InvalidStateError
            raise InvalidStateError(f"AddSubFolder refused bin {name!r} (already exists?)")
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
        proj, _ = self._require_project()
        tl = proj.GetCurrentTimeline()
        if tl is None:
            from .backend import InvalidStateError
            raise InvalidStateError("no timeline selected")
        before = self._hydrate_timeline_state(tl)
        nvideo, naudio = self._track_counts(tl)
        if timeline_track_index < 1 or timeline_track_index > nvideo + naudio:
            from .backend import NotFoundError
            msg = (
                f"track index out of range: {timeline_track_index}/{nvideo + naudio} "
                f"({nvideo} video + {naudio} audio)"
            )
            raise NotFoundError(msg)
        pool_item = self._lookup_pool_item(proj.GetMediaPool(), media_clip_id)
        if pool_item is None:
            from .backend import NotFoundError
            msg = f"media clip {media_clip_id!r} not resolvable on the live pool"
            raise NotFoundError(msg)
        fps_v = self._fps(fps_of_timeline(tl))
        info = {
            "mediaPoolItem": pool_item,
            "mediaType": 1 if timeline_track_index <= nvideo else 2,
            "trackIndex": timeline_track_index if timeline_track_index <= nvideo else timeline_track_index - nvideo,
            "startFrame": round(source_in_seconds * fps_v),
            "endFrame": round((source_in_seconds + duration_seconds) * fps_v),
            "recordFrame": round(start_seconds * fps_v),
        }
        media_pool = proj.GetMediaPool()
        items = media_pool.AppendToTimeline([info])
        if not items:
            from .backend import InvalidStateError
            raise InvalidStateError("AppendToTimeline returned no items")
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
        # Documented placement: clipInfo["recordFrame"] lands the clip at a given
        # record position; no shifting of later items is performed (honest API).
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
        item = self._item_by_id.get(timeline_item_id)
        if item is None:
            from .backend import NotFoundError
            msg = f"timeline item {timeline_item_id!r} not found"
            raise NotFoundError(msg)
        item.Delete()
        after = self._hydrate_timeline_state(tl)
        return self._state_delta_from_before_after(
            before, after, changed_path=f"timelines.{tl.GetName()}.tracks"
        )

    def move_clip(self, timeline_item_id: str, new_position_seconds: float) -> StateDelta:
        proj, _ = self._require_project()
        tl = proj.GetCurrentTimeline()
        if tl is None:
            from .backend import InvalidStateError
            raise InvalidStateError("no timeline selected")
        before = self._hydrate_timeline_state(tl)
        item = self._item_by_id.get(timeline_item_id)
        if item is None:
            from .backend import NotFoundError
            msg = f"timeline item {timeline_item_id!r} not found"
            raise NotFoundError(msg)
        pool_item = item.GetMediaPoolItem()
        if pool_item is None:
            from .backend import InvalidStateError
            msg = f"timeline item {timeline_item_id!r} has no resolvable source clip"
            raise InvalidStateError(msg)
        src_in = int(item.GetSourceStartFrame())
        src_end = int(item.GetSourceEndFrame())
        key = self._item_registry.get(timeline_item_id)
        if key is None:
            from .backend import InvalidStateError
            msg = f"timeline item {timeline_item_id!r} missing registry identity"
            raise InvalidStateError(msg)
        _, tr_type, tr_idx, _ = key
        item.Delete()
        info = {
            "mediaPoolItem": pool_item,
            "mediaType": 1 if tr_type == "video" else 2,
            "trackIndex": tr_idx,
            "startFrame": src_in,
            "endFrame": src_end,
            "recordFrame": round(new_position_seconds * self._fps(fps_of_timeline(tl))),
        }
        items = proj.GetMediaPool().AppendToTimeline([info])
        if not items:
            from .backend import InvalidStateError
            raise InvalidStateError("AppendToTimeline returned no items on move")
        after = self._hydrate_timeline_state(tl)
        return self._state_delta_from_before_after(
            before, after, changed_path=f"timelines.{tl.GetName()}.tracks"
        )
    # ---- Phase 2 LIVE — per-item mutations ----------------------------------

    # ---- Phase 2 LIVE — transitions & render ---------------------------------

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
        clip = self._lookup_item(tl, timeline_item_id)
        self._set_props(
            clip,
            {
                "Pan": float(pan_x),
                "Tilt": float(pan_y),
                "ZoomX": float(zoom_x),
                "ZoomY": float(zoom_y),
                "RotationAngle": float(rotation),
            },
        )
        after = self._hydrate_timeline_state(tl)
        return self._state_delta_from_before_after(
            before, after, changed_path=f"timelines.{tl.GetName()}.items[{timeline_item_id}].transform"
        )

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
        clip = self._lookup_item(tl, timeline_item_id)
        self._set_props(
            clip,
            {
                "CropLeft": float(left),
                "CropRight": float(right),
                "CropTop": float(top),
                "CropBottom": float(bottom),
            },
        )
        after = self._hydrate_timeline_state(tl)
        return self._state_delta_from_before_after(
            before, after, changed_path=f"timelines.{tl.GetName()}.items[{timeline_item_id}].crop"
        )

    def set_composite_mode(self, timeline_item_id: str, mode: CompositeMode | str) -> StateDelta:
        proj, _ = self._require_project()
        tl = proj.GetCurrentTimeline()
        if tl is None:
            from .backend import InvalidStateError
            raise InvalidStateError("no timeline selected")
        before = self._hydrate_timeline_state(tl)
        clip = self._lookup_item(tl, timeline_item_id)
        mode_str = mode.value if isinstance(mode, CompositeMode) else str(mode)
        self._set_props(clip, {"CompositeMode": _COMPOSITE_MAP.get(mode_str.lower(),
                        mode_str.replace("_", " ").title().replace(" ", ""))})
        after = self._hydrate_timeline_state(tl)
        return self._state_delta_from_before_after(
            before, after, changed_path=f"timelines.{tl.GetName()}.items[{timeline_item_id}].composite_mode"
        )

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
        clip = self._lookup_item(tl, timeline_item_id)
        # Resolve property "Opacity" is a percentage 0-100.
        self._set_props(clip, {"Opacity": opacity * 100.0})
        after = self._hydrate_timeline_state(tl)
        return self._state_delta_from_before_after(
            before, after, changed_path=f"timelines.{tl.GetName()}.items[{timeline_item_id}].opacity"
        )

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
        clip = self._lookup_item(tl, timeline_item_id)
        fps_v = self._fps(fps_of_timeline(tl))
        duration_frames = max(int(clip.GetDuration()), 1)
        fade_in_frames = min(round(fade_in_seconds * fps_v), duration_frames)
        fade_out_frames = min(round(fade_out_seconds * fps_v), duration_frames)
        # Fade handles are frame positions relative to the item start, one key per call.
        self._set_props(
            clip,
            {
                "FadeInStart": 0 if fade_in_frames else -1,
                "FadeInEnd": fade_in_frames if fade_in_frames else -1,
                "FadeOutStart": (duration_frames - fade_out_frames) if fade_out_frames else -1,
                "FadeOutEnd": duration_frames if fade_out_frames else -1,
            },
        )
        after = self._hydrate_timeline_state(tl)
        return self._state_delta_from_before_after(
            before, after, changed_path=f"timelines.{tl.GetName()}.items[{timeline_item_id}].fades"
        )

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
        clip = self._lookup_item(tl, timeline_item_id)
        self._set_props(clip, {"Speed": float(speed)})
        after = self._hydrate_timeline_state(tl)
        return self._state_delta_from_before_after(
            before, after, changed_path=f"timelines.{tl.GetName()}.items[{timeline_item_id}].speed"
        )

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
        clip = self._lookup_item(tl, timeline_item_id)
        color_str = color.value if isinstance(color, MarkerColor) else str(color)
        color_cap = _MARKER_COLOR_MAP.get(color_str.lower(), color_str.title())
        frame = round(position_seconds * self._fps(fps_of_timeline(tl)))
        if not clip.AddMarker(frame, color_cap, label, note):
            from .backend import InvalidStateError
            raise InvalidStateError(f"AddMarker rejected frame={frame} color={color_cap!r}")
        after = self._hydrate_timeline_state(tl)
        return self._state_delta_from_before_after(
            before, after, changed_path=f"timelines.{tl.GetName()}.items[{timeline_item_id}].markers"
        )

    # ---- Phase 2 LIVE — transitions & render ---------------------------------

    def add_transition(
        self,
        timeline_item_id: str,
        track_index: int,
        style: TransitionStyle | str,
        duration_seconds: float,
        alignment: TransitionAlignment | str,
    ) -> StateDelta:
        # Resolve's documented scripting API has NO transition mutation entry point.
        # Honest degradation: refuse instead of fabricating success. The fake backend
        # supports transitions for offline planning; real Resolve needs manual work.
        msg = (
            "add_transition is not scriptable via Resolve's documented API "
            f"(style={style!r}, track={track_index}). Apply transitions in Resolve UI "
            "or via the offline (fake) backend."
        )
        raise ResolveUnavailableError(msg)

    def add_render_job(
        self,
        timeline_name: str,
        format: RenderJobFormat | str,
        output_path: str,
    ) -> RenderJob:
        proj, _ = self._require_project()
        fmt_str = format.value if isinstance(format, RenderJobFormat) else str(format)
        count, target = int(proj.GetTimelineCount() or 0), None
        for i in range(1, count + 1):
            cand = proj.GetTimelineByIndex(i)
            if cand is not None and cand.GetName() == timeline_name:
                target = cand
                break
        if target is None:
            from .backend import NotFoundError
            msg = f"timeline {timeline_name!r} not found in current project"
            raise NotFoundError(msg)
        if not proj.SetCurrentTimeline(target):
            from .backend import InvalidStateError
            raise InvalidStateError(f"could not activate timeline {timeline_name!r}")
        out = pathlib.Path(output_path)
        settings: dict[str, str] = {"TargetDir": str(out.parent), "CustomName": out.name}
        if not proj.SetRenderSettings(settings):
            from .backend import InvalidStateError
            raise InvalidStateError(f"SetRenderSettings rejected {settings!r}")
        job_id = proj.AddRenderJob()
        if not job_id:
            from .backend import InvalidStateError
            raise InvalidStateError("AddRenderJob returned no job id")
        return RenderJob(
            id=str(job_id),
            timeline_name=timeline_name,
            format=fmt_str,  # type: ignore[arg-type]
            output_path=output_path,
            status=RenderJobStatus.QUEUED,
            progress=0.0,
        )

    def start_render(self, job_id: str) -> RenderJob:
        proj, _ = self._require_project()
        known = {str(j.get("JobId")) for j in (proj.GetRenderJobList() or []) if isinstance(j, dict)}
        if job_id not in known:
            from .backend import NotFoundError
            msg = f"render job {job_id!r} not found"
            raise NotFoundError(msg)
        if not proj.StartRendering(job_id):
            from .backend import InvalidStateError
            raise InvalidStateError(f"StartRendering refused job {job_id!r}")
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
            info = proj.GetRenderJobStatus(job_id)
        except Exception as exc:
            from .backend import NotFoundError
            raise NotFoundError(f"render job {job_id!r} not found: {exc}") from exc
        if not isinstance(info, dict) or not info:
            from .backend import NotFoundError
            raise NotFoundError(f"render job {job_id!r} not found")
        status_text = str(info.get("JobStatus") or info.get("Status") or "queued")
        try:
            progress = float(info.get("CompletionPercentage", 0.0)) / 100.0
        except (TypeError, ValueError):
            progress = 0.0
        return RenderJob(
            id=job_id,
            timeline_name="",
            format=RenderJobFormat.MP4,
            output_path="",
            status=_resolve_render_status_enum(status_text),
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
        count, target = int(proj.GetTimelineCount() or 0), None
        before_state = None
        for i in range(1, count + 1):
            cand = proj.GetTimelineByIndex(i)
            if cand is not None and cand.GetName() == name:
                target = cand
                before_state = self._hydrate_timeline_state(cand)
                break
        if target is None or before_state is None:
            from .backend import NotFoundError
            msg = f"timeline {name!r} not found in current project"
            raise NotFoundError(msg)
        mp = proj.GetMediaPool()
        deleter = getattr(mp, "DeleteTimelines", None)
        if not callable(deleter):
            msg = (
                "MediaPool.DeleteTimelines is unavailable (Resolve earlier than 18.x); "
                f"delete timeline {name!r} manually in the UI."
            )
            raise ResolveUnavailableError(msg)
        if not deleter([target]):
            from .backend import InvalidStateError
            raise InvalidStateError(f"DeleteTimelines refused timeline {name!r}")
        after_state = before_state.model_copy(update={"tracks": [], "duration_seconds": 0.0})
        return StateDelta(
            before=before_state.model_dump(mode="json"),
            after=after_state.model_dump(mode="json"),
            changed_paths=[f"timelines.{name}.deleted"],
        )

    def delete_media(self, media_clip_id: str, confirm: bool = False) -> dict[str, Any]:
        if not confirm:
            from .backend import DestructiveDisabledError
            raise DestructiveDisabledError("destructive call requires confirm=true")
        proj, _ = self._require_project()
        mp = proj.GetMediaPool()
        if mp is None:
            from .backend import InvalidStateError
            raise InvalidStateError("project has no media pool")
        clip = self._lookup_pool_item(mp, media_clip_id)
        if clip is None:
            from .backend import NotFoundError
            raise NotFoundError(f"media clip {media_clip_id!r} not resolvable in pool")
        if not mp.DeleteClips([clip]):
            from .backend import InvalidStateError
            raise InvalidStateError(f"DeleteClips refused media {media_clip_id!r}")
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
        """FrameRate for a Resolve Timeline object via documented GetSetting."""
        try:
            v = timeline_obj.GetSetting("timelineFrameRate")
            if v:
                return FrameRate(fps=float(v), drop_frame=False)
        except Exception:
            pass
        try:
            return self._safe_fps(self._require_project()[0])
        except Exception:
            return FrameRate(fps=24.0, drop_frame=False)

    def _fps(self, frame_rate: FrameRate) -> float:
        """Return the nominal integer fps (drop-frame math uses 30 internally)."""
        fps = float(frame_rate.fps)
        if abs(fps) < 1e-6:
            return 24.0
        return round(fps) if abs(fps - round(fps)) < 1e-3 else fps

    def _hydrate_timeline_state(self, tl: Any) -> TimelineState:
        """Read a Resolve Timeline into a TimelineState with session-stable item ids.

        Resolve's scripting API exposes no uuid for timeline items, so we mint
        opaque ids (``tl_item_<n>``) keyed by ``(clip name, media_type,
        track_index_within_type, start_frame)`` and reuse them across scans: ids
        handed out by one hydrate stay resolvable for the life of this process.
        They are not stable across server restarts — documented in README.
        """
        name = tl.GetName()
        fr = self._timeline_fr(tl)
        nvideo, naudio = self._track_counts(tl)
        fps_v = max(self._fps(fr), 1.0)
        tracks: list[Track] = []
        duration_end = 0.0
        fresh: dict[str, Any] = {}
        wire_index = 0
        for media_type, count in (("video", nvideo), ("audio", naudio)):
            kind = TrackKind.VIDEO if media_type == "video" else TrackKind.AUDIO
            for intra_track in range(1, count + 1):
                wire_index += 1
                try:
                    raw_items = tl.GetItemListInTrack(media_type, intra_track) or []
                except Exception:
                    raw_items = []
                track_items: list[TimelineItem] = []
                for it in raw_items:
                    start_frame = int(it.GetStart())
                    dur_frames = int(it.GetDuration())
                    start_seconds = round(start_frame / fps_v, 3)
                    duration_seconds = round(dur_frames / fps_v, 3)
                    duration_end = max(duration_end, start_seconds + duration_seconds)
                    key = (str(it.GetName()), media_type, intra_track, start_frame)
                    ident = next(
                        (sid for sid, k in self._item_registry.items() if k == key), None
                    )
                    if ident is None:
                        ident = f"tl_item_{self._next_item_num}"
                        self._next_item_num += 1
                        self._item_registry[ident] = key
                    track_items.append(
                        TimelineItem(
                            id=ident,
                            track_index=wire_index,
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
                    )
                    fresh[ident] = it
                tracks.append(Track(index=wire_index, kind=kind, items=track_items))
        self._item_by_id = fresh
        return TimelineState(
            name=name,
            frame_rate=fr,
            duration_seconds=duration_end,
            tracks=tracks,
            transitions=[],
        )

    def _track_counts(self, tl: Any) -> tuple[int, int]:
        """Return ``(video_track_count, audio_track_count)`` via the real API."""
        def _count(kind: str) -> int:
            try:
                return max(0, int(tl.GetTrackCount(kind)))
            except Exception:
                return 0
        return _count("video"), _count("audio")

    def _set_props(self, clip: Any, props: dict[str, Any]) -> None:
        """Apply documented per-key properties; refuse silent failure."""
        for key, value in props.items():
            ok = clip.SetProperty(key, value)
            if not ok:
                from .backend import InvalidStateError
                raise InvalidStateError(f"SetProperty({key!r}) rejected value {value!r}")

    def _lookup_item(self, tl: Any, timeline_item_id: str) -> Any:
        """Return the live TimelineItem for a stable id, or raise NotFoundError."""
        self._hydrate_timeline_state(tl)  # refresh object map; ids stay stable via registry
        item = self._item_by_id.get(timeline_item_id)
        if item is None:
            from .backend import NotFoundError
            msg = f"timeline item {timeline_item_id!r} not found"
            raise NotFoundError(msg)
        return item

    @staticmethod
    def _find_folder(folder: Any, name: str) -> Any | None:
        if folder is None:
            return None
        try:
            if folder.GetName() == name:
                return folder
        except Exception:
            return None
        try:
            subs = folder.GetSubFolderList() or []
        except Exception:
            return None
        for sub in subs:
            hit = DaVinciResolveBackend._find_folder(sub, name)
            if hit is not None:
                return hit
        return None

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
