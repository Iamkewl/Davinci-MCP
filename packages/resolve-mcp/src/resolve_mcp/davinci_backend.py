"""Real backend: thin wrapper over DaVinci Resolve's scripting API.

Every call in this module uses ONLY the documented Resolve scripting surface
(Blackmagic's "DaVinci Resolve Scripting API" README, v18+, as re-verified in
``scratchpad/resolve-api-audit/resolve_api_reference.md``). Anything without a
documented entry point raises :class:`~resolve_mcp.backend.UnsupportedOperationError`
instead of fabricating success -- see :meth:`DaVinciResolveBackend.add_fade`,
:meth:`~DaVinciResolveBackend.set_speed`, :meth:`~DaVinciResolveBackend.add_transition`,
and :meth:`~DaVinciResolveBackend.restart_app`.

Key facts this module encodes (see the reference doc for citations):

* IDs are Resolve's own ``GetUniqueId()`` (``GetMediaId()`` fallback for media
  pool items), never Python object identity -- the scripting bridge hands out a
  fresh proxy wrapper on every ``Get*`` call, so ``id(obj)`` never survives
  between calls. Every lookup walks the pool/tracks comparing unique ids.
* Frame numbers returned by ``TimelineItem.GetStart()/GetEnd()`` are in the
  same absolute coordinate space as ``Timeline.GetStartFrame()`` (commonly
  86400 @ 24fps for a 01:00:00:00 start) -- every read and write offsets by it.
* There is no documented API to reposition an item in place or to shift later
  items on ripple-insert; both are implemented as capture -> ``DeleteClips`` ->
  ``AppendToTimeline`` -> restore, which mints a new ``TimelineItem`` (hence
  ``StateDelta.id_remap``) and does not carry over grades, versions, or Fusion
  compositions -- Resolve's scripting API has no call that reads or clones them.

This module is *deliberately* defensive about importing the Resolve scripting
module:

* The ``import DaVinciResolveScript`` line lives inside a guarded block so the
  server can boot even when Resolve is absent (CI, dev laptops without Studio
  installed).
* ``_require()`` lazily (re)connects on every call, so starting Resolve *after*
  this server process has already booted still works without a restart.

This module is the ONLY place in ``resolve-mcp`` that imports
``DaVinciResolveScript``. If you find yourself importing it elsewhere,
something is wrong with the boundary.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import platform
import re
import sys
from typing import Any

from .backend import (
    AlreadyExistsError,
    DestructiveDisabledError,
    InvalidStateError,
    NotFoundError,
    ResolveMCPError,
    ResolveUnavailableError,
    UnsupportedOperationError,
)
from .schemas import (
    COMPOSITE_CONSTANT_NAMES,
    Bin,
    CompositeMode,
    Crop,
    FrameRate,
    Marker,
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
    TimelineSummary,
    Track,
    TrackKind,
    Transform,
    TransitionAlignment,
    TransitionStyle,
)
from .timecode import TimeConverter

# --- module-level constants & small pure helpers --------------------------------

#: NTSC-ish rates that take SMPTE drop-frame timecode and Resolve's decimal spelling.
_NTSC_RATES: tuple[float, ...] = (23.976, 29.97, 59.94)

_FPS_RE = re.compile(r"[-+]?\d*\.?\d+")

#: Resolve render-job status strings (documented values: Ready/Rendering/Complete/
#: Failed/Cancelled) mapped onto our RenderJobStatus enum.
_RENDER_STATUS_MAP: dict[str, RenderJobStatus] = {
    "ready": RenderJobStatus.QUEUED,
    "rendering": RenderJobStatus.RUNNING,
    "complete": RenderJobStatus.COMPLETED,
    "failed": RenderJobStatus.FAILED,
    "cancelled": RenderJobStatus.CANCELLED,
    "canceled": RenderJobStatus.CANCELLED,
}

#: RenderJobFormat -> (acceptable file extensions from GetRenderFormats(), codec
#: description substring to look for in GetRenderCodecs()). Extensions are tried
#: in order; the first one Resolve actually offers wins.
_RENDER_FORMAT_HINTS: dict[RenderJobFormat, tuple[tuple[str, ...], str]] = {
    RenderJobFormat.MP4: (("mp4",), "H.264"),
    RenderJobFormat.MOV: (("mov",), "H.264"),
    RenderJobFormat.PRORES: (("mov",), "ProRes"),
    RenderJobFormat.DNXHR: (("mov", "mxf"), "DNxHR"),
}


def _default_modules_dir() -> str | None:
    """Per-OS default ``.../Developer/Scripting/Modules`` dir (reference doc section 1)."""
    system = platform.system()
    if system == "Windows":
        root = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
        path = pathlib.Path(root) / "Blackmagic Design" / "DaVinci Resolve"
        path = path / "Support" / "Developer" / "Scripting" / "Modules"
        return str(path)
    if system == "Darwin":
        return "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules"
    if system == "Linux":
        return "/opt/resolve/Developer/Scripting/Modules"
    return None


def _bootstrap_sys_path() -> None:
    """Ensure ``DaVinciResolveScript.py`` is importable before we try.

    Per the bootstrap contract (README.md 'Resolve bootstrap'; reference doc
    section 1), ``PYTHONPATH`` must include ``$RESOLVE_SCRIPT_API/Modules``.
    We add that plus the per-OS default install location so the server still
    works when the caller forgot to export ``PYTHONPATH`` -- everything else
    (``RESOLVE_SCRIPT_LIB``, the External-scripting preference) has no
    sys.path equivalent and is only surfaced in the connection-error message.
    """
    candidates: list[str] = []
    api_root = os.environ.get("RESOLVE_SCRIPT_API")
    if api_root:
        candidates.append(str(pathlib.Path(api_root) / "Modules"))
    default_dir = _default_modules_dir()
    if default_dir:
        candidates.append(default_dir)
    for candidate in candidates:
        if candidate not in sys.path and pathlib.Path(candidate).is_dir():
            sys.path.append(candidate)


def _try_import_resolve() -> Any | None:
    """Best-effort import; returns the module or ``None`` if unavailable."""
    _bootstrap_sys_path()
    try:
        import DaVinciResolveScript as dvr
    except Exception:
        return None
    return dvr


def _parse_fps(value: Any) -> float:
    """Parse a Resolve fps-ish setting ('24', '29.97', '29.97 DF') into a float.

    Tolerant of a trailing drop-frame marker or unit suffix; defaults to 24.0
    when nothing numeric can be found (never silently produces 0).
    """
    if value is None:
        return 24.0
    match = _FPS_RE.match(str(value).strip())
    if not match:
        return 24.0
    try:
        parsed = float(match.group(0))
    except ValueError:
        return 24.0
    return parsed if parsed > 0 else 24.0


def _sanitize_frame_rate(fps: float, drop_frame: bool) -> FrameRate:
    """Drop-frame is only meaningful at 29.97/59.94 (``TimeConverter`` rejects
    anything else); silently correct rather than raise on hydration.
    """
    if drop_frame and not any(abs(fps - rate) < 0.01 for rate in (29.97, 59.94)):
        drop_frame = False
    return FrameRate(fps=fps, drop_frame=drop_frame)


def _fps_to_setting_str(fps: float) -> str:
    """Format fps the way Resolve's ``timelineFrameRate`` setting documents
    ('24', '23.976', '29.97', '59.94').
    """
    for candidate in _NTSC_RATES:
        if abs(fps - candidate) < 0.01:
            return f"{candidate:g}"
    if abs(fps - round(fps)) < 1e-6:
        return str(round(fps))
    return f"{fps:g}"


def _approx_equal(actual: Any, expected: Any, tol: float = 1e-3) -> bool:
    """Numeric-tolerant equality for verifying a Resolve property read-back."""
    try:
        return abs(float(actual) - float(expected)) <= tol
    except (TypeError, ValueError):
        return bool(actual == expected)


def _parse_resolution(value: Any) -> tuple[int | None, int | None]:
    """Parse ``GetClipProperty()['Resolution']`` ('1920x1080') into (w, h)."""
    if not value:
        return None, None
    try:
        w_str, h_str = str(value).lower().split("x")
        return int(w_str.strip()), int(h_str.strip())
    except (ValueError, AttributeError):
        return None, None


class DaVinciResolveBackend:
    """Real Resolve backend. Class exists regardless of whether Resolve is installed.

    Construction does NOT raise. Methods raise
    :class:`~resolve_mcp.backend.ResolveUnavailableError` only when invoked
    without a working Resolve connection, and
    :class:`~resolve_mcp.backend.UnsupportedOperationError` for the four
    operations Resolve's scripting API has no entry point for at all
    (see :meth:`unsupported_tools`).
    """

    def __init__(self, *, allow_destructive: bool = False) -> None:
        self._resolve_module: Any | None = _try_import_resolve()
        self._resolve: Any | None = None
        self._project_manager: Any | None = None
        self._allow_destructive: bool = allow_destructive
        self._render_jobs: dict[str, RenderJob] = {}
        #: Lazily-built ``resolve.COMPOSITE_*`` int -> CompositeMode reverse map.
        self._composite_reverse: dict[Any, CompositeMode] | None = None
        self._connect()

    # --- capabilities ----------------------------------------------------------

    def unsupported_tools(self) -> frozenset[str]:
        """Tools with no documented Resolve scripting entry point.

        ``add_fade``/``set_speed``: ``TimelineItem.SetProperty``'s documented
        key list has no Fade*/Speed key (reference verdict #4). ``add_transition``:
        no transition-insertion call exists anywhere in the scripting surface
        (verdict re: add_transition). ``restart_app``: the Resolve object exposes
        ``Quit()`` but no Restart/Relaunch call (verdict #11).
        """
        return frozenset({"add_fade", "set_speed", "add_transition", "restart_app"})

    # --- connection --------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._resolve is not None and self._project_manager is not None

    def _connect(self) -> None:
        """(Re)attempt ``DaVinciResolveScript.scriptapp("Resolve")`` +
        ``Resolve.GetProjectManager()``.

        Called from every :meth:`_require`, so starting Resolve after this
        server process has already booted still works without a restart.
        """
        if self._resolve is not None and self._project_manager is not None:
            return
        if self._resolve_module is None:
            self._resolve_module = _try_import_resolve()
        if self._resolve_module is None:
            return
        try:
            resolve_app = self._resolve_module.scriptapp("Resolve")
        except Exception:
            resolve_app = None
        if resolve_app is None:
            return
        try:
            project_manager = resolve_app.GetProjectManager()
        except Exception:
            project_manager = None
        if project_manager is None:
            return
        self._resolve = resolve_app
        self._project_manager = project_manager

    def _require(self) -> tuple[Any, Any]:
        """Ensure a live connection (retrying via :meth:`_connect`) or raise."""
        self._connect()
        if self._resolve is None or self._project_manager is None:
            env = {
                "RESOLVE_SCRIPT_API": os.environ.get("RESOLVE_SCRIPT_API", "<unset>"),
                "RESOLVE_SCRIPT_LIB": os.environ.get("RESOLVE_SCRIPT_LIB", "<unset>"),
            }
            msg = (
                "DaVinci Resolve is not reachable via its scripting API. Checklist: "
                "(1) DaVinci Resolve STUDIO must be running -- external scripting is a "
                "Studio-only feature, the free edition has no scripting API; "
                "(2) Preferences -> General -> External scripting using = Local; "
                f"(3) env vars must point at the Resolve install: {env!r} "
                "(see README.md 'Resolve bootstrap')."
            )
            raise ResolveUnavailableError(msg)
        return self._resolve, self._project_manager

    def _require_project(self) -> Any:
        """Return the current ``Project`` (via ``ProjectManager.GetCurrentProject``)."""
        _, pm = self._require()
        proj = pm.GetCurrentProject()
        if proj is None:
            raise InvalidStateError("no project open")
        return proj

    def _require_timeline(self, proj: Any) -> Any:
        """Return the current ``Timeline`` (via ``Project.GetCurrentTimeline``)."""
        tl = proj.GetCurrentTimeline()
        if tl is None:
            raise InvalidStateError("no timeline selected")
        return tl

    # --- projects ------------------------------------------------------------

    def list_projects(self) -> list[str]:
        """``ProjectManager.GetProjectListInCurrentFolder()``; falls back to the
        current project's name alone if the folder listing comes back empty.
        """
        _, pm = self._require()
        names = pm.GetProjectListInCurrentFolder()
        if not isinstance(names, list) or not names:
            proj = pm.GetCurrentProject()
            return [str(proj.GetName())] if proj is not None else []
        return [str(n) for n in names]

    def create_project(
        self,
        name: str,
        frame_rate: FrameRate | dict[str, Any] | float,
        width: int,
        height: int,
    ) -> ProjectInfo:
        """``ProjectManager.CreateProject(name)``, then apply fps/resolution/
        drop-frame via ``Project.SetSetting`` (reference verdict #9) and read
        every value back via ``GetSetting`` -- never fabricate what we asked for.
        """
        _, pm = self._require()
        fr = self._coerce_frame_rate(frame_rate)
        proj = pm.CreateProject(name)
        if proj is None:
            msg = f"failed to create project {name!r} -- does it already exist?"
            raise AlreadyExistsError(msg)
        proj.SetSetting("timelineFrameRate", _fps_to_setting_str(fr.fps))
        proj.SetSetting("timelineResolutionWidth", str(width))
        proj.SetSetting("timelineResolutionHeight", str(height))
        proj.SetSetting("timelineDropFrameTimecode", "1" if fr.drop_frame else "0")
        return self._read_project_info(proj)

    def open_project(self, name: str) -> ProjectInfo:
        """``ProjectManager.LoadProject(name)``."""
        _, pm = self._require()
        proj = pm.LoadProject(name)
        if proj is None:
            msg = f"project {name!r} not found at project-manager default path"
            raise NotFoundError(msg)
        return self._read_project_info(proj)

    def save_project(self) -> ProjectInfo:
        """``ProjectManager.SaveProject()`` (lives on the manager, not the project)."""
        _, pm = self._require()
        proj = pm.GetCurrentProject()
        if proj is None:
            raise InvalidStateError("no project open")
        if not pm.SaveProject():
            raise ResolveMCPError("ProjectManager.SaveProject() returned False")
        return self._read_project_info(proj)

    def current_project(self) -> ProjectInfo:
        proj = self._require_project()
        return self._read_project_info(proj)

    def _read_project_info(self, proj: Any) -> ProjectInfo:
        """Read settings back via ``Project.GetSetting``.

        ``Project.GetResolutionWidth/Height`` were not independently confirmed
        in the API reference audit -- we deliberately avoid them and read the
        same ``timelineResolutionWidth/Height`` settings we (or Resolve's
        project defaults) wrote, exactly like ``timelineFrameRate``.
        """
        fps = _parse_fps(self._safe_get_setting(proj, "timelineFrameRate", "24"))
        drop_raw = self._safe_get_setting(proj, "timelineDropFrameTimecode", "0")
        drop_frame = str(drop_raw).strip() in ("1", "true", "True")
        fr = _sanitize_frame_rate(fps, drop_frame)
        width = self._safe_get_setting_int(proj, "timelineResolutionWidth", 1920)
        height = self._safe_get_setting_int(proj, "timelineResolutionHeight", 1080)
        return ProjectInfo(
            name=str(proj.GetName()),
            frame_rate=fr,
            resolution_width=width,
            resolution_height=height,
            path=None,
            is_modified=False,
        )

    @staticmethod
    def _safe_get_setting(proj: Any, key: str, default: str) -> str:
        try:
            v = proj.GetSetting(key)
            return str(v) if v else default
        except Exception:
            return default

    @staticmethod
    def _safe_get_setting_int(proj: Any, key: str, default: int) -> int:
        try:
            v = proj.GetSetting(key)
            return int(float(v)) if v else default
        except (TypeError, ValueError):
            return default
        except Exception:
            return default

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

    # --- media pool ----------------------------------------------------------

    def list_media_pool(self) -> MediaPoolState:
        """Walk ``MediaPool.GetRootFolder()`` via ``GetClipList``/``GetSubFolderList``."""
        proj = self._require_project()
        mp = proj.GetMediaPool()
        if mp is None:
            return MediaPoolState(bins=[], clips=[])
        root = mp.GetRootFolder()
        bins: list[Bin] = []
        clips: list[MediaClip] = []
        if root is not None:
            self._walk_bin(str(root.GetName()), root, bins, clips)
        return MediaPoolState(bins=bins, clips=clips)

    def _walk_bin(self, name: str, folder: Any, bins: list[Bin], clips: list[MediaClip]) -> None:
        try:
            clip_list = folder.GetClipList() or []
        except Exception:
            clip_list = []
        ids: list[str] = []
        for c in clip_list:
            clip = self._cv_to_clip(c, name)
            clips.append(clip)
            ids.append(clip.id)
        bins.append(Bin(name=name, clip_ids=ids))
        try:
            subs = folder.GetSubFolderList() or []
        except Exception:
            subs = []
        for sub in subs:
            self._walk_bin(str(sub.GetName()), sub, bins, clips)

    def import_media(self, paths: list[str], bin: str | None = None) -> list[MediaClip]:
        """``MediaPool.ImportMedia(paths)``, optionally into a named bin via
        ``AddSubFolder``/``SetCurrentFolder`` (created if it doesn't exist).
        """
        proj = self._require_project()
        mp = proj.GetMediaPool()
        if mp is None:
            raise InvalidStateError("project has no media pool")
        root = mp.GetRootFolder()
        bin_name = str(root.GetName()) if root is not None else "Master"
        if bin is not None:
            folder = self._find_folder(root, bin) if root is not None else None
            if folder is None and root is not None:
                folder = mp.AddSubFolder(root, bin)
            if folder is None:
                msg = f"could not find or create bin {bin!r}"
                raise InvalidStateError(msg)
            if not mp.SetCurrentFolder(folder):
                msg = f"could not select bin {bin!r}"
                raise InvalidStateError(msg)
            bin_name = bin
        items = mp.ImportMedia(list(paths))
        if not items:
            raise ResolveMCPError("ImportMedia returned no clips (empty paths or unsupported files)")
        return [self._cv_to_clip(it, bin_name) for it in items]

    def create_bin(self, name: str) -> Bin:
        """``MediaPool.AddSubFolder(root, name)``."""
        proj = self._require_project()
        mp = proj.GetMediaPool()
        if mp is None:
            raise InvalidStateError("project has no media pool")
        root = mp.GetRootFolder()
        created = mp.AddSubFolder(root, name) if root is not None else None
        if not created:
            msg = f"AddSubFolder refused bin {name!r} (already exists?)"
            raise InvalidStateError(msg)
        return Bin(name=name, clip_ids=[])

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

    @staticmethod
    def _pool_item_id(item: Any) -> str:
        """MediaPoolItem id: ``GetUniqueId()``, falling back to ``GetMediaId()``
        (spec section C) -- never Python object identity.
        """
        try:
            uid = item.GetUniqueId()
            if uid:
                return str(uid)
        except Exception:
            pass
        try:
            mid = item.GetMediaId()
            if mid:
                return str(mid)
        except Exception:
            pass
        raise InvalidStateError("media pool item exposes neither GetUniqueId() nor GetMediaId()")

    @staticmethod
    def _media_fps(pool_item: Any) -> float | None:
        """Media's own fps from ``MediaPoolItem.GetClipProperty()['FPS']``, or
        ``None`` when unknown (callers fall back to the timeline's fps).
        """
        try:
            prop = pool_item.GetClipProperty() or {}
        except Exception:
            return None
        raw = prop.get("FPS")
        if not raw:
            return None
        match = _FPS_RE.match(str(raw).strip())
        if not match:
            return None
        try:
            value = float(match.group(0))
        except ValueError:
            return None
        return value if value > 0 else None

    @staticmethod
    def _media_duration_frames(pool_item: Any) -> int | None:
        """``MediaPoolItem.GetClipProperty()['Frames']``, or ``None`` when unknown."""
        try:
            prop = pool_item.GetClipProperty() or {}
        except Exception:
            return None
        raw = prop.get("Frames")
        if raw is None:
            return None
        try:
            return int(float(str(raw)))
        except ValueError:
            return None

    def _cv_to_clip(self, cv: Any, bin_name: str) -> MediaClip:
        """MediaPoolItem -> MediaClip using real ``GetClipProperty()`` metadata:
        duration = Frames/FPS (0.0 when unknown, never fabricated), kind from
        'Type' (Still->image, Audio->audio, else video), id via
        :meth:`_pool_item_id`.
        """
        clip_id = self._pool_item_id(cv)
        try:
            prop = cv.GetClipProperty() or {}
        except Exception:
            prop = {}
        path = str(prop.get("File Path", ""))
        try:
            name = str(cv.GetName())
        except Exception:
            name = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        type_str = str(prop.get("Type", "")).strip().lower()
        if "still" in type_str:
            kind = MediaKind.IMAGE
        elif type_str == "audio":
            kind = MediaKind.AUDIO
        else:
            kind = MediaKind.VIDEO
        fps = self._media_fps(cv)
        frames = self._media_duration_frames(cv)
        duration_seconds = (frames / fps) if (frames is not None and fps) else 0.0
        width, height = _parse_resolution(prop.get("Resolution"))
        return MediaClip(
            id=clip_id,
            bin=bin_name,
            name=name,
            path=path,
            kind=kind,
            duration_seconds=duration_seconds,
            frame_rate=FrameRate(fps=fps, drop_frame=False) if fps else None,
            resolution_width=width,
            resolution_height=height,
        )

    def _lookup_pool_item(self, mp: Any, media_clip_id: str) -> Any:
        try:
            root = mp.GetRootFolder()
        except Exception:
            root = None
        hit = self._find_pool_item(root, media_clip_id) if root is not None else None
        if hit is None:
            msg = f"media clip {media_clip_id!r} not resolvable on the live media pool"
            raise NotFoundError(msg)
        return hit

    def _find_pool_item(self, folder: Any, target_id: str) -> Any | None:
        try:
            clips = folder.GetClipList() or []
        except Exception:
            clips = []
        for c in clips:
            try:
                if self._pool_item_id(c) == target_id:
                    return c
            except Exception:
                continue
        try:
            subs = folder.GetSubFolderList() or []
        except Exception:
            subs = []
        for sub in subs:
            hit = self._find_pool_item(sub, target_id)
            if hit is not None:
                return hit
        return None

    # ---- timeline ------------------------------------------------------------

    def create_timeline(self, name: str, frame_rate: FrameRate | dict[str, Any] | float) -> TimelineState:
        """``MediaPool.CreateEmptyTimeline`` + ``Project.SetCurrentTimeline``;
        if the requested fps differs from the project's, opt the new timeline
        into per-timeline settings via ``Timeline.SetSetting('useCustomSettings', '1')``
        before overriding ``timelineFrameRate`` (reference verdict #9 -- plausible
        ordering from example code, not independently re-verified against a live
        Resolve).
        """
        proj = self._require_project()
        fr = self._coerce_frame_rate(frame_rate)
        mp = proj.GetMediaPool()
        if mp is None:
            raise InvalidStateError("project has no media pool")
        existing = self._find_timeline_by_name(proj, name)
        tl = mp.CreateEmptyTimeline(name)
        if tl is None:
            if existing is not None:
                raise AlreadyExistsError(f"timeline {name!r} already exists")
            raise ResolveMCPError(f"failed to create timeline {name!r}")
        if not proj.SetCurrentTimeline(tl):
            msg = f"could not activate newly-created timeline {name!r}"
            raise InvalidStateError(msg)
        project_fps = _parse_fps(self._safe_get_setting(proj, "timelineFrameRate", "24"))
        if abs(project_fps - fr.fps) > 1e-3:
            with contextlib.suppress(Exception):
                tl.SetSetting("useCustomSettings", "1")
            with contextlib.suppress(Exception):
                tl.SetSetting("timelineFrameRate", _fps_to_setting_str(fr.fps))
        return self._hydrate_timeline_state(tl)

    def list_timelines(self) -> list[TimelineSummary]:
        """``Project.GetTimelineCount()``/``GetTimelineByIndex(1-based)``."""
        proj = self._require_project()
        current = proj.GetCurrentTimeline()
        current_name = str(current.GetName()) if current is not None else None
        count = int(proj.GetTimelineCount() or 0)
        out: list[TimelineSummary] = []
        for i in range(1, count + 1):
            tl = proj.GetTimelineByIndex(i)
            if tl is None:
                continue
            tl_name = str(tl.GetName())
            out.append(TimelineSummary(name=tl_name, is_current=(tl_name == current_name)))
        return out

    def set_current_timeline(self, name: str) -> TimelineState:
        """``Project.SetCurrentTimeline`` on a timeline found via :meth:`_find_timeline_by_name`."""
        proj = self._require_project()
        tl = self._find_timeline_by_name(proj, name)
        if tl is None:
            raise NotFoundError(f"timeline {name!r} not found in current project")
        if not proj.SetCurrentTimeline(tl):
            raise InvalidStateError(f"Project.SetCurrentTimeline refused timeline {name!r}")
        return self._hydrate_timeline_state(tl)

    def _find_timeline_by_name(self, proj: Any, name: str) -> Any | None:
        count = int(proj.GetTimelineCount() or 0)
        for i in range(1, count + 1):
            cand = proj.GetTimelineByIndex(i)
            if cand is not None and cand.GetName() == name:
                return cand
        return None

    def get_timeline_state(self) -> TimelineState:
        proj = self._require_project()
        tl = self._require_timeline(proj)
        return self._hydrate_timeline_state(tl)

    def append_clip(
        self,
        media_clip_id: str,
        timeline_track_index: int,
        start_seconds: float,
        duration_seconds: float,
        source_in_seconds: float = 0.0,
    ) -> StateDelta:
        """``MediaPool.AppendToTimeline([clipInfo])`` with the documented
        ``mediaPoolItem``/``startFrame``/``endFrame``/``mediaType``/``trackIndex``/
        ``recordFrame`` keys. Source-range frames use the media's own fps when
        known (else the timeline's); ``recordFrame`` uses the timeline's fps and
        is offset by ``Timeline.GetStartFrame()``.
        """
        proj = self._require_project()
        tl = self._require_timeline(proj)
        before = self._hydrate_timeline_state(tl)
        nvideo, naudio = self._track_counts(tl)
        if timeline_track_index < 1 or timeline_track_index > nvideo + naudio:
            msg = (
                f"track index out of range: {timeline_track_index}/{nvideo + naudio} "
                f"({nvideo} video + {naudio} audio)"
            )
            raise NotFoundError(msg)
        mp = proj.GetMediaPool()
        pool_item = self._lookup_pool_item(mp, media_clip_id)
        media_type = 1 if timeline_track_index <= nvideo else 2
        intra_track = timeline_track_index if media_type == 1 else timeline_track_index - nvideo
        info = self._build_clip_info(tl, pool_item, media_type, intra_track, start_seconds, duration_seconds, source_in_seconds)
        items = mp.AppendToTimeline([info])
        if not items:
            raise InvalidStateError("AppendToTimeline returned no items (overlapping placement or invalid range?)")
        after = self._hydrate_timeline_state(tl)
        return self._delta(before, after, f"timelines.{tl.GetName()}.tracks[{timeline_track_index}].items")

    def _build_clip_info(
        self,
        tl: Any,
        pool_item: Any,
        media_type: int,
        intra_track: int,
        start_seconds: float,
        duration_seconds: float,
        source_in_seconds: float,
    ) -> dict[str, Any]:
        """Assemble one ``AppendToTimeline`` clipInfo dict; raises if the
        requested source range would exceed the media's own known duration.
        """
        media_fps = self._media_fps(pool_item)
        src_conv = TimeConverter(FrameRate(fps=media_fps, drop_frame=False)) if media_fps else self._converter_of(tl)
        start_frame_src = src_conv.seconds_to_frames_round(source_in_seconds)
        end_frame_src = src_conv.seconds_to_frames_round(source_in_seconds + duration_seconds)
        media_duration_frames = self._media_duration_frames(pool_item)
        if media_duration_frames is not None and end_frame_src > media_duration_frames:
            msg = (
                f"source range [{source_in_seconds}, {source_in_seconds + duration_seconds}]s "
                f"exceeds the media's own duration ({media_duration_frames} frames)"
            )
            raise InvalidStateError(msg)
        tl_conv = self._converter_of(tl)
        record_frame = self._start_frame(tl) + tl_conv.seconds_to_frames_round(start_seconds)
        return {
            "mediaPoolItem": pool_item,
            "startFrame": start_frame_src,
            "endFrame": end_frame_src,
            "mediaType": media_type,
            "trackIndex": intra_track,
            "recordFrame": record_frame,
        }

    def insert_clip(
        self,
        media_clip_id: str,
        timeline_track_index: int,
        timeline_position_seconds: float,
        duration_seconds: float,
        source_in_seconds: float = 0.0,
    ) -> StateDelta:
        """Ripple insert: ``AppendToTimeline`` has no shift-later-items option
        (no documented parameter or side effect does this), so every item on the
        target track whose start is at or after the insertion point is moved
        right by the inserted duration -- via the same delete+re-append
        :meth:`_move_clip_internal` uses for ``move_clip`` -- processed
        furthest-start-first so each move always lands in already-vacated space,
        then the new clip is appended at the freed position.
        """
        proj = self._require_project()
        tl = self._require_timeline(proj)
        before = self._hydrate_timeline_state(tl)
        nvideo, naudio = self._track_counts(tl)
        if timeline_track_index < 1 or timeline_track_index > nvideo + naudio:
            msg = (
                f"track index out of range: {timeline_track_index}/{nvideo + naudio} "
                f"({nvideo} video + {naudio} audio)"
            )
            raise NotFoundError(msg)
        track = next((t for t in before.tracks if t.index == timeline_track_index), None)
        if track is None:
            raise NotFoundError(f"track index {timeline_track_index} does not exist")
        to_shift = sorted(
            (it for it in track.items if it.start_seconds >= timeline_position_seconds),
            key=lambda it: it.start_seconds,
            reverse=True,
        )
        id_remap: dict[str, str] = {}
        for it in to_shift:
            id_remap.update(self._move_clip_internal(proj, tl, it.id, it.start_seconds + duration_seconds))
        self.append_clip(
            media_clip_id=media_clip_id,
            timeline_track_index=timeline_track_index,
            start_seconds=timeline_position_seconds,
            duration_seconds=duration_seconds,
            source_in_seconds=source_in_seconds,
        )
        after = self._hydrate_timeline_state(tl)
        return StateDelta(
            before=before.model_dump(mode="json"),
            after=after.model_dump(mode="json"),
            changed_paths=[f"timelines.{tl.GetName()}.tracks[{timeline_track_index}].items"],
            id_remap=id_remap,
        )

    def delete_clip(self, timeline_item_id: str) -> StateDelta:
        """``Timeline.DeleteClips([item], False)`` -- ``TimelineItem.Delete()``
        does not exist (reference verdict #1, confirmed absent in 4 independent
        method-enumeration sources).
        """
        proj = self._require_project()
        tl = self._require_timeline(proj)
        before = self._hydrate_timeline_state(tl)
        it = self._find_timeline_item(tl, timeline_item_id)
        if not tl.DeleteClips([it], False):
            raise InvalidStateError(f"Timeline.DeleteClips refused to remove item {timeline_item_id!r}")
        after = self._hydrate_timeline_state(tl)
        return self._delta(before, after, f"timelines.{tl.GetName()}.tracks")

    def move_clip(self, timeline_item_id: str, new_position_seconds: float) -> StateDelta:
        """No documented API repositions an item in place (reference verdict
        #5): capture readable state, ``Timeline.DeleteClips`` +
        ``MediaPool.AppendToTimeline`` at the new record frame, then restore.
        Grades, versions, and Fusion compositions are NOT carried over.
        """
        proj = self._require_project()
        tl = self._require_timeline(proj)
        before = self._hydrate_timeline_state(tl)
        remap = self._move_clip_internal(proj, tl, timeline_item_id, new_position_seconds)
        after = self._hydrate_timeline_state(tl)
        return StateDelta(
            before=before.model_dump(mode="json"),
            after=after.model_dump(mode="json"),
            changed_paths=[f"timelines.{tl.GetName()}.tracks"],
            id_remap=remap,
        )

    def _move_clip_internal(self, proj: Any, tl: Any, timeline_item_id: str, new_position_seconds: float) -> dict[str, str]:
        """Shared delete+re-append engine for :meth:`move_clip` and the ripple
        shift inside :meth:`insert_clip`. Returns ``{old_id: new_id}``.
        """
        it, media_type_str, intra_track, _wire = self._locate_timeline_item(tl, timeline_item_id)
        pool_item = it.GetMediaPoolItem()
        if pool_item is None:
            msg = f"timeline item {timeline_item_id!r} has no resolvable source clip (title/generator?)"
            raise InvalidStateError(msg)
        src_in = int(it.GetSourceStartFrame())
        src_out = int(it.GetSourceEndFrame())
        captured_transform = self._read_transform(it)
        captured_crop = self._read_crop(it)
        captured_opacity = self._read_opacity(it)
        captured_composite = self._read_composite_mode(it)
        conv = self._converter_of(tl)
        captured_markers = self._read_markers(it, conv)

        if not tl.DeleteClips([it], False):
            raise InvalidStateError(f"Timeline.DeleteClips refused to remove item {timeline_item_id!r}")

        media_type = 1 if media_type_str == "video" else 2
        record_frame = self._start_frame(tl) + conv.seconds_to_frames_round(new_position_seconds)
        info = {
            "mediaPoolItem": pool_item,
            "startFrame": src_in,
            "endFrame": src_out,
            "mediaType": media_type,
            "trackIndex": intra_track,
            "recordFrame": record_frame,
        }
        new_items = proj.GetMediaPool().AppendToTimeline([info])
        if not new_items:
            msg = (
                f"AppendToTimeline returned no items while moving {timeline_item_id!r}; "
                "the old item was already deleted, so the timeline now has one fewer clip"
            )
            raise InvalidStateError(msg)
        new_item = new_items[0]
        new_id = self._item_id(new_item)

        if media_type_str == "video":
            resolve_app, _ = self._require()
            props: dict[str, Any] = {
                "Pan": captured_transform.pan_x,
                "Tilt": captured_transform.pan_y,
                "ZoomX": captured_transform.zoom_x,
                "ZoomY": captured_transform.zoom_y,
                "RotationAngle": captured_transform.rotation,
                "AnchorPointX": captured_transform.anchor_x,
                "AnchorPointY": captured_transform.anchor_y,
                "CropLeft": captured_crop.left,
                "CropRight": captured_crop.right,
                "CropTop": captured_crop.top,
                "CropBottom": captured_crop.bottom,
                "Opacity": captured_opacity * 100.0,
            }
            composite_const = COMPOSITE_CONSTANT_NAMES[captured_composite]
            if hasattr(resolve_app, composite_const):
                props["CompositeMode"] = getattr(resolve_app, composite_const)
            self._set_props(new_item, props)
        for marker in captured_markers:
            frame = conv.seconds_to_frames_round(marker.position_seconds)
            color_cap = marker.color.value.capitalize()
            if not new_item.AddMarker(frame, color_cap, marker.label, marker.note, 1, ""):
                msg = f"AddMarker rejected restoring a marker at {marker.position_seconds}s onto moved item {new_id!r}"
                raise InvalidStateError(msg)
        return {timeline_item_id: new_id}

    # ---- item (per-item mutations) -------------------------------------------

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
        """``TimelineItem.SetProperty`` for Pan/Tilt/ZoomX/ZoomY/RotationAngle/
        AnchorPointX/AnchorPointY (documented key list), read back via
        ``GetProperty`` to confirm the edit actually landed.
        """
        if not (0.0 < zoom_x <= 100.0 and 0.0 < zoom_y <= 100.0):
            raise ValueError(f"zoom must be in (0, 100], got zoom_x={zoom_x} zoom_y={zoom_y}")
        if not -360.0 <= rotation <= 360.0:
            raise ValueError(f"rotation must be within [-360, 360] degrees, got {rotation}")
        proj = self._require_project()
        tl = self._require_timeline(proj)
        before = self._hydrate_timeline_state(tl)
        it = self._find_timeline_item(tl, timeline_item_id)
        self._set_props(
            it,
            {
                "Pan": float(pan_x),
                "Tilt": float(pan_y),
                "ZoomX": float(zoom_x),
                "ZoomY": float(zoom_y),
                "RotationAngle": float(rotation),
                "AnchorPointX": float(anchor_x),
                "AnchorPointY": float(anchor_y),
            },
        )
        after = self._hydrate_timeline_state(tl)
        return self._delta(before, after, f"timelines.{tl.GetName()}.items[{timeline_item_id}].transform")

    def set_crop(self, timeline_item_id: str, left: float, right: float, top: float, bottom: float) -> StateDelta:
        """``TimelineItem.SetProperty`` for CropLeft/CropRight/CropTop/CropBottom."""
        if min(left, right, top, bottom) < 0:
            msg = f"crop values must be >= 0, got left={left} right={right} top={top} bottom={bottom}"
            raise ValueError(msg)
        proj = self._require_project()
        tl = self._require_timeline(proj)
        before = self._hydrate_timeline_state(tl)
        it = self._find_timeline_item(tl, timeline_item_id)
        self._set_props(
            it,
            {
                "CropLeft": float(left),
                "CropRight": float(right),
                "CropTop": float(top),
                "CropBottom": float(bottom),
            },
        )
        after = self._hydrate_timeline_state(tl)
        return self._delta(before, after, f"timelines.{tl.GetName()}.items[{timeline_item_id}].crop")

    def set_composite_mode(self, timeline_item_id: str, mode: CompositeMode | str) -> StateDelta:
        """``TimelineItem.SetProperty("CompositeMode", ...)`` requires a
        ``resolve.COMPOSITE_*`` int constant, not a free-form string (reference
        verdict #3) -- resolved via ``COMPOSITE_CONSTANT_NAMES`` +
        ``getattr(resolve_app, name)``; raises ``UnsupportedOperationError``
        naming the attribute if this Resolve build doesn't expose it.
        """
        mode_enum = mode if isinstance(mode, CompositeMode) else CompositeMode(str(mode))
        resolve_app, _ = self._require()
        const_name = COMPOSITE_CONSTANT_NAMES[mode_enum]
        if not hasattr(resolve_app, const_name):
            msg = f"resolve.{const_name} is not exposed by this Resolve build (mode={mode_enum.value!r})"
            raise UnsupportedOperationError(msg)
        const_value = getattr(resolve_app, const_name)
        proj = self._require_project()
        tl = self._require_timeline(proj)
        before = self._hydrate_timeline_state(tl)
        it = self._find_timeline_item(tl, timeline_item_id)
        self._set_props(it, {"CompositeMode": const_value})
        after = self._hydrate_timeline_state(tl)
        return self._delta(before, after, f"timelines.{tl.GetName()}.items[{timeline_item_id}].composite_mode")

    def set_opacity(self, timeline_item_id: str, opacity: float) -> StateDelta:
        """``TimelineItem.SetProperty("Opacity", pct)`` -- Resolve stores 0-100,
        the wire contract is [0, 1].
        """
        if not 0.0 <= opacity <= 1.0:
            raise ValueError(f"opacity must be in [0, 1], got {opacity}")
        proj = self._require_project()
        tl = self._require_timeline(proj)
        before = self._hydrate_timeline_state(tl)
        it = self._find_timeline_item(tl, timeline_item_id)
        self._set_props(it, {"Opacity": opacity * 100.0})
        after = self._hydrate_timeline_state(tl)
        return self._delta(before, after, f"timelines.{tl.GetName()}.items[{timeline_item_id}].opacity")

    def add_fade(self, timeline_item_id: str, fade_in_seconds: float, fade_out_seconds: float) -> StateDelta:
        msg = (
            "add_fade has no documented Resolve scripting entry point: "
            "TimelineItem.SetProperty's documented key list has no "
            "FadeIn*/FadeOut* key (verified absent across 3 independent API "
            "mirrors). Apply fades in the Resolve UI, or use the fake backend "
            "for offline planning."
        )
        raise UnsupportedOperationError(msg)

    def set_speed(self, timeline_item_id: str, speed: float) -> StateDelta:
        msg = (
            "set_speed has no documented Resolve scripting entry point: "
            "TimelineItem.SetProperty's documented key list has no 'Speed' or "
            "other retime key. Apply retiming in the Resolve UI, or use the "
            "fake backend for offline planning."
        )
        raise UnsupportedOperationError(msg)

    def add_marker(
        self,
        timeline_item_id: str,
        position_seconds: float,
        label: str,
        color: MarkerColor | str,
        note: str = "",
    ) -> StateDelta:
        """``TimelineItem.AddMarker(frameId, color, name, note, duration,
        customData)`` -- 6-arg, item-relative ``frameId`` (spec section A);
        ``duration`` is fixed at 1 frame and ``customData`` at ``""``.
        """
        if position_seconds < 0:
            raise ValueError(f"position_seconds must be >= 0, got {position_seconds}")
        color_enum = color if isinstance(color, MarkerColor) else MarkerColor(str(color).lower())
        proj = self._require_project()
        tl = self._require_timeline(proj)
        before = self._hydrate_timeline_state(tl)
        it = self._find_timeline_item(tl, timeline_item_id)
        conv = self._converter_of(tl)
        frame = conv.seconds_to_frames_round(position_seconds)
        duration_frames = int(it.GetDuration())
        if duration_frames > 0 and frame >= duration_frames:
            msg = f"marker position {position_seconds}s (frame {frame}) is outside the item duration ({duration_frames} frames)"
            raise ValueError(msg)
        color_cap = color_enum.value.capitalize()
        if not it.AddMarker(frame, color_cap, label, note, 1, ""):
            msg = f"AddMarker rejected frame={frame} color={color_cap!r} (a marker may already exist there)"
            raise InvalidStateError(msg)
        after = self._hydrate_timeline_state(tl)
        return self._delta(before, after, f"timelines.{tl.GetName()}.items[{timeline_item_id}].markers")

    # ---- effects --------------------------------------------------------------

    def add_transition(
        self,
        timeline_item_id: str,
        track_index: int,
        style: TransitionStyle | str,
        duration_seconds: float,
        alignment: TransitionAlignment | str,
    ) -> StateDelta:
        msg = (
            f"add_transition has no documented Resolve scripting entry point "
            f"(style={style!r}, track={track_index}): the scripting API can read "
            "timelines but has no transition-insertion call. Apply transitions "
            "in the Resolve UI, or use the fake backend for offline planning."
        )
        raise UnsupportedOperationError(msg)

    # ---- render ---------------------------------------------------------------

    def add_render_job(self, timeline_name: str, format: RenderJobFormat | str, output_path: str) -> RenderJob:
        """Activate the timeline, pick format+codec via
        ``Project.GetRenderFormats()``/``GetRenderCodecs(format)`` and
        ``SetCurrentRenderFormatAndCodec`` (documented but never called by the
        previous implementation -- reference verdict #7), then
        ``SetRenderSettings`` + ``AddRenderJob``.
        """
        proj = self._require_project()
        target = self._find_timeline_by_name(proj, timeline_name)
        if target is None:
            raise NotFoundError(f"timeline {timeline_name!r} not found in current project")
        if not proj.SetCurrentTimeline(target):
            raise InvalidStateError(f"could not activate timeline {timeline_name!r}")
        fmt_enum = format if isinstance(format, RenderJobFormat) else RenderJobFormat(str(format))
        format_key, codec_key = self._pick_format_and_codec(proj, fmt_enum)
        if not proj.SetCurrentRenderFormatAndCodec(format_key, codec_key):
            msg = f"SetCurrentRenderFormatAndCodec({format_key!r}, {codec_key!r}) was refused"
            raise InvalidStateError(msg)
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        settings: dict[str, bool | str] = {
            "SelectAllFrames": True,
            "TargetDir": str(out.parent),
            "CustomName": out.stem,
        }
        if not proj.SetRenderSettings(settings):
            raise InvalidStateError(f"SetRenderSettings rejected {settings!r}")
        raw_job_id = proj.AddRenderJob()
        if not raw_job_id:
            raise InvalidStateError("AddRenderJob returned no job id")
        job_id = str(raw_job_id)
        record = RenderJob(
            id=job_id,
            timeline_name=timeline_name,
            format=fmt_enum,
            output_path=output_path,
            status=RenderJobStatus.QUEUED,
            progress=0.0,
        )
        self._render_jobs[job_id] = record
        return record

    def _pick_format_and_codec(self, proj: Any, fmt: RenderJobFormat) -> tuple[str, str]:
        """``GetRenderFormats()`` -> ``{format: extension}`` and
        ``GetRenderCodecs(format)`` -> ``{codec description: codec name}`` are
        documented to exist (reference verdict #7), but their exact key casing
        was not independently re-verified in the audit doc -- so we discover
        the real format/codec keys from the live API responses instead of
        hardcoding an assumed spelling.
        """
        try:
            formats = proj.GetRenderFormats() or {}
        except Exception as exc:
            raise InvalidStateError(f"GetRenderFormats() failed: {exc}") from exc
        extensions, codec_substr = _RENDER_FORMAT_HINTS[fmt]
        # GetRenderFormats() maps a human label -> file extension ("QuickTime" -> "mov").
        # SetCurrentRenderFormatAndCodec wants the EXTENSION, not the label.
        format_label: str | None = None
        format_ext: str | None = None
        for ext in extensions:
            format_label = next((k for k, v in formats.items() if str(v).lower() == ext), None)
            if format_label is not None:
                format_ext = str(formats[format_label])
                break
        if format_label is None or format_ext is None:
            msg = (
                f"no render format with extension in {extensions} is available "
                f"(GetRenderFormats()={formats!r})"
            )
            raise InvalidStateError(msg)
        # Resolve's own examples query codecs by the label; some builds accept the
        # extension instead, so try both before giving up.
        codecs: dict[str, Any] = {}
        for probe in (format_label, format_ext):
            try:
                codecs = proj.GetRenderCodecs(probe) or {}
            except Exception as exc:
                raise InvalidStateError(f"GetRenderCodecs({probe!r}) failed: {exc}") from exc
            if codecs:
                break
        needle = codec_substr.lower().replace(".", "")
        # The codec argument is the codec NAME (the dict value, e.g. "H264"), never
        # the description (e.g. "H.264") — Resolve rejects the description.
        codec_name: str | None = None
        for description, name in codecs.items():
            haystack = f"{description} {name}".lower().replace(".", "")
            if needle in haystack:
                codec_name = str(name)
                break
        if codec_name is None:
            msg = (
                f"no {codec_substr!r} codec available for format {format_ext!r} "
                f"(GetRenderCodecs()={codecs!r})"
            )
            raise InvalidStateError(msg)
        return format_ext, codec_name

    def start_render(self, job_id: str) -> RenderJob:
        """Confirm membership via ``Project.GetRenderJobList()``, then
        ``Project.StartRendering(job_id)`` (single-job-id call form).
        """
        proj = self._require_project()
        known = {str(j.get("JobId")) for j in (proj.GetRenderJobList() or []) if isinstance(j, dict)}
        if job_id not in known:
            raise NotFoundError(f"render job {job_id!r} not found")
        if not proj.StartRendering(job_id):
            raise InvalidStateError(f"StartRendering refused job {job_id!r}")
        base = self._render_jobs.get(job_id) or RenderJob(id=job_id, timeline_name="", output_path="")
        updated = base.model_copy(update={"status": RenderJobStatus.RUNNING, "progress": 0.0})
        self._render_jobs[job_id] = updated
        return updated

    def get_render_status(self, job_id: str) -> RenderJob:
        """``Project.GetRenderJobStatus(job_id)`` -> ``{'JobStatus':
        'Ready'|'Rendering'|'Complete'|'Failed'|'Cancelled', 'CompletionPercentage': 0-100}``.
        """
        proj = self._require_project()
        try:
            info = proj.GetRenderJobStatus(job_id)
        except Exception as exc:
            raise NotFoundError(f"render job {job_id!r} not found: {exc}") from exc
        if not isinstance(info, dict) or not info:
            raise NotFoundError(f"render job {job_id!r} not found")
        status_text = str(info.get("JobStatus", "Ready"))
        try:
            progress = float(info.get("CompletionPercentage", 0.0)) / 100.0
        except (TypeError, ValueError):
            progress = 0.0
        status_enum = _RENDER_STATUS_MAP.get(status_text.lower(), RenderJobStatus.QUEUED)
        base = self._render_jobs.get(job_id) or RenderJob(id=job_id, timeline_name="", output_path="")
        return base.model_copy(update={"status": status_enum, "progress": max(0.0, min(1.0, progress))})

    # ---- destructive (gated) ---------------------------------------------------

    def _require_destructive(self, confirm: bool) -> None:
        """Mirror ``FakeResolveBackend``'s gate: both ``--allow-destructive`` at
        process start and ``confirm=True`` on the call must be true.
        """
        if not (self._allow_destructive and confirm):
            msg = "destructive tool is disabled. Run the server with --allow-destructive and pass confirm=true."
            raise DestructiveDisabledError(msg)

    def quit_app(self, confirm: bool = False) -> dict[str, Any]:
        """``Resolve.Quit()``."""
        self._require_destructive(confirm)
        resolve_app, _ = self._require()
        try:
            resolve_app.Quit()
        except Exception as exc:
            raise ResolveMCPError(f"Resolve.Quit failed: {exc}") from exc
        self._resolve = None
        self._project_manager = None
        return {"quit": True, "project_after": None}

    def restart_app(self, confirm: bool = False) -> dict[str, Any]:
        msg = (
            "restart_app has no documented Resolve scripting entry point: the "
            "Resolve object exposes Quit() but no Restart/Relaunch call "
            "(reference verdict #11). Use quit_app and have your process "
            "supervisor relaunch Resolve."
        )
        raise UnsupportedOperationError(msg)

    def delete_timeline(self, name: str, confirm: bool = False) -> StateDelta:
        """``MediaPool.DeleteTimelines([timeline])``."""
        self._require_destructive(confirm)
        proj = self._require_project()
        target = self._find_timeline_by_name(proj, name)
        if target is None:
            raise NotFoundError(f"timeline {name!r} not found in current project")
        before_state = self._hydrate_timeline_state(target)
        mp = proj.GetMediaPool()
        deleter = getattr(mp, "DeleteTimelines", None)
        if not callable(deleter):
            msg = "MediaPool.DeleteTimelines is unavailable on this Resolve build; delete the timeline manually in the UI."
            raise UnsupportedOperationError(msg)
        if not deleter([target]):
            raise InvalidStateError(f"DeleteTimelines refused timeline {name!r}")
        after_state = before_state.model_copy(update={"tracks": [], "duration_seconds": 0.0})
        return StateDelta(
            before=before_state.model_dump(mode="json"),
            after=after_state.model_dump(mode="json"),
            changed_paths=[f"timelines.{name}.deleted"],
        )

    def delete_media(self, media_clip_id: str, confirm: bool = False) -> dict[str, Any]:
        """``MediaPool.DeleteClips([MediaPoolItem])`` -- removes from the media
        pool/bin (not the timeline).
        """
        self._require_destructive(confirm)
        proj = self._require_project()
        mp = proj.GetMediaPool()
        if mp is None:
            raise InvalidStateError("project has no media pool")
        clip = self._lookup_pool_item(mp, media_clip_id)
        if not mp.DeleteClips([clip]):
            raise InvalidStateError(f"DeleteClips refused media {media_clip_id!r}")
        return {"deleted": {"id": media_clip_id}}

    # --- convert / frame math ----------------------------------------------------

    def _frame_rate_of_timeline(self, tl: Any) -> FrameRate:
        """FrameRate via ``Timeline.GetSetting('timelineFrameRate' /
        'timelineDropFrameTimecode')``.
        """
        try:
            fps = _parse_fps(tl.GetSetting("timelineFrameRate"))
        except Exception:
            fps = 24.0
        try:
            df_raw = tl.GetSetting("timelineDropFrameTimecode")
            drop_frame = str(df_raw).strip() in ("1", "true", "True")
        except Exception:
            drop_frame = False
        return _sanitize_frame_rate(fps, drop_frame)

    def _converter_of(self, tl: Any) -> TimeConverter:
        return TimeConverter(self._frame_rate_of_timeline(tl))

    def _start_frame(self, tl: Any) -> int:
        """``Timeline.GetStartFrame()`` -- the absolute frame offset of the
        timeline's start timecode (often 86400 @ 24fps for 01:00:00:00). Every
        frame we read or write is offset by this (reference verdict #12).
        """
        try:
            return int(tl.GetStartFrame())
        except Exception:
            return 0

    def _track_counts(self, tl: Any) -> tuple[int, int]:
        """``(video_track_count, audio_track_count)`` via ``Timeline.GetTrackCount``."""

        def _count(kind: str) -> int:
            try:
                return max(0, int(tl.GetTrackCount(kind)))
            except Exception:
                return 0

        return _count("video"), _count("audio")

    # --- lookup -------------------------------------------------------------------

    @staticmethod
    def _item_id(it: Any) -> str:
        """TimelineItem id: ``GetUniqueId()`` -- never Python object identity."""
        try:
            uid = it.GetUniqueId()
            if uid:
                return str(uid)
        except Exception:
            pass
        raise InvalidStateError("timeline item exposes no GetUniqueId()")

    def _find_timeline_item(self, tl: Any, item_id: str) -> Any:
        it, _media_type, _intra, _wire = self._locate_timeline_item(tl, item_id)
        return it

    def _locate_timeline_item(self, tl: Any, item_id: str) -> tuple[Any, str, int, int]:
        """Walk ``GetItemListInTrack`` for every track and match
        ``GetUniqueId()``; we never cache the wrapper across calls because the
        scripting bridge mints a fresh proxy per fetch (reference verdict #8).
        Returns ``(item, media_type, intra_track_index, wire_track_index)``.
        """
        nvideo, naudio = self._track_counts(tl)
        wire_index = 0
        for media_type, count in (("video", nvideo), ("audio", naudio)):
            for intra_track in range(1, count + 1):
                wire_index += 1
                try:
                    items = tl.GetItemListInTrack(media_type, intra_track) or []
                except Exception:
                    items = []
                for it in items:
                    try:
                        uid = str(it.GetUniqueId())
                    except Exception:
                        continue
                    if uid == item_id:
                        return it, media_type, intra_track, wire_index
        raise NotFoundError(f"timeline item {item_id!r} not found")

    # --- hydrate --------------------------------------------------------------

    def _hydrate_timeline_state(self, tl: Any) -> TimelineState:
        """Read a Resolve Timeline into a ``TimelineState`` using
        ``GetName``/``GetSetting``/``GetTrackCount``/``GetItemListInTrack``/
        ``GetStartFrame`` and, per item, everything :meth:`_hydrate_item` reads.
        Ids are Resolve's own ``GetUniqueId()``, never object identity.
        """
        name = str(tl.GetName())
        fr = self._frame_rate_of_timeline(tl)
        conv = TimeConverter(fr)
        start_frame = self._start_frame(tl)
        nvideo, naudio = self._track_counts(tl)
        tracks: list[Track] = []
        duration_end = 0.0
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
                    item_model = self._hydrate_item(it, wire_index, conv, start_frame)
                    duration_end = max(duration_end, item_model.start_seconds + item_model.duration_seconds)
                    track_items.append(item_model)
                tracks.append(Track(index=wire_index, kind=kind, items=track_items))
        return TimelineState(name=name, frame_rate=fr, duration_seconds=duration_end, tracks=tracks, transitions=[])

    def _hydrate_item(self, it: Any, wire_track_index: int, conv: TimeConverter, start_frame: int) -> TimelineItem:
        """One ``TimelineItem`` -> our wire model: position/duration via
        ``GetStart``/``GetEnd``/``GetDuration`` (offset by ``start_frame``),
        source in/out via ``GetSourceStartFrame``/``GetSourceEndFrame`` (media's
        own fps when the source clip is known), transform/crop/opacity/
        composite via ``GetProperty``, markers via ``GetMarkers``, media id via
        ``GetMediaPoolItem().GetUniqueId()``.
        """
        item_id = self._item_id(it)
        start_abs = int(it.GetStart())
        dur_frames = int(it.GetDuration())
        start_seconds = conv.frames_to_seconds(max(start_abs - start_frame, 0))
        duration_seconds = conv.frames_to_seconds(dur_frames)

        try:
            media_item = it.GetMediaPoolItem()
        except Exception:
            media_item = None
        media_clip_id = ""
        media_conv = conv
        if media_item is not None:
            try:
                media_clip_id = self._pool_item_id(media_item)
            except Exception:
                media_clip_id = ""
            media_fps = self._media_fps(media_item)
            if media_fps is not None:
                media_conv = TimeConverter(FrameRate(fps=media_fps, drop_frame=False))

        source_in_seconds = 0.0
        source_out_seconds = duration_seconds
        try:
            src_in = int(it.GetSourceStartFrame())
            src_out = int(it.GetSourceEndFrame())
            source_in_seconds = media_conv.frames_to_seconds(src_in)
            source_out_seconds = media_conv.frames_to_seconds(src_out)
        except Exception:
            pass

        return TimelineItem(
            id=item_id,
            track_index=wire_track_index,
            media_clip_id=media_clip_id,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            source_in_seconds=source_in_seconds,
            source_out_seconds=source_out_seconds,
            transform=self._read_transform(it),
            crop=self._read_crop(it),
            composite_mode=self._read_composite_mode(it),
            opacity=self._read_opacity(it),
            markers=self._read_markers(it, conv),
        )

    def _read_transform(self, it: Any) -> Transform:
        """Pan/Tilt/ZoomX/ZoomY/RotationAngle/AnchorPointX/AnchorPointY via
        ``TimelineItem.GetProperty`` (documented ``SetProperty``/``GetProperty``
        key set). Values are clamped into the wire schema's documented bounds
        rather than crashing hydration if Resolve ever returns something
        outside them (a known gap -- see the final report).
        """

        def _get(key: str, default: float) -> float:
            try:
                v = it.GetProperty(key)
                return float(v) if v is not None else default
            except Exception:
                return default

        zoom_x = _get("ZoomX", 1.0)
        zoom_y = _get("ZoomY", 1.0)
        rotation = _get("RotationAngle", 0.0)
        return Transform(
            pan_x=_get("Pan", 0.0),
            pan_y=_get("Tilt", 0.0),
            zoom_x=min(max(zoom_x, 0.0001), 100.0),
            zoom_y=min(max(zoom_y, 0.0001), 100.0),
            rotation=min(max(rotation, -360.0), 360.0),
            anchor_x=_get("AnchorPointX", 0.0),
            anchor_y=_get("AnchorPointY", 0.0),
        )

    def _read_crop(self, it: Any) -> Crop:
        """CropLeft/CropRight/CropTop/CropBottom via ``TimelineItem.GetProperty``."""

        def _get(key: str) -> float:
            try:
                v = it.GetProperty(key)
                return max(0.0, float(v)) if v is not None else 0.0
            except Exception:
                return 0.0

        return Crop(left=_get("CropLeft"), right=_get("CropRight"), top=_get("CropTop"), bottom=_get("CropBottom"))

    def _read_opacity(self, it: Any) -> float:
        """Opacity via ``TimelineItem.GetProperty("Opacity")`` -- Resolve stores
        0-100, the wire contract is [0, 1].
        """
        try:
            v = it.GetProperty("Opacity")
            pct = float(v) if v is not None else 100.0
        except Exception:
            pct = 100.0
        return max(0.0, min(1.0, pct / 100.0))

    def _composite_mode_reverse_map(self) -> dict[Any, CompositeMode]:
        """Build (once) the ``resolve.COMPOSITE_*`` int -> CompositeMode map."""
        if self._composite_reverse is not None:
            return self._composite_reverse
        mapping: dict[Any, CompositeMode] = {}
        resolve_app = self._resolve
        if resolve_app is not None:
            for mode, const_name in COMPOSITE_CONSTANT_NAMES.items():
                value = getattr(resolve_app, const_name, None)
                if value is not None:
                    mapping[value] = mode
        self._composite_reverse = mapping
        return mapping

    def _read_composite_mode(self, it: Any) -> CompositeMode:
        """CompositeMode via ``TimelineItem.GetProperty("CompositeMode")``,
        mapped back from the ``resolve.COMPOSITE_*`` int constant.
        """
        try:
            v = it.GetProperty("CompositeMode")
        except Exception:
            return CompositeMode.NORMAL
        return self._composite_mode_reverse_map().get(v, CompositeMode.NORMAL)

    def _read_markers(self, it: Any, conv: TimeConverter) -> list[Marker]:
        """``TimelineItem.GetMarkers()`` -> ``{frameId: {"color", "name",
        "note", ...}}``; ``frameId`` is item-relative (spec section A). The
        exact dict shape is corroborated indirectly (the documented
        ``AddMarker`` signature and marker-delete-by-frame methods enumerated
        for TimelineItem in one source) rather than independently re-verified
        letter-for-letter in the audit doc -- flagged in the final report.
        """
        try:
            raw = it.GetMarkers() or {}
        except Exception:
            raw = {}
        item_id = self._item_id(it)
        markers: list[Marker] = []
        for frame_id, info in raw.items():
            if not isinstance(info, dict):
                continue
            try:
                frame = int(frame_id)
            except (TypeError, ValueError):
                continue
            position_seconds = conv.frames_to_seconds(max(frame, 0))
            color_raw = str(info.get("color", "Blue")).lower()
            try:
                color = MarkerColor(color_raw)
            except ValueError:
                color = MarkerColor.BLUE
            markers.append(
                Marker(
                    id=f"{item_id}:{frame}",
                    timeline_item_id=item_id,
                    position_seconds=position_seconds,
                    label=str(info.get("name", "")),
                    color=color,
                    note=str(info.get("note", "")),
                )
            )
        return markers

    # --- property io ------------------------------------------------------------

    def _set_props(self, it: Any, props: dict[str, Any]) -> None:
        """Apply each key via ``TimelineItem.SetProperty``, then read it back
        via ``GetProperty`` to confirm the edit actually landed -- refuses
        silent failure (mismatch or a ``False`` return both raise).
        """
        for key, value in props.items():
            try:
                ok = it.SetProperty(key, value)
            except Exception as exc:
                raise InvalidStateError(f"SetProperty({key!r}, {value!r}) raised: {exc}") from exc
            if not ok:
                raise InvalidStateError(f"SetProperty({key!r}, {value!r}) was rejected by Resolve")
        for key, expected in props.items():
            try:
                actual = it.GetProperty(key)
            except Exception as exc:
                raise InvalidStateError(f"GetProperty({key!r}) raised after SetProperty: {exc}") from exc
            if actual is None or not _approx_equal(actual, expected):
                msg = f"SetProperty({key!r}, {expected!r}) did not take: GetProperty returned {actual!r}"
                raise InvalidStateError(msg)

    # --- state delta --------------------------------------------------------------

    def _delta(self, before: TimelineState, after: TimelineState, changed_path: str) -> StateDelta:
        return StateDelta(
            before=before.model_dump(mode="json"),
            after=after.model_dump(mode="json"),
            changed_paths=[changed_path],
        )

    # --- test convenience -----------------------------------------------------

    def root_folder(self) -> Any:
        """Public alias for tests; returns the current project's media pool root."""
        proj = self._require_project()
        mp = proj.GetMediaPool()
        return mp.GetRootFolder() if mp is not None else None


__all__ = ["DaVinciResolveBackend"]
