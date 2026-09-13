"""Offline stand-in for DaVinci Resolve's scripting API.

This harness is written FROM Blackmagic's documented scripting API (see
``docs/resolve-scripting-api.md`` for the sourced
method inventory), deliberately NOT from what ``davinci_backend.py`` happens to
call. The previous harness was written the other way round and therefore
happily modelled ``TimelineItem.Delete()``, a method Resolve does not have —
the live-backend tests passed while the real call would have raised
``AttributeError`` against Resolve.

Rules this harness enforces, because each one caught a real bug class:

* **Fresh wrappers.** Every accessor returns a NEW wrapper object around shared
  underlying state, exactly like Resolve's Python bridge. Python ``id()`` of a
  returned object is therefore worthless as an identity key; only
  ``GetUniqueId()`` survives. (The old backend keyed the media pool on
  ``hex(id(obj))``.)
* **Only documented methods exist.** Anything else raises ``AttributeError``,
  and ``SetProperty`` returns ``False`` for keys outside the documented list.
* **Absolute frame numbers.** A timeline starts at 01:00:00:00 (frame 86400 at
  24 fps) and ``GetStart()``/``GetEnd()`` live in that space, so any code that
  forgets ``GetStartFrame()`` is off by an hour.
* **Exact arities.** ``AddMarker`` needs its five positional arguments,
  ``DeleteClips`` takes the item list plus the ripple flag.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

# --- documented value sets -------------------------------------------------------

#: TimelineItem.SetProperty keys, per the scripting README's property table.
DOCUMENTED_PROPERTY_KEYS: frozenset[str] = frozenset(
    {
        "Pan", "Tilt", "ZoomX", "ZoomY", "ZoomGang", "RotationAngle",
        "AnchorPointX", "AnchorPointY", "Pitch", "Yaw", "FlipX", "FlipY",
        "CropLeft", "CropRight", "CropTop", "CropBottom", "CropSoftness",
        "CropRetain", "DynamicZoomEase", "CompositeMode", "Opacity",
        "Distortion", "RetimeProcess", "MotionEstimation", "Scaling",
        "ResizeFilter",
    }
)

_PROPERTY_DEFAULTS: dict[str, Any] = {
    "Pan": 0.0, "Tilt": 0.0, "ZoomX": 1.0, "ZoomY": 1.0, "ZoomGang": True,
    "RotationAngle": 0.0, "AnchorPointX": 0.0, "AnchorPointY": 0.0,
    "CropLeft": 0.0, "CropRight": 0.0, "CropTop": 0.0, "CropBottom": 0.0,
    "Opacity": 100.0, "CompositeMode": 0,
}

#: resolve.COMPOSITE_* constants, in documented order.
COMPOSITE_CONSTANTS: tuple[str, ...] = (
    "COMPOSITE_NORMAL", "COMPOSITE_ADD", "COMPOSITE_SUBTRACT", "COMPOSITE_DIFF",
    "COMPOSITE_MULTIPLY", "COMPOSITE_SCREEN", "COMPOSITE_OVERLAY",
    "COMPOSITE_HARDLIGHT", "COMPOSITE_SOFTLIGHT", "COMPOSITE_DARKEN",
    "COMPOSITE_LIGHTEN", "COMPOSITE_COLOR_DODGE", "COMPOSITE_COLOR_BURN",
    "COMPOSITE_EXCLUSION", "COMPOSITE_HUE", "COMPOSITE_SATURATE",
    "COMPOSITE_COLORIZE", "COMPOSITE_LUMA_MASK", "COMPOSITE_DIVIDE",
    "COMPOSITE_LINEAR_DODGE", "COMPOSITE_LINEAR_BURN", "COMPOSITE_LINEAR_LIGHT",
    "COMPOSITE_VIVID_LIGHT", "COMPOSITE_PIN_LIGHT", "COMPOSITE_HARD_MIX",
    "COMPOSITE_LIGHTER_COLOR", "COMPOSITE_DARKER_COLOR", "COMPOSITE_FOREGROUND",
    "COMPOSITE_ALPHA", "COMPOSITE_INVERTED_ALPHA", "COMPOSITE_LUM",
    "COMPOSITE_INVERTED_LUM",
)

MARKER_COLORS: frozenset[str] = frozenset(
    {
        "Blue", "Cyan", "Green", "Yellow", "Red", "Pink", "Purple", "Fuchsia",
        "Rose", "Lavender", "Sky", "Mint", "Lemon", "Sand", "Cocoa", "Cream",
    }
)

#: GetRenderFormats(): human label -> file extension, per the documented example
#: ``{'MP4': 'mp4', 'QuickTime': 'mov', ...}``.
RENDER_FORMATS: dict[str, str] = {"MP4": "mp4", "QuickTime": "mov", "MXF OP1A": "mxf"}
#: GetRenderCodecs(format): codec description -> codec name. Resolve accepts either the
#: label or the extension as the lookup key, and SetCurrentRenderFormatAndCodec requires
#: the extension plus the codec NAME (never the description).
RENDER_CODECS: dict[str, dict[str, str]] = {
    "mp4": {"H.264": "H264", "H.265": "H265"},
    "mov": {"H.264": "H264", "Apple ProRes 422 HQ": "ProRes422HQ", "DNxHR HQ": "DNxHRHQ"},
    "mxf": {"DNxHR HQ": "DNxHRHQ"},
}


# --- call log --------------------------------------------------------------------


@dataclass
class CallLog:
    """Sequential record of every scripting call the backend made."""

    entries: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def record(self, method: str, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            self.entries.append({"method": method, "args": list(args), "kwargs": dict(kwargs)})

    def methods_called(self) -> list[str]:
        with self._lock:
            return [e["method"] for e in self.entries]

    def args_for(self, method: str) -> list[list[Any]]:
        with self._lock:
            return [e["args"] for e in self.entries if e["method"] == method]

    def reset(self) -> None:
        with self._lock:
            self.entries.clear()


# --- underlying state (never handed to callers) ----------------------------------


@dataclass
class _MediaState:
    uid: str
    name: str
    path: str
    frames: int
    fps: float
    kind: str  # "Video" | "Audio" | "Video + Audio" | "Still"
    width: int = 1920
    height: int = 1080


@dataclass
class _ItemState:
    uid: str
    media: _MediaState
    media_type: int  # 1 = video, 2 = audio
    track_index: int
    record_frame: int  # absolute, includes the timeline start offset
    source_start: int
    source_end: int  # exclusive
    properties: dict[str, Any] = field(default_factory=dict)
    markers: dict[int, dict[str, Any]] = field(default_factory=dict)

    @property
    def duration(self) -> int:
        return max(1, self.source_end - self.source_start)


@dataclass
class _TimelineData:
    uid: str
    name: str
    start_frame: int = 86400  # 01:00:00:00 at 24fps, Resolve's default
    settings: dict[str, str] = field(default_factory=dict)
    items: list[_ItemState] = field(default_factory=list)
    video_tracks: int = 1
    audio_tracks: int = 1


@dataclass
class _FolderData:
    name: str
    clips: list[_MediaState] = field(default_factory=list)
    subfolders: list[_FolderData] = field(default_factory=list)


@dataclass
class _ProjectData:
    uid: str
    name: str
    settings: dict[str, str] = field(default_factory=dict)
    root: _FolderData = field(default_factory=lambda: _FolderData(name="Master"))
    current_folder: _FolderData | None = None
    timelines: list[_TimelineData] = field(default_factory=list)
    current_timeline: _TimelineData | None = None
    render_settings: dict[str, Any] = field(default_factory=dict)
    render_format: tuple[str, str] | None = None
    render_jobs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _World:
    """Shared mutable state plus the id counters."""

    log: CallLog
    projects: list[_ProjectData] = field(default_factory=list)
    current_project: _ProjectData | None = None
    library: dict[str, _MediaState] = field(default_factory=dict)  # path -> media on "disk"
    counters: dict[str, int] = field(default_factory=dict)
    quit_called: bool = False

    def next_id(self, prefix: str) -> str:
        self.counters[prefix] = self.counters.get(prefix, 0) + 1
        return f"{prefix}_{self.counters[prefix]}"


# --- wrappers (fresh object per accessor, like the real bridge) -------------------


class _Wrapper:
    __slots__ = ("_world",)

    def __init__(self, world: _World) -> None:
        self._world = world

    @property
    def _log(self) -> CallLog:
        return self._world.log


class MediaPoolItem(_Wrapper):
    """Documented ``MediaPoolItem``."""

    __slots__ = ("_state",)

    def __init__(self, world: _World, state: _MediaState) -> None:
        super().__init__(world)
        self._state = state

    def GetName(self) -> str:
        self._log.record("MediaPoolItem.GetName")
        return self._state.name

    def GetUniqueId(self) -> str:
        self._log.record("MediaPoolItem.GetUniqueId")
        return self._state.uid

    def GetMediaId(self) -> str:
        self._log.record("MediaPoolItem.GetMediaId")
        return self._state.uid

    def GetClipProperty(self, key: str | None = None) -> Any:
        self._log.record("MediaPoolItem.GetClipProperty", key)
        props: dict[str, Any] = {
            "Clip Name": self._state.name,
            "File Path": self._state.path,
            "Type": self._state.kind,
            "Frames": str(self._state.frames),
            "FPS": str(self._state.fps),
            "Resolution": f"{self._state.width}x{self._state.height}",
            "Duration": _timecode(self._state.frames, self._state.fps),
        }
        return props if key is None else props.get(key, "")


class TimelineItem(_Wrapper):
    """Documented ``TimelineItem``. Note there is no ``Delete()``."""

    __slots__ = ("_state", "_timeline")

    def __init__(self, world: _World, state: _ItemState, timeline: _TimelineData) -> None:
        super().__init__(world)
        self._state = state
        self._timeline = timeline

    def GetName(self) -> str:
        self._log.record("TimelineItem.GetName")
        return self._state.media.name

    def GetUniqueId(self) -> str:
        self._log.record("TimelineItem.GetUniqueId")
        return self._state.uid

    def GetStart(self) -> int:
        self._log.record("TimelineItem.GetStart")
        return self._state.record_frame

    def GetEnd(self) -> int:
        self._log.record("TimelineItem.GetEnd")
        return self._state.record_frame + self._state.duration

    def GetDuration(self) -> int:
        self._log.record("TimelineItem.GetDuration")
        return self._state.duration

    def GetLeftOffset(self) -> int:
        self._log.record("TimelineItem.GetLeftOffset")
        return self._state.source_start

    def GetRightOffset(self) -> int:
        self._log.record("TimelineItem.GetRightOffset")
        return self._state.source_end

    def GetSourceStartFrame(self) -> int:
        self._log.record("TimelineItem.GetSourceStartFrame")
        return self._state.source_start

    def GetSourceEndFrame(self) -> int:
        self._log.record("TimelineItem.GetSourceEndFrame")
        return self._state.source_end

    def GetMediaPoolItem(self) -> MediaPoolItem:
        self._log.record("TimelineItem.GetMediaPoolItem")
        return MediaPoolItem(self._world, self._state.media)

    def SetProperty(self, key: str, value: Any) -> bool:
        self._log.record("TimelineItem.SetProperty", key, value)
        if key not in DOCUMENTED_PROPERTY_KEYS:
            return False
        if key == "CompositeMode" and not isinstance(value, int):
            return False  # Resolve wants the COMPOSITE_* constant, not a string
        self._state.properties[key] = value
        return True

    def GetProperty(self, key: str | None = None) -> Any:
        self._log.record("TimelineItem.GetProperty", key)
        if key is None:
            merged = dict(_PROPERTY_DEFAULTS)
            merged.update(self._state.properties)
            return merged
        if key not in DOCUMENTED_PROPERTY_KEYS:
            return None
        return self._state.properties.get(key, _PROPERTY_DEFAULTS.get(key))

    def AddMarker(
        self,
        frameId: int,
        color: str,
        name: str,
        note: str,
        duration: int,
        customData: str = "",
    ) -> bool:
        self._log.record("TimelineItem.AddMarker", frameId, color, name, note, duration, customData)
        if color not in MARKER_COLORS or duration < 1:
            return False
        if frameId < 0 or frameId >= self._state.duration or frameId in self._state.markers:
            return False
        self._state.markers[int(frameId)] = {
            "color": color,
            "duration": duration,
            "note": note,
            "name": name,
            "customData": customData,
        }
        return True

    def GetMarkers(self) -> dict[int, dict[str, Any]]:
        self._log.record("TimelineItem.GetMarkers")
        return {k: dict(v) for k, v in self._state.markers.items()}


class Folder(_Wrapper):
    """Documented ``Folder``."""

    __slots__ = ("_state",)

    def __init__(self, world: _World, state: _FolderData) -> None:
        super().__init__(world)
        self._state = state

    def GetName(self) -> str:
        self._log.record("Folder.GetName")
        return self._state.name

    def GetClipList(self) -> list[MediaPoolItem]:
        self._log.record("Folder.GetClipList")
        return [MediaPoolItem(self._world, clip) for clip in self._state.clips]

    def GetSubFolderList(self) -> list[Folder]:
        self._log.record("Folder.GetSubFolderList")
        return [Folder(self._world, sub) for sub in self._state.subfolders]


class Timeline(_Wrapper):
    """Documented ``Timeline``."""

    __slots__ = ("_state",)

    def __init__(self, world: _World, state: _TimelineData) -> None:
        super().__init__(world)
        self._state = state

    def GetName(self) -> str:
        self._log.record("Timeline.GetName")
        return self._state.name

    def GetUniqueId(self) -> str:
        self._log.record("Timeline.GetUniqueId")
        return self._state.uid

    def GetStartFrame(self) -> int:
        self._log.record("Timeline.GetStartFrame")
        return self._state.start_frame

    def GetEndFrame(self) -> int:
        self._log.record("Timeline.GetEndFrame")
        end = self._state.start_frame
        for item in self._state.items:
            end = max(end, item.record_frame + item.duration)
        return end

    def GetSetting(self, key: str | None = None) -> Any:
        self._log.record("Timeline.GetSetting", key)
        project = self._world.current_project
        inherited = dict(project.settings) if project is not None else {}
        if self._state.settings.get("useCustomSettings") == "1":
            inherited.update(self._state.settings)
        else:
            inherited.setdefault("useCustomSettings", "0")
        return inherited if key is None else inherited.get(key, "")

    def SetSetting(self, key: str, value: str) -> bool:
        self._log.record("Timeline.SetSetting", key, value)
        self._state.settings[key] = value
        return True

    def GetTrackCount(self, trackType: str) -> int:
        self._log.record("Timeline.GetTrackCount", trackType)
        if trackType == "video":
            return self._state.video_tracks
        if trackType == "audio":
            return self._state.audio_tracks
        if trackType == "subtitle":
            return 0
        return 0

    def GetItemListInTrack(self, trackType: str, index: int) -> list[TimelineItem]:
        self._log.record("Timeline.GetItemListInTrack", trackType, index)
        media_type = 1 if trackType == "video" else 2
        items = [
            it
            for it in self._state.items
            if it.media_type == media_type and it.track_index == index
        ]
        items.sort(key=lambda it: it.record_frame)
        return [TimelineItem(self._world, it, self._state) for it in items]

    def DeleteClips(self, timelineItems: list[TimelineItem], rippleDelete: bool = False) -> bool:
        self._log.record("Timeline.DeleteClips", len(timelineItems), rippleDelete)
        doomed = {item._state.uid for item in timelineItems}
        if not doomed:
            return False
        kept = [it for it in self._state.items if it.uid not in doomed]
        if len(kept) == len(self._state.items):
            return False
        self._state.items = kept
        return True

    def AddMarker(
        self,
        frameId: int,
        color: str,
        name: str,
        note: str,
        duration: int,
        customData: str = "",
    ) -> bool:
        self._log.record("Timeline.AddMarker", frameId, color, name, note, duration, customData)
        return color in MARKER_COLORS


class MediaPool(_Wrapper):
    """Documented ``MediaPool``."""

    __slots__ = ("_project",)

    def __init__(self, world: _World, project: _ProjectData) -> None:
        super().__init__(world)
        self._project = project

    def GetRootFolder(self) -> Folder:
        self._log.record("MediaPool.GetRootFolder")
        return Folder(self._world, self._project.root)

    def AddSubFolder(self, folder: Folder, name: str) -> Folder | bool:
        self._log.record("MediaPool.AddSubFolder", folder.GetName(), name)
        parent = folder._state
        if any(sub.name == name for sub in parent.subfolders):
            return False
        sub = _FolderData(name=name)
        parent.subfolders.append(sub)
        return Folder(self._world, sub)

    def SetCurrentFolder(self, folder: Folder) -> bool:
        self._log.record("MediaPool.SetCurrentFolder", folder.GetName())
        self._project.current_folder = folder._state
        return True

    def GetCurrentFolder(self) -> Folder:
        self._log.record("MediaPool.GetCurrentFolder")
        return Folder(self._world, self._project.current_folder or self._project.root)

    def ImportMedia(self, paths: list[str]) -> list[MediaPoolItem]:
        self._log.record("MediaPool.ImportMedia", list(paths))
        target = self._project.current_folder or self._project.root
        imported: list[MediaPoolItem] = []
        for path in paths:
            source = self._world.library.get(path)
            if source is None:
                continue  # Resolve silently skips files it cannot read
            clip = _MediaState(
                uid=self._world.next_id("mpi"),
                name=source.name,
                path=source.path,
                frames=source.frames,
                fps=source.fps,
                kind=source.kind,
                width=source.width,
                height=source.height,
            )
            target.clips.append(clip)
            imported.append(MediaPoolItem(self._world, clip))
        return imported

    def CreateEmptyTimeline(self, name: str) -> Timeline | None:
        self._log.record("MediaPool.CreateEmptyTimeline", name)
        if any(tl.name == name for tl in self._project.timelines):
            return None
        data = _TimelineData(uid=self._world.next_id("tl"), name=name)
        self._project.timelines.append(data)
        self._project.current_timeline = data
        return Timeline(self._world, data)

    def AppendToTimeline(self, clipInfos: list[dict[str, Any]]) -> list[TimelineItem]:
        self._log.record("MediaPool.AppendToTimeline", _loggable(clipInfos))
        timeline = self._project.current_timeline
        if timeline is None:
            return []
        created: list[TimelineItem] = []
        for info in clipInfos:
            pool_item = info.get("mediaPoolItem")
            if not isinstance(pool_item, MediaPoolItem):
                continue
            media = pool_item._state
            media_type = int(info.get("mediaType", 1))
            track_index = int(info.get("trackIndex", 1))
            source_start = int(info.get("startFrame", 0))
            source_end = int(info.get("endFrame", media.frames))
            if source_end <= source_start:
                continue
            duration = source_end - source_start
            if "recordFrame" in info:
                record_frame = int(info["recordFrame"])
            else:
                same_track = [
                    it
                    for it in timeline.items
                    if it.media_type == media_type and it.track_index == track_index
                ]
                record_frame = max(
                    (it.record_frame + it.duration for it in same_track),
                    default=timeline.start_frame,
                )
            if record_frame < timeline.start_frame:
                continue  # cannot place before the start of the timeline
            if any(
                it.media_type == media_type
                and it.track_index == track_index
                and record_frame < it.record_frame + it.duration
                and it.record_frame < record_frame + duration
                for it in timeline.items
            ):
                continue  # Resolve will not stack two clips in the same spot
            state = _ItemState(
                uid=self._world.next_id("tli"),
                media=media,
                media_type=media_type,
                track_index=track_index,
                record_frame=record_frame,
                source_start=source_start,
                source_end=source_end,
            )
            timeline.items.append(state)
            if media_type == 1:
                timeline.video_tracks = max(timeline.video_tracks, track_index)
            else:
                timeline.audio_tracks = max(timeline.audio_tracks, track_index)
            created.append(TimelineItem(self._world, state, timeline))
        return created

    def DeleteClips(self, clips: list[MediaPoolItem]) -> bool:
        self._log.record("MediaPool.DeleteClips", len(clips))
        doomed = {clip._state.uid for clip in clips}
        removed = False

        def _walk(folder: _FolderData) -> None:
            nonlocal removed
            before = len(folder.clips)
            folder.clips = [c for c in folder.clips if c.uid not in doomed]
            removed = removed or len(folder.clips) != before
            for sub in folder.subfolders:
                _walk(sub)

        _walk(self._project.root)
        return removed

    def DeleteTimelines(self, timelines: list[Timeline]) -> bool:
        self._log.record("MediaPool.DeleteTimelines", len(timelines))
        doomed = {tl._state.uid for tl in timelines}
        before = len(self._project.timelines)
        self._project.timelines = [tl for tl in self._project.timelines if tl.uid not in doomed]
        if (
            self._project.current_timeline is not None
            and self._project.current_timeline.uid in doomed
        ):
            self._project.current_timeline = (
                self._project.timelines[0] if self._project.timelines else None
            )
        return len(self._project.timelines) != before


class Project(_Wrapper):
    """Documented ``Project``."""

    __slots__ = ("_state",)

    def __init__(self, world: _World, state: _ProjectData) -> None:
        super().__init__(world)
        self._state = state

    def GetName(self) -> str:
        self._log.record("Project.GetName")
        return self._state.name

    def GetUniqueId(self) -> str:
        self._log.record("Project.GetUniqueId")
        return self._state.uid

    def GetSetting(self, key: str | None = None) -> Any:
        self._log.record("Project.GetSetting", key)
        return dict(self._state.settings) if key is None else self._state.settings.get(key, "")

    def SetSetting(self, key: str, value: str) -> bool:
        self._log.record("Project.SetSetting", key, value)
        self._state.settings[key] = str(value)
        return True

    def GetMediaPool(self) -> MediaPool:
        self._log.record("Project.GetMediaPool")
        return MediaPool(self._world, self._state)

    def GetCurrentTimeline(self) -> Timeline | None:
        self._log.record("Project.GetCurrentTimeline")
        current = self._state.current_timeline
        return Timeline(self._world, current) if current is not None else None

    def GetTimelineCount(self) -> int:
        self._log.record("Project.GetTimelineCount")
        return len(self._state.timelines)

    def GetTimelineByIndex(self, index: int) -> Timeline | None:
        self._log.record("Project.GetTimelineByIndex", index)
        if 1 <= index <= len(self._state.timelines):
            return Timeline(self._world, self._state.timelines[index - 1])
        return None

    def SetCurrentTimeline(self, timeline: Timeline) -> bool:
        self._log.record("Project.SetCurrentTimeline", timeline.GetName())
        for data in self._state.timelines:
            if data.uid == timeline._state.uid:
                self._state.current_timeline = data
                return True
        return False

    def GetRenderFormats(self) -> dict[str, str]:
        self._log.record("Project.GetRenderFormats")
        return dict(RENDER_FORMATS)

    def GetRenderCodecs(self, renderFormat: str) -> dict[str, str]:
        self._log.record("Project.GetRenderCodecs", renderFormat)
        extension = RENDER_FORMATS.get(renderFormat, renderFormat)
        return dict(RENDER_CODECS.get(extension, {}))

    def SetCurrentRenderFormatAndCodec(self, format: str, codec: str) -> bool:
        self._log.record("Project.SetCurrentRenderFormatAndCodec", format, codec)
        # Resolve wants the file extension and the codec NAME; the human-readable
        # label ("QuickTime") or description ("H.264") is refused.
        if format not in RENDER_FORMATS.values():
            return False
        if codec not in RENDER_CODECS.get(format, {}).values():
            return False
        self._state.render_format = (format, codec)
        return True

    def SetRenderSettings(self, settings: dict[str, Any]) -> bool:
        self._log.record("Project.SetRenderSettings", dict(settings))
        self._state.render_settings.update(settings)
        return True

    def AddRenderJob(self) -> str:
        self._log.record("Project.AddRenderJob")
        job_id = self._world.next_id("job")
        self._state.render_jobs.append(
            {
                "JobId": job_id,
                "JobStatus": "Ready",
                "CompletionPercentage": 0,
                "TargetDir": self._state.render_settings.get("TargetDir", ""),
                "OutputFilename": self._state.render_settings.get("CustomName", ""),
            }
        )
        return job_id

    def GetRenderJobList(self) -> list[dict[str, Any]]:
        self._log.record("Project.GetRenderJobList")
        return [dict(job) for job in self._state.render_jobs]

    def StartRendering(self, *jobIds: Any, **kwargs: Any) -> bool:
        self._log.record("Project.StartRendering", *jobIds, **kwargs)
        wanted: set[str] = set()
        for entry in jobIds:
            if isinstance(entry, list):
                wanted.update(str(x) for x in entry)
            elif isinstance(entry, str):
                wanted.add(entry)
        started = False
        for job in self._state.render_jobs:
            if not wanted or job["JobId"] in wanted:
                job["JobStatus"] = "Rendering"
                job["CompletionPercentage"] = 10
                started = True
        return started

    def GetRenderJobStatus(self, jobId: str) -> dict[str, Any]:
        self._log.record("Project.GetRenderJobStatus", jobId)
        for job in self._state.render_jobs:
            if job["JobId"] == jobId:
                return {
                    "JobStatus": job["JobStatus"],
                    "CompletionPercentage": job["CompletionPercentage"],
                }
        return {}

    def IsRenderingInProgress(self) -> bool:
        self._log.record("Project.IsRenderingInProgress")
        return any(job["JobStatus"] == "Rendering" for job in self._state.render_jobs)


class ProjectManager(_Wrapper):
    """Documented ``ProjectManager``."""

    def GetCurrentProject(self) -> Project | None:
        self._log.record("ProjectManager.GetCurrentProject")
        current = self._world.current_project
        return Project(self._world, current) if current is not None else None

    def CreateProject(self, name: str) -> Project | None:
        self._log.record("ProjectManager.CreateProject", name)
        if any(p.name == name for p in self._world.projects):
            return None
        data = _ProjectData(uid=self._world.next_id("prj"), name=name)
        # A brand-new project has NO timeline.
        self._world.projects.append(data)
        self._world.current_project = data
        return Project(self._world, data)

    def LoadProject(self, name: str) -> Project | None:
        self._log.record("ProjectManager.LoadProject", name)
        for data in self._world.projects:
            if data.name == name:
                self._world.current_project = data
                return Project(self._world, data)
        return None

    def SaveProject(self) -> bool:
        self._log.record("ProjectManager.SaveProject")
        return self._world.current_project is not None

    def GetProjectListInCurrentFolder(self) -> list[str]:
        self._log.record("ProjectManager.GetProjectListInCurrentFolder")
        return [p.name for p in self._world.projects]


class Resolve(_Wrapper):
    """Documented ``Resolve`` app object, including the COMPOSITE_* constants."""

    def __init__(self, world: _World) -> None:
        super().__init__(world)
        for value, name in enumerate(COMPOSITE_CONSTANTS):
            object.__setattr__(self, name, value)

    def __getattr__(self, item: str) -> Any:
        if item in COMPOSITE_CONSTANTS:
            return COMPOSITE_CONSTANTS.index(item)
        raise AttributeError(f"Resolve has no attribute {item!r}")

    def GetProjectManager(self) -> ProjectManager:
        self._log.record("Resolve.GetProjectManager")
        return ProjectManager(self._world)

    def GetProductName(self) -> str:
        self._log.record("Resolve.GetProductName")
        return "DaVinci Resolve Studio"

    def GetVersionString(self) -> str:
        self._log.record("Resolve.GetVersionString")
        return "19.0.0"

    def Quit(self) -> None:
        self._log.record("Resolve.Quit")
        self._world.quit_called = True


# --- test-facing façade ----------------------------------------------------------


class FakeResolveApp:
    """Controls the simulated Resolve instance from a test."""

    def __init__(self) -> None:
        self.log = CallLog()
        self.world = _World(log=self.log)
        self.running = True

    # -- seeding ---------------------------------------------------------------

    def add_media_file(
        self,
        path: str,
        *,
        frames: int = 240,
        fps: float = 24.0,
        kind: str = "Video",
        name: str | None = None,
        width: int = 1920,
        height: int = 1080,
    ) -> str:
        """Put a file on the simulated disk so ImportMedia can find it."""
        media = _MediaState(
            uid=f"disk_{len(self.world.library) + 1}",
            name=name or path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0],
            path=path,
            frames=frames,
            fps=fps,
            kind=kind,
            width=width,
            height=height,
        )
        self.world.library[path] = media
        return path

    def app(self) -> Resolve | None:
        return Resolve(self.world) if self.running else None

    # -- inspection ------------------------------------------------------------

    @property
    def current_project(self) -> _ProjectData | None:
        return self.world.current_project

    def timeline(self, name: str | None = None) -> _TimelineData | None:
        project = self.world.current_project
        if project is None:
            return None
        if name is None:
            return project.current_timeline
        return next((tl for tl in project.timelines if tl.name == name), None)

    def items_on(self, track_type: str, index: int, timeline: str | None = None) -> list[_ItemState]:
        data = self.timeline(timeline)
        if data is None:
            return []
        media_type = 1 if track_type == "video" else 2
        found = [
            it for it in data.items if it.media_type == media_type and it.track_index == index
        ]
        return sorted(found, key=lambda it: it.record_frame)

    @property
    def quit_called(self) -> bool:
        return self.world.quit_called


def install_fake_resolve(monkeypatch: Any) -> tuple[FakeResolveApp, CallLog]:
    """Install the harness as the importable ``DaVinciResolveScript`` module."""
    import sys

    fake = FakeResolveApp()

    def scriptapp(name: str) -> Any:
        fake.log.record("scriptapp", name)
        return fake.app()

    module = type("FakeDaVinciResolveScript", (), {"scriptapp": staticmethod(scriptapp)})
    monkeypatch.setitem(sys.modules, "DaVinciResolveScript", module)
    return fake, fake.log


# --- helpers ----------------------------------------------------------------------


def _timecode(frames: int, fps: float) -> str:
    rate = max(1, round(fps))
    total_seconds, frame = divmod(int(frames), rate)
    minutes, second = divmod(total_seconds, 60)
    hour, minute = divmod(minutes, 60)
    return f"{hour:02d}:{minute:02d}:{second:02d}:{frame:02d}"


def _loggable(clip_infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Log clipInfo dicts without the unprintable wrapper objects."""
    out: list[dict[str, Any]] = []
    for info in clip_infos:
        entry = {k: v for k, v in info.items() if k != "mediaPoolItem"}
        item = info.get("mediaPoolItem")
        entry["mediaPoolItem"] = item._state.uid if isinstance(item, MediaPoolItem) else None
        out.append(entry)
    return out
