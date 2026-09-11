"""Record/replay harness modeling the DOCUMENTED DaVinci Resolve scripting API.

Every method here corresponds to an entry in Blackmagic's
"DaVinci Resolve Scripting API" README (v18+):

* ``Timeline.GetTrackCount(trackType)`` — one call per kind ("video"/"audio").
* ``Timeline.GetItemListInTrack(trackType, index)`` — 1-based index per kind.
* ``MediaPool.*`` — AppendToTimeline, CreateEmptyTimeline, ImportMedia, AddSubFolder,
  SetCurrentFolder, DeleteClips, DeleteTimelines.
* ``TimelineItem.*`` — GetStart/GetDuration/GetName/GetClipProperty/
  GetMediaPoolItem/GetSourceStartFrame/GetSourceEndFrame/SetProperty/AddMarker/Delete.
* ``Project.*`` — GetTimelineCount, GetTimelineByIndex, SetCurrentTimeline,
  SetRenderSettings, AddRenderJob, StartRendering, GetRenderJobStatus,
  GetRenderJobList, GetMediaPool, GetCurrentTimeline, GetSetting/SetSetting.
* ``ProjectManager.*`` — CreateProject, LoadProject, SaveProject,
  GetCurrentProject, GetProjectListInCurrentFolder.
* ``Resolve.*`` — scriptapp, GetProjectManager, Quit.

Each fake class takes its parent :class:`CallLog` explicitly so the same log is
shared anywhere in the call graph.

.. note:: Behavioural fidelity (frame placement on append, fade handle frames,
   etc.) is exercised by the offline tests; the live-Resolve smoke checklist in
   plan.md still runs manually because a harness can never verify reality.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CallLog:
    """Sequential, dict-of-positional-args list of every method called."""

    entries: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def record(self, method: str, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            self.entries.append({"method": method, "args": list(args), "kwargs": dict(kwargs)})

    def methods_called(self) -> list[str]:
        with self._lock:
            return [e["method"] for e in self.entries]

    def reset(self) -> None:
        with self._lock:
            self.entries.clear()


class _LogHolder:
    def __init__(self, log: CallLog) -> None:
        self._log = log


# ---------------------------------------------------------------------------
# Timeline items
# ---------------------------------------------------------------------------


class FakeTimelineItem(_LogHolder):
    """Documented ``TimelineItem``: frame ranges, properties, markers, delete."""

    def __init__(
        self,
        name: str,
        *,
        log: CallLog,
        track_type: str = "video",
        track_index: int = 1,
        start_frame: int = 0,
        duration_frames: int = 24,
        pool_item: FakePoolItem | None = None,
        parent: FakeTimeline | None = None,
    ) -> None:
        super().__init__(log)
        self._name = name
        self._track_type = track_type
        self._track_index = track_index
        self._start_frame = start_frame
        self._duration_frames = duration_frames
        self._pool_item = pool_item
        self._parent = parent

    # -- introspection -------------------------------------------------------

    def GetName(self) -> str:
        self._log.record("TimelineItem.GetName")
        return self._name

    def GetStart(self) -> int:
        self._log.record("TimelineItem.GetStart")
        return self._start_frame

    def GetDuration(self) -> int:
        self._log.record("TimelineItem.GetDuration")
        return self._duration_frames

    def GetClipProperty(self) -> dict[str, Any]:
        self._log.record("TimelineItem.GetClipProperty")
        return {"Clip Name": self._name}

    def GetMediaPoolItem(self) -> FakePoolItem | None:
        self._log.record("TimelineItem.GetMediaPoolItem")
        return self._pool_item

    def GetSourceStartFrame(self) -> int:
        self._log.record("TimelineItem.GetSourceStartFrame")
        return 0

    def GetSourceEndFrame(self) -> int:
        self._log.record("TimelineItem.GetSourceEndFrame")
        return self._duration_frames

    # -- mutation ------------------------------------------------------------

    def SetProperty(self, key: str, value: Any) -> bool:
        self._log.record("TimelineItem.SetProperty", key, value)
        return True

    def AddMarker(self, frame: int, color: str, name: str, note: str) -> bool:
        self._log.record("TimelineItem.AddMarker", frame, color, name, note)
        return True

    def Delete(self) -> bool:
        self._log.record("TimelineItem.Delete")
        if self._parent is not None:
            self._parent._remove_item(self)
            return True
        return False


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


class FakeTimeline(_LogHolder):
    """Documented ``Timeline``: GetTrackCount/GetItemListInTrack take a type."""

    def __init__(self, name: str, log: CallLog) -> None:
        super().__init__(log)
        self.name = name
        # media_type -> intra-type 1-based track index -> items
        self._tracks: dict[str, dict[int, list[FakeTimelineItem]]] = {
            "video": {1: []},
            "audio": {1: []},
        }

    def GetName(self) -> str:
        self._log.record("Timeline.GetName")
        return self.name

    def GetSetting(self, key: str) -> str | None:
        self._log.record("Timeline.GetSetting", key)
        if key == "timelineFrameRate":
            return "24.0"
        return ""

    def SetSetting(self, key: str, value: str) -> bool:
        self._log.record("Timeline.SetSetting", key, value)
        return True

    def GetTrackCount(self, track_type: str) -> int:
        self._log.record("Timeline.GetTrackCount", track_type)
        return len(self._tracks.get(track_type, {}))

    def GetItemListInTrack(self, track_type: str, index: int) -> list[FakeTimelineItem]:
        self._log.record("Timeline.GetItemListInTrack", track_type, index)
        return list(self._tracks.get(track_type, {}).get(index, []))

    # -- harness-only mutation ------------------------------------------------

    def _place_item(self, info: dict[str, Any], pool_item: FakePoolItem) -> FakeTimelineItem:
        media_type = "video" if info.get("mediaType", 1) == 1 else "audio"
        track_index = int(info.get("trackIndex", 1))
        track = self._tracks.setdefault(media_type, {}).setdefault(track_index, [])
        start_frame = int(info.get("recordFrame", 0))
        if "recordFrame" not in info:
            start_frame = max((it._start_frame + it._duration_frames for it in track), default=0)
        duration = max(1, int(info.get("endFrame", 1)) - int(info.get("startFrame", 0)))
        item = FakeTimelineItem(
            name=pool_item.GetName(),
            log=self._log,
            track_type=media_type,
            track_index=track_index,
            start_frame=start_frame,
            duration_frames=duration,
            pool_item=pool_item,
            parent=self,
        )
        track.append(item)
        track.sort(key=lambda i: i._start_frame)
        return item

    def _remove_item(self, item: FakeTimelineItem) -> None:
        track = self._tracks.get(item._track_type, {}).get(item._track_index, [])
        if item in track:
            track.remove(item)


# ---------------------------------------------------------------------------
# Media pool tree
# ---------------------------------------------------------------------------


class FakeFolder(_LogHolder):
    def __init__(self, name: str, *, log: CallLog) -> None:
        super().__init__(log)
        self.name = name
        self._clips: list[FakePoolItem] = []
        self._sub: list[FakeFolder] = []

    def GetName(self) -> str:
        self._log.record("Folder.GetName")
        return self.name

    def GetClipList(self) -> list[FakePoolItem]:
        self._log.record("Folder.GetClipList")
        return list(self._clips)

    def GetSubFolderList(self) -> list[FakeFolder]:
        self._log.record("Folder.GetSubFolderList")
        return list(self._sub)

    def _add_clip(self, item: FakePoolItem) -> None:
        self._clips.append(item)


class FakePoolItem(_LogHolder):
    def __init__(self, name: str, log: CallLog) -> None:
        super().__init__(log)
        self._name = name

    def GetName(self) -> str:
        self._log.record("PoolItem.GetName")
        return self._name

    def GetClipProperty(self) -> dict[str, Any]:
        self._log.record("PoolItem.GetClipProperty")
        return {"Clip Name": self._name, "File Name": self._name, "File Path": self._name}


class FakeMediaPool(_LogHolder):
    """Documented ``MediaPool`` incl. AppendToTimeline & DeleteTimelines."""

    def __init__(self, log: CallLog) -> None:
        super().__init__(log)
        self._root = FakeFolder("Master", log=log)
        self._current: FakeFolder = self._root
        self._timelines: list[FakeTimeline] = []
        self._current_timeline: FakeTimeline | None = None

    def GetRootFolder(self) -> FakeFolder:
        self._log.record("MediaPool.GetRootFolder")
        return self._root

    def AddSubFolder(self, folder: FakeFolder, name: str) -> FakeFolder | bool:
        self._log.record("MediaPool.AddSubFolder", folder.GetName(), name)
        if any(sub.GetName() == name for sub in folder._sub):
            return False
        sub = FakeFolder(name, log=self._log)
        folder._sub.append(sub)
        return sub

    def SetCurrentFolder(self, folder: FakeFolder) -> bool:
        self._log.record("MediaPool.SetCurrentFolder", folder.GetName())
        self._current = folder
        return True

    def ImportMedia(self, paths: list[str]) -> list[FakePoolItem]:
        self._log.record("MediaPool.ImportMedia", list(paths))
        items = [FakePoolItem(name=p.rsplit("/", 1)[-1], log=self._log) for p in paths]
        for it in items:
            self._current._add_clip(it)
        return items

    def CreateEmptyTimeline(self, name: str) -> FakeTimeline:
        self._log.record("MediaPool.CreateEmptyTimeline", name)
        tl = FakeTimeline(name, self._log)
        self._timelines.append(tl)
        self._current_timeline = tl
        return tl

    def AppendToTimeline(self, infos: list[dict[str, Any]]) -> list[FakeTimelineItem]:
        self._log.record("MediaPool.AppendToTimeline", infos)
        tl = self._current_timeline or (self._timelines[0] if self._timelines else None)
        if tl is None:
            return []
        return [tl._place_item(info, info["mediaPoolItem"]) for info in infos]

    def DeleteClips(self, clips: list[FakePoolItem]) -> bool:
        self._log.record("MediaPool.DeleteClips", clips)
        doomed = set(map(id, clips))
        for folder in self._walk_folders(self._root):
            folder._clips = [c for c in folder._clips if id(c) not in doomed]
        return True

    def DeleteTimelines(self, timelines: list[FakeTimeline]) -> bool:
        self._log.record("MediaPool.DeleteTimelines", timelines)
        doomed = set(map(id, timelines))
        self._timelines = [t for t in self._timelines if id(t) not in doomed]
        if self._current_timeline is not None and id(self._current_timeline) in doomed:
            self._current_timeline = self._timelines[0] if self._timelines else None
        return True

    def _walk_folders(self, folder: FakeFolder):
        yield folder
        for sub in folder._sub:
            yield from self._walk_folders(sub)


# ---------------------------------------------------------------------------
# Project, ProjectManager, Resolve
# ---------------------------------------------------------------------------


class FakeProject(_LogHolder):
    def __init__(self, name: str, *, log: CallLog) -> None:
        super().__init__(log)
        self.name = name
        self._mp = FakeMediaPool(log)
        # Every project starts with one empty timeline, like real Resolve.
        first_tl = FakeTimeline("Timeline 1", log)
        self._mp._timelines.append(first_tl)
        self._mp._current_timeline = first_tl
        self._render_settings: dict[str, Any] = {}
        self._render_jobs: dict[str, dict[str, Any]] = {}
        self._job_seq = 1

    def GetName(self) -> str:
        self._log.record("Project.GetName")
        return self.name

    def GetSetting(self, key: str) -> str | None:
        self._log.record("Project.GetSetting", key)
        if key == "timelineFrameRate":
            return "24.0"
        return ""

    def SetSetting(self, key: str, value: str) -> bool:
        self._log.record("Project.SetSetting", key, value)
        return True

    def GetMediaPool(self) -> FakeMediaPool:
        self._log.record("Project.GetMediaPool")
        return self._mp

    def GetCurrentTimeline(self) -> FakeTimeline | None:
        self._log.record("Project.GetCurrentTimeline")
        return self._mp._current_timeline

    def GetTimelineCount(self) -> int:
        self._log.record("Project.GetTimelineCount")
        return len(self._mp._timelines)

    def GetTimelineByIndex(self, index: int) -> FakeTimeline | None:
        self._log.record("Project.GetTimelineByIndex", index)
        try:
            return self._mp._timelines[index - 1]
        except IndexError:
            return None

    def SetCurrentTimeline(self, timeline: FakeTimeline) -> bool:
        self._log.record("Project.SetCurrentTimeline", timeline.GetName())
        if timeline not in self._mp._timelines:
            return False
        self._mp._current_timeline = timeline
        return True

    def SetRenderSettings(self, settings: dict[str, Any]) -> bool:
        self._log.record("Project.SetRenderSettings", settings)
        self._render_settings.update(settings)
        return True

    def AddRenderJob(self) -> str:
        self._log.record("Project.AddRenderJob")
        job_id = str(self._job_seq)
        self._job_seq += 1
        self._render_jobs[job_id] = {"JobStatus": "queued", "CompletionPercentage": 0.0}
        return job_id

    def StartRendering(self, job_id: str) -> bool:
        self._log.record("Project.StartRendering", job_id)
        if job_id not in self._render_jobs:
            return False
        self._render_jobs[job_id]["JobStatus"] = "rendering"
        return True

    def GetRenderJobList(self) -> list[dict[str, Any]]:
        self._log.record("Project.GetRenderJobList")
        return [{"JobId": jid, **job} for jid, job in self._render_jobs.items()]

    def GetRenderJobStatus(self, job_id: str) -> dict[str, Any]:
        self._log.record("Project.GetRenderJobStatus", job_id)
        if job_id not in self._render_jobs:
            return {}
        return dict(self._render_jobs[job_id])


class FakeProjectManager(_LogHolder):
    def __init__(self, log: CallLog) -> None:
        super().__init__(log)
        self._projects: dict[str, FakeProject] = {"Timeline 1": FakeProject("Timeline 1", log=log)}
        self._current: FakeProject = next(iter(self._projects.values()))

    def GetCurrentProject(self) -> FakeProject:
        self._log.record("ProjectManager.GetCurrentProject")
        return self._current

    def CreateProject(self, name: str) -> FakeProject:
        self._log.record("ProjectManager.CreateProject", name)
        proj = FakeProject(name, log=self._log)
        self._projects[name] = proj
        self._current = proj
        return proj

    def LoadProject(self, name: str) -> FakeProject:
        self._log.record("ProjectManager.LoadProject", name)
        if name in self._projects:
            self._current = self._projects[name]
        return self._current if self._current.name == name else FakeProject(name, log=self._log)

    def SaveProject(self) -> bool:
        self._log.record("ProjectManager.SaveProject")
        return True

    def GetProjectListInCurrentFolder(self) -> list[str]:
        self._log.record("ProjectManager.GetProjectListInCurrentFolder")
        return list(self._projects)


class FakeResolve(_LogHolder):
    def __init__(self, log: CallLog) -> None:
        super().__init__(log)
        self._pm = FakeProjectManager(log)
        self._quit_called = False

    def GetProjectManager(self) -> FakeProjectManager:
        self._log.record("Resolve.GetProjectManager")
        return self._pm

    def Quit(self) -> None:
        self._log.record("Resolve.Quit")
        self._quit_called = True


def install_fake_resolve(monkeypatch: Any) -> tuple[FakeResolve, CallLog]:
    """Install the harness as ``DaVinciResolveScript`` so the backend imports it
    without a real Resolve install. Same :class:`FakeResolve` + :class:`CallLog`
    shared between fixture and backend."""
    log = CallLog()
    fake = FakeResolve(log)

    def fake_scriptapp(_name: str) -> Any:
        log.record("scriptapp", _name)
        return fake

    fake_module = type("FakeModule", (), {"scriptapp": staticmethod(fake_scriptapp)})

    monkeypatch.setitem(__import__("sys").modules, "DaVinciResolveScript", fake_module)
    return fake, log
