"""Record/replay harness for the live DaVinci Resolve scripting API surface.

Each fake class takes its parent ``CallLog`` explicitly so the same log is
shared anywhere in the call graph. This keeps the daemon's constructor and
the test fixture pointing at the same recorder.
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


class FakeTimeline(_LogHolder):
    def __init__(self, name: str, log: CallLog) -> None:
        super().__init__(log)
        self.name = name
        self._items: list[FakeTimelineItem] = []

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

    def GetTrackCount(self) -> int:
        self._log.record("Timeline.GetTrackCount")
        return 2

    def GetTrackType(self, idx: int) -> str:
        self._log.record("Timeline.GetTrackType", idx)
        return "video" if idx == 1 else "audio"

    def GetItemListInTrack(self, tr_type: str) -> list[FakeTimelineItem]:
        self._log.record("Timeline.GetItemListInTrack", tr_type)
        return self._items

    def AppendItemsInTimeline(self, items: list[tuple[Any, dict[str, Any]]]) -> bool:
        self._log.record("Timeline.AppendItemsInTimeline", items)
        for pool_item, info in items:
            _name = "?"
            try:
                _name = pool_item.GetName()
            except Exception:
                try:
                    _name = getattr(pool_item, "name", "?")
                except Exception:
                    _name = "?"
            entry_name = _name
            self._items.append(
                FakeTimelineItem(
                    name=entry_name,
                    log=self._log,
                    track_index=int(info.get("trackIndex", 1)),
                    start_frame=int(info.get("startFrame", 0)),
                    duration_frames=max(int(info.get("endFrame", 0)) - int(info.get("startFrame", 0)), 1),
                )
            )
        return True

    def DeleteClips(self, items: list[dict[str, Any]]) -> bool:
        self._log.record("Timeline.DeleteClips", items)
        return True

    def AddTransition(self, track_index: int, info: dict[str, Any]) -> Any:
        self._log.record("Timeline.AddTransition", track_index, info)
        return True


class FakeTimelineItem(_LogHolder):
    def __init__(
        self,
        name: str,
        *,
        log: CallLog,
        track_index: int = 1,
        start_frame: int = 0,
        duration_frames: int = 24,
    ) -> None:
        super().__init__(log)
        self._name = name
        self._track_index = track_index
        self._start_frame = start_frame
        self._duration_frames = duration_frames

    def GetStart(self) -> int:
        self._log.record("TimelineItem.GetStart")
        return self._start_frame

    def GetDuration(self) -> int:
        self._log.record("TimelineItem.GetDuration")
        return self._duration_frames

    def GetName(self) -> str:
        self._log.record("TimelineItem.GetName")
        return self._name

    def GetClipProperty(self) -> dict[str, Any]:
        self._log.record("TimelineItem.GetClipProperty")
        return {"Clip Name": self._name}

    def SetProperty(self, kind: str, props: dict[str, Any]) -> bool:
        self._log.record("TimelineItem.SetProperty", kind, props)
        return True

    def AddMarker(self, frame: int, color: str, name: str, note: str) -> bool:
        self._log.record("TimelineItem.AddMarker", frame, color, name, note)
        return True


class FakeMediaPool(_LogHolder):
    def __init__(self, log: CallLog) -> None:
        super().__init__(log)
        self._root = FakeFolder("Master", log=log, pool=self)

    def GetRootFolder(self) -> FakeFolder:
        self._log.record("MediaPool.GetRootFolder")
        return self._root

    def GetClipList(self) -> list[FakePoolItem]:
        self._log.record("MediaPool.GetClipList")
        return self._root.GetClipList()

    def ImportMedia(self, paths: list[str]) -> list[FakePoolItem]:
        self._log.record("MediaPool.ImportMedia", list(paths))
        items: list[FakePoolItem] = []
        for p in paths:
            it = FakePoolItem(name=p.rsplit("/", 1)[-1], log=self._log)
            self._root.AddClip(it)
            items.append(it)
        return items

    def AddSubFolder(self, folder: FakeFolder, name: str) -> FakeFolder:
        self._log.record("MediaPool.AddSubFolder", folder.GetName(), name)
        return folder

    def CreateEmptyTimeline(self, name: str) -> FakeTimeline:
        self._log.record("MediaPool.CreateEmptyTimeline", name)
        return FakeTimeline(name, self._log)


class FakeFolder(_LogHolder):
    def __init__(self, name: str, *, log: CallLog, pool: FakeMediaPool | None = None) -> None:
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

    def AddClip(self, item: FakePoolItem) -> None:
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

    def Delete(self) -> bool:
        self._log.record("PoolItem.Delete")
        return True


class FakeProject(_LogHolder):
    def __init__(self, name: str, *, log: CallLog) -> None:
        super().__init__(log)
        self.name = name
        self._mp = FakeMediaPool(log)
        self._timeline = FakeTimeline("Timeline 1", log)

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
        return self._timeline

    def GetResolutionWidth(self) -> int:
        self._log.record("Project.GetResolutionWidth")
        return 1920

    def GetResolutionHeight(self) -> int:
        self._log.record("Project.GetResolutionHeight")
        return 1080

    def SaveProject(self) -> bool:
        self._log.record("Project.SaveProject")
        return True

    def AddRenderJob(self) -> int:
        self._log.record("Project.AddRenderJob")
        return 1

    def StartRendering(self, job: Any) -> bool:
        self._log.record("Project.StartRendering", job)
        return True

    def GetRenderJobs(self) -> Any:
        self._log.record("Project.GetRenderJobs")
        return {"1": "queued"}

    def GetRenderJob(self, job_id: str) -> Any:
        self._log.record("Project.GetRenderJob", job_id)
        return {"JobId": job_id, "Status": "Rendering", "Progress": 0.0}


class FakeProjectManager(_LogHolder):
    def __init__(self, log: CallLog) -> None:
        super().__init__(log)
        self._current = FakeProject("DefaultProject", log=log)

    def GetCurrentProject(self) -> FakeProject:
        self._log.record("ProjectManager.GetCurrentProject")
        return self._current

    def CreateProject(self, name: str) -> FakeProject:
        self._log.record("ProjectManager.CreateProject", name)
        self._current = FakeProject(name, log=self._log)
        return self._current

    def LoadProject(self, name: str) -> FakeProject:
        self._log.record("ProjectManager.LoadProject", name)
        self._current = FakeProject(name, log=self._log)
        return self._current


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
    """Install a fake Resolve scripting module so backend can import it without the
    real SDK present. The same :class:`FakeResolve` instance is shared between the
    fixture and the backend so the call log records every backend-invoked method."""
    log = CallLog()
    fake = FakeResolve(log)

    def fake_scriptapp(_name: str) -> Any:
        log.record("scriptapp", _name)
        return fake

    fake_module = type("FakeModule", (), {"scriptapp": staticmethod(fake_scriptapp)})

    monkeypatch.setitem(__import__("sys").modules, "DaVinciResolveScript", fake_module)
    return fake, log
