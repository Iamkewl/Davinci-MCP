"""Live-backend tests against the documented-API harness.

These are the only tests that exercise ``DaVinciResolveBackend``. Because no
Resolve install exists in CI, the harness in ``fake_resolve.py`` stands in for
it — but that harness is written from Blackmagic's documented API rather than
from this backend, so a call the backend invents fails here the same way it
would fail against Resolve.

What the suite pins down:

* ids survive across calls (Resolve hands out fresh wrapper objects every time);
* seconds on the wire are relative to the timeline start, while Resolve's frame
  numbers are absolute (a 01:00:00:00 start = frame 86400);
* mutations are verified by reading the property back;
* operations Resolve genuinely cannot do fail loudly instead of pretending;
* every scripting call the backend makes is on the documented allow-list.
"""

from __future__ import annotations

from typing import Any

import pytest
from resolve_mcp.backend import (
    DestructiveDisabledError,
    NotFoundError,
    ResolveUnavailableError,
    UnsupportedOperationError,
)
from resolve_mcp.davinci_backend import DaVinciResolveBackend
from resolve_mcp.schemas import CompositeMode, MarkerColor, MediaKind, RenderJobStatus

from .fake_resolve import COMPOSITE_CONSTANTS, CallLog, FakeResolveApp, install_fake_resolve

# Methods the backend is allowed to call, derived from the scripting API reference
# in scratchpad/resolve-api-audit/resolve_api_reference.md. Anything outside this
# set is either invented or undocumented — both are bugs.
DOCUMENTED_API: frozenset[str] = frozenset(
    {
        "scriptapp",
        "Resolve.GetProjectManager", "Resolve.Quit", "Resolve.GetProductName",
        "Resolve.GetVersionString",
        "ProjectManager.CreateProject", "ProjectManager.LoadProject",
        "ProjectManager.SaveProject", "ProjectManager.GetCurrentProject",
        "ProjectManager.GetProjectListInCurrentFolder",
        "Project.GetName", "Project.GetUniqueId", "Project.GetSetting", "Project.SetSetting",
        "Project.GetMediaPool", "Project.GetCurrentTimeline", "Project.GetTimelineCount",
        "Project.GetTimelineByIndex", "Project.SetCurrentTimeline",
        "Project.SetRenderSettings", "Project.AddRenderJob", "Project.StartRendering",
        "Project.GetRenderJobList", "Project.GetRenderJobStatus",
        "Project.GetRenderFormats", "Project.GetRenderCodecs",
        "Project.SetCurrentRenderFormatAndCodec", "Project.IsRenderingInProgress",
        "MediaPool.GetRootFolder", "MediaPool.AddSubFolder", "MediaPool.SetCurrentFolder",
        "MediaPool.GetCurrentFolder", "MediaPool.ImportMedia", "MediaPool.CreateEmptyTimeline",
        "MediaPool.AppendToTimeline", "MediaPool.DeleteClips", "MediaPool.DeleteTimelines",
        "Folder.GetName", "Folder.GetClipList", "Folder.GetSubFolderList",
        "MediaPoolItem.GetName", "MediaPoolItem.GetUniqueId", "MediaPoolItem.GetMediaId",
        "MediaPoolItem.GetClipProperty",
        "Timeline.GetName", "Timeline.GetUniqueId", "Timeline.GetSetting", "Timeline.SetSetting",
        "Timeline.GetTrackCount", "Timeline.GetItemListInTrack", "Timeline.GetStartFrame",
        "Timeline.GetEndFrame", "Timeline.DeleteClips", "Timeline.AddMarker",
        "TimelineItem.GetName", "TimelineItem.GetUniqueId", "TimelineItem.GetStart",
        "TimelineItem.GetEnd", "TimelineItem.GetDuration", "TimelineItem.GetLeftOffset",
        "TimelineItem.GetRightOffset", "TimelineItem.GetSourceStartFrame",
        "TimelineItem.GetSourceEndFrame", "TimelineItem.GetMediaPoolItem",
        "TimelineItem.SetProperty", "TimelineItem.GetProperty", "TimelineItem.AddMarker",
        "TimelineItem.GetMarkers",
    }
)

CLIP_A = "/media/a.mov"
CLIP_B = "/media/b.mov"
MUSIC = "/media/track.wav"


@pytest.fixture
def resolve(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeResolveApp, CallLog]:
    fake, log = install_fake_resolve(monkeypatch)
    fake.add_media_file(CLIP_A, frames=240, fps=24.0, kind="Video")  # 10s
    fake.add_media_file(CLIP_B, frames=480, fps=24.0, kind="Video")  # 20s
    fake.add_media_file(MUSIC, frames=720, fps=24.0, kind="Audio")  # 30s
    return fake, log


@pytest.fixture
def backend(resolve: tuple[FakeResolveApp, CallLog]) -> DaVinciResolveBackend:
    return DaVinciResolveBackend(allow_destructive=True)


def _seed(backend: DaVinciResolveBackend, *, fps: float = 24.0) -> tuple[str, str]:
    """A project with the two clips imported and an empty timeline, ready to edit."""
    backend.create_project("reel", {"fps": fps, "drop_frame": False}, 1920, 1080)
    clips = backend.import_media([CLIP_A, CLIP_B])
    backend.create_timeline("main", {"fps": fps, "drop_frame": False})
    return clips[0].id, clips[1].id


# --- connection ------------------------------------------------------------------


def test_methods_fail_clearly_when_resolve_is_not_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, _ = install_fake_resolve(monkeypatch)
    fake.running = False
    backend = DaVinciResolveBackend()
    assert backend.is_connected is False
    with pytest.raises(ResolveUnavailableError) as excinfo:
        backend.list_projects()
    message = str(excinfo.value)
    assert "RESOLVE_SCRIPT" in message or "scripting" in message.lower()


def test_connects_lazily_when_resolve_starts_later(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, _ = install_fake_resolve(monkeypatch)
    fake.running = False
    backend = DaVinciResolveBackend()
    with pytest.raises(ResolveUnavailableError):
        backend.list_projects()
    fake.running = True  # the user starts Resolve after the server
    assert backend.list_projects() == []


# --- projects --------------------------------------------------------------------


def test_create_project_applies_settings_and_reports_them(
    backend: DaVinciResolveBackend, resolve: tuple[FakeResolveApp, CallLog]
) -> None:
    fake, _ = resolve
    info = backend.create_project("reel", {"fps": 29.97, "drop_frame": True}, 1280, 720)
    assert (info.name, info.resolution_width, info.resolution_height) == ("reel", 1280, 720)
    assert info.frame_rate.fps == pytest.approx(29.97)
    assert info.frame_rate.drop_frame is True
    settings = fake.current_project.settings  # type: ignore[union-attr]
    assert settings["timelineFrameRate"] in {"29.97", "29.970"}
    assert settings["timelineResolutionWidth"] == "1280"
    assert settings["timelineResolutionHeight"] == "720"
    assert settings["timelineDropFrameTimecode"] == "1"


def test_create_duplicate_project_is_reported(backend: DaVinciResolveBackend) -> None:
    backend.create_project("reel", 24.0, 1920, 1080)
    with pytest.raises(Exception) as excinfo:
        backend.create_project("reel", 24.0, 1920, 1080)
    assert "reel" in str(excinfo.value)


def test_open_and_list_projects(backend: DaVinciResolveBackend) -> None:
    backend.create_project("one", 24.0, 1920, 1080)
    backend.create_project("two", 25.0, 1920, 1080)
    assert set(backend.list_projects()) == {"one", "two"}
    reopened = backend.open_project("one")
    assert reopened.name == "one"
    assert backend.current_project().name == "one"
    with pytest.raises(NotFoundError):
        backend.open_project("nope")


def test_save_project(backend: DaVinciResolveBackend, resolve: tuple[FakeResolveApp, CallLog]) -> None:
    _, log = resolve
    backend.create_project("reel", 24.0, 1920, 1080)
    assert backend.save_project().name == "reel"
    assert "ProjectManager.SaveProject" in log.methods_called()


# --- media pool ------------------------------------------------------------------


def test_import_media_reads_real_metadata(backend: DaVinciResolveBackend) -> None:
    backend.create_project("reel", 24.0, 1920, 1080)
    clips = backend.import_media([CLIP_A, MUSIC])
    by_name = {c.name: c for c in clips}
    assert by_name["a"].duration_seconds == pytest.approx(10.0)  # 240 frames @ 24
    assert by_name["a"].kind is MediaKind.VIDEO
    assert by_name["track"].kind is MediaKind.AUDIO
    assert by_name["track"].duration_seconds == pytest.approx(30.0)
    assert by_name["a"].path == CLIP_A


def test_import_into_a_bin(backend: DaVinciResolveBackend, resolve: tuple[FakeResolveApp, CallLog]) -> None:
    fake, _ = resolve
    backend.create_project("reel", 24.0, 1920, 1080)
    clips = backend.import_media([CLIP_A], bin="b-roll")
    assert clips[0].bin == "b-roll"
    pool = backend.list_media_pool()
    assert any(b.name == "b-roll" for b in pool.bins)
    root = fake.current_project.root  # type: ignore[union-attr]
    assert [sub.name for sub in root.subfolders] == ["b-roll"]


def test_media_ids_survive_across_calls(backend: DaVinciResolveBackend) -> None:
    """Resolve returns a NEW wrapper object per call; ids must not be object identity."""
    backend.create_project("reel", 24.0, 1920, 1080)
    imported = backend.import_media([CLIP_A])[0]
    listed = backend.list_media_pool().clips
    assert [c.id for c in listed] == [imported.id]
    backend.create_timeline("main", 24.0)
    # The id handed out at import time still resolves several calls later.
    backend.append_clip(imported.id, 1, 0.0, 2.0)
    assert len(backend.get_timeline_state().tracks[0].items) == 1


def test_unknown_media_id_is_reported(backend: DaVinciResolveBackend) -> None:
    _seed(backend)
    with pytest.raises(NotFoundError):
        backend.append_clip("mpi_does_not_exist", 1, 0.0, 1.0)


# --- timelines -------------------------------------------------------------------


def test_create_timeline_becomes_current(
    backend: DaVinciResolveBackend, resolve: tuple[FakeResolveApp, CallLog]
) -> None:
    fake, _ = resolve
    backend.create_project("reel", 24.0, 1920, 1080)
    state = backend.create_timeline("main", 24.0)
    assert state.name == "main"
    assert fake.current_project.current_timeline.name == "main"  # type: ignore[union-attr]
    assert [t.kind.value for t in state.tracks] == ["video", "audio"]


def test_list_and_switch_timelines(backend: DaVinciResolveBackend) -> None:
    backend.create_project("reel", 24.0, 1920, 1080)
    backend.create_timeline("first", 24.0)
    backend.create_timeline("second", 24.0)
    listed = {t.name: t.is_current for t in backend.list_timelines()}
    assert listed == {"first": False, "second": True}
    assert backend.set_current_timeline("first").name == "first"
    assert backend.get_timeline_state().name == "first"
    with pytest.raises(NotFoundError):
        backend.set_current_timeline("missing")


# --- placement and frame math ----------------------------------------------------


def test_append_places_at_absolute_record_frame(
    backend: DaVinciResolveBackend, resolve: tuple[FakeResolveApp, CallLog]
) -> None:
    fake, _ = resolve
    clip_a, _ = _seed(backend)
    backend.append_clip(clip_a, 1, start_seconds=2.0, duration_seconds=3.0, source_in_seconds=1.0)
    placed = fake.items_on("video", 1)
    assert len(placed) == 1
    item = placed[0]
    # 01:00:00:00 (86400) + 2s * 24fps
    assert item.record_frame == 86400 + 48
    assert item.source_start == 24  # 1.0s into the source
    assert item.duration == 72  # 3.0s
    # ...and the wire format reports it relative to the timeline start again.
    state = backend.get_timeline_state()
    wire = state.tracks[0].items[0]
    assert wire.start_seconds == pytest.approx(2.0)
    assert wire.duration_seconds == pytest.approx(3.0)
    assert wire.source_in_seconds == pytest.approx(1.0)


def test_audio_goes_to_the_audio_track(
    backend: DaVinciResolveBackend, resolve: tuple[FakeResolveApp, CallLog]
) -> None:
    fake, _ = resolve
    backend.create_project("reel", 24.0, 1920, 1080)
    music = backend.import_media([MUSIC])[0]
    backend.create_timeline("main", 24.0)
    backend.append_clip(music.id, 2, 0.0, 5.0)  # wire index 2 == A1
    assert fake.items_on("video", 1) == []
    audio = fake.items_on("audio", 1)
    assert len(audio) == 1 and audio[0].media_type == 2


def test_frame_math_at_23_976(backend: DaVinciResolveBackend, resolve: tuple[FakeResolveApp, CallLog]) -> None:
    fake, _ = resolve
    backend.create_project("reel", {"fps": 23.976, "drop_frame": False}, 1920, 1080)
    clip = backend.import_media([CLIP_A])[0]
    backend.create_timeline("main", {"fps": 23.976, "drop_frame": False})
    backend.append_clip(clip.id, 1, start_seconds=1.0, duration_seconds=2.0)
    item = fake.items_on("video", 1)[0]
    start_offset = item.record_frame - fake.timeline().start_frame  # type: ignore[union-attr]
    assert start_offset == 24  # nearest frame to 1.0s at 23.976fps
    assert item.duration == 48


def test_track_index_out_of_range(backend: DaVinciResolveBackend) -> None:
    clip_a, _ = _seed(backend)
    with pytest.raises(NotFoundError):
        backend.append_clip(clip_a, 9, 0.0, 1.0)


# --- item lifecycle --------------------------------------------------------------


def test_delete_clip_uses_timeline_delete_clips(
    backend: DaVinciResolveBackend, resolve: tuple[FakeResolveApp, CallLog]
) -> None:
    fake, log = resolve
    clip_a, _ = _seed(backend)
    backend.append_clip(clip_a, 1, 0.0, 2.0)
    item_id = backend.get_timeline_state().tracks[0].items[0].id
    backend.delete_clip(item_id)
    assert fake.items_on("video", 1) == []
    assert "Timeline.DeleteClips" in log.methods_called()


def test_move_clip_recreates_the_item_and_reports_the_new_id(
    backend: DaVinciResolveBackend, resolve: tuple[FakeResolveApp, CallLog]
) -> None:
    fake, _ = resolve
    clip_a, _ = _seed(backend)
    backend.append_clip(clip_a, 1, 0.0, 2.0)
    item = backend.get_timeline_state().tracks[0].items[0]
    backend.set_opacity(item.id, 0.5)
    backend.add_marker(item.id, 0.5, "hook", MarkerColor.RED, "note")

    delta = backend.move_clip(item.id, 6.0)

    assert item.id in delta.id_remap, "a recreated item must report its new id"
    new_id = delta.id_remap[item.id]
    state = backend.get_timeline_state()
    moved = state.tracks[0].items[0]
    assert moved.id == new_id
    assert moved.start_seconds == pytest.approx(6.0)
    assert moved.duration_seconds == pytest.approx(2.0)
    # Properties and markers that Resolve lets us read are carried over.
    assert moved.opacity == pytest.approx(0.5)
    assert [m.label for m in moved.markers] == ["hook"]
    assert fake.items_on("video", 1)[0].record_frame == 86400 + 144


def test_insert_clip_ripples_later_items(
    backend: DaVinciResolveBackend, resolve: tuple[FakeResolveApp, CallLog]
) -> None:
    fake, _ = resolve
    clip_a, clip_b = _seed(backend)
    backend.append_clip(clip_a, 1, 0.0, 2.0)
    backend.append_clip(clip_b, 1, 2.0, 2.0)
    before_ids = [i.id for i in backend.get_timeline_state().tracks[0].items]

    delta = backend.insert_clip(clip_a, 1, 0.0, 1.0)

    starts = [i.start_seconds for i in backend.get_timeline_state().tracks[0].items]
    assert starts == pytest.approx([0.0, 1.0, 3.0])
    # Both original items were recreated further right, so both must be remapped.
    assert set(delta.id_remap) == set(before_ids)
    assert len(fake.items_on("video", 1)) == 3


# --- properties ------------------------------------------------------------------


def test_transform_writes_resolve_units_and_reads_back(
    backend: DaVinciResolveBackend, resolve: tuple[FakeResolveApp, CallLog]
) -> None:
    fake, log = resolve
    clip_a, _ = _seed(backend)
    backend.append_clip(clip_a, 1, 0.0, 2.0)
    item_id = backend.get_timeline_state().tracks[0].items[0].id

    delta = backend.set_transform(
        item_id, pan_x=120.0, pan_y=-40.0, zoom_x=1.5, zoom_y=1.5, rotation=12.0,
        anchor_x=10.0, anchor_y=-10.0,
    )

    props = fake.items_on("video", 1)[0].properties
    assert props["Pan"] == pytest.approx(120.0)
    assert props["Tilt"] == pytest.approx(-40.0)
    assert props["ZoomX"] == pytest.approx(1.5)
    assert props["RotationAngle"] == pytest.approx(12.0)
    assert props["AnchorPointX"] == pytest.approx(10.0)
    after = delta.after["tracks"][0]["items"][0]["transform"]
    assert after["pan_x"] == pytest.approx(120.0)
    assert after["zoom_x"] == pytest.approx(1.5)
    assert "TimelineItem.GetProperty" in log.methods_called(), "the edit must be read back"


def test_crop_and_opacity_use_resolve_scales(
    backend: DaVinciResolveBackend, resolve: tuple[FakeResolveApp, CallLog]
) -> None:
    fake, _ = resolve
    clip_a, _ = _seed(backend)
    backend.append_clip(clip_a, 1, 0.0, 2.0)
    item_id = backend.get_timeline_state().tracks[0].items[0].id
    backend.set_crop(item_id, left=100.0, right=50.0, top=10.0, bottom=5.0)
    backend.set_opacity(item_id, 0.25)
    props = fake.items_on("video", 1)[0].properties
    assert props["CropLeft"] == pytest.approx(100.0)
    assert props["Opacity"] == pytest.approx(25.0), "Resolve opacity is a percentage"
    item = backend.get_timeline_state().tracks[0].items[0]
    assert item.opacity == pytest.approx(0.25)
    assert item.crop.left == pytest.approx(100.0)


def test_composite_mode_uses_documented_constants(
    backend: DaVinciResolveBackend, resolve: tuple[FakeResolveApp, CallLog]
) -> None:
    fake, _ = resolve
    clip_a, _ = _seed(backend)
    backend.append_clip(clip_a, 1, 0.0, 2.0)
    item_id = backend.get_timeline_state().tracks[0].items[0].id
    backend.set_composite_mode(item_id, CompositeMode.SCREEN)
    stored = fake.items_on("video", 1)[0].properties["CompositeMode"]
    assert stored == COMPOSITE_CONSTANTS.index("COMPOSITE_SCREEN")
    assert backend.get_timeline_state().tracks[0].items[0].composite_mode is CompositeMode.SCREEN


def test_add_marker_passes_six_arguments(
    backend: DaVinciResolveBackend, resolve: tuple[FakeResolveApp, CallLog]
) -> None:
    _fake, log = resolve
    clip_a, _ = _seed(backend)
    backend.append_clip(clip_a, 1, 0.0, 4.0)
    item_id = backend.get_timeline_state().tracks[0].items[0].id
    backend.add_marker(item_id, 1.0, "hook", MarkerColor.CREAM, "the drop")
    (call,) = log.args_for("TimelineItem.AddMarker")
    frame_id, color, name, note, duration = call[:5]
    assert frame_id == 24, "marker frames are relative to the item"
    assert color == "Cream", "Resolve wants capitalized colour names"
    assert (name, note) == ("hook", "the drop")
    assert duration >= 1
    markers = backend.get_timeline_state().tracks[0].items[0].markers
    assert [(m.label, m.color) for m in markers] == [("hook", MarkerColor.CREAM)]


# --- honest degradation ----------------------------------------------------------


def test_unsupported_operations_refuse_instead_of_faking(backend: DaVinciResolveBackend) -> None:
    clip_a, _ = _seed(backend)
    backend.append_clip(clip_a, 1, 0.0, 4.0)
    item_id = backend.get_timeline_state().tracks[0].items[0].id
    with pytest.raises(UnsupportedOperationError):
        backend.add_fade(item_id, 0.5, 0.5)
    with pytest.raises(UnsupportedOperationError):
        backend.set_speed(item_id, 2.0)
    with pytest.raises(UnsupportedOperationError):
        backend.add_transition(item_id, 1, "cross_dissolve", 1.0, "mid")
    with pytest.raises(UnsupportedOperationError):
        backend.restart_app(confirm=True)


def test_unsupported_tools_are_declared(backend: DaVinciResolveBackend) -> None:
    assert set(backend.unsupported_tools()) == {
        "add_fade",
        "set_speed",
        "add_transition",
        "restart_app",
    }


# --- render ----------------------------------------------------------------------


def test_render_job_lifecycle(
    backend: DaVinciResolveBackend, resolve: tuple[FakeResolveApp, CallLog], tmp_path: Any
) -> None:
    fake, log = resolve
    clip_a, _unused = _seed(backend)
    backend.append_clip(clip_a, 1, 0.0, 2.0)
    out = tmp_path / "exports" / "reel.mp4"

    job = backend.add_render_job("main", "mp4", str(out))
    assert job.status is RenderJobStatus.QUEUED
    settings = fake.current_project.render_settings  # type: ignore[union-attr]
    assert settings["TargetDir"] == str(out.parent)
    assert settings["CustomName"] in {"reel", "reel.mp4"}
    assert fake.current_project.render_format is not None  # type: ignore[union-attr]
    assert "Project.SetCurrentRenderFormatAndCodec" in log.methods_called()

    started = backend.start_render(job.id)
    assert started.status is RenderJobStatus.RUNNING
    status = backend.get_render_status(job.id)
    assert status.status in {RenderJobStatus.RUNNING, RenderJobStatus.COMPLETED}
    assert 0.0 <= status.progress <= 1.0


def test_render_unknown_job(backend: DaVinciResolveBackend) -> None:
    _seed(backend)
    with pytest.raises(NotFoundError):
        backend.start_render("job_nope")
    with pytest.raises(NotFoundError):
        backend.get_render_status("job_nope")


def test_render_unknown_timeline(backend: DaVinciResolveBackend) -> None:
    _seed(backend)
    with pytest.raises(NotFoundError):
        backend.add_render_job("absent", "mp4", "/tmp/x.mp4")


# --- destructive -----------------------------------------------------------------


def test_destructive_requires_flag_and_confirm(
    resolve: tuple[FakeResolveApp, CallLog],
) -> None:
    gated = DaVinciResolveBackend(allow_destructive=False)
    gated.create_project("reel", 24.0, 1920, 1080)
    gated.create_timeline("main", 24.0)
    with pytest.raises(DestructiveDisabledError):
        gated.quit_app(confirm=True)
    with pytest.raises(DestructiveDisabledError):
        gated.delete_timeline("main", confirm=True)

    allowed = DaVinciResolveBackend(allow_destructive=True)
    with pytest.raises(DestructiveDisabledError):
        allowed.quit_app(confirm=False)


def test_delete_timeline_and_media(
    backend: DaVinciResolveBackend, resolve: tuple[FakeResolveApp, CallLog]
) -> None:
    fake, log = resolve
    clip_a, _ = _seed(backend)
    backend.delete_timeline("main", confirm=True)
    assert fake.current_project.timelines == []  # type: ignore[union-attr]
    assert "MediaPool.DeleteTimelines" in log.methods_called()

    backend.delete_media(clip_a, confirm=True)
    assert backend.list_media_pool().clips[0].id != clip_a


def test_quit_app(backend: DaVinciResolveBackend, resolve: tuple[FakeResolveApp, CallLog]) -> None:
    fake, _ = resolve
    backend.create_project("reel", 24.0, 1920, 1080)
    backend.quit_app(confirm=True)
    assert fake.quit_called is True


# --- the guard rail --------------------------------------------------------------


def test_calls_only_documented_api(
    backend: DaVinciResolveBackend, resolve: tuple[FakeResolveApp, CallLog], tmp_path: Any
) -> None:
    """Exercise the whole surface, then assert every call was a documented one."""
    _fake, log = resolve
    clip_a, clip_b = _seed(backend)
    backend.list_projects()
    backend.current_project()
    backend.save_project()
    backend.create_bin("b-roll")
    backend.list_media_pool()
    backend.list_timelines()
    backend.append_clip(clip_a, 1, 0.0, 2.0)
    backend.append_clip(clip_b, 1, 2.0, 2.0)
    first = backend.get_timeline_state().tracks[0].items[0].id
    backend.set_transform(first, 0.0, 0.0, 1.2, 1.2, 0.0)
    backend.set_crop(first, 1.0, 1.0, 1.0, 1.0)
    backend.set_opacity(first, 0.8)
    backend.set_composite_mode(first, CompositeMode.ADD)
    backend.add_marker(first, 0.5, "m", MarkerColor.BLUE, "")
    backend.move_clip(first, 6.0)
    backend.insert_clip(clip_a, 1, 0.0, 1.0)
    remaining = backend.get_timeline_state().tracks[0].items[-1].id
    backend.delete_clip(remaining)
    job = backend.add_render_job("main", "mov", str(tmp_path / "out.mov"))
    backend.start_render(job.id)
    backend.get_render_status(job.id)
    backend.delete_media(clip_b, confirm=True)
    backend.delete_timeline("main", confirm=True)

    undocumented = sorted(set(log.methods_called()) - DOCUMENTED_API)
    assert not undocumented, f"backend called undocumented Resolve API: {undocumented}"


def test_a_wrong_length_from_resolve_fails_loudly_and_leaves_nothing_behind(
    backend: DaVinciResolveBackend,
    resolve: tuple[FakeResolveApp, CallLog],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blackmagic's docs never say whether clipInfo's ``endFrame`` is inclusive.

    We assume exclusive. If a build disagrees, every clip is one frame too long
    and shots that tile perfectly start colliding several appends later — so the
    backend checks the length Resolve actually built and reports the mismatch
    with both numbers, instead of leaving a wrong clip on the timeline.
    """
    fake, _ = resolve
    clip_a, _ = _seed(backend)
    from .fake_resolve import MediaPool

    real_append = MediaPool.AppendToTimeline

    def one_frame_long(self: Any, clipInfos: list[dict[str, Any]]) -> list[Any]:
        stretched = [{**info, "endFrame": int(info["endFrame"]) + 1} for info in clipInfos]
        return real_append(self, stretched)

    monkeypatch.setattr(MediaPool, "AppendToTimeline", one_frame_long)

    with pytest.raises(Exception, match="INCLUSIVE") as caught:
        backend.append_clip(clip_a, 1, start_seconds=0.0, duration_seconds=2.0)
    assert "49-frame item for a 48-frame request" in str(caught.value)
    assert fake.items_on("video", 1) == []


def test_render_status_survives_a_different_key_casing(
    backend: DaVinciResolveBackend,
    resolve: tuple[FakeResolveApp, CallLog],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reference documents GetRenderJobStatus by example only. If a build
    spells the keys differently, defaulting would report a finished render as
    queued for ever — the worst kind of wrong answer for a long operation."""
    from .fake_resolve import Project

    clip_a, _ = _seed(backend)
    backend.append_clip(clip_a, 1, 0.0, 2.0)
    job = backend.add_render_job("main", "mp4", "/tmp/out/reel.mp4")
    backend.start_render(job.id)

    monkeypatch.setattr(
        Project,
        "GetRenderJobStatus",
        lambda self, jid: {"jobStatus": "Complete", "completion percentage": 100},
    )
    status = backend.get_render_status(job.id)
    assert status.status == RenderJobStatus.COMPLETED
    assert status.progress == pytest.approx(1.0)


def test_render_status_without_any_status_field_is_an_error(
    backend: DaVinciResolveBackend,
    resolve: tuple[FakeResolveApp, CallLog],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from .fake_resolve import Project

    clip_a, _ = _seed(backend)
    backend.append_clip(clip_a, 1, 0.0, 2.0)
    job = backend.add_render_job("main", "mp4", "/tmp/out/reel.mp4")

    monkeypatch.setattr(Project, "GetRenderJobStatus", lambda self, jid: {"Unexpected": 1})
    with pytest.raises(Exception, match="no status field"):
        backend.get_render_status(job.id)
