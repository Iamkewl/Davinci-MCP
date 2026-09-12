"""In-memory fake backend: a believable DaVinci Resolve stand-in.

This backend is what CI, the director's offline mode, and anyone without Resolve
Studio actually run against, so it aims for *behavioural* fidelity rather than a
convenient stub:

* State is scoped per project — switching projects switches media pool,
  timelines and render jobs, exactly like Resolve.
* Positions and durations are quantized to whole frames through
  :mod:`resolve_mcp.timecode`, so an edit placed at 1.234s reports back the
  frame-aligned value a real NLE would give you.
* Media is probed with ``ffprobe`` when it is on PATH, so clip durations and
  kinds are real; a file that does not exist cannot be imported.
* Clips may not silently overlap on a track, fades may not exceed the item, and
  drop-frame is rejected for frame rates that have no drop-frame form.

All mutations are single-threaded behind an ``RLock`` so the server can spawn
worker tasks without corrupting state.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import uuid
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backend import (
    AlreadyExistsError,
    DestructiveDisabledError,
    InvalidStateError,
    NotFoundError,
)
from .schemas import (
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
    Transition,
    TransitionAlignment,
    TransitionStyle,
)
from .timecode import TimeConverter

_PROBE_TIMEOUT_SECONDS = 15.0


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class _ProjectState:
    """Everything scoped to one project (Resolve scopes all of this per project)."""

    info: ProjectInfo
    clips: dict[str, MediaClip] = field(default_factory=dict)
    bins: dict[str, Bin] = field(default_factory=dict)
    timelines: dict[str, TimelineState] = field(default_factory=dict)
    current_timeline: str | None = None
    render_jobs: dict[str, RenderJob] = field(default_factory=dict)


class FakeResolveBackend:
    """A complete, deterministic, in-memory Resolve backend."""

    def __init__(self, *, allow_destructive: bool = False) -> None:
        self._lock = threading.RLock()
        self._projects: dict[str, _ProjectState] = {}
        self._current_project: str | None = None
        self._allow_destructive: bool = allow_destructive

    # ---- capabilities --------------------------------------------------------

    def unsupported_tools(self) -> frozenset[str]:
        """The fake backend implements every tool — that is its whole point."""
        return frozenset()

    # ---- projects ------------------------------------------------------------

    def list_projects(self) -> list[str]:
        with self._lock:
            return sorted(self._projects.keys())

    def create_project(
        self,
        name: str,
        frame_rate: FrameRate | dict[str, Any] | float,
        width: int,
        height: int,
    ) -> ProjectInfo:
        with self._lock:
            if name in self._projects:
                msg = f"project {name!r} already exists"
                raise AlreadyExistsError(msg)
            fr = self._coerce_frame_rate(frame_rate)
            info = ProjectInfo(
                name=name,
                frame_rate=fr,
                resolution_width=width,
                resolution_height=height,
                path=None,
                is_modified=False,
            )
            state = _ProjectState(info=info)
            state.bins["Master"] = Bin(name="Master", clip_ids=[])
            self._projects[name] = state
            self._current_project = name
            return info

    def open_project(self, name: str) -> ProjectInfo:
        with self._lock:
            state = self._projects.get(name)
            if state is None:
                known = ", ".join(sorted(self._projects)) or "<none>"
                msg = f"project {name!r} not found (have: {known})"
                raise NotFoundError(msg)
            self._current_project = name
            return state.info

    def save_project(self) -> ProjectInfo:
        with self._lock:
            state = self._require_project()
            state.info = state.info.model_copy(update={"is_modified": False})
            return state.info

    def current_project(self) -> ProjectInfo:
        with self._lock:
            return self._require_project().info

    # ---- media pool ----------------------------------------------------------

    def list_media_pool(self) -> MediaPoolState:
        with self._lock:
            state = self._require_project()
            return MediaPoolState(
                bins=sorted(state.bins.values(), key=lambda b: b.name),
                clips=sorted(state.clips.values(), key=lambda c: c.name),
            )

    def create_bin(self, name: str) -> Bin:
        with self._lock:
            state = self._require_project()
            if name in state.bins:
                msg = f"bin {name!r} already exists"
                raise AlreadyExistsError(msg)
            bin_ = Bin(name=name, clip_ids=[])
            state.bins[name] = bin_
            self._touch(state)
            return bin_

    def import_media(self, paths: Iterable[str], bin: str | None = None) -> list[MediaClip]:
        with self._lock:
            state = self._require_project()
            bin_name = bin or "Master"
            if bin_name not in state.bins:
                msg = f"bin {bin_name!r} does not exist"
                raise NotFoundError(msg)
            path_list = list(paths)
            if not path_list:
                msg = "import_media needs at least one path"
                raise InvalidStateError(msg)
            missing = [p for p in path_list if not Path(p).is_file()]
            if missing:
                msg = f"cannot import, file(s) not found: {', '.join(sorted(missing))}"
                raise NotFoundError(msg)
            out: list[MediaClip] = []
            for path in path_list:
                probe = _probe_media(path)
                clip = MediaClip(
                    id=_new_id("clip"),
                    bin=bin_name,
                    name=self._unique_clip_name(state, Path(path).stem or Path(path).name),
                    path=path,
                    kind=probe.kind,
                    duration_seconds=probe.duration_seconds,
                    frame_rate=FrameRate(fps=probe.fps, drop_frame=False) if probe.fps else None,
                    resolution_width=probe.width,
                    resolution_height=probe.height,
                )
                state.clips[clip.id] = clip
                state.bins[bin_name].clip_ids.append(clip.id)
                out.append(clip)
            self._touch(state)
            return out

    # ---- timeline ------------------------------------------------------------

    def create_timeline(
        self, name: str, frame_rate: FrameRate | dict[str, Any] | float
    ) -> TimelineState:
        with self._lock:
            state = self._require_project()
            fr = self._coerce_frame_rate(frame_rate)
            if name in state.timelines:
                msg = f"timeline {name!r} already exists"
                raise AlreadyExistsError(msg)
            tl = TimelineState(
                name=name,
                frame_rate=fr,
                duration_seconds=0.0,
                tracks=[
                    Track(index=1, kind=TrackKind.VIDEO, items=[]),
                    Track(index=2, kind=TrackKind.AUDIO, items=[]),
                ],
            )
            state.timelines[name] = tl
            state.current_timeline = name
            self._touch(state)
            return tl

    def list_timelines(self) -> list[TimelineSummary]:
        with self._lock:
            state = self._require_project()
            return [
                TimelineSummary(name=name, is_current=(name == state.current_timeline))
                for name in sorted(state.timelines)
            ]

    def set_current_timeline(self, name: str) -> TimelineState:
        with self._lock:
            state = self._require_project()
            tl = state.timelines.get(name)
            if tl is None:
                known = ", ".join(sorted(state.timelines)) or "<none>"
                msg = f"timeline {name!r} not found in project {state.info.name!r} (have: {known})"
                raise NotFoundError(msg)
            state.current_timeline = name
            return tl

    def get_timeline_state(self) -> TimelineState:
        with self._lock:
            return self._require_timeline()

    def append_clip(
        self,
        media_clip_id: str,
        timeline_track_index: int,
        start_seconds: float,
        duration_seconds: float,
        source_in_seconds: float = 0.0,
    ) -> StateDelta:
        return self._place_clip(
            media_clip_id=media_clip_id,
            timeline_track_index=timeline_track_index,
            position_seconds=start_seconds,
            duration_seconds=duration_seconds,
            source_in_seconds=source_in_seconds,
            ripple=False,
        )

    def insert_clip(
        self,
        media_clip_id: str,
        timeline_track_index: int,
        timeline_position_seconds: float,
        duration_seconds: float,
        source_in_seconds: float = 0.0,
    ) -> StateDelta:
        return self._place_clip(
            media_clip_id=media_clip_id,
            timeline_track_index=timeline_track_index,
            position_seconds=timeline_position_seconds,
            duration_seconds=duration_seconds,
            source_in_seconds=source_in_seconds,
            ripple=True,
        )

    def _place_clip(
        self,
        *,
        media_clip_id: str,
        timeline_track_index: int,
        position_seconds: float,
        duration_seconds: float,
        source_in_seconds: float,
        ripple: bool,
    ) -> StateDelta:
        with self._lock:
            state = self._require_project()
            tl = self._require_timeline()
            media = state.clips.get(media_clip_id)
            if media is None:
                known = ", ".join(sorted(state.clips)) or "<pool empty>"
                msg = f"media clip {media_clip_id!r} not found (pool has: {known})"
                raise NotFoundError(msg)
            track = self._get_track(tl, timeline_track_index)
            if not self._fits_track_kind(track.kind, media.kind):
                msg = (
                    f"media kind {media.kind.value} cannot live on "
                    f"{track.kind.value} track {track.index}"
                )
                raise InvalidStateError(msg)
            if duration_seconds <= 0:
                msg = f"duration_seconds must be > 0, got {duration_seconds}"
                raise InvalidStateError(msg)
            conv = TimeConverter(tl.frame_rate)
            # Snap both ends of the span: rounding start and duration separately
            # can push a shot one frame into the next one, turning a perfectly
            # tiled edit into an overlap.
            start = self._quantize(conv, position_seconds)
            end = self._quantize(conv, position_seconds + duration_seconds)
            duration = round(end - start, 6)
            source_in = self._quantize(conv, source_in_seconds)
            if duration <= 0:
                msg = (
                    f"duration_seconds {duration_seconds} is shorter than one frame at "
                    f"{tl.frame_rate.fps} fps"
                )
                raise InvalidStateError(msg)
            if media.duration_seconds > 0 and source_in + duration > media.duration_seconds + 1e-6:
                msg = (
                    f"source range {source_in:.3f}..{source_in + duration:.3f}s exceeds "
                    f"{media.name!r} duration {media.duration_seconds:.3f}s"
                )
                raise InvalidStateError(msg)
            before = deepcopy(tl)
            items = list(track.items)
            if ripple:
                straddler = next(
                    (
                        it
                        for it in items
                        if it.start_seconds < start < it.start_seconds + it.duration_seconds
                    ),
                    None,
                )
                if straddler is not None:
                    msg = (
                        f"insert position {start}s falls inside item {straddler.id} "
                        f"({straddler.start_seconds}.."
                        f"{straddler.start_seconds + straddler.duration_seconds}s); "
                        "insert at a clip boundary or use append_clip"
                    )
                    raise InvalidStateError(msg)
                items = [
                    it.model_copy(update={"start_seconds": it.start_seconds + duration})
                    if it.start_seconds >= start
                    else it
                    for it in items
                ]
            clash = self._overlapping(items, start, duration)
            if clash is not None:
                msg = (
                    f"clip would overlap item {clash.id} at {clash.start_seconds}.."
                    f"{clash.start_seconds + clash.duration_seconds}s on track "
                    f"{track.index}; move it or use insert_clip"
                )
                raise InvalidStateError(msg)
            item = TimelineItem(
                id=_new_id("item"),
                track_index=track.index,
                media_clip_id=media.id,
                start_seconds=start,
                duration_seconds=duration,
                source_in_seconds=source_in,
                source_out_seconds=source_in + duration,
            )
            new_items = sorted([*items, item], key=lambda it: it.start_seconds)
            new_tl = self._replace_track(tl, track.index, new_items)
            self._store_timeline(state, new_tl)
            return _delta(
                before,
                state.timelines[new_tl.name],
                changed_path=f"timelines.{new_tl.name}.tracks[{track.index}].items",
            )

    def delete_clip(self, timeline_item_id: str) -> StateDelta:
        with self._lock:
            state = self._require_project()
            tl = self._require_timeline()
            before = deepcopy(tl)
            removed_path = ""
            new_tracks: list[Track] = []
            for track in tl.tracks:
                kept = [it for it in track.items if it.id != timeline_item_id]
                if len(kept) != len(track.items):
                    removed_path = f"timelines.{tl.name}.tracks[{track.index}].items"
                new_tracks.append(track.model_copy(update={"items": kept}))
            if not removed_path:
                msg = f"timeline item {timeline_item_id!r} not found on timeline {tl.name!r}"
                raise NotFoundError(msg)
            new_tl = tl.model_copy(update={"tracks": new_tracks})
            self._store_timeline(state, new_tl)
            return _delta(before, state.timelines[new_tl.name], changed_path=removed_path)

    def move_clip(self, timeline_item_id: str, new_position_seconds: float) -> StateDelta:
        with self._lock:
            state = self._require_project()
            tl = self._require_timeline()
            conv = TimeConverter(tl.frame_rate)
            target = self._quantize(conv, new_position_seconds)
            before = deepcopy(tl)
            moved_path = ""
            new_tracks: list[Track] = []
            for track in tl.tracks:
                match = next((it for it in track.items if it.id == timeline_item_id), None)
                if match is None:
                    new_tracks.append(track)
                    continue
                others = [it for it in track.items if it.id != timeline_item_id]
                clash = self._overlapping(others, target, match.duration_seconds)
                if clash is not None:
                    msg = (
                        f"moving {timeline_item_id} to {target}s would overlap item "
                        f"{clash.id} at {clash.start_seconds}.."
                        f"{clash.start_seconds + clash.duration_seconds}s"
                    )
                    raise InvalidStateError(msg)
                moved = match.model_copy(update={"start_seconds": target})
                moved_path = f"timelines.{tl.name}.tracks[{track.index}].items"
                new_tracks.append(
                    track.model_copy(
                        update={"items": sorted([*others, moved], key=lambda i2: i2.start_seconds)}
                    )
                )
            if not moved_path:
                msg = f"timeline item {timeline_item_id!r} not found on timeline {tl.name!r}"
                raise NotFoundError(msg)
            new_tl = tl.model_copy(update={"tracks": new_tracks})
            self._store_timeline(state, new_tl)
            return _delta(before, state.timelines[new_tl.name], changed_path=moved_path)

    # ---- per-item mutations --------------------------------------------------

    def _update_item(
        self,
        timeline_item_id: str,
        updater: Callable[[TimelineItem], TimelineItem],
        path: str,
    ) -> StateDelta:
        with self._lock:
            state = self._require_project()
            tl = self._require_timeline()
            before = deepcopy(tl)
            found = False
            new_tracks: list[Track] = []
            for track in tl.tracks:
                replaced: list[TimelineItem] = []
                for it in track.items:
                    if it.id == timeline_item_id:
                        replaced.append(updater(it))
                        found = True
                    else:
                        replaced.append(it)
                new_tracks.append(track.model_copy(update={"items": replaced}))
            if not found:
                msg = f"timeline item {timeline_item_id!r} not found on timeline {tl.name!r}"
                raise NotFoundError(msg)
            new_tl = tl.model_copy(update={"tracks": new_tracks})
            self._store_timeline(state, new_tl)
            return _delta(before, state.timelines[new_tl.name], changed_path=path)

    def set_transform(
        self,
        timeline_item_id: str,
        pan_x: float,
        pan_y: float,
        zoom_x: float,
        zoom_y: float,
        rotation: float,
        anchor_x: float = 0.0,
        anchor_y: float = 0.0,
    ) -> StateDelta:
        new_t = Transform(
            pan_x=pan_x,
            pan_y=pan_y,
            zoom_x=zoom_x,
            zoom_y=zoom_y,
            rotation=rotation,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
        )
        return self._update_item(
            timeline_item_id,
            updater=lambda it: it.model_copy(update={"transform": new_t}),
            path=f"timelines.*.tracks.*.items[{timeline_item_id}].transform",
        )

    def set_crop(
        self,
        timeline_item_id: str,
        left: float,
        right: float,
        top: float,
        bottom: float,
    ) -> StateDelta:
        new_c = Crop(left=left, right=right, top=top, bottom=bottom)
        return self._update_item(
            timeline_item_id,
            updater=lambda it: it.model_copy(update={"crop": new_c}),
            path=f"timelines.*.tracks.*.items[{timeline_item_id}].crop",
        )

    def set_composite_mode(self, timeline_item_id: str, mode: CompositeMode | str) -> StateDelta:
        resolved = CompositeMode(mode) if isinstance(mode, str) else mode
        return self._update_item(
            timeline_item_id,
            updater=lambda it: it.model_copy(update={"composite_mode": resolved}),
            path=f"timelines.*.tracks.*.items[{timeline_item_id}].composite_mode",
        )

    def set_opacity(self, timeline_item_id: str, opacity: float) -> StateDelta:
        if not 0.0 <= opacity <= 1.0:
            msg = f"opacity must be in [0, 1], got {opacity}"
            raise InvalidStateError(msg)
        return self._update_item(
            timeline_item_id,
            updater=lambda it: it.model_copy(update={"opacity": opacity}),
            path=f"timelines.*.tracks.*.items[{timeline_item_id}].opacity",
        )

    def add_fade(
        self,
        timeline_item_id: str,
        fade_in_seconds: float,
        fade_out_seconds: float,
    ) -> StateDelta:
        if fade_in_seconds < 0 or fade_out_seconds < 0:
            msg = "fade durations must be non-negative"
            raise InvalidStateError(msg)

        def _apply(it: TimelineItem) -> TimelineItem:
            if fade_in_seconds + fade_out_seconds > it.duration_seconds + 1e-9:
                msg = (
                    f"fades {fade_in_seconds}s + {fade_out_seconds}s exceed item duration "
                    f"{it.duration_seconds}s"
                )
                raise InvalidStateError(msg)
            return it.model_copy(
                update={
                    "fade_in_seconds": fade_in_seconds,
                    "fade_out_seconds": fade_out_seconds,
                }
            )

        return self._update_item(
            timeline_item_id,
            updater=_apply,
            path=f"timelines.*.tracks.*.items[{timeline_item_id}].fade",
        )

    def set_speed(self, timeline_item_id: str, speed: float) -> StateDelta:
        if speed <= 0:
            msg = f"speed must be > 0, got {speed}"
            raise InvalidStateError(msg)
        return self._update_item(
            timeline_item_id,
            updater=lambda it: it.model_copy(update={"speed": speed}),
            path=f"timelines.*.tracks.*.items[{timeline_item_id}].speed",
        )

    def add_marker(
        self,
        timeline_item_id: str,
        position_seconds: float,
        label: str,
        color: MarkerColor | str,
        note: str = "",
    ) -> StateDelta:
        resolved = MarkerColor(color) if isinstance(color, str) else color

        def _apply(it: TimelineItem) -> TimelineItem:
            if position_seconds < 0 or position_seconds >= it.duration_seconds:
                msg = (
                    f"marker at {position_seconds}s is outside item {it.id} "
                    f"(0..{it.duration_seconds}s); positions are relative to the item start"
                )
                raise InvalidStateError(msg)
            marker = Marker(
                id=_new_id("marker"),
                timeline_item_id=it.id,
                position_seconds=position_seconds,
                label=label,
                color=resolved,
                note=note,
            )
            return it.model_copy(update={"markers": [*it.markers, marker]})

        return self._update_item(
            timeline_item_id,
            updater=_apply,
            path=f"timelines.*.tracks.*.items[{timeline_item_id}].markers",
        )

    # ---- transitions & render ------------------------------------------------

    def add_transition(
        self,
        timeline_item_id: str,
        track_index: int,
        style: TransitionStyle | str,
        duration_seconds: float,
        alignment: TransitionAlignment | str,
    ) -> StateDelta:
        resolved_style = TransitionStyle(style) if isinstance(style, str) else style
        resolved_alignment = (
            TransitionAlignment(alignment) if isinstance(alignment, str) else alignment
        )
        if duration_seconds <= 0:
            msg = f"transition duration must be > 0, got {duration_seconds}"
            raise InvalidStateError(msg)
        with self._lock:
            state = self._require_project()
            tl = self._require_timeline()
            before = deepcopy(tl)
            target = next(
                (
                    it
                    for track in tl.tracks
                    if track.index == track_index
                    for it in track.items
                    if it.id == timeline_item_id
                ),
                None,
            )
            if target is None:
                msg = f"timeline item {timeline_item_id!r} not found on track {track_index}"
                raise NotFoundError(msg)
            if duration_seconds > target.duration_seconds:
                msg = (
                    f"transition {duration_seconds}s is longer than item {target.id} "
                    f"({target.duration_seconds}s)"
                )
                raise InvalidStateError(msg)
            transition = Transition(
                id=_new_id("trans"),
                timeline_item_id=timeline_item_id,
                track_index=track_index,
                style=resolved_style,
                duration_seconds=duration_seconds,
                alignment=resolved_alignment,
            )
            new_tl = tl.model_copy(update={"transitions": [*tl.transitions, transition]})
            self._store_timeline(state, new_tl)
            return _delta(
                before,
                state.timelines[new_tl.name],
                changed_path=f"timelines.{tl.name}.transitions",
            )

    def add_render_job(
        self,
        timeline_name: str,
        format: RenderJobFormat | str,
        output_path: str,
    ) -> RenderJob:
        with self._lock:
            state = self._require_project()
            if timeline_name not in state.timelines:
                known = ", ".join(sorted(state.timelines)) or "<none>"
                msg = f"timeline {timeline_name!r} not found (have: {known})"
                raise NotFoundError(msg)
            resolved = RenderJobFormat(format) if isinstance(format, str) else format
            job = RenderJob(
                id=_new_id("render"),
                timeline_name=timeline_name,
                format=resolved,
                output_path=output_path,
                status=RenderJobStatus.QUEUED,
            )
            state.render_jobs[job.id] = job
            return job

    def start_render(self, job_id: str) -> RenderJob:
        with self._lock:
            state = self._require_project()
            job = state.render_jobs.get(job_id)
            if job is None:
                msg = f"render job {job_id!r} not found"
                raise NotFoundError(msg)
            if job.status not in (RenderJobStatus.QUEUED, RenderJobStatus.FAILED):
                msg = f"render job {job_id!r} is {job.status.value}; cannot start"
                raise InvalidStateError(msg)
            updated = job.model_copy(update={"status": RenderJobStatus.RUNNING, "progress": 0.0})
            state.render_jobs[job_id] = updated
            return updated

    def get_render_status(self, job_id: str) -> RenderJob:
        with self._lock:
            state = self._require_project()
            job = state.render_jobs.get(job_id)
            if job is None:
                msg = f"render job {job_id!r} not found"
                raise NotFoundError(msg)
            if job.status == RenderJobStatus.RUNNING:
                # A fake render is finished by the time you ask, so polling clients
                # terminate instead of spinning forever.
                job = job.model_copy(
                    update={"status": RenderJobStatus.COMPLETED, "progress": 1.0}
                )
                state.render_jobs[job_id] = job
            return job

    # ---- destructive (gated) -------------------------------------------------

    def _require_destructive(self, confirm: bool) -> None:
        if not (self._allow_destructive and confirm):
            msg = (
                "destructive tool is disabled. Run the server with --allow-destructive "
                "and pass confirm=true."
            )
            raise DestructiveDisabledError(msg)

    def quit_app(self, confirm: bool = False) -> dict[str, Any]:
        with self._lock:
            self._require_destructive(confirm)
            self._current_project = None
            return {"quit": True, "project_after": None}

    def restart_app(self, confirm: bool = False) -> dict[str, Any]:
        with self._lock:
            self._require_destructive(confirm)
            self._current_project = None
            return {"restart": True, "project_after": None}

    def delete_timeline(self, name: str, confirm: bool = False) -> StateDelta:
        with self._lock:
            self._require_destructive(confirm)
            state = self._require_project()
            before = state.timelines.get(name)
            if before is None:
                msg = f"timeline {name!r} not found"
                raise NotFoundError(msg)
            del state.timelines[name]
            if state.current_timeline == name:
                state.current_timeline = None
            self._touch(state)
            after = before.model_copy(
                update={"tracks": [], "transitions": [], "duration_seconds": 0.0}
            )
            return _delta(before, after, changed_path=f"timelines.{name}.deleted")

    def delete_media(self, media_clip_id: str, confirm: bool = False) -> dict[str, Any]:
        with self._lock:
            self._require_destructive(confirm)
            state = self._require_project()
            clip = state.clips.pop(media_clip_id, None)
            if clip is None:
                msg = f"media clip {media_clip_id!r} not found"
                raise NotFoundError(msg)
            bin_ = state.bins.get(clip.bin)
            if bin_ is not None:
                state.bins[clip.bin] = bin_.model_copy(
                    update={"clip_ids": [x for x in bin_.clip_ids if x != media_clip_id]}
                )
            self._touch(state)
            return {"deleted": {"id": media_clip_id, "name": clip.name}}

    # ---- helpers -------------------------------------------------------------

    def _require_project(self) -> _ProjectState:
        if self._current_project is None:
            msg = "no project open — call create_project or open_project first"
            raise InvalidStateError(msg)
        return self._projects[self._current_project]

    def _require_timeline(self) -> TimelineState:
        state = self._require_project()
        if state.current_timeline is None:
            msg = "no timeline selected — call create_timeline or set_current_timeline first"
            raise InvalidStateError(msg)
        return state.timelines[state.current_timeline]

    def _store_timeline(self, state: _ProjectState, tl: TimelineState) -> None:
        state.timelines[tl.name] = self._recompute_duration(tl)
        self._touch(state)

    def _touch(self, state: _ProjectState) -> None:
        if not state.info.is_modified:
            state.info = state.info.model_copy(update={"is_modified": True})

    @staticmethod
    def _unique_clip_name(state: _ProjectState, base: str) -> str:
        taken = {c.name for c in state.clips.values()}
        if base not in taken:
            return base
        n = 2
        while f"{base} ({n})" in taken:
            n += 1
        return f"{base} ({n})"

    @staticmethod
    def _quantize(conv: TimeConverter, seconds: float) -> float:
        """Snap a wire value onto the timeline's frame grid, like a real NLE."""
        if seconds < 0:
            msg = f"seconds must be >= 0, got {seconds}"
            raise InvalidStateError(msg)
        return round(conv.frames_to_seconds(conv.seconds_to_frames_round(seconds)), 6)

    @staticmethod
    def _overlapping(
        items: Iterable[TimelineItem], start: float, duration: float
    ) -> TimelineItem | None:
        end = start + duration
        for it in items:
            it_end = it.start_seconds + it.duration_seconds
            if start < it_end - 1e-9 and it.start_seconds < end - 1e-9:
                return it
        return None

    @staticmethod
    def _replace_track(tl: TimelineState, index: int, items: list[TimelineItem]) -> TimelineState:
        return tl.model_copy(
            update={
                "tracks": [
                    t.model_copy(update={"items": items}) if t.index == index else t
                    for t in tl.tracks
                ]
            }
        )

    @staticmethod
    def _get_track(tl: TimelineState, index: int) -> Track:
        for t in tl.tracks:
            if t.index == index:
                return t
        available = ", ".join(f"{t.index}={t.kind.value}" for t in tl.tracks)
        msg = f"track index {index} does not exist on timeline {tl.name!r} (have: {available})"
        raise NotFoundError(msg)

    @staticmethod
    def _recompute_duration(tl: TimelineState) -> TimelineState:
        end = 0.0
        for t in tl.tracks:
            for it in t.items:
                end = max(end, it.start_seconds + it.duration_seconds)
        return tl.model_copy(update={"duration_seconds": round(end, 6)})

    @staticmethod
    def _coerce_frame_rate(frame_rate: FrameRate | dict[str, Any] | float) -> FrameRate:
        if isinstance(frame_rate, FrameRate):
            fr = frame_rate
        elif isinstance(frame_rate, dict):
            fr = FrameRate(**frame_rate)
        elif isinstance(frame_rate, int | float):
            fr = FrameRate(fps=float(frame_rate), drop_frame=False)
        else:
            msg = f"unsupported frame_rate type {type(frame_rate).__name__}"
            raise TypeError(msg)
        try:
            TimeConverter(fr)  # validates fps > 0 and drop-frame/fps compatibility
        except ValueError as exc:
            raise InvalidStateError(str(exc)) from exc
        return fr

    @staticmethod
    def _fits_track_kind(track_kind: TrackKind, media_kind: MediaKind) -> bool:
        if track_kind == TrackKind.VIDEO:
            return media_kind in (MediaKind.VIDEO, MediaKind.IMAGE)
        return media_kind == MediaKind.AUDIO


# --- media probing ----------------------------------------------------------------


@dataclass(frozen=True)
class _Probe:
    duration_seconds: float
    kind: MediaKind
    fps: float | None = None
    width: int | None = None
    height: int | None = None


def _probe_media(path: str) -> _Probe:
    """Read real duration/kind/fps via ffprobe when available.

    resolve-mcp stays dependency-light, so this shells out to ffprobe if it is on
    PATH and otherwise falls back to extension sniffing with duration 0.0
    ("unknown"), which callers must treat as "don't know", never as "empty".
    """
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return _Probe(duration_seconds=0.0, kind=_kind_from_extension(path))
    try:
        proc = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-show_entries",
                "stream=codec_type,avg_frame_rate,width,height",
                "-of",
                "json",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
        if proc.returncode != 0:
            return _Probe(duration_seconds=0.0, kind=_kind_from_extension(path))
        data = json.loads(proc.stdout or "{}")
    except (OSError, ValueError, subprocess.SubprocessError):
        return _Probe(duration_seconds=0.0, kind=_kind_from_extension(path))

    streams = data.get("streams") or []
    try:
        duration = float((data.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    kind = _kind_from_extension(path)
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    if video is not None:
        kind = MediaKind.IMAGE if duration == 0.0 and audio is None else MediaKind.VIDEO
        fps = _parse_fraction(video.get("avg_frame_rate"))
        width = _as_int(video.get("width"))
        height = _as_int(video.get("height"))
    elif audio is not None:
        kind = MediaKind.AUDIO
    return _Probe(
        duration_seconds=max(0.0, duration), kind=kind, fps=fps, width=width, height=height
    )


def _kind_from_extension(path: str) -> MediaKind:
    lower = path.lower()
    if lower.endswith((".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".aif", ".aiff")):
        return MediaKind.AUDIO
    if lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")):
        return MediaKind.IMAGE
    return MediaKind.VIDEO


def _parse_fraction(value: Any) -> float | None:
    if not isinstance(value, str) or "/" not in value:
        return None
    num, _, den = value.partition("/")
    try:
        n, d = float(num), float(den)
    except ValueError:
        return None
    return round(n / d, 6) if d else None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _delta(before: TimelineState, after: TimelineState, *, changed_path: str) -> StateDelta:
    return StateDelta(
        before=before.model_dump(mode="json"),
        after=after.model_dump(mode="json"),
        changed_paths=[changed_path],
    )


__all__ = ["FakeResolveBackend"]
