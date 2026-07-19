"""In-memory fake backend used by every unit test in this package.

The shape of the data (projects, bins, media clips, timelines, tracks, items,
transforms, fades, markers, transitions, render jobs) mirrors what the real
``DaVinciResolveBackend`` returns — the two impls are intentionally kept small enough
that ``Protocol`` conformance is checked in tests, never by inheritance.

All mutations are single-threaded and locked behind an ``RLock`` so the server can
potentially spawn worker tasks without corrupting state in later phases.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Iterable
from copy import deepcopy
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
    Track,
    TrackKind,
    Transform,
    Transition,
    TransitionAlignment,
    TransitionStyle,
)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class FakeResolveBackend:
    """A complete, deterministic, in-memory Resolve backend."""

    def __init__(self, *, allow_destructive: bool = False) -> None:
        self._lock = threading.RLock()
        self._projects: dict[str, ProjectInfo] = {}
        self._current_project: str | None = None
        self._media_clips: dict[str, MediaClip] = {}
        self._bins: dict[str, Bin] = {}
        self._timelines: dict[str, TimelineState] = {}
        self._current_timeline: str | None = None
        self._render_jobs: dict[str, RenderJob] = {}
        self._allow_destructive: bool = allow_destructive

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
            self._projects[name] = info
            self._current_project = name
            self._bins.setdefault("Master", Bin(name="Master", clip_ids=[]))
            return info

    def open_project(self, name: str) -> ProjectInfo:
        with self._lock:
            info = self._projects.get(name)
            if info is None:
                msg = f"project {name!r} not found"
                raise NotFoundError(msg)
            self._current_project = name
            self._current_timeline = None
            return info

    def save_project(self) -> ProjectInfo:
        with self._lock:
            info = self._require_current_project()
            self._projects[info.name] = info.model_copy(update={"is_modified": False})
            return self._projects[info.name]

    def current_project(self) -> ProjectInfo:
        with self._lock:
            return self._require_current_project()

    # ---- media pool ----------------------------------------------------------

    def list_media_pool(self) -> MediaPoolState:
        with self._lock:
            self._require_current_project()
            return MediaPoolState(
                bins=sorted(self._bins.values(), key=lambda b: b.name),
                clips=sorted(self._media_clips.values(), key=lambda c: c.name),
            )

    def create_bin(self, name: str) -> Bin:
        with self._lock:
            self._require_current_project()
            if name in self._bins:
                msg = f"bin {name!r} already exists"
                raise AlreadyExistsError(msg)
            bin = Bin(name=name, clip_ids=[])
            self._bins[name] = bin
            return bin

    def import_media(self, paths: Iterable[str], bin: str | None = None) -> list[MediaClip]:
        with self._lock:
            self._require_current_project()
            bin_name = bin or "Master"
            if bin_name not in self._bins:
                msg = f"bin {bin_name!r} does not exist"
                raise NotFoundError(msg)
            out: list[MediaClip] = []
            existing_names = {c.name for c in self._media_clips.values()}
            for path in paths:
                filename = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                base, _, _ = filename.rpartition(".")
                name = base or filename
                # Disambiguate duplicates within the pool.
                if name in existing_names:
                    name = f"{name}_{len(existing_names) + 1}"
                existing_names.add(name)
                kind = self._infer_kind(path)
                clip = MediaClip(
                    id=_new_id("clip"),
                    bin=bin_name,
                    name=name,
                    path=path,
                    kind=kind,
                    duration_seconds=10.0,  # unknown without probing; tests may override
                )
                self._media_clips[clip.id] = clip
                self._bins[bin_name].clip_ids.append(clip.id)
                out.append(clip)
            return out

    # ---- timeline ------------------------------------------------------------

    def create_timeline(self, name: str, frame_rate: FrameRate | dict[str, Any] | float) -> TimelineState:
        with self._lock:
            self._require_current_project()
            fr = self._coerce_frame_rate(frame_rate)
            if name in self._timelines:
                msg = f"timeline {name!r} already exists"
                raise AlreadyExistsError(msg)
            tl = TimelineState(
                name=name,
                frame_rate=fr,
                duration_seconds=0.0,
                tracks=[
                    Track(index=0, kind=TrackKind.VIDEO, items=[]),
                    Track(index=1, kind=TrackKind.AUDIO, items=[]),
                ],
            )
            self._timelines[name] = tl
            self._current_timeline = name
            return tl

    def get_timeline_state(self) -> TimelineState:
        with self._lock:
            return self._require_current_timeline()

    def append_clip(
        self,
        media_clip_id: str,
        timeline_track_index: int,
        start_seconds: float,
        duration_seconds: float,
        source_in_seconds: float = 0.0,
    ) -> StateDelta:
        with self._lock:
            tl = self._require_current_timeline()
            media = self._media_clips.get(media_clip_id)
            if media is None:
                msg = f"media clip {media_clip_id!r} not found"
                raise NotFoundError(msg)
            track = self._get_track(tl, timeline_track_index)
            before = deepcopy(tl)
            if not self._fits_track_kind(track.kind, media.kind):
                msg = f"media kind {media.kind} cannot live on {track.kind} track {track.index}"
                raise InvalidStateError(msg)
            item = TimelineItem(
                id=_new_id("item"),
                track_index=track.index,
                media_clip_id=media.id,
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
                source_in_seconds=source_in_seconds,
                source_out_seconds=source_in_seconds + duration_seconds,
                transform=Transform(),
                crop=Crop(),
                composite_mode=CompositeMode.NORMAL,
                opacity=1.0,
            )
            new_items = sorted((*track.items, item), key=lambda it: it.start_seconds)
            track_replaced = track.model_copy(update={"items": new_items})
            self._timelines[tl.name] = tl.model_copy(
                update={"tracks": [t if t.index != track.index else track_replaced for t in tl.tracks]}
            )
            tl = self._recompute_duration(self._timelines[tl.name])
            self._timelines[tl.name] = tl
            return _delta(before, tl, changed_path=f"timelines.{tl.name}.tracks[{track.index}].items")

    def insert_clip(
        self,
        media_clip_id: str,
        timeline_track_index: int,
        timeline_position_seconds: float,
        duration_seconds: float,
        source_in_seconds: float = 0.0,
    ) -> StateDelta:
        with self._lock:
            tl = self._require_current_timeline()
            media = self._media_clips.get(media_clip_id)
            if media is None:
                msg = f"media clip {media_clip_id!r} not found"
                raise NotFoundError(msg)
            track = self._get_track(tl, timeline_track_index)
            before = deepcopy(tl)
            if not self._fits_track_kind(track.kind, media.kind):
                msg = f"media kind {media.kind} cannot live on {track.kind} track {track.index}"
                raise InvalidStateError(msg)
            new_item = TimelineItem(
                id=_new_id("item"),
                track_index=track.index,
                media_clip_id=media.id,
                start_seconds=timeline_position_seconds,
                duration_seconds=duration_seconds,
                source_in_seconds=source_in_seconds,
                source_out_seconds=source_in_seconds + duration_seconds,
            )
            shifted_items = [
                it.model_copy(update={"start_seconds": it.start_seconds + duration_seconds})
                if it.start_seconds >= timeline_position_seconds
                else it
                for it in track.items
            ]
            new_items = sorted((*shifted_items, new_item), key=lambda it: it.start_seconds)
            track_replaced = track.model_copy(update={"items": new_items})
            self._timelines[tl.name] = tl.model_copy(
                update={"tracks": [t if t.index != track.index else track_replaced for t in tl.tracks]}
            )
            tl = self._recompute_duration(self._timelines[tl.name])
            self._timelines[tl.name] = tl
            return _delta(before, tl, changed_path=f"timelines.{tl.name}.tracks[{track.index}].items")

    def delete_clip(self, timeline_item_id: str) -> StateDelta:
        with self._lock:
            tl = self._require_current_timeline()
            before = deepcopy(tl)
            removed_path = ""
            new_tracks: list[Track] = []
            for track in tl.tracks:
                kept = [it for it in track.items if it.id != timeline_item_id]
                if len(kept) != len(track.items):
                    removed_path = f"timelines.{tl.name}.tracks[{track.index}].items"
                new_track = track.model_copy(update={"items": kept}) if kept is not track.items else track
                new_tracks.append(new_track)
            if not removed_path:
                msg = f"timeline item {timeline_item_id!r} not found"
                raise NotFoundError(msg)
            new_tl = tl.model_copy(update={"tracks": new_tracks})
            new_tl = self._recompute_duration(new_tl)
            self._timelines[tl.name] = new_tl
            return _delta(before, new_tl, changed_path=removed_path)

    def move_clip(self, timeline_item_id: str, new_position_seconds: float) -> StateDelta:
        with self._lock:
            tl = self._require_current_timeline()
            before = deepcopy(tl)
            moved_path = ""
            new_tracks: list[Track] = []
            for track in tl.tracks:
                items: list[TimelineItem] = []
                moved_here = False
                for it in track.items:
                    if it.id == timeline_item_id:
                        items.append(it.model_copy(update={"start_seconds": new_position_seconds}))
                        moved_here = True
                    else:
                        items.append(it)
                if moved_here:
                    moved_path = f"timelines.{tl.name}.tracks[{track.index}].items"
                    items.sort(key=lambda i2: i2.start_seconds)
                    new_tracks.append(track.model_copy(update={"items": items}))
                else:
                    new_tracks.append(track)
            if not moved_path:
                msg = f"timeline item {timeline_item_id!r} not found"
                raise NotFoundError(msg)
            new_tl = tl.model_copy(update={"tracks": new_tracks})
            new_tl = self._recompute_duration(new_tl)
            self._timelines[tl.name] = new_tl
            return _delta(before, new_tl, changed_path=moved_path)

    # ---- Phase 2: per-item mutations ----------------------------------------

    def _update_item(
        self,
        timeline_item_id: str,
        updater: Callable[[TimelineItem], TimelineItem],
        path: str,
    ) -> StateDelta:
        with self._lock:
            tl = self._require_current_timeline()
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
                msg = f"timeline item {timeline_item_id!r} not found"
                raise NotFoundError(msg)
            new_tl = tl.model_copy(update={"tracks": new_tracks})
            self._timelines[tl.name] = new_tl
            return _delta(before, new_tl, changed_path=path)

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
        if isinstance(mode, str):
            mode = CompositeMode(mode)
        return self._update_item(
            timeline_item_id,
            updater=lambda it: it.model_copy(update={"composite_mode": mode}),
            path=f"timelines.*.tracks.*.items[{timeline_item_id}].composite_mode",
        )

    def set_opacity(self, timeline_item_id: str, opacity: float) -> StateDelta:
        if not 0.0 <= opacity <= 1.0:
            msg = f"opacity must be in [0, 1], got {opacity}"
            raise ValueError(msg)
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
            raise ValueError(msg)

        def _apply(it: TimelineItem) -> TimelineItem:
            cap = it.duration_seconds
            if fade_in_seconds + fade_out_seconds > cap:
                msg = (
                    f"fade durations {fade_in_seconds} + {fade_out_seconds} exceed "
                    f"item duration {cap}; clamp in the caller."
                )
                raise ValueError(msg)
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
            msg = "speed must be > 0"
            raise ValueError(msg)
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
        if isinstance(color, str):
            color = MarkerColor(color)

        def _apply(it: TimelineItem) -> TimelineItem:
            if position_seconds > it.duration_seconds:
                msg = "marker position exceeds item duration"
                raise ValueError(msg)
            new_marker = Marker(
                id=_new_id("marker"),
                timeline_item_id=it.id,
                position_seconds=position_seconds,
                label=label,
                color=color,
                note=note,
            )
            return it.model_copy(update={"markers": [*it.markers, new_marker]})

        return self._update_item(
            timeline_item_id,
            updater=_apply,
            path=f"timelines.*.tracks.*.items[{timeline_item_id}].markers",
        )

    # ---- Phase 2: transitions & render --------------------------------------

    def add_transition(
        self,
        timeline_item_id: str,
        track_index: int,
        style: TransitionStyle | str,
        duration_seconds: float,
        alignment: TransitionAlignment | str,
    ) -> StateDelta:
        if isinstance(style, str):
            style = TransitionStyle(style)
        if isinstance(alignment, str):
            alignment = TransitionAlignment(alignment)
        if duration_seconds < 0:
            msg = "transition duration must be non-negative"
            raise ValueError(msg)
        with self._lock:
            tl = self._require_current_timeline()
            before = deepcopy(tl)
            # Verify the item exists on the requested track.
            target_track_index: int | None = None
            for track in tl.tracks:
                if track.index == track_index:
                    for it in track.items:
                        if it.id == timeline_item_id:
                            target_track_index = track.index
                            break
            if target_track_index is None:
                msg = f"timeline item {timeline_item_id!r} not found on track {track_index}"
                raise NotFoundError(msg)
            new_transition = Transition(
                id=_new_id("trans"),
                timeline_item_id=timeline_item_id,
                track_index=target_track_index,
                style=style,
                duration_seconds=duration_seconds,
                alignment=alignment,
            )
            new_tl = tl.model_copy(
                update={"transitions": [*tl.transitions, new_transition]}
            )
            self._timelines[tl.name] = new_tl
            return _delta(
                before,
                new_tl,
                changed_path=f"timelines.{tl.name}.transitions",
            )

    def add_render_job(
        self,
        timeline_name: str,
        format: RenderJobFormat | str,
        output_path: str,
    ) -> RenderJob:
        with self._lock:
            if timeline_name not in self._timelines:
                msg = f"timeline {timeline_name!r} not found"
                raise NotFoundError(msg)
            if isinstance(format, str):
                format = RenderJobFormat(format)
            job = RenderJob(
                id=_new_id("render"),
                timeline_name=timeline_name,
                format=format,
                output_path=output_path,
                status=RenderJobStatus.QUEUED,
            )
            self._render_jobs[job.id] = job
            return job

    def start_render(self, job_id: str) -> RenderJob:
        with self._lock:
            job = self._render_jobs.get(job_id)
            if job is None:
                msg = f"render job {job_id!r} not found"
                raise NotFoundError(msg)
            if job.status not in (RenderJobStatus.QUEUED, RenderJobStatus.FAILED):
                msg = f"render job {job_id!r} is {job.status.value}; cannot start"
                raise InvalidStateError(msg)
            updated = job.model_copy(
                update={
                    "status": RenderJobStatus.RUNNING,
                    "progress": 0.0,
                }
            )
            self._render_jobs[job_id] = updated
            return updated

    def get_render_status(self, job_id: str) -> RenderJob:
        with self._lock:
            job = self._render_jobs.get(job_id)
            if job is None:
                msg = f"render job {job_id!r} not found"
                raise NotFoundError(msg)
            return job

    # ---- Phase 2: destructive (gated) ---------------------------------------

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
            # Pretend to quit: project becomes un-set.
            self._current_project = None
            self._current_timeline = None
            return {"quit": True, "project_after": None}

    def restart_app(self, confirm: bool = False) -> dict[str, Any]:
        with self._lock:
            self._require_destructive(confirm)
            self._current_project = None
            self._current_timeline = None
            return {"restart": True, "project_after": None}

    def delete_timeline(self, name: str, confirm: bool = False) -> StateDelta:
        with self._lock:
            self._require_destructive(confirm)
            if name not in self._timelines:
                msg = f"timeline {name!r} not found"
                raise NotFoundError(msg)
            before = self._timelines[name]
            del self._timelines[name]
            if self._current_timeline == name:
                self._current_timeline = None
            after_state = TimelineState(
                name=name,
                frame_rate=before.frame_rate,
                duration_seconds=0.0,
                tracks=[
                    Track(index=0, kind=TrackKind.VIDEO, items=[]),
                    Track(index=1, kind=TrackKind.AUDIO, items=[]),
                ],
            )
            return _delta(before, after_state, changed_path=f"timelines.{name}")

    def delete_media(self, media_clip_id: str, confirm: bool = False) -> dict[str, Any]:
        with self._lock:
            self._require_destructive(confirm)
            clip = self._media_clips.pop(media_clip_id, None)
            if clip is None:
                msg = f"media clip {media_clip_id!r} not found"
                raise NotFoundError(msg)
            bin = self._bins.get(clip.bin)
            if bin is not None:
                self._bins[clip.bin] = bin.model_copy(
                    update={"clip_ids": [x for x in bin.clip_ids if x != media_clip_id]}
                )
            return {"deleted": {"id": media_clip_id, "name": clip.name}}

    # ---- helpers -------------------------------------------------------------

    def _require_current_project(self) -> ProjectInfo:
        if self._current_project is None:
            msg = "no project open"
            raise InvalidStateError(msg)
        info = self._projects[self._current_project]
        return info

    def _require_current_timeline(self) -> TimelineState:
        if self._current_timeline is None:
            msg = "no timeline selected"
            raise InvalidStateError(msg)
        return self._timelines[self._current_timeline]

    def _get_track(self, tl: TimelineState, index: int) -> Track:
        for t in tl.tracks:
            if t.index == index:
                return t
        msg = f"track index {index} does not exist on timeline {tl.name!r}"
        raise NotFoundError(msg)

    def _recompute_duration(self, tl: TimelineState) -> TimelineState:
        end = 0.0
        for t in tl.tracks:
            for it in t.items:
                end = max(end, it.start_seconds + it.duration_seconds)
        return tl.model_copy(update={"duration_seconds": end})

    def _coerce_frame_rate(self, frame_rate: FrameRate | dict[str, Any] | float) -> FrameRate:
        if isinstance(frame_rate, FrameRate):
            return frame_rate
        if isinstance(frame_rate, dict):
            return FrameRate(**frame_rate)
        if isinstance(frame_rate, int | float):
            return FrameRate(fps=float(frame_rate), drop_frame=False)
        msg = f"unsupported frame_rate type {type(frame_rate).__name__}"
        raise TypeError(msg)

    @staticmethod
    def _fits_track_kind(track_kind: TrackKind, media_kind: MediaKind) -> bool:
        if track_kind == TrackKind.VIDEO:
            return media_kind in (MediaKind.VIDEO, MediaKind.IMAGE)
        return media_kind == MediaKind.AUDIO

    @staticmethod
    def _infer_kind(path: str) -> MediaKind:
        lower = path.lower()
        for ext in (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"):
            if lower.endswith(ext):
                return MediaKind.AUDIO
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"):
            if lower.endswith(ext):
                return MediaKind.IMAGE
        return MediaKind.VIDEO


def _delta(before: TimelineState, after: TimelineState, *, changed_path: str) -> StateDelta:
    return StateDelta(
        before=before.model_dump(mode="json"),
        after=after.model_dump(mode="json"),
        changed_paths=[changed_path],
    )


__all__ = ["FakeResolveBackend"]
