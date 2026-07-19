"""Pydantic schemas shared across the resolve-mcp package.

These are the wire-types used both by FastMCP (auto-generated JSON schemas) and by the
backend implementations (Fake and DaVinci). Keeping a single canonical model tree
prevents the fake and the real backend from drifting apart.
"""

from __future__ import annotations

import datetime
import enum
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
)


class StrictModel(BaseModel):
    """Base model with strict validation: no silent coercion, no extra fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)


# --- Time -----------------------------------------------------------------------


class FrameRate(StrictModel):
    """Frames-per-second for a project. Drop-frame handled separately."""

    fps: NonNegativeFloat = Field(..., description="Frames per second (e.g., 23.976, 29.97, 30, 60).")
    drop_frame: bool = Field(False, description="True for SMPTE drop-frame timecode at 29.97/59.94.")


class Timecode(StrictModel):
    """A SMPTE-style timecode string (e.g., '00:00:12:15', '00:01:00;02')."""

    value: Annotated[str, Field(pattern=r"^\d{2,}:\d{2}:\d{2}[:;]\d{2}$")]


# --- Project --------------------------------------------------------------------


class ProjectInfo(StrictModel):
    name: str
    frame_rate: FrameRate
    resolution_width: NonNegativeInt
    resolution_height: NonNegativeInt
    path: str | None = None
    is_modified: bool = False


# --- Media pool -----------------------------------------------------------------


class MediaKind(enum.StrEnum):
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"


class MediaClip(StrictModel):
    id: str = Field(..., description="Stable identifier assigned by the backend.")
    bin: str
    name: str
    path: str
    kind: MediaKind
    duration_seconds: NonNegativeFloat
    frame_rate: FrameRate | None = None
    resolution_width: NonNegativeInt | None = None
    resolution_height: NonNegativeInt | None = None


class Bin(StrictModel):
    name: str
    clip_ids: list[str]


class MediaPoolState(StrictModel):
    bins: list[Bin]
    clips: list[MediaClip]


# --- Timeline / items -----------------------------------------------------------


class TrackKind(enum.StrEnum):
    VIDEO = "video"
    AUDIO = "audio"


class Transform(StrictModel):
    """Resolve item transform: pan + zoom. Stored in normalized units where applicable."""

    pan_x: float = 0.0
    pan_y: float = 0.0
    zoom_x: float = 1.0
    zoom_y: float = 1.0
    rotation: float = 0.0
    anchor_x: float = 0.5
    anchor_y: float = 0.5


class Crop(StrictModel):
    left: float = 0.0
    right: float = 0.0
    top: float = 0.0
    bottom: float = 0.0


class CompositeMode(enum.StrEnum):
    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    SOFT_LIGHT = "soft_light"
    HARD_LIGHT = "hard_light"


class MarkerColor(enum.StrEnum):
    BLUE = "blue"
    CYAN = "cyan"
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    PINK = "pink"
    PURPLE = "purple"
    FUCHSIA = "fuchsia"
    ROSE = "rose"
    LAVENDER = "lavender"
    SKY = "sky"
    MINT = "mint"
    LEMON = "lemon"
    SAND = "sand"
    TAWNY = "tawny"
    COCOA = "cocoa"


class Marker(StrictModel):
    """A point marker on a timeline item."""

    id: str
    timeline_item_id: str
    position_seconds: NonNegativeFloat
    label: str = ""
    color: MarkerColor = MarkerColor.BLUE
    note: str = ""


class TransitionStyle(enum.StrEnum):
    CROSS_DISSOLVE = "cross_dissolve"
    DIP_TO_BLACK = "dip_to_black"
    DIP_TO_WHITE = "dip_to_white"
    PUSH_LEFT = "push_left"
    PUSH_RIGHT = "push_right"
    WIPE_LEFT = "wipe_left"
    WIPE_RIGHT = "wipe_right"


class TransitionAlignment(enum.StrEnum):
    """Where the transition sits on the item boundary."""

    START = "start"
    END = "end"
    MID = "mid"


class Transition(StrictModel):
    """A cross-clip transition attached to an item."""

    id: str
    timeline_item_id: str
    track_index: NonNegativeInt
    style: TransitionStyle = TransitionStyle.CROSS_DISSOLVE
    duration_seconds: NonNegativeFloat
    alignment: TransitionAlignment = TransitionAlignment.MID


class TimelineItem(StrictModel):
    id: str
    track_index: NonNegativeInt
    media_clip_id: str
    start_seconds: NonNegativeFloat
    duration_seconds: NonNegativeFloat
    source_in_seconds: NonNegativeFloat
    source_out_seconds: NonNegativeFloat = Field(..., ge=0)
    transform: Transform = Field(default_factory=Transform)
    crop: Crop = Field(default_factory=Crop)
    composite_mode: CompositeMode = CompositeMode.NORMAL
    opacity: float = 1.0
    speed: float = 1.0
    fade_in_seconds: NonNegativeFloat = 0.0
    fade_out_seconds: NonNegativeFloat = 0.0
    markers: list[Marker] = Field(default_factory=list)


class Track(StrictModel):
    index: NonNegativeInt
    kind: TrackKind
    items: list[TimelineItem]


class TimelineState(StrictModel):
    name: str
    frame_rate: FrameRate
    duration_seconds: NonNegativeFloat
    tracks: list[Track]
    transitions: list[Transition] = Field(default_factory=list)


# --- Render jobs ----------------------------------------------------------------


class RenderJobFormat(enum.StrEnum):
    MP4 = "mp4"
    MOV = "mov"
    DNXHR = "dnxhr"
    PRORES = "prores"


class RenderJobStatus(enum.StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RenderJob(StrictModel):
    id: str
    timeline_name: str
    format: RenderJobFormat = RenderJobFormat.MP4
    output_path: str
    status: RenderJobStatus = RenderJobStatus.QUEUED
    progress: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    submitted_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")
    )


# --- State delta ----------------------------------------------------------------


class StateDelta(StrictModel):
    """A merged snapshot returned by every mutating tool.

    Includes before/after of the affected slice so the caller can verify the change
    actually landed (the key invariant that was missing in the prior build).
    """

    before: dict[str, Any]
    after: dict[str, Any]
    changed_paths: list[str]
