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

TrackIndex = Annotated[int, Field(ge=1, description="1-based track index (Resolve V1/A1 semantics).")]



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
    """Resolve item transform, in Resolve's own Inspector units.

    ``pan_*`` / ``anchor_*`` are pixel offsets from the frame centre (Resolve
    ``Pan``/``Tilt``/``AnchorPointX``/``AnchorPointY``), ``zoom_*`` is a scale
    multiplier where 1.0 = 100% (``ZoomX``/``ZoomY``), ``rotation`` is degrees
    (``RotationAngle``).
    """

    pan_x: float = 0.0
    pan_y: float = 0.0
    zoom_x: Annotated[float, Field(gt=0.0, le=100.0)] = 1.0
    zoom_y: Annotated[float, Field(gt=0.0, le=100.0)] = 1.0
    rotation: Annotated[float, Field(ge=-360.0, le=360.0)] = 0.0
    anchor_x: float = 0.0
    anchor_y: float = 0.0


class Crop(StrictModel):
    """Crop in pixels removed from each edge (Resolve ``CropLeft``/``CropRight``/...)."""

    left: NonNegativeFloat = 0.0
    right: NonNegativeFloat = 0.0
    top: NonNegativeFloat = 0.0
    bottom: NonNegativeFloat = 0.0


class CompositeMode(enum.StrEnum):
    """Blend modes; each maps to a documented ``resolve.COMPOSITE_*`` constant."""

    NORMAL = "normal"
    ADD = "add"
    SUBTRACT = "subtract"
    DIFFERENCE = "difference"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    HARD_LIGHT = "hard_light"
    SOFT_LIGHT = "soft_light"
    DARKEN = "darken"
    LIGHTEN = "lighten"
    COLOR_DODGE = "color_dodge"
    COLOR_BURN = "color_burn"
    EXCLUSION = "exclusion"
    LINEAR_DODGE = "linear_dodge"
    LINEAR_BURN = "linear_burn"
    LINEAR_LIGHT = "linear_light"
    VIVID_LIGHT = "vivid_light"
    PIN_LIGHT = "pin_light"
    HARD_MIX = "hard_mix"
    LIGHTER_COLOR = "lighter_color"
    DARKER_COLOR = "darker_color"
    HUE = "hue"
    SATURATION = "saturation"
    COLOR = "color"
    LUMINOSITY = "luminosity"


#: CompositeMode -> name of the documented Resolve constant (``getattr(resolve, name)``).
COMPOSITE_CONSTANT_NAMES: dict[CompositeMode, str] = {
    CompositeMode.NORMAL: "COMPOSITE_NORMAL",
    CompositeMode.ADD: "COMPOSITE_ADD",
    CompositeMode.SUBTRACT: "COMPOSITE_SUBTRACT",
    CompositeMode.DIFFERENCE: "COMPOSITE_DIFF",
    CompositeMode.MULTIPLY: "COMPOSITE_MULTIPLY",
    CompositeMode.SCREEN: "COMPOSITE_SCREEN",
    CompositeMode.OVERLAY: "COMPOSITE_OVERLAY",
    CompositeMode.HARD_LIGHT: "COMPOSITE_HARDLIGHT",
    CompositeMode.SOFT_LIGHT: "COMPOSITE_SOFTLIGHT",
    CompositeMode.DARKEN: "COMPOSITE_DARKEN",
    CompositeMode.LIGHTEN: "COMPOSITE_LIGHTEN",
    CompositeMode.COLOR_DODGE: "COMPOSITE_COLOR_DODGE",
    CompositeMode.COLOR_BURN: "COMPOSITE_COLOR_BURN",
    CompositeMode.EXCLUSION: "COMPOSITE_EXCLUSION",
    CompositeMode.LINEAR_DODGE: "COMPOSITE_LINEAR_DODGE",
    CompositeMode.LINEAR_BURN: "COMPOSITE_LINEAR_BURN",
    CompositeMode.LINEAR_LIGHT: "COMPOSITE_LINEAR_LIGHT",
    CompositeMode.VIVID_LIGHT: "COMPOSITE_VIVID_LIGHT",
    CompositeMode.PIN_LIGHT: "COMPOSITE_PIN_LIGHT",
    CompositeMode.HARD_MIX: "COMPOSITE_HARD_MIX",
    CompositeMode.LIGHTER_COLOR: "COMPOSITE_LIGHTER_COLOR",
    CompositeMode.DARKER_COLOR: "COMPOSITE_DARKER_COLOR",
    CompositeMode.HUE: "COMPOSITE_HUE",
    CompositeMode.SATURATION: "COMPOSITE_SATURATE",
    CompositeMode.COLOR: "COMPOSITE_COLORIZE",
    CompositeMode.LUMINOSITY: "COMPOSITE_LUM",
}


class MarkerColor(enum.StrEnum):
    """Resolve's 16 marker colours (passed to Resolve Capitalized, e.g. ``"Blue"``)."""

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
    COCOA = "cocoa"
    CREAM = "cream"


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
    track_index: TrackIndex
    style: TransitionStyle = TransitionStyle.CROSS_DISSOLVE
    duration_seconds: NonNegativeFloat
    alignment: TransitionAlignment = TransitionAlignment.MID


class TimelineItem(StrictModel):
    id: str
    track_index: TrackIndex
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
    index: TrackIndex
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
    #: Old timeline-item id -> new id for items the backend had to recreate
    #: (the live backend's move/ripple-insert). Empty when ids are stable.
    id_remap: dict[str, str] = Field(default_factory=dict)


# --- Timeline listing -----------------------------------------------------------


class TimelineSummary(StrictModel):
    """One entry of ``list_timelines``."""

    name: str
    is_current: bool = False
