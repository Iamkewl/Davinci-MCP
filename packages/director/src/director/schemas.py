"""Pydantic schemas for the director pipeline.

These are the wire-types between the Orchestrator and agents (Contextualizer →
Planner ↔ Director → Editor) and the on-disk run store. Keeping a strict,
versioned model tree lets the editor safely verify acts against the plan.
"""

from __future__ import annotations

import datetime
import enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat


class StrictModel(BaseModel):
    """Base model with no extra fields and frozen by default."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)


# --- Ingestion: per-clip map --------------------------------------------------


class ClipMood(enum.StrEnum):  # placeholder — not used yet but reserved for v2
    UPCALM = "upcalm"
    UNKNOWN = "unknown"


class PerClipMap(StrictModel):
    """Output of the Contextualizer for one source clip."""

    clip_id: str
    source_path: str
    duration_seconds: NonNegativeFloat = 0.0
    fps: float | None = None
    # Vision-derived fields
    visual_summary: str = ""
    dominant_shot_type: str = ""
    key_moments: list[KeyMoment] = Field(default_factory=list)
    # Audio-derived fields (filled only when the source has audio)
    tempo_bpm: Annotated[float, Field(gt=0.0)] | None = None
    beat_frames: list[NonNegativeFloat] = Field(default_factory=list)
    onset_strength: list[NonNegativeFloat] = Field(default_factory=list)
    has_audio: bool = False


class KeyMoment(StrictModel):
    """A point in a clip that the orchestrator should consider (cut, peak, etc.)."""

    position_seconds: NonNegativeFloat
    kind: Annotated[str, Field(pattern=r"^[a-z_]+$")] = "highlight"
    description: str = ""


# --- Plan: beat-synced timeline plan -----------------------------------------


class PlanOpKind(enum.StrEnum):
    """The verb of a single timeline-act in a plan."""

    APPEND_CLIP = "append_clip"
    INSERT_CLIP = "insert_clip"
    MOVE_CLIP = "move_clip"
    DELETE_CLIP = "delete_clip"
    SET_TRANSFORM = "set_transform"
    SET_CROP = "set_crop"
    SET_OPACITY = "set_opacity"
    SET_COMPOSITE_MODE = "set_composite_mode"
    ADD_FADE = "add_fade"
    SET_SPEED = "set_speed"
    ADD_MARKER = "add_marker"
    ADD_TRANSITION = "add_transition"


class PlanOp(StrictModel):
    """A single ordered write in a plan. Stable id is only for traceability."""

    id: str
    kind: PlanOpKind
    # Tool-call arguments (already deserialized into JSON for MCP dispatch).
    args: dict[str, Any] = Field(default_factory=dict)
    # Human-written rationale for the act (filled by Planner; aids the Director).
    rationale: str = ""


class Plan(StrictModel):
    """A plan is an ordered list of :class:`PlanOp` entries with a target."""

    plan_id: str
    version: int = 1
    target_timeline: str
    target_project: str
    ops: list[PlanOp] = Field(default_factory=list)
    summary: str = ""
    created_at: str = Field(default_factory=lambda: _now_iso())


def _now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")


# --- Director verdict ---------------------------------------------------------


class DirectorAxisScore(StrictModel):
    """A single axis score from the Director agent."""

    name: Annotated[str, Field(pattern=r"^[a-z_]+$")]
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    rationale: str = ""


class DirectorVerdict(enum.StrEnum):
    APPROVED = "APPROVED"
    ACCEPTED_WITH_WARNINGS = "ACCEPTED_WITH_WARNINGS"
    FAILED = "FAILED"


class DirectorEvaluation(StrictModel):
    """Structured output from the Director agent."""

    verdict: DirectorVerdict
    overall: Annotated[float, Field(ge=0.0, le=1.0)]
    axes: list[DirectorAxisScore] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: _now_iso())


# --- Run store ----------------------------------------------------------------


class RunMode(enum.StrEnum):
    AUTO = "auto"
    INTERACTIVE = "interactive"


class RunStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_REVIEW = "awaiting_review"
    COMPLETED_APPROVED = "completed_approved"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"


class RunRecord(StrictModel):
    """Top-level record for a single orchestrator run."""

    run_id: str
    mode: RunMode = RunMode.AUTO
    input_clips: list[str] = Field(default_factory=list)
    input_music: str | None = None
    user_prompt: str = ""
    status: RunStatus = RunStatus.PENDING
    iterations: int = 0
    last_verdict: DirectorVerdict | None = None
    final_plan_id: str | None = None
    created_at: str = Field(default_factory=lambda: _now_iso())
    updated_at: str = Field(default_factory=lambda: _now_iso())


class ToolCallRecord(StrictModel):
    """One MCP-tool call dispatched by the Editor."""

    run_id: str
    iteration: int
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    error: str | None = None
    recorded_at: str = Field(default_factory=lambda: _now_iso())


# --- Event log ----------------------------------------------------------------


class EventKind(enum.StrEnum):
    PLAN_COMPILED = "plan_compiled"
    DIRECTOR_VERDICT = "director_verdict"
    TOOL_CALLED = "tool_called"
    TOOL_OBSERVED = "tool_observed"
    PLAN_APPLIED = "plan_applied"
    CHECKPOINT = "checkpoint"
    ERROR = "error"


class OrchestratorEvent(StrictModel):
    """Single entry in the append-only JSONL event log."""

    run_id: str
    iteration: int
    kind: EventKind
    payload: dict[str, Any]
    recorded_at: str = Field(default_factory=lambda: _now_iso())


# Late-binding: forward refs
PerClipMap.model_rebuild()
