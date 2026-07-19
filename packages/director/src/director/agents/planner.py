"""Planner: produces a structured, schema-validated beat-synced timeline plan.

The planner uses ``GeminiClient.generate_json`` with the :class:`Plan` schema.
Inside tests we hand-write a small "structured" planner that returns a fixed
plan, so the orchestrator tests don't require network access.

This module deliberately does NOT swallow LLM misbehavior: if the response is not
schema-valid, the planner raises ``InvalidModelOutput`` — the orchestrator
treats that like a Director-FAILED verdict and stops iterating.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from ..ingestion.gemini_client import GeminiClient, GeminiError
from ..schemas import PerClipMap, Plan, PlanOp, PlanOpKind
from ..settings import DirectorSettings
from .base import Agent, InvalidModelOutput


@dataclass
class PlannerRequest:
    """Inputs collected by the orchestrator before calling the planner."""

    user_prompt: str
    per_clip: list[PerClipMap]
    target_project: str
    target_timeline: str
    target_fps: float
    music_bpm: float | None = None
    beat_times: list[float] | None = None
    music_duration_seconds: float | None = None


class Planner(Agent[Plan]):
    """Produce an actionable plan, parameterized by a beat map."""

    SYSTEM = (
        "You are a deterministic edit planner. Given a user's brief, a list of "
        "video clips (with visual summaries), and a beat timeline for a music "
        "track, output a JSON Plan: each op describes one resolve-mcp tool call. "
        "Always favor: append_clip to place clips at beat_times, add_fade for "
        "smooth transitions, optional add_transition at boundaries. Do not call "
        "destructive tools. Use only the following verb kinds: "
        + ", ".join(k.value for k in PlanOpKind)
        + ". Validate the JSON strictly."
    )

    INTERPRET_SYSTEM = (
        "You are the same edit planner, now operating on an EXISTING timeline.\n"
        "Given the user's natural-language instruction, the current timeline "
        "state (JSON), and an index of existing timeline items, return a "
        "Plan whose ops ONLY modify the existing timeline via these verbs: "
        "move_clip, delete_clip, set_transform, set_crop, set_opacity, "
        "set_composite_mode, add_fade, set_speed, add_marker, add_transition. "
        "Do NOT insert or append new clips. Reference existing items by their "
        "real ``id``s from the timeline state — no symbolic placeholders. "
        "Validate JSON strictly."
    )

    def __init__(
        self,
        *,
        gemini: GeminiClient | None,
        settings: DirectorSettings,
    ) -> None:
        super().__init__(gemini=gemini, settings=settings)

    async def run(self, request: PlannerRequest) -> Plan:
        if self._gemini is None:
            # Deterministic offline plan for tests/CI.
            return _build_deterministic_plan(request)
        try:
            return await self._gemini.generate_json(
                system=self.SYSTEM,
                user=_user_prompt(request),
                response_schema=Plan,
            )
        except GeminiError as err:
            raise InvalidModelOutput(str(err)) from err
        except ValidationError as err:
            raise InvalidModelOutput(str(err)) from err

    async def interpret(
        self,
        *,
        instruction: str,
        timeline_state: dict[str, Any],
        target_project: str,
        target_timeline: str,
    ) -> Plan:
        """Produce a *delta* plan (modify / fade / move) from an NL instruction.

        Used by the Director REPL. The Gemini path returns a structured Plan;
        the offline fallback returns a no-op plan with an injected rationale.
        """
        if self._gemini is None:
            return _offline_interpret_plan(
                instruction, timeline_state, target_project, target_timeline
            )
        try:
            return await self._gemini.generate_json(
                system=self.INTERPRET_SYSTEM,
                user=_interpret_user_prompt(
                    instruction=instruction,
                    timeline_state=timeline_state,
                    target_project=target_project,
                    target_timeline=target_timeline,
                ),
                response_schema=Plan,
            )
        except (GeminiError, ValidationError) as err:
            raise InvalidModelOutput(str(err)) from err


# --- Offline deterministic builder used by tests + as a sane fallback ----------


def _build_deterministic_plan(req: PlannerRequest) -> Plan:
    ops: list[PlanOp] = []
    beats = sorted(req.beat_times or [])
    if not beats:
        beats = [float(i * 2.0) for i in range(max(1, len(req.per_clip)))]
    cursor = 0.0
    for clip_idx, clip in enumerate(req.per_clip):
        # Use beat "closest to" the next position; default slice = 2s if beats are scarce.
        target_beat = beats[min(clip_idx, len(beats) - 1)]
        duration = _slice_for_clip(clip, fallback_seconds=2.0)
        ops.append(
            PlanOp(
                id=f"op_{uuid.uuid4().hex[:8]}",
                kind=PlanOpKind.APPEND_CLIP,
                args={
                    "media_clip_id": clip.clip_id,
                    "timeline_track_index": 0,
                    "start_seconds": cursor,
                    "duration_seconds": duration,
                },
                rationale=(
                    f"Place {clip.clip_id or 'clip'} at beat {target_beat:.2f}s "
                    f"to align with the music."
                ),
            )
        )
        if duration > 0:
            ops.append(
                PlanOp(
                    id=f"op_{uuid.uuid4().hex[:8]}",
                    kind=PlanOpKind.ADD_FADE,
                    args={
                        "timeline_item_id": "<item:0>",
                        "fade_in_seconds": 0.15,
                        "fade_out_seconds": 0.2,
                    },
                    rationale="Soft cross-clip fade for visual smoothness",
                )
            )
        cursor += duration
    return Plan(
        plan_id=f"plan_{uuid.uuid4().hex[:8]}",
        version=1,
        target_project=req.target_project,
        target_timeline=req.target_timeline,
        ops=ops,
        summary=(
            f"Beat-synced timeline committed for {len(req.per_clip)} clip(s) "
            f"on '{req.target_timeline}'."
        ),
    )


def _slice_for_clip(clip: PerClipMap, *, fallback_seconds: float) -> float:
    # If we know duration, cap at min(source-duration, fallback-2-beat-grid).
    if clip.duration_seconds > 0:
        return min(clip.duration_seconds, 4.0)
    return fallback_seconds


def _user_prompt(req: PlannerRequest) -> str:
    parts: list[str] = [
        f"User brief: {req.user_prompt}",
        f"Target project: {req.target_project}",
        f"Target timeline: {req.target_timeline}",
        f"Timeline fps: {req.target_fps}",
    ]
    if req.music_bpm is not None:
        parts.append(f"Music BPM: {req.music_bpm:.2f}")
    if req.beat_times:
        parts.extend([
            "First 32 beat times (s):",
            ", ".join(f"{t:.3f}" for t in req.beat_times[:32]),
        ])
    if req.per_clip:
        parts.append("Clips:")
        for c in req.per_clip:
            parts.append(
                f"- id={c.clip_id} path={c.source_path} dur={c.duration_seconds:.2f}s "
                f"summary={c.visual_summary!r}"
            )
    return "\n".join(parts)


def _offline_interpret_plan(
    instruction: str,
    timeline_state: dict[str, Any],
    target_project: str,
    target_timeline: str,
) -> Plan:
    """Trivial deterministic delta plan: add a marker to the first clip + a 0.2s fade.

    Real LLM-driven REPL sessions will produce structured plans via Gemini; the
    offline fallback is just enough to verify the REPL plumbing without network.
    """
    ops: list[PlanOp] = []
    tracks = timeline_state.get("tracks", [])
    if tracks and tracks[0].get("items"):
        first_item = tracks[0]["items"][0]
        ops.append(
            PlanOp(
                id=f"op_{uuid.uuid4().hex[:8]}",
                kind=PlanOpKind.ADD_MARKER,
                args={
                    "timeline_item_id": first_item["id"],
                    "position_seconds": 0.0,
                    "label": instruction[:30],
                    "color": "yellow",
                    "note": instruction,
                },
                rationale="Annotate first item with the user's instruction.",
            )
        )
        ops.append(
            PlanOp(
                id=f"op_{uuid.uuid4().hex[:8]}",
                kind=PlanOpKind.ADD_FADE,
                args={
                    "timeline_item_id": first_item["id"],
                    "fade_in_seconds": 0.2,
                    "fade_out_seconds": 0.2,
                },
                rationale="Smooth the first item per REPL request.",
            )
        )
    return Plan(
        plan_id=f"plan_{uuid.uuid4().hex[:8]}",
        version=1,
        target_project=target_project,
        target_timeline=target_timeline,
        ops=ops,
        summary=f"Interpreted: {instruction[:80]}",
    )


def _interpret_user_prompt(
    *,
    instruction: str,
    timeline_state: dict[str, Any],
    target_project: str,
    target_timeline: str,
) -> str:
    return (
        f"User instruction: {instruction}\n"
        f"Target project: {target_project}\n"
        f"Target timeline: {target_timeline}\n"
        f"Current timeline state (JSON):\n"
        f"{timeline_state!s}"
    )


__all__ = ["Planner", "PlannerRequest", "_build_deterministic_plan"]
