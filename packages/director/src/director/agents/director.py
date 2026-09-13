"""Director: score a plan on what it actually produces, then approve or reject.

The offline scorer measures the plan itself rather than counting ops: where the
cuts land relative to the detected beats, whether the timeline is covered
end-to-end, whether the shots respect the source media, whether the brief's
requested length and pacing are met, and whether the plan is structurally
executable at all. (The previous version scored ``appends / beat_count``, which
made a 3-shot plan against a 60-beat track look like a 5% failure while never
checking a single cut point.)

Honest verdict rules
--------------------

* APPROVED: overall >= ``director_min_overall`` AND every axis >=
  ``director_min_per_axis``.
* ACCEPTED_WITH_WARNINGS: overall clears the floor but some axis doesn't.
* FAILED: overall below the floor.

A plan with structural problems (see :mod:`director.plan_validation`) can never
be APPROVED, whoever produced it — including an LLM reviewer that liked it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ..errors import ProviderError
from ..ingestion.gemini_client import GeminiClient
from ..schemas import (
    DirectorAxisScore,
    DirectorEvaluation,
    DirectorVerdict,
    PerClipMap,
    Plan,
    PlanOp,
    PlanOpKind,
)
from ..settings import DirectorSettings
from .base import Agent, InvalidModelOutput

if TYPE_CHECKING:
    from ..llm.base import LLMClient

__all__ = ["Director", "DirectorOutcome", "PlanContext"]

#: A cut counts as "on the beat" within this distance (about half a frame at 24fps).
BEAT_TOLERANCE_SECONDS = 0.05

#: Shots closer than this count as contiguous — below one frame at 24fps, so
#: millisecond rounding in a client's numbers is not reported as a hole.
GAP_TOLERANCE_SECONDS = 0.021

_DESTRUCTIVE_KINDS = {PlanOpKind.DELETE_CLIP}


@dataclass(frozen=True)
class PlanContext:
    """What the plan was supposed to achieve — needed to judge whether it does."""

    beat_times: list[float] = field(default_factory=list)
    per_clip: list[PerClipMap] = field(default_factory=list)
    target_duration_seconds: float | None = None
    music_duration_seconds: float | None = None
    music_path: str | None = None
    available_tools: frozenset[str] = frozenset()
    validation_issues: list[str] = field(default_factory=list)


@dataclass
class DirectorOutcome:
    """Verdict plus the raw axes used to compute it."""

    evaluation: DirectorEvaluation


class Director(Agent[DirectorOutcome]):
    """Pure scorer agent. No state, repeatable."""

    SYSTEM = (
        "You are a strict but fair director reviewing an auto-generated edit "
        "plan against a user's brief and a music beat grid. Score 0..1 on each "
        "axis: beat_sync, coverage, validity, prompt_fidelity, variety, safety. "
        "Judge what the plan actually does — where cuts land relative to the "
        "beats, whether the timeline is covered, whether it matches the brief. "
        "Return APPROVED only if every axis clears its floor; otherwise "
        "ACCEPTED_WITH_WARNINGS, or FAILED when the plan is unusable. Issues and "
        "suggestions must be specific enough for the planner to act on. Output "
        "JSON conforming to the schema."
    )

    def __init__(
        self,
        *,
        gemini: GeminiClient | None = None,
        llm: LLMClient | None = None,
        settings: DirectorSettings,
    ) -> None:
        super().__init__(gemini=gemini, llm=llm, settings=settings)

    async def run(
        self,
        *,
        plan: Plan,
        user_prompt: str,
        context: PlanContext | None = None,
    ) -> DirectorOutcome:
        ctx = context or PlanContext()
        if self._llm is None:
            evaluation = evaluate_plan(plan, ctx, self._settings)
        else:
            try:
                evaluation = await self._llm.generate_json(
                    system=self.SYSTEM,
                    user=_build_user(plan, user_prompt, ctx),
                    response_schema=DirectorEvaluation,
                )
            except (ProviderError, ValidationError) as err:
                raise InvalidModelOutput(str(err)) from err
            evaluation = _enforce_validation(evaluation, ctx)
        return DirectorOutcome(evaluation=evaluation)


# --- offline scoring --------------------------------------------------------------


def evaluate_plan(
    plan: Plan, context: PlanContext, settings: DirectorSettings
) -> DirectorEvaluation:
    """Deterministic scoring used offline and in CI."""
    appends = [op for op in plan.ops if op.kind == PlanOpKind.APPEND_CLIP]
    video = [op for op in appends if _track(op) == 1]
    issues: list[str] = list(context.validation_issues)
    suggestions: list[str] = []
    axes: list[DirectorAxisScore] = []

    if not plan.ops:
        return DirectorEvaluation(
            verdict=DirectorVerdict.FAILED,
            overall=0.0,
            axes=[DirectorAxisScore(name="completeness", score=0.0, rationale="empty plan")],
            issues=["the plan contains no operations"],
            suggestions=["produce at least one append_clip covering the requested length"],
        )

    # --- beat sync
    if context.beat_times and len(video) > 1:
        cuts = sorted(_start(op) for op in video)[1:]  # the first shot starts the timeline
        on_beat = sum(1 for cut in cuts if _distance_to_beat(cut, context.beat_times) <= BEAT_TOLERANCE_SECONDS)
        score = on_beat / len(cuts) if cuts else 1.0
        rationale = f"{on_beat}/{len(cuts)} cuts land on a detected beat"
        if score < 1.0:
            issues.append(f"beat_sync: {len(cuts) - on_beat} cut(s) do not land on a beat")
            suggestions.append("snap every cut to a beat time from the analysis")
    else:
        score, rationale = 1.0, "no music/beat grid supplied — not applicable"
    axes.append(DirectorAxisScore(name="beat_sync", score=score, rationale=rationale))

    # --- coverage
    target = context.target_duration_seconds
    if video:
        covered, gaps = _coverage(video)
        if target and target > 0:
            score = max(0.0, min(1.0, covered / target))
            rationale = f"covers {covered:.1f}s of the {target:.1f}s target"
            if score < 0.95:
                issues.append(f"coverage: the cut is {covered:.1f}s but {target:.1f}s was asked for")
                suggestions.append("extend or add shots so the timeline reaches the target length")
        else:
            score, rationale = 1.0, f"covers {covered:.1f}s (no target length given)"
        if gaps:
            score = min(score, 0.6)
            rationale += f"; {len(gaps)} gap(s) between shots"
            issues.append(f"coverage: {len(gaps)} gap(s) between shots")
            suggestions.append("make each shot start where the previous one ends")
    else:
        score, rationale = (1.0, "delta plan — no new shots") if not appends else (0.0, "no video shots")
    axes.append(DirectorAxisScore(name="coverage", score=score, rationale=rationale))

    # --- validity
    if context.validation_issues:
        axes.append(
            DirectorAxisScore(
                name="validity",
                score=0.0,
                rationale=f"{len(context.validation_issues)} structural problem(s)",
            )
        )
        suggestions.append("fix the structural problems before anything else")
    else:
        axes.append(DirectorAxisScore(name="validity", score=1.0, rationale="plan is executable"))

    # --- prompt fidelity
    if target and video:
        covered, _ = _coverage(video)
        drift = abs(covered - target) / target
        score = max(0.0, 1.0 - drift * 2)  # 10% drift -> 0.8
        rationale = f"length is {covered:.1f}s against a {target:.1f}s brief"
        if drift > 0.1:
            issues.append(f"prompt_fidelity: length differs from the brief by {drift:.0%}")
    else:
        score, rationale = 1.0, "no explicit length in the brief"
    axes.append(DirectorAxisScore(name="prompt_fidelity", score=score, rationale=rationale))

    # --- variety
    if video and len(context.per_clip) > 1:
        used = [op.args.get("media_clip_id") for op in video]
        distinct = len({u for u in used if isinstance(u, str)})
        repeats = sum(1 for a, b in pairwise(used) if a == b)
        score = distinct / min(len(context.per_clip), len(video))
        score = max(0.0, min(1.0, score) - 0.2 * repeats)
        rationale = f"{distinct} of {len(context.per_clip)} clips used, {repeats} back-to-back repeat(s)"
        if score < 1.0:
            issues.append(f"variety: {rationale}")
            suggestions.append("rotate through every clip before reusing one")
    else:
        score, rationale = 1.0, "single clip or delta plan — not applicable"
    axes.append(DirectorAxisScore(name="variety", score=score, rationale=rationale))

    # --- safety
    destructive = [op for op in plan.ops if op.kind in _DESTRUCTIVE_KINDS]
    if destructive:
        axes.append(
            DirectorAxisScore(
                name="safety", score=0.5, rationale=f"{len(destructive)} destructive op(s)"
            )
        )
    else:
        axes.append(DirectorAxisScore(name="safety", score=1.0, rationale="no destructive ops"))

    overall = sum(a.score for a in axes) / len(axes)
    below = [a for a in axes if a.score < settings.director_min_per_axis]
    for axis in below:
        issues.append(f"{axis.name} below floor ({axis.score:.2f}): {axis.rationale}")
    if overall < settings.director_min_overall:
        verdict = DirectorVerdict.FAILED
        issues.append(f"overall {overall:.2f} below floor {settings.director_min_overall:.2f}")
    elif below:
        verdict = DirectorVerdict.ACCEPTED_WITH_WARNINGS
    else:
        verdict = DirectorVerdict.APPROVED
    return DirectorEvaluation(
        verdict=verdict,
        overall=overall,
        axes=axes,
        issues=_dedupe(issues),
        suggestions=_dedupe(suggestions),
    )


def _enforce_validation(
    evaluation: DirectorEvaluation, context: PlanContext
) -> DirectorEvaluation:
    """A model may not approve a plan that cannot execute."""
    if not context.validation_issues or evaluation.verdict != DirectorVerdict.APPROVED:
        return evaluation
    return evaluation.model_copy(
        update={
            "verdict": DirectorVerdict.ACCEPTED_WITH_WARNINGS,
            "issues": _dedupe([*evaluation.issues, *context.validation_issues]),
        }
    )


# --- helpers ----------------------------------------------------------------------


def _track(op: PlanOp) -> int:
    value = op.args.get("timeline_track_index", 1)
    return value if isinstance(value, int) and not isinstance(value, bool) else 1


def _start(op: PlanOp) -> float:
    value = op.args.get("start_seconds", op.args.get("timeline_position_seconds", 0.0))
    return float(value) if isinstance(value, int | float) else 0.0


def _duration(op: PlanOp) -> float:
    value = op.args.get("duration_seconds", 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _coverage(video_ops: Sequence[PlanOp]) -> tuple[float, list[tuple[float, float]]]:
    """Total covered seconds and any gaps between consecutive shots."""
    spans = sorted((_start(op), _start(op) + _duration(op)) for op in video_ops)
    covered = 0.0
    gaps: list[tuple[float, float]] = []
    previous_end = 0.0
    for start, end in spans:
        covered += max(0.0, end - start)
        if start > previous_end + GAP_TOLERANCE_SECONDS:
            gaps.append((previous_end, start))
        previous_end = max(previous_end, end)
    return covered, gaps


def _distance_to_beat(position: float, beats: list[float]) -> float:
    return min((abs(position - beat) for beat in beats), default=float("inf"))


def _dedupe(values: list[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return seen


def _build_user(plan: Plan, user_prompt: str, context: PlanContext) -> str:
    beats = context.beat_times[:200]
    parts = [
        f"User brief: {user_prompt}",
        f"Target length: {context.target_duration_seconds or 'unspecified'}",
        f"Beats ({len(context.beat_times)}): " + ", ".join(f"{b:.2f}" for b in beats),
        "Clips: "
        + "; ".join(f"{c.clip_id}={c.duration_seconds:.1f}s" for c in context.per_clip),
    ]
    if context.validation_issues:
        parts.append("Structural problems already found: " + "; ".join(context.validation_issues))
    parts.append(f"Plan summary: {plan.summary}")
    parts.append(f"Plan ops: {plan.model_dump_json()}")
    return "\n".join(parts)
