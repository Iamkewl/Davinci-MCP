"""Director: scores a plan on a fixed axes set; returns APPROVED / ACCEPTED / FAILED.

Honest verdict rules
--------------------

* APPROVED: overall >= settings.director_min_overall AND every per-axis score >=
  settings.director_min_per_axis.
* ACCEPTED_WITH_WARNINGS: overall >= min_overall but at least one axis is below
  per-axis floor. We do not stop; we relay the suggestion list to the planner.
* FAILED: overall below floor OR critical issue surfaced.

We never silently force-accept on max iterations. If the planner hits
``max_planner_iterations`` with no APPROVED verdict, the orchestrator surfaces
the latest verdict and exits with status ``failed``.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from ..ingestion.gemini_client import GeminiClient, GeminiError
from ..schemas import DirectorAxisScore, DirectorEvaluation, DirectorVerdict, Plan
from ..settings import DirectorSettings
from .base import Agent, InvalidModelOutput


@dataclass
class DirectorOutcome:
    """Verdict plus the raw axes used to compute it."""

    evaluation: DirectorEvaluation


class Director(Agent[DirectorOutcome]):
    """Pure scorer agent. No state, repeatable."""

    SYSTEM = (
        "You are a strict but fair director reviewing an auto-generated edit "
        "plan against a user's brief and a music beat grid. Score 0..1 on each "
        "axis: beat_sync, narrative, completeness, prompt_fidelity, safety. "
        "Return APPROVED if overall >= min_overall and no axis below "
        "min_per_axis; ACCEPTED_WITH_WARNINGS otherwise if overall >= "
        "min_overall; FAILED if overall < min_overall. Output JSON conforming "
        "to the schema."
    )

    def __init__(
        self,
        *,
        gemini: GeminiClient | None,
        settings: DirectorSettings,
    ) -> None:
        super().__init__(gemini=gemini, settings=settings)

    async def run(
        self,
        *,
        plan: Plan,
        user_prompt: str,
        beat_count: int,
    ) -> DirectorOutcome:
        if self._gemini is None:
            evaluation = _offline_evaluation(plan, beat_count, self._settings)
        else:
            try:
                evaluation = await self._gemini.generate_json(
                    system=self.SYSTEM,
                    user=_build_user(plan, user_prompt, beat_count),
                    response_schema=DirectorEvaluation,
                )
            except (GeminiError, ValidationError) as err:
                raise InvalidModelOutput(str(err)) from err
        return DirectorOutcome(evaluation=evaluation)


# --- Pure helpers -------------------------------------------------------------


def _offline_evaluation(
    plan: Plan,
    beat_count: int,
    settings: DirectorSettings,
) -> DirectorEvaluation:
    """Deterministic scoring for tests + offline CI."""

    axes = [
        DirectorAxisScore(name="beat_sync", score=_score_beat_sync(plan, beat_count), rationale=""),
        DirectorAxisScore(
            name="narrative", score=_score_narrative(plan), rationale="",
        ),
        DirectorAxisScore(
            name="completeness",
            score=_score_completeness(plan),
            rationale="",
        ),
        DirectorAxisScore(
            name="prompt_fidelity", score=_score_prompt_fidelity(plan), rationale="",
        ),
        DirectorAxisScore(name="safety", score=1.0, rationale=""),
    ]
    overall = sum(a.score for a in axes) / len(axes)
    issues: list[str] = []
    suggestions: list[str] = []
    if any(a.score < settings.director_min_per_axis for a in axes):
        for a in axes:
            if a.score < settings.director_min_per_axis:
                issues.append(f"{a.name} below floor ({a.score:.2f})")
                suggestions.append(f"Improve {a.name} in the next planner iteration.")
    verdict = (
        DirectorVerdict.APPROVED
        if (overall >= settings.director_min_overall
            and not any(a.score < settings.director_min_per_axis for a in axes))
        else DirectorVerdict.ACCEPTED_WITH_WARNINGS
        if overall >= settings.director_min_overall
        else DirectorVerdict.FAILED
    )
    if verdict == DirectorVerdict.FAILED:
        issues.append(f"overall {overall:.2f} below floor {settings.director_min_overall:.2f}")
    return DirectorEvaluation(
        verdict=verdict,
        overall=overall,
        axes=axes,
        issues=issues,
        suggestions=suggestions,
    )


def _score_beat_sync(plan: Plan, beat_count: int) -> float:
    """1.0 if appends >= beats, else a ratio cap."""
    appended = sum(1 for op in plan.ops if op.kind.value == "append_clip")
    if beat_count <= 0:
        return 0.5
    return min(1.0, appended / beat_count)


def _score_narrative(plan: Plan) -> float:
    return min(1.0, len(plan.ops) / 4.0)


def _score_completeness(plan: Plan) -> float:
    return 1.0 if plan.ops else 0.5


def _score_prompt_fidelity(plan: Plan) -> float:
    return 1.0 if plan.summary else 0.7


def _build_user(plan: Plan, user_prompt: str, beat_count: int) -> str:
    return (
        f"User brief: {user_prompt}\n"
        f"Beat count: {beat_count}\n"
        f"Plan summary: {plan.summary}\n"
        f"Plan ops: {plan.model_dump_json()}"
    )


__all__ = ["Director", "DirectorOutcome"]
