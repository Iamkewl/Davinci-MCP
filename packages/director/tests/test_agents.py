"""Unit tests for the planner + director offline paths and pure helpers."""

from __future__ import annotations

from director.agents import (
    Director,
    DirectorOutcome,
    Planner,
    PlannerRequest,
)
from director.ingestion.audio_analyzer import first_n_beats_close_to_times
from director.schemas import (
    DirectorVerdict,
    PerClipMap,
    Plan,
    PlanOp,
    PlanOpKind,
)
from director.settings import DirectorSettings


async def test_planner_offline_produces_plan() -> None:
    settings = DirectorSettings(gemini_api_key=None, max_planner_iterations=5)
    planner = Planner(gemini=None, settings=settings)
    req = PlannerRequest(
        user_prompt="hi",
        per_clip=[
            PerClipMap(clip_id="c1", source_path="/a.mp4", duration_seconds=2.0),
            PerClipMap(clip_id="c2", source_path="/b.mp4", duration_seconds=2.0),
        ],
        target_project="p",
        target_timeline="t",
        target_fps=24.0,
        beat_times=[0.0, 2.0],
        music_bpm=120.0,
    )
    plan = await planner.run(req)
    assert isinstance(plan, Plan)
    assert plan.ops
    kinds = [op.kind for op in plan.ops]
    assert PlanOpKind.APPEND_CLIP in kinds


async def test_director_offline_returns_verdict() -> None:
    settings = DirectorSettings(
        gemini_api_key=None,
        director_min_overall=0.4,
        director_min_per_axis=0.3,
    )
    director = Director(gemini=None, settings=settings)
    plan = Plan(
        plan_id="p1",
        target_project="p",
        target_timeline="t",
        ops=[
            PlanOp(
                id="op1",
                kind=PlanOpKind.APPEND_CLIP,
                args={"media_clip_id": "c1", "duration_seconds": 1.0},
            ),
        ],
        summary="x",
    )
    out: DirectorOutcome = await director.run(
        plan=plan,
        user_prompt="x",
        beat_count=4,
    )
    assert out.evaluation.verdict in (
        DirectorVerdict.APPROVED,
        DirectorVerdict.ACCEPTED_WITH_WARNINGS,
        DirectorVerdict.FAILED,
    )


def test_first_n_beats_close_to_times() -> None:
    beats = [0.0, 1.0, 2.0, 3.0, 4.0]
    out = first_n_beats_close_to_times(beats, [0.05, 1.1, 2.95], tolerance_seconds=0.2)
    # Each target is within 0.2 of a beat.
    assert out[0] == 0.0
    assert out[1] == 1.0
    assert out[2] == 3.0


def test_first_n_beats_falls_back_when_outside_tolerance() -> None:
    beats = [0.0, 1.0, 2.0]
    out = first_n_beats_close_to_times(beats, [99.0], tolerance_seconds=0.1)
    # 99.0 has no nearby beat — fall back to the target itself.
    assert out == [99.0]
