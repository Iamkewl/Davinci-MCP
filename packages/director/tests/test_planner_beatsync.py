"""The offline planner must actually cut to the beat.

The old planner placed clips back to back at a running cursor and mentioned the
beat only in its rationale text, so these tests assert the numbers a real editor
would check: where each cut lands, whether the timeline is covered, whether the
music is on the timeline at all, and whether the shots stay inside their source.
"""

from __future__ import annotations

from itertools import pairwise

import pytest
from director.agents.planner import (
    AUDIO_TRACK,
    VIDEO_TRACK,
    PlannerRequest,
    build_deterministic_plan,
    parse_target_duration,
)
from director.plan_validation import validate_plan
from director.schemas import KeyMoment, PerClipMap, PlanOpKind

BEAT = 0.5  # 120 BPM
BEATS = [round(i * BEAT, 3) for i in range(1, 121)]  # 60s of beats


def _clips(*durations: float) -> list[PerClipMap]:
    return [
        PerClipMap(clip_id=f"clip_{index}", source_path=f"/m/{index}.mp4", duration_seconds=duration)
        for index, duration in enumerate(durations)
    ]


def _request(**overrides: object) -> PlannerRequest:
    base: dict[str, object] = {
        "user_prompt": "high-energy 20s reel",
        "per_clip": _clips(8.0, 12.0, 6.0),
        "target_project": "p",
        "target_timeline": "t",
        "target_fps": 24.0,
        "music_bpm": 120.0,
        "beat_times": BEATS,
        "music_duration_seconds": 60.0,
        "music_path": "/m/track.wav",
    }
    base.update(overrides)
    return PlannerRequest(**base)  # type: ignore[arg-type]


def _appends(plan: object, track: int) -> list[dict]:
    return [
        op.args
        for op in plan.ops  # type: ignore[attr-defined]
        if op.kind == PlanOpKind.APPEND_CLIP and op.args.get("timeline_track_index") == track
    ]


# --- the headline claim ---------------------------------------------------------


def test_every_cut_lands_on_a_beat() -> None:
    plan = build_deterministic_plan(_request())
    shots = sorted(_appends(plan, VIDEO_TRACK), key=lambda a: a["start_seconds"])
    cuts = [a["start_seconds"] for a in shots[1:]]
    assert cuts, "a 20s reel should contain several cuts"
    for cut in cuts:
        nearest = min(BEATS, key=lambda b: abs(b - cut))
        assert abs(nearest - cut) <= 0.02, f"cut at {cut}s is off the beat grid"


def test_shots_tile_the_target_length_without_gaps_or_overlaps() -> None:
    plan = build_deterministic_plan(_request())
    shots = sorted(_appends(plan, VIDEO_TRACK), key=lambda a: a["start_seconds"])
    assert shots[0]["start_seconds"] == 0.0
    for previous, current in pairwise(shots):
        end = previous["start_seconds"] + previous["duration_seconds"]
        assert current["start_seconds"] == pytest.approx(end, abs=1e-6), "gap or overlap between shots"
    total = shots[-1]["start_seconds"] + shots[-1]["duration_seconds"]
    assert total == pytest.approx(20.0, abs=0.05)


def test_music_is_placed_on_the_audio_track() -> None:
    plan = build_deterministic_plan(_request())
    music = _appends(plan, AUDIO_TRACK)
    assert len(music) == 1, "the music must be laid on the timeline, not just analysed"
    assert music[0]["media_clip_id"] == "/m/track.wav"
    assert music[0]["start_seconds"] == 0.0
    assert music[0]["duration_seconds"] == pytest.approx(20.0, abs=0.05)


def test_plan_is_executable() -> None:
    request = _request()
    issues = validate_plan(
        build_deterministic_plan(request),
        per_clip=request.per_clip,
        music_path=request.music_path,
    )
    assert issues == []


# --- clip handling ---------------------------------------------------------------


def test_uses_every_clip_and_avoids_back_to_back_repeats() -> None:
    plan = build_deterministic_plan(_request())
    used = [a["media_clip_id"] for a in _appends(plan, VIDEO_TRACK)]
    assert set(used) == {"clip_0", "clip_1", "clip_2"}
    assert all(a != b for a, b in pairwise(used))


def test_never_reads_past_the_end_of_a_clip() -> None:
    clips = _clips(3.0, 4.0)
    plan = build_deterministic_plan(_request(per_clip=clips))
    by_id = {c.clip_id: c for c in clips}
    for args in _appends(plan, VIDEO_TRACK):
        clip = by_id[args["media_clip_id"]]
        end = args["source_in_seconds"] + args["duration_seconds"]
        assert end <= clip.duration_seconds + 1e-6, f"{clip.clip_id}: reads {end}s of {clip.duration_seconds}s"


def test_reused_clip_shows_different_footage() -> None:
    plan = build_deterministic_plan(_request(per_clip=_clips(12.0)))
    starts = [a["source_in_seconds"] for a in _appends(plan, VIDEO_TRACK)]
    assert len(set(starts)) > 1, "a repeated clip replayed the same seconds every time"


def test_prefers_a_key_moment_when_the_vision_pass_found_one() -> None:
    clip = PerClipMap(
        clip_id="clip_0",
        source_path="/m/0.mp4",
        duration_seconds=30.0,
        key_moments=[KeyMoment(position_seconds=12.0, kind="highlight")],
    )
    plan = build_deterministic_plan(_request(per_clip=[clip]))
    starts = [a["source_in_seconds"] for a in _appends(plan, VIDEO_TRACK)]
    assert 12.0 in starts


# --- brief interpretation ---------------------------------------------------------


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("high-energy 30s reel", 30.0),
        ("make me a 45 second cut", 45.0),
        ("1 minute montage", 60.0),
        ("a 1:30 piece", 90.0),
        ("no length mentioned", None),
    ],
)
def test_parse_target_duration(prompt: str, expected: float | None) -> None:
    assert parse_target_duration(prompt) == expected


def test_pacing_follows_the_brief() -> None:
    fast = build_deterministic_plan(_request(user_prompt="high-energy 20s reel"))
    slow = build_deterministic_plan(_request(user_prompt="moody cinematic 20s piece"))
    fast_shots = _appends(fast, VIDEO_TRACK)
    slow_shots = _appends(slow, VIDEO_TRACK)
    assert len(fast_shots) > len(slow_shots), "an energetic brief should cut faster"


def test_target_length_never_exceeds_the_music() -> None:
    plan = build_deterministic_plan(
        _request(user_prompt="60s reel", music_duration_seconds=10.0, beat_times=BEATS[:20])
    )
    shots = _appends(plan, VIDEO_TRACK)
    total = shots[-1]["start_seconds"] + shots[-1]["duration_seconds"]
    assert total <= 10.0 + 1e-6


# --- capability and feedback awareness --------------------------------------------


def test_respects_the_tools_the_backend_offers() -> None:
    """The live DaVinci backend has no fades, so a plan for it must not use them."""
    request = _request(available_tools=frozenset({"append_clip", "get_timeline_state"}))
    plan = build_deterministic_plan(request)
    assert all(op.kind != PlanOpKind.ADD_FADE for op in plan.ops)
    assert validate_plan(
        plan,
        per_clip=request.per_clip,
        available_tools=request.available_tools,
        music_path=request.music_path,
    ) == []


def test_fades_top_and_tail_only() -> None:
    plan = build_deterministic_plan(_request())
    fades = [op for op in plan.ops if op.kind == PlanOpKind.ADD_FADE]
    shots = _appends(plan, VIDEO_TRACK)
    assert len(fades) < len(shots), "fading every shot dips to black on every cut"
    assert any(op.args.get("fade_in_seconds") for op in fades)
    assert any(op.args.get("fade_out_seconds") for op in fades)


def test_feedback_changes_the_next_plan() -> None:
    before = build_deterministic_plan(_request())
    after = build_deterministic_plan(
        _request(feedback=["beat_sync: 3 cut(s) do not land on a beat"])
    )
    assert [op.args for op in after.ops] != [op.args for op in before.ops]
    # "snap to the beat" means cutting on every beat, so there are more shots.
    assert len(_appends(after, VIDEO_TRACK)) > len(_appends(before, VIDEO_TRACK))


def test_summary_states_real_numbers() -> None:
    plan = build_deterministic_plan(_request())
    shots = _appends(plan, VIDEO_TRACK)
    assert f"{len(shots)} shot(s)" in plan.summary
    assert "music on A1" in plan.summary


def test_works_without_music_or_durations() -> None:
    request = _request(
        beat_times=[],
        music_bpm=None,
        music_duration_seconds=None,
        music_path=None,
        per_clip=[PerClipMap(clip_id="c", source_path="/m/c.mp4")],  # unknown length
        user_prompt="10 second cut",
    )
    plan = build_deterministic_plan(request)
    shots = _appends(plan, VIDEO_TRACK)
    assert shots, "an even grid should still produce shots"
    total = shots[-1]["start_seconds"] + shots[-1]["duration_seconds"]
    assert total == pytest.approx(10.0, abs=0.05)
    assert validate_plan(plan, per_clip=request.per_clip) == []


# --- regressions -------------------------------------------------------------------


def test_clips_shorter_than_the_minimum_segment_do_not_over_read() -> None:
    """When nothing is long enough the shot is shortened — but the floor used to
    be applied on top of the clip's real length, planning a read past the end of
    the footage that the validator then rejected on every iteration."""
    req = _request(per_clip=_clips(0.3, 0.25), user_prompt="high-energy 10s reel")
    plan = build_deterministic_plan(req)
    assert validate_plan(
        plan,
        per_clip=req.per_clip,
        available_tools=req.available_tools,
        music_path=req.music_path,
    ) == []
    by_id = {clip.clip_id: clip.duration_seconds for clip in req.per_clip}
    for op in plan.ops:
        if op.kind != PlanOpKind.APPEND_CLIP:
            continue
        media = op.args["media_clip_id"]
        if media not in by_id:
            continue  # the music bed
        end = float(op.args.get("source_in_seconds", 0.0)) + float(op.args["duration_seconds"])
        assert end <= by_id[media] + 1e-6, f"{op.args} reads past a {by_id[media]}s clip"


def test_target_length_is_capped_by_the_music() -> None:
    """A 30s brief over a 10s track is a 10s cut — and the reviewer has to be
    told 10, or it fails a correct plan for "only" covering a third of it."""
    from director.agents.planner import effective_target_duration

    clips = _clips(8.0, 12.0)
    assert effective_target_duration("30s reel", clips, 10.0) == pytest.approx(10.0)
    assert effective_target_duration("30s reel", clips, None) == pytest.approx(30.0)
    assert effective_target_duration("30s reel", clips, 60.0) == pytest.approx(30.0)

    req = _request(user_prompt="30s reel", music_duration_seconds=10.0, per_clip=clips)
    plan = build_deterministic_plan(req)
    video = [op for op in plan.ops if op.kind == PlanOpKind.APPEND_CLIP and op.args.get("timeline_track_index") == VIDEO_TRACK]
    covered = max(float(op.args["start_seconds"]) + float(op.args["duration_seconds"]) for op in video)
    assert covered == pytest.approx(effective_target_duration("30s reel", clips, 10.0), abs=0.05)
