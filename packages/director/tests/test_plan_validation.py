"""Validation is the gate that keeps a broken plan away from Resolve.

Each test represents a way a plan (usually a model's) goes wrong in practice —
an invented argument name, a 0-based track index, a shot longer than its source,
a reference to an append that never happens — and asserts the problem is caught
with a message the planner can act on.
"""

from __future__ import annotations

import pytest
from director.plan_validation import VERB_SPECS, describe_verbs, validate_plan
from director.schemas import PerClipMap, Plan, PlanOp, PlanOpKind

CLIPS = [
    PerClipMap(clip_id="clip_a", source_path="/m/a.mp4", duration_seconds=10.0),
    PerClipMap(clip_id="clip_b", source_path="/m/b.mp4", duration_seconds=4.0),
]
ALL_TOOLS = frozenset(spec.tool for spec in VERB_SPECS.values())


def _plan(*ops: PlanOp) -> Plan:
    return Plan(
        plan_id="p", version=1, target_project="proj", target_timeline="tl",
        ops=list(ops), summary="s",
    )


def _append(**args: object) -> PlanOp:
    base: dict[str, object] = {
        "media_clip_id": "clip_a",
        "timeline_track_index": 1,
        "start_seconds": 0.0,
        "duration_seconds": 2.0,
    }
    base.update(args)
    return PlanOp(id="op", kind=PlanOpKind.APPEND_CLIP, args=base)


def _issues(plan: Plan, **kwargs: object) -> list[str]:
    return validate_plan(plan, per_clip=CLIPS, **kwargs)  # type: ignore[arg-type]


def test_a_good_plan_has_no_issues() -> None:
    plan = _plan(
        _append(start_seconds=0.0, duration_seconds=2.0),
        _append(media_clip_id="clip_b", start_seconds=2.0, duration_seconds=2.0),
    )
    assert _issues(plan) == []


def test_empty_plan_is_reported() -> None:
    assert _issues(_plan()) == ["plan contains no operations"]


def test_unknown_argument_name() -> None:
    issues = _issues(_plan(_append(duration=2.0)))
    assert any("unknown argument 'duration'" in i for i in issues)
    assert any("duration_seconds" in i for i in issues), "should suggest the real name"


def test_missing_required_argument() -> None:
    op = PlanOp(id="op", kind=PlanOpKind.APPEND_CLIP, args={"media_clip_id": "clip_a"})
    assert any("missing required argument 'duration_seconds'" in i for i in _issues(_plan(op)))


def test_zero_based_track_index_is_rejected() -> None:
    assert any("1-based" in i for i in _issues(_plan(_append(timeline_track_index=0))))


def test_non_positive_duration_is_rejected() -> None:
    assert any("duration_seconds must be a number > 0" in i for i in _issues(_plan(_append(duration_seconds=0))))


def test_negative_position_is_rejected() -> None:
    assert any("start_seconds must be a number >= 0" in i for i in _issues(_plan(_append(start_seconds=-1.0))))


def test_source_range_beyond_the_clip() -> None:
    plan = _plan(_append(media_clip_id="clip_b", source_in_seconds=3.0, duration_seconds=3.0))
    assert any("of a 4.00s clip" in i for i in _issues(plan))


def test_unknown_media_id() -> None:
    assert any("not one of the run's clips" in i for i in _issues(_plan(_append(media_clip_id="nope"))))


def test_source_path_counts_as_a_known_id() -> None:
    assert _issues(_plan(_append(media_clip_id="/m/a.mp4"))) == []


def test_music_path_counts_as_a_known_id() -> None:
    plan = _plan(_append(media_clip_id="/m/track.wav", timeline_track_index=2))
    assert _issues(plan, music_path="/m/track.wav") == []


def test_overlapping_shots_on_one_track() -> None:
    plan = _plan(
        _append(start_seconds=0.0, duration_seconds=3.0),
        _append(start_seconds=2.0, duration_seconds=3.0),
    )
    assert any("overlap on track 1" in i for i in _issues(plan))


def test_same_position_on_different_tracks_is_fine() -> None:
    plan = _plan(
        _append(start_seconds=0.0, duration_seconds=3.0),
        _append(media_clip_id="/m/track.wav", timeline_track_index=2, start_seconds=0.0, duration_seconds=3.0),
    )
    assert _issues(plan, music_path="/m/track.wav") == []


def test_dangling_symbolic_reference() -> None:
    plan = _plan(
        _append(),
        PlanOp(
            id="op2",
            kind=PlanOpKind.ADD_FADE,
            args={"timeline_item_id": "<item:5>", "fade_in_seconds": 0.5},
        ),
    )
    assert any("only appends 1 clip(s)" in i for i in _issues(plan))


def test_tool_the_backend_does_not_offer() -> None:
    plan = _plan(
        _append(),
        PlanOp(id="op2", kind=PlanOpKind.ADD_FADE, args={"timeline_item_id": "<item:0>"}),
    )
    issues = _issues(plan, available_tools=frozenset({"append_clip"}))
    assert any("'add_fade' is not available" in i for i in issues)


def test_opacity_out_of_range() -> None:
    plan = _plan(
        _append(),
        PlanOp(
            id="op2",
            kind=PlanOpKind.SET_OPACITY,
            args={"timeline_item_id": "<item:0>", "opacity": 150},
        ),
    )
    assert any("opacity must be within 0.0..1.0" in i for i in _issues(plan))


# --- the prompt side of the same table -------------------------------------------


def test_describe_verbs_lists_argument_names_and_units() -> None:
    text = describe_verbs()
    assert "append_clip" in text
    assert "duration_seconds (required)" in text
    assert "1-based" in text
    assert "seconds" in text


def test_describe_verbs_hides_unavailable_tools() -> None:
    text = describe_verbs(frozenset({"append_clip"}))
    assert "append_clip" in text
    assert "add_fade" not in text


@pytest.mark.parametrize("kind", list(PlanOpKind))
def test_every_verb_has_a_spec(kind: PlanOpKind) -> None:
    """A verb without a spec would be unvalidated and undocumented."""
    assert kind in VERB_SPECS
    assert VERB_SPECS[kind].tool
    assert VERB_SPECS[kind].summary
