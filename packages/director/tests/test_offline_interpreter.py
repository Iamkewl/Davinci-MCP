"""The offline REPL must do what the instruction says — or admit it can't.

"Refine the cut conversationally" used to mean "add a marker and a fade to the
first clip, whatever you typed". These tests pin the actual parsing, including
the honest refusal that replaced the pretence.
"""

from __future__ import annotations

import pytest
from director.agents.offline_interpreter import UNPARSED_SUMMARY_PREFIX, interpret_offline
from director.schemas import PlanOpKind


def _state(items: int = 3, *, with_music: bool = False) -> dict:
    video = [
        {
            "id": f"item_{index}",
            "start_seconds": float(index) * 2.0,
            "duration_seconds": 2.0,
        }
        for index in range(items)
    ]
    tracks = [{"index": 1, "kind": "video", "items": video}]
    if with_music:
        tracks.append(
            {
                "index": 2,
                "kind": "audio",
                "items": [{"id": "item_music", "start_seconds": 0.0, "duration_seconds": 20.0}],
            }
        )
    return {"name": "tl", "tracks": tracks}


def _plan(instruction: str, state: dict | None = None):
    return interpret_offline(
        instruction=instruction,
        timeline_state=state if state is not None else _state(),
        target_project="p",
        target_timeline="tl",
    )


def _kinds(plan) -> list[PlanOpKind]:
    return [op.kind for op in plan.ops]


# --- understood instructions ------------------------------------------------------


def test_fade_in_with_a_duration() -> None:
    plan = _plan("fade in the first clip 1s")
    assert _kinds(plan) == [PlanOpKind.ADD_FADE]
    args = plan.ops[0].args
    assert args["timeline_item_id"] == "item_0"
    assert args["fade_in_seconds"] == 1.0
    assert args["fade_out_seconds"] == 0.0


def test_fade_out_on_the_last_clip() -> None:
    args = _plan("fade out the last clip").ops[0].args
    assert args["timeline_item_id"] == "item_2"
    assert args["fade_out_seconds"] > 0


def test_fade_is_capped_at_half_the_item() -> None:
    args = _plan("fade in the first clip 10s").ops[0].args
    assert args["fade_in_seconds"] <= 1.0  # item is 2s long


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        ("slow clip 2 to 0.5x", 0.5),
        ("speed up clip 2", 2.0),
        ("slow down clip 2", 0.5),
    ],
)
def test_speed_instructions(instruction: str, expected: float) -> None:
    plan = _plan(instruction)
    assert _kinds(plan) == [PlanOpKind.SET_SPEED]
    assert plan.ops[0].args["speed"] == expected


def test_opacity_percentage() -> None:
    plan = _plan("set opacity of clip 3 to 60%")
    assert _kinds(plan) == [PlanOpKind.SET_OPACITY]
    assert plan.ops[0].args["timeline_item_id"] == "item_2"
    assert plan.ops[0].args["opacity"] == pytest.approx(0.6)


def test_zoom_percentage() -> None:
    plan = _plan("zoom clip 1 to 120%")
    assert _kinds(plan) == [PlanOpKind.SET_TRANSFORM]
    assert plan.ops[0].args["zoom_x"] == pytest.approx(1.2)
    assert plan.ops[0].args["zoom_y"] == pytest.approx(1.2)


def test_rotate() -> None:
    plan = _plan("rotate clip 2 by -15")
    assert plan.ops[0].args["rotation"] == -15.0


def test_marker_with_label_and_position() -> None:
    plan = _plan("mark clip 2 'hook' at 0.5s")
    assert _kinds(plan) == [PlanOpKind.ADD_MARKER]
    args = plan.ops[0].args
    assert args["timeline_item_id"] == "item_1"
    assert args["label"] == "hook"
    assert args["position_seconds"] == 0.5


def test_marker_position_stays_inside_the_item() -> None:
    args = _plan("mark clip 1 'late' at 99s").ops[0].args
    assert args["position_seconds"] < 2.0


def test_move() -> None:
    plan = _plan("move clip 3 to 12s")
    assert _kinds(plan) == [PlanOpKind.MOVE_CLIP]
    assert plan.ops[0].args == {"timeline_item_id": "item_2", "new_position_seconds": 12.0}


def test_delete() -> None:
    plan = _plan("delete clip 2")
    assert _kinds(plan) == [PlanOpKind.DELETE_CLIP]
    assert plan.ops[0].args["timeline_item_id"] == "item_1"


def test_all_clips_targets_every_item() -> None:
    plan = _plan("fade out all clips")
    assert len(plan.ops) == 3


def test_blend_mode() -> None:
    plan = _plan("blend mode screen on clip 1")
    assert _kinds(plan) == [PlanOpKind.SET_COMPOSITE_MODE]
    assert plan.ops[0].args["mode"] == "screen"


def test_clip_numbering_ignores_the_music_bed() -> None:
    """The music also starts at 0.0; "clip 1" must still mean the first picture."""
    plan = _plan("fade in clip 1 0.5s", _state(with_music=True))
    assert plan.ops[0].args["timeline_item_id"] == "item_0"


# --- honest refusal ----------------------------------------------------------------


def test_unparseable_instruction_produces_no_ops() -> None:
    plan = _plan("make it feel dreamier and more cinematic")
    assert plan.ops == []
    assert plan.summary.startswith(UNPARSED_SUMMARY_PREFIX)
    assert "LLM provider" in plan.summary


def test_empty_timeline_is_explained() -> None:
    plan = _plan("fade in the first clip", {"name": "tl", "tracks": []})
    assert plan.ops == []
    assert "empty" in plan.summary
