"""Check a plan before it touches Resolve.

A plan is a list of tool calls with free-form argument dicts, which means a
model (or a bug) can produce something that looks plausible and fails halfway
through, leaving a half-built timeline. Validation is therefore a gate: the
orchestrator refuses to execute a plan with issues, and feeds the issues back to
the planner as concrete instructions.

The same :data:`VERB_SPECS` table drives three things — validation, the prompt
text that tells a model exactly which arguments each verb takes, and the
documentation of units — so they cannot drift apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import pairwise

from .schemas import PerClipMap, Plan, PlanOp, PlanOpKind

__all__ = [
    "SYMBOLIC_ITEM_RE",
    "VERB_SPECS",
    "VerbSpec",
    "describe_verbs",
    "validate_plan",
]

SYMBOLIC_ITEM_RE = re.compile(r"^<item:(\d+)>$")

_NUMBER = (int, float)


@dataclass(frozen=True)
class VerbSpec:
    """One plan verb: the MCP tool it maps to and the arguments it accepts."""

    tool: str
    summary: str
    required: dict[str, str] = field(default_factory=dict)
    optional: dict[str, str] = field(default_factory=dict)

    @property
    def all_args(self) -> dict[str, str]:
        return {**self.required, **self.optional}


_ITEM_REF = "timeline item id, or <item:N> referring to the Nth append_clip in this plan"

VERB_SPECS: dict[PlanOpKind, VerbSpec] = {
    PlanOpKind.APPEND_CLIP: VerbSpec(
        tool="append_clip",
        summary="place a clip on a track at a position (fails if it would overlap)",
        required={
            "media_clip_id": "clip id from the context, or the clip's source path",
            "duration_seconds": "length on the timeline, seconds > 0",
        },
        optional={
            "timeline_track_index": "1-based; 1 = first video track, 2 = first audio track (default 1)",
            "start_seconds": "position from the timeline start, seconds >= 0 (default 0)",
            "source_in_seconds": "in-point inside the source clip, seconds >= 0 (default 0)",
            "__symbolic_id__": '"<item:N>" label so later ops can address this item',
        },
    ),
    PlanOpKind.INSERT_CLIP: VerbSpec(
        tool="insert_clip",
        summary="ripple-insert a clip; items at/after the position shift right",
        required={
            "media_clip_id": "clip id or source path",
            "timeline_track_index": "1-based track index",
            "timeline_position_seconds": "insert position in seconds (must be a clip boundary)",
            "duration_seconds": "length in seconds > 0",
        },
        optional={"source_in_seconds": "in-point inside the source, seconds >= 0"},
    ),
    PlanOpKind.MOVE_CLIP: VerbSpec(
        tool="move_clip",
        summary="move an existing item to a new position",
        required={
            "timeline_item_id": _ITEM_REF,
            "new_position_seconds": "new start position in seconds >= 0",
        },
    ),
    PlanOpKind.DELETE_CLIP: VerbSpec(
        tool="delete_clip",
        summary="remove an item from the timeline",
        required={"timeline_item_id": _ITEM_REF},
    ),
    PlanOpKind.SET_TRANSFORM: VerbSpec(
        tool="set_transform",
        summary="pan/zoom/rotate an item (Resolve Inspector units)",
        required={"timeline_item_id": _ITEM_REF},
        optional={
            "pan_x": "pixels from centre (default 0)",
            "pan_y": "pixels from centre (default 0)",
            "zoom_x": "scale multiplier, 1.0 = 100% (default 1.0)",
            "zoom_y": "scale multiplier, 1.0 = 100% (default 1.0)",
            "rotation": "degrees -360..360 (default 0)",
            "anchor_x": "anchor offset in pixels (default 0)",
            "anchor_y": "anchor offset in pixels (default 0)",
        },
    ),
    PlanOpKind.SET_CROP: VerbSpec(
        tool="set_crop",
        summary="crop an item by pixels per edge",
        required={"timeline_item_id": _ITEM_REF},
        optional={
            "left": "pixels >= 0",
            "right": "pixels >= 0",
            "top": "pixels >= 0",
            "bottom": "pixels >= 0",
        },
    ),
    PlanOpKind.SET_OPACITY: VerbSpec(
        tool="set_opacity",
        summary="set item opacity",
        required={"timeline_item_id": _ITEM_REF, "opacity": "0.0..1.0"},
    ),
    PlanOpKind.SET_COMPOSITE_MODE: VerbSpec(
        tool="set_composite_mode",
        summary="set the blend mode",
        required={
            "timeline_item_id": _ITEM_REF,
            "mode": 'e.g. "normal", "screen", "multiply", "add"',
        },
    ),
    PlanOpKind.ADD_FADE: VerbSpec(
        tool="add_fade",
        summary="fade an item in/out (not available on live Resolve)",
        required={"timeline_item_id": _ITEM_REF},
        optional={
            "fade_in_seconds": "seconds >= 0 (default 0)",
            "fade_out_seconds": "seconds >= 0 (default 0)",
        },
    ),
    PlanOpKind.SET_SPEED: VerbSpec(
        tool="set_speed",
        summary="retime an item (not available on live Resolve)",
        required={"timeline_item_id": _ITEM_REF, "speed": "multiplier > 0"},
    ),
    PlanOpKind.ADD_MARKER: VerbSpec(
        tool="add_marker",
        summary="add a marker on an item",
        required={
            "timeline_item_id": _ITEM_REF,
            "position_seconds": "offset from the ITEM's start, seconds >= 0",
            "label": "short marker name",
        },
        optional={
            "color": 'marker colour, e.g. "blue", "red", "cream" (default blue)',
            "note": "longer note",
        },
    ),
    PlanOpKind.ADD_TRANSITION: VerbSpec(
        tool="add_transition",
        summary="attach a transition (not available on live Resolve)",
        required={
            "timeline_item_id": _ITEM_REF,
            "track_index": "1-based index of the item's track",
            "duration_seconds": "seconds > 0",
        },
        optional={
            "style": '"cross_dissolve" | "dip_to_black" | "dip_to_white" | "push_left" | ...',
            "alignment": '"start" | "end" | "mid"',
        },
    ),
}

_POSITIVE_ARGS = {"duration_seconds", "speed"}
_NON_NEGATIVE_ARGS = {
    "start_seconds",
    "source_in_seconds",
    "timeline_position_seconds",
    "new_position_seconds",
    "position_seconds",
    "fade_in_seconds",
    "fade_out_seconds",
    "left",
    "right",
    "top",
    "bottom",
}


def describe_verbs(available_tools: frozenset[str] = frozenset()) -> str:
    """Prompt-ready spec of every usable verb, with exact argument names."""
    lines: list[str] = []
    for kind, spec in VERB_SPECS.items():
        if available_tools and spec.tool not in available_tools:
            continue
        lines.append(f"- {kind.value}: {spec.summary}")
        for name, meaning in spec.required.items():
            lines.append(f"    {name} (required): {meaning}")
        for name, meaning in spec.optional.items():
            lines.append(f"    {name} (optional): {meaning}")
    return "\n".join(lines)


def validate_plan(
    plan: Plan,
    *,
    per_clip: list[PerClipMap] | None = None,
    available_tools: frozenset[str] = frozenset(),
    music_path: str | None = None,
) -> list[str]:
    """Return every reason this plan would not execute cleanly.

    An empty list means "safe to run": the ops name real tools, carry the right
    arguments, reference media and items that exist, and do not collide on the
    timeline.
    """
    issues: list[str] = []
    known_media = _known_media_ids(per_clip, music_path)
    append_count = sum(1 for op in plan.ops if op.kind == PlanOpKind.APPEND_CLIP)
    placements: list[tuple[int, float, float, str]] = []
    append_index = -1

    if not plan.ops:
        return ["plan contains no operations"]

    for position, op in enumerate(plan.ops, start=1):
        spec = VERB_SPECS.get(op.kind)
        label = f"op {position} ({op.kind.value})"
        if spec is None:
            issues.append(f"{label}: unknown verb")
            continue
        if available_tools and spec.tool not in available_tools:
            issues.append(
                f"{label}: tool {spec.tool!r} is not available on this backend; "
                f"usable verbs are {sorted(_usable_kinds(available_tools))}"
            )
            continue
        if op.kind == PlanOpKind.APPEND_CLIP:
            append_index += 1

        issues.extend(_check_args(label, op, spec))
        issues.extend(_check_media(label, op, known_media))
        issues.extend(_check_item_refs(label, op, append_count))
        issues.extend(_check_ranges(label, op, per_clip, known_media))

        if op.kind in (PlanOpKind.APPEND_CLIP, PlanOpKind.INSERT_CLIP):
            track, start, duration = _placement(op)
            if duration > 0:
                placements.append((track, start, duration, label))

    issues.extend(_check_overlaps(placements))
    return issues


# --- individual checks ------------------------------------------------------------


def _usable_kinds(available_tools: frozenset[str]) -> set[str]:
    return {k.value for k, spec in VERB_SPECS.items() if spec.tool in available_tools}


def _check_args(label: str, op: PlanOp, spec: VerbSpec) -> list[str]:
    issues: list[str] = []
    known = set(spec.all_args)
    for name in op.args:
        if name not in known:
            hint = _closest(name, known)
            suffix = f" (did you mean {hint!r}?)" if hint else ""
            issues.append(f"{label}: unknown argument {name!r}{suffix}")
    for name in spec.required:
        if name == "__symbolic_id__":
            continue
        if op.args.get(name) is None:
            issues.append(f"{label}: missing required argument {name!r} ({spec.required[name]})")
    for name, value in op.args.items():
        numeric = isinstance(value, _NUMBER) and not isinstance(value, bool)
        if name in _POSITIVE_ARGS and (not numeric or float(value) <= 0):
            issues.append(f"{label}: {name} must be a number > 0, got {value!r}")
        elif name in _NON_NEGATIVE_ARGS and (not numeric or float(value) < 0):
            issues.append(f"{label}: {name} must be a number >= 0, got {value!r}")
    track = op.args.get("timeline_track_index", op.args.get("track_index"))
    if track is not None and (not isinstance(track, int) or isinstance(track, bool) or track < 1):
        issues.append(f"{label}: track index is 1-based, got {track!r}")
    opacity = op.args.get("opacity")
    if opacity is not None and (not isinstance(opacity, _NUMBER) or not 0.0 <= float(opacity) <= 1.0):
        issues.append(f"{label}: opacity must be within 0.0..1.0, got {opacity!r}")
    return issues


def _check_media(label: str, op: PlanOp, known_media: set[str]) -> list[str]:
    media_id = op.args.get("media_clip_id")
    if media_id is None or not known_media:
        return []
    if not isinstance(media_id, str) or media_id not in known_media:
        return [f"{label}: media_clip_id {media_id!r} is not one of the run's clips"]
    return []


def _check_item_refs(label: str, op: PlanOp, append_count: int) -> list[str]:
    issues: list[str] = []
    for name in ("timeline_item_id",):
        value = op.args.get(name)
        if not isinstance(value, str):
            continue
        match = SYMBOLIC_ITEM_RE.match(value)
        if match and int(match.group(1)) >= append_count:
            issues.append(
                f"{label}: {value} refers to append #{int(match.group(1)) + 1} "
                f"but the plan only appends {append_count} clip(s)"
            )
    return issues


def _check_ranges(
    label: str,
    op: PlanOp,
    per_clip: list[PerClipMap] | None,
    known_media: set[str],
) -> list[str]:
    if op.kind not in (PlanOpKind.APPEND_CLIP, PlanOpKind.INSERT_CLIP) or not per_clip:
        return []
    media_id = op.args.get("media_clip_id")
    duration = op.args.get("duration_seconds")
    source_in = op.args.get("source_in_seconds", 0.0)
    if not isinstance(media_id, str) or not isinstance(duration, _NUMBER):
        return []
    clip = next(
        (c for c in per_clip if media_id in (c.clip_id, c.source_path)),
        None,
    )
    if clip is None or clip.duration_seconds <= 0:
        return []  # unknown length: the editor clamps at execution time
    if not isinstance(source_in, _NUMBER):
        return []
    end = float(source_in) + float(duration)
    if end > clip.duration_seconds + 1e-6:
        return [
            f"{label}: uses {source_in:.2f}..{end:.2f}s of a "
            f"{clip.duration_seconds:.2f}s clip ({clip.source_path})"
        ]
    return []


def _placement(op: PlanOp) -> tuple[int, float, float]:
    track = op.args.get("timeline_track_index", 1)
    start = op.args.get(
        "start_seconds", op.args.get("timeline_position_seconds", 0.0)
    )
    duration = op.args.get("duration_seconds", 0.0)
    return (
        track if isinstance(track, int) and not isinstance(track, bool) else 1,
        float(start) if isinstance(start, _NUMBER) else 0.0,
        float(duration) if isinstance(duration, _NUMBER) else 0.0,
    )


def _check_overlaps(placements: list[tuple[int, float, float, str]]) -> list[str]:
    issues: list[str] = []
    by_track: dict[int, list[tuple[float, float, str]]] = {}
    for track, start, duration, label in placements:
        by_track.setdefault(track, []).append((start, duration, label))
    for track, entries in by_track.items():
        ordered = sorted(entries)
        for (start_a, dur_a, label_a), (start_b, _dur_b, label_b) in pairwise(ordered):
            if start_b < start_a + dur_a - 1e-6:
                issues.append(
                    f"{label_a} and {label_b} overlap on track {track} "
                    f"({start_a:.2f}..{start_a + dur_a:.2f}s vs {start_b:.2f}s)"
                )
    return issues


def _known_media_ids(per_clip: list[PerClipMap] | None, music_path: str | None) -> set[str]:
    known: set[str] = set()
    for clip in per_clip or []:
        known.add(clip.clip_id)
        known.add(clip.source_path)
    if music_path:
        known.add(music_path)
    return known


def _closest(name: str, candidates: set[str]) -> str | None:
    """Cheap typo hint: same prefix or a one-word overlap."""
    lowered = name.lower()
    for candidate in sorted(candidates):
        if candidate.lower().startswith(lowered[:4]) or lowered.startswith(candidate.lower()[:4]):
            return candidate
    return None
