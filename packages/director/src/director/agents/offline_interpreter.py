"""Offline interpretation of natural-language edit instructions.

The interactive REPL is usable without any LLM key, so a small deterministic
parser turns common instructions into real plan ops. It understands the edits
people actually type at a timeline ("fade in the first clip 1s", "delete clip 2",
"slow clip 1 to 0.5x", "move clip 3 to 12s", "mark clip 1 'hook'") and refuses
honestly when it cannot: an unparsed instruction produces an EMPTY plan whose
summary says so, rather than a plausible-looking edit nobody asked for.

Clip references are 1-based in the order items appear on the timeline, matching
what ``state`` prints; "first"/"last"/"all" work too.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any

from ..schemas import Plan, PlanOp, PlanOpKind

__all__ = ["UNPARSED_SUMMARY_PREFIX", "interpret_offline"]

UNPARSED_SUMMARY_PREFIX = "could not interpret offline"

_NUMBER = r"(\d+(?:\.\d+)?)"


@dataclass(frozen=True)
class _Item:
    """One timeline item, flattened out of the timeline state."""

    id: str
    index: int  # 1-based, in timeline order
    track_index: int
    start_seconds: float
    duration_seconds: float


def interpret_offline(
    *,
    instruction: str,
    timeline_state: dict[str, Any],
    target_project: str,
    target_timeline: str,
) -> Plan:
    """Turn one instruction into a delta plan against the current timeline."""
    text = instruction.strip().lower()
    items = _flatten_items(timeline_state)
    ops: list[PlanOp] = []
    summary = ""

    if not items:
        summary = "timeline is empty — build a cut first (director auto) before refining it"
    else:
        ops, summary = _parse(text, items)

    if not ops and not summary:
        summary = (
            f"{UNPARSED_SUMMARY_PREFIX}: {instruction.strip()!r}. Offline mode understands "
            "fade / opacity / speed / zoom / rotate / marker / move / delete instructions; "
            "configure an LLM provider for free-form direction."
        )
    return Plan(
        plan_id=f"plan_{uuid.uuid4().hex[:8]}",
        version=1,
        target_project=target_project,
        target_timeline=target_timeline,
        ops=ops,
        summary=summary,
    )


# --- parsing ------------------------------------------------------------------


def _parse(text: str, items: list[_Item]) -> tuple[list[PlanOp], str]:
    targets = _targets(text, items)
    ops: list[PlanOp] = []

    # delete
    if re.search(r"\b(delete|remove|drop|cut out)\b", text):
        for it in targets:
            ops.append(_op(PlanOpKind.DELETE_CLIP, {"timeline_item_id": it.id}, f"delete item {it.index}"))
        return ops, f"delete {_names(targets)}"

    # move
    move = re.search(rf"\bmove\b.*?\bto\s+{_NUMBER}\s*(s|sec|secs|seconds)?\b", text)
    if move:
        position = float(move.group(1))
        for it in targets:
            ops.append(
                _op(
                    PlanOpKind.MOVE_CLIP,
                    {"timeline_item_id": it.id, "new_position_seconds": position},
                    f"move item {it.index} to {position}s",
                )
            )
        return ops, f"move {_names(targets)} to {position}s"

    # fades
    fade_in = _fade_amount(text, r"fade[- ]?in|fade up")
    fade_out = _fade_amount(text, r"fade[- ]?out|fade down|fade to black")
    if fade_in is not None or fade_out is not None:
        for it in targets:
            cap = max(0.0, it.duration_seconds / 2.0)
            ops.append(
                _op(
                    PlanOpKind.ADD_FADE,
                    {
                        "timeline_item_id": it.id,
                        "fade_in_seconds": min(fade_in or 0.0, cap),
                        "fade_out_seconds": min(fade_out or 0.0, cap),
                    },
                    f"fade item {it.index}",
                )
            )
        return ops, f"fade {_names(targets)}"

    # speed
    speed = _speed(text)
    if speed is not None:
        for it in targets:
            ops.append(
                _op(
                    PlanOpKind.SET_SPEED,
                    {"timeline_item_id": it.id, "speed": speed},
                    f"set item {it.index} speed to {speed}x",
                )
            )
        return ops, f"set {_names(targets)} to {speed}x"

    # opacity
    opacity = _percent(text, r"opacity|transparen\w*|fade level")
    if opacity is not None:
        for it in targets:
            ops.append(
                _op(
                    PlanOpKind.SET_OPACITY,
                    {"timeline_item_id": it.id, "opacity": opacity},
                    f"set item {it.index} opacity to {opacity:.2f}",
                )
            )
        return ops, f"set opacity of {_names(targets)} to {opacity:.0%}"

    # zoom / punch in
    zoom = _zoom(text)
    if zoom is not None:
        for it in targets:
            ops.append(
                _op(
                    PlanOpKind.SET_TRANSFORM,
                    {
                        "timeline_item_id": it.id,
                        "pan_x": 0.0,
                        "pan_y": 0.0,
                        "zoom_x": zoom,
                        "zoom_y": zoom,
                        "rotation": 0.0,
                    },
                    f"zoom item {it.index} to {zoom:.2f}x",
                )
            )
        return ops, f"zoom {_names(targets)} to {zoom:.2f}x"

    # rotate
    # "rotate clip 2 by -15": the clip number must not be read as the angle.
    rotate = re.search(rf"\brotate\b.*?\bby\s+(-?{_NUMBER})", text) or re.search(
        rf"\brotate\s+(-?{_NUMBER})", text
    )
    if rotate:
        degrees = max(-360.0, min(360.0, float(rotate.group(1))))
        for it in targets:
            ops.append(
                _op(
                    PlanOpKind.SET_TRANSFORM,
                    {
                        "timeline_item_id": it.id,
                        "pan_x": 0.0,
                        "pan_y": 0.0,
                        "zoom_x": 1.0,
                        "zoom_y": 1.0,
                        "rotation": degrees,
                    },
                    f"rotate item {it.index} by {degrees} degrees",
                )
            )
        return ops, f"rotate {_names(targets)} by {degrees} degrees"

    # blend mode
    blend = re.search(
        r"\b(?:blend|composite)(?:\s+mode)?\s+(?:to\s+)?"
        r"(normal|add|subtract|difference|multiply|screen|overlay|hard[ _-]?light|"
        r"soft[ _-]?light|darken|lighten|color[ _-]?dodge|color[ _-]?burn|exclusion|"
        r"hue|saturation|color|luminosity)\b",
        text,
    )
    if blend:
        mode = re.sub(r"[ -]", "_", blend.group(1).strip())
        for it in targets:
            ops.append(
                _op(
                    PlanOpKind.SET_COMPOSITE_MODE,
                    {"timeline_item_id": it.id, "mode": mode},
                    f"set item {it.index} blend mode to {mode}",
                )
            )
        return ops, f"set blend mode of {_names(targets)} to {mode}"

    # marker
    if re.search(r"\b(marker|mark|note|flag)\b", text):
        label = _quoted(text) or "marker"
        at = re.search(rf"\bat\s+{_NUMBER}\s*(s|sec|secs|seconds)?\b", text)
        for it in targets:
            position = float(at.group(1)) if at else 0.0
            position = min(position, max(0.0, it.duration_seconds - 0.001))
            ops.append(
                _op(
                    PlanOpKind.ADD_MARKER,
                    {
                        "timeline_item_id": it.id,
                        "position_seconds": position,
                        "label": label[:30],
                        "color": _marker_color(text),
                        "note": label,
                    },
                    f"mark item {it.index}",
                )
            )
        return ops, f"add marker to {_names(targets)}"

    return [], ""


# --- helpers ------------------------------------------------------------------


def _flatten_items(timeline_state: dict[str, Any]) -> list[_Item]:
    tracks = timeline_state.get("tracks")
    flat: list[_Item] = []
    if not isinstance(tracks, list):
        return flat
    raw: list[tuple[int, dict[str, Any]]] = []
    video_only: list[tuple[int, dict[str, Any]]] = []
    for track in tracks:
        if not isinstance(track, dict):
            continue
        index = track.get("index")
        track_index = int(index) if isinstance(index, int) else 1
        for item in track.get("items") or []:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                raw.append((track_index, item))
                if track.get("kind", "video") == "video":
                    video_only.append((track_index, item))
    # "clip 2" means the second picture clip; the music bed shares start 0.0 and
    # would otherwise steal the numbering.
    if video_only:
        raw = video_only
    raw.sort(key=lambda pair: (float(pair[1].get("start_seconds") or 0.0), pair[0]))
    for position, (track_index, item) in enumerate(raw, start=1):
        flat.append(
            _Item(
                id=str(item["id"]),
                index=position,
                track_index=track_index,
                start_seconds=float(item.get("start_seconds") or 0.0),
                duration_seconds=float(item.get("duration_seconds") or 0.0),
            )
        )
    return flat


def _targets(text: str, items: list[_Item]) -> list[_Item]:
    """Which items the instruction refers to; defaults to the first item."""
    if re.search(r"\b(all|every|each)\b", text):
        return items
    if re.search(r"\blast\b", text):
        return items[-1:]
    if re.search(r"\bfirst\b", text):
        return items[:1]
    numbered = re.search(r"\b(?:clip|item|shot)\s*#?\s*(\d+)", text)
    if numbered:
        wanted = int(numbered.group(1))
        return [it for it in items if it.index == wanted] or items[:1]
    return items[:1]


def _names(items: list[_Item]) -> str:
    if not items:
        return "no items"
    if len(items) == 1:
        return f"clip {items[0].index}"
    return f"{len(items)} clips"


def _fade_amount(text: str, pattern: str) -> float | None:
    match = re.search(rf"(?:{pattern})[^\d]*{_NUMBER}\s*(?:s|sec|secs|seconds)?", text)
    if match:
        return float(match.group(1))
    return 0.5 if re.search(pattern, text) else None


def _speed(text: str) -> float | None:
    explicit = re.search(rf"{_NUMBER}\s*x\b", text)
    if explicit and re.search(r"\b(speed|slow|fast|faster|slower|retime)\b", text):
        return float(explicit.group(1))
    percent = re.search(rf"speed[^\d]*{_NUMBER}\s*%", text)
    if percent:
        return float(percent.group(1)) / 100.0
    if re.search(r"\bslow (?:it |them )?down\b|\bslow motion\b|\bslo-?mo\b", text):
        return 0.5
    if re.search(r"\bspeed (?:it |them )?up\b|\bfaster\b", text):
        return 2.0
    return None


def _percent(text: str, pattern: str) -> float | None:
    if not re.search(pattern, text):
        return None
    percent = re.search(rf"{_NUMBER}\s*%", text)
    if percent:
        return max(0.0, min(1.0, float(percent.group(1)) / 100.0))
    fraction = re.search(rf"(?:to|at)\s+{_NUMBER}\b", text)
    if fraction:
        value = float(fraction.group(1))
        return max(0.0, min(1.0, value if value <= 1.0 else value / 100.0))
    return None


def _zoom(text: str) -> float | None:
    if not re.search(r"\bzoom|punch in|scale\b", text):
        return None
    percent = re.search(rf"{_NUMBER}\s*%", text)
    if percent:
        return max(0.01, float(percent.group(1)) / 100.0)
    multiplier = re.search(rf"{_NUMBER}\s*x\b", text)
    if multiplier:
        return max(0.01, float(multiplier.group(1)))
    if re.search(r"\bzoom out\b", text):
        return 0.9
    return 1.2


def _quoted(text: str) -> str | None:
    match = re.search(r"['\"]([^'\"]+)['\"]", text)
    return match.group(1) if match else None


def _marker_color(text: str) -> str:
    for color in (
        "blue", "cyan", "green", "yellow", "red", "pink", "purple", "fuchsia",
        "rose", "lavender", "sky", "mint", "lemon", "sand", "cocoa", "cream",
    ):
        if re.search(rf"\b{color}\b", text):
            return color
    return "blue"


def _op(kind: PlanOpKind, args: dict[str, Any], rationale: str) -> PlanOp:
    return PlanOp(id=f"op_{uuid.uuid4().hex[:8]}", kind=kind, args=args, rationale=rationale)
