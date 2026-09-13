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


#: "the second clip" has to mean clip 2. Without this the ordinal fell through
#: to the "no target named" default and silently edited clip 1.
_ORDINALS = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
    "sixth": 6, "6th": 6,
    "seventh": 7, "7th": 7,
    "eighth": 8, "8th": 8,
    "ninth": 9, "9th": 9,
    "tenth": 10, "10th": 10,
}
_ORDINAL_RE = re.compile(r"\b(" + "|".join(_ORDINALS) + r")\b")

#: Phrasing that means "change it relative to what it is now" rather than
#: "set it to this value". "increase opacity by 20%" must not become 0.2.
_RELATIVE_RE = re.compile(r"\b(by|more|less|increase|decrease|reduce|raise|lower|boost|dim)\b")
_DOWNWARD_RE = re.compile(r"\b(decrease|reduce|lower|dim|less|down|out|darker)\b")


@dataclass(frozen=True)
class _Item:
    """One timeline item, flattened out of the timeline state."""

    id: str
    index: int  # 1-based, in timeline order
    track_index: int
    start_seconds: float
    duration_seconds: float
    #: current values, needed to resolve relative adjustments ("by 20%")
    opacity: float = 1.0
    zoom: float = 1.0


@dataclass(frozen=True)
class _Adjust:
    """Either an absolute target value or a change relative to the current one."""

    absolute: float | None = None
    factor: float | None = None  # multiply the current value
    delta: float | None = None  # add to the current value

    def resolve(self, current: float, low: float, high: float) -> float:
        if self.absolute is not None:
            value = self.absolute
        elif self.factor is not None:
            value = current * self.factor
        elif self.delta is not None:
            value = current + self.delta
        else:  # pragma: no cover - constructed with exactly one field set
            value = current
        return max(low, min(high, value))


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
    _FADE_IN = r"fade[- ]?in|fade up"
    _FADE_OUT = r"fade[- ]?out|fade down|fade to black"
    fade_in = _fade_amount(text, _FADE_IN, other=_FADE_OUT)
    fade_out = _fade_amount(text, _FADE_OUT, other=_FADE_IN)
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
        values: list[float] = []
        for it in targets:
            value = opacity.resolve(it.opacity, 0.0, 1.0)
            values.append(value)
            ops.append(
                _op(
                    PlanOpKind.SET_OPACITY,
                    {"timeline_item_id": it.id, "opacity": value},
                    f"set item {it.index} opacity to {value:.2f}",
                )
            )
        return ops, f"set opacity of {_names(targets)} to {_amounts(values, '{:.0%}')}"

    # zoom / punch in
    zoom = _zoom(text)
    if zoom is not None:
        zooms: list[float] = []
        for it in targets:
            value = zoom.resolve(it.zoom, 0.01, 100.0)
            zooms.append(value)
            ops.append(
                _op(
                    PlanOpKind.SET_TRANSFORM,
                    {
                        "timeline_item_id": it.id,
                        "pan_x": 0.0,
                        "pan_y": 0.0,
                        "zoom_x": value,
                        "zoom_y": value,
                        "rotation": 0.0,
                    },
                    f"zoom item {it.index} to {value:.2f}x",
                )
            )
        return ops, f"zoom {_names(targets)} to {_amounts(zooms, '{:.2f}x')}"

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
        transform = item.get("transform")
        zoom = transform.get("zoom_x") if isinstance(transform, dict) else None
        flat.append(
            _Item(
                id=str(item["id"]),
                index=position,
                track_index=track_index,
                start_seconds=float(item.get("start_seconds") or 0.0),
                duration_seconds=float(item.get("duration_seconds") or 0.0),
                opacity=_as_float(item.get("opacity"), 1.0),
                zoom=_as_float(zoom, 1.0),
            )
        )
    return flat


def _as_float(value: Any, default: float) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else default


def _targets(text: str, items: list[_Item]) -> list[_Item]:
    """Which items the instruction refers to; defaults to the first item."""
    if re.search(r"\b(all|every|each)\b", text):
        return items
    if re.search(r"\blast\b", text):
        return items[-1:]
    numbered = re.search(r"\b(?:clip|item|shot)\s*#?\s*(\d+)", text)
    if numbered:
        wanted = int(numbered.group(1))
        return [it for it in items if it.index == wanted] or items[:1]
    ordinal = _ORDINAL_RE.search(text)
    if ordinal:
        wanted = _ORDINALS[ordinal.group(1)]
        return [it for it in items if it.index == wanted] or items[:1]
    return items[:1]


def _amounts(values: list[float], fmt: str) -> str:
    """One value when every item ended up the same, otherwise a range — a
    relative adjustment lands differently on items that started differently."""
    if not values:
        return "nothing"
    low, high = min(values), max(values)
    if abs(high - low) < 1e-9:
        return fmt.format(low)
    return f"{fmt.format(low)}..{fmt.format(high)}"


def _names(items: list[_Item]) -> str:
    if not items:
        return "no items"
    if len(items) == 1:
        return f"clip {items[0].index}"
    return f"{len(items)} clips"


def _fade_amount(text: str, pattern: str, *, other: str) -> float | None:
    """Seconds for one fade direction, or 0.5 when the direction is named
    without a number, or None when it isn't mentioned at all.

    The number search is bounded by the *other* direction's keyword: in
    "fade in then fade out 2s" the 2 belongs to the out-fade only, and an
    unbounded skip would hand it to both.
    """
    found = re.search(pattern, text)
    if not found:
        return None
    rest = text[found.end():]
    boundary = re.search(other, rest)
    if boundary:
        rest = rest[: boundary.start()]
    match = re.match(rf"[^\d]*{_NUMBER}\s*(?:s|sec|secs|seconds)?", rest)
    return float(match.group(1)) if match else 0.5


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


def _is_relative(text: str) -> bool:
    """"increase opacity by 20%" adjusts; "set opacity to 20%" assigns.

    An explicit "to"/"at" wins, because "reduce it to 20%" is an assignment
    despite the "reduce".
    """
    if re.search(rf"\b(?:to|at)\s+{_NUMBER}", text):
        return False
    return bool(_RELATIVE_RE.search(text))


def _percent(text: str, pattern: str) -> _Adjust | None:
    if not re.search(pattern, text):
        return None
    relative = _is_relative(text)
    percent = re.search(rf"{_NUMBER}\s*%", text)
    if percent:
        value = float(percent.group(1)) / 100.0
        if relative:
            return _Adjust(delta=-value if _DOWNWARD_RE.search(text) else value)
        return _Adjust(absolute=value)
    fraction = re.search(rf"(?:to|at)\s+{_NUMBER}\b", text)
    if fraction:
        raw = float(fraction.group(1))
        return _Adjust(absolute=raw if raw <= 1.0 else raw / 100.0)
    return None


def _zoom(text: str) -> _Adjust | None:
    if not re.search(r"\bzoom|punch in|scale\b", text):
        return None
    relative = _is_relative(text)
    percent = re.search(rf"{_NUMBER}\s*%", text)
    if percent:
        value = float(percent.group(1)) / 100.0
        if relative:
            # "zoom in by 20%" -> 1.2x the current zoom; "out by 20%" -> 0.8x.
            return _Adjust(factor=max(0.01, 1.0 - value if _DOWNWARD_RE.search(text) else 1.0 + value))
        return _Adjust(absolute=max(0.01, value))
    multiplier = re.search(rf"{_NUMBER}\s*x\b", text)
    if multiplier:
        return _Adjust(absolute=max(0.01, float(multiplier.group(1))))
    if re.search(r"\bzoom out\b", text):
        return _Adjust(factor=0.9)
    return _Adjust(factor=1.2) if relative else _Adjust(absolute=1.2)


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
