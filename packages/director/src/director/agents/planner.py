"""Planner: produce a beat-synced, executable timeline plan.

Offline (no LLM) the planner is fully deterministic and genuinely beat-driven:
cut points ARE beat times taken from the music analysis, segments tile the
timeline without gaps, clips rotate so every one is used, source in-points walk
through each clip so a repeat shows different footage, and the music itself is
laid on the audio track. (The previous implementation placed clips back-to-back
at a running cursor and only mentioned beats in the rationale text.)

With an LLM the same information plus the exact verb/argument specification
(:func:`director.plan_validation.describe_verbs`) goes into the prompt, and the
result is validated before it is returned — a model that invents an argument
name gets one repair attempt, then the run fails honestly rather than half
executing.

Either way the plan is checked with :func:`validate_plan`, so what leaves the
planner is executable.
"""

from __future__ import annotations

import re
import statistics
import uuid
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from ..ingestion.gemini_client import GeminiClient, GeminiError
from ..plan_validation import describe_verbs, validate_plan
from ..schemas import PerClipMap, Plan, PlanOp, PlanOpKind
from ..settings import DirectorSettings
from .base import Agent, InvalidModelOutput
from .offline_interpreter import interpret_offline

if TYPE_CHECKING:
    from ..llm.base import LLMClient

__all__ = ["Planner", "PlannerRequest", "parse_target_duration"]

# Wire convention: track 1 is the first video track, track 2 the first audio track.
VIDEO_TRACK = 1
AUDIO_TRACK = 2

MAX_FADE_SECONDS = 0.5
MIN_SEGMENT_SECONDS = 0.4
DEFAULT_TOTAL_SECONDS = 30.0
MAX_TOTAL_SECONDS = 300.0

_FAST_WORDS = (
    "high-energy", "high energy", "energetic", "fast", "faster", "hype", "punchy",
    "upbeat", "frantic", "action", "workout", "trailer", "tiktok", "reel",
)
_SLOW_WORDS = (
    "moody", "cinematic", "slow", "calm", "dreamy", "emotional", "ambient",
    "chill", "documentary", "gentle", "romantic",
)


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
    music_path: str | None = None
    available_tools: frozenset[str] = frozenset()
    feedback: list[str] = field(default_factory=list)
    previous_plan: Plan | None = None

    def can_use(self, tool: str) -> bool:
        """An empty tool set means "unknown, assume everything is available"."""
        return not self.available_tools or tool in self.available_tools


def parse_target_duration(prompt: str) -> float | None:
    """Pull a requested length out of a brief: "30s reel", "1 minute", "90-second"."""
    text = prompt.lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|\s)?\s*(seconds|second|secs|sec|s)\b", text)
    if match:
        return _clamp_total(float(match.group(1)))
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|\s)?\s*(minutes|minute|mins|min|m)\b", text)
    if match:
        return _clamp_total(float(match.group(1)) * 60.0)
    match = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    if match:
        return _clamp_total(int(match.group(1)) * 60.0 + int(match.group(2)))
    return None


class Planner(Agent[Plan]):
    """Produce an actionable plan, parameterized by a beat map."""

    SYSTEM_TEMPLATE = (
        "You are an edit planner driving a video editor through a fixed set of "
        "tool calls. Return a JSON Plan: an ordered list of ops, each one tool "
        "call.\n\n"
        "Rules:\n"
        "* Times are seconds. Track 1 is the first video track, track 2 the "
        "first audio track.\n"
        "* Cut ON the beat times you are given — a cut that lands between beats "
        "is a defect.\n"
        "* Cover the whole requested length with no gaps and no overlapping "
        "clips on a track.\n"
        "* Never use more of a clip than it actually contains "
        "(source_in + duration <= its duration).\n"
        "* Put the music on the audio track as one append covering the edit.\n"
        "* Label each append with \"__symbolic_id__\": \"<item:N>\" (N counts "
        "appends from 0) and reference those items later by that label.\n"
        "* Use ONLY these verbs and argument names:\n{verbs}\n"
    )

    INTERPRET_SYSTEM = (
        "You are the same edit planner, now modifying an EXISTING timeline.\n"
        "Given the user's instruction and the current timeline state (JSON), "
        "return a Plan whose ops only modify what is already there: move_clip, "
        "delete_clip, set_transform, set_crop, set_opacity, set_composite_mode, "
        "add_fade, set_speed, add_marker, add_transition. Do NOT append or "
        "insert clips. Reference items by their real ``id`` from the state — no "
        "symbolic placeholders. Marker positions are relative to the item start."
    )

    def __init__(
        self,
        *,
        gemini: GeminiClient | None = None,
        llm: LLMClient | None = None,
        settings: DirectorSettings,
    ) -> None:
        super().__init__(gemini=gemini, llm=llm, settings=settings)

    async def run(self, request: PlannerRequest) -> Plan:
        if self._llm is None:
            return build_deterministic_plan(request)
        plan = await self._generate(request, extra_issues=[])
        issues = validate_plan(
            plan,
            per_clip=request.per_clip,
            available_tools=request.available_tools,
            music_path=request.music_path,
        )
        if issues:
            # One repair attempt with the concrete problems spelled out.
            plan = await self._generate(request, extra_issues=issues)
            issues = validate_plan(
                plan,
                per_clip=request.per_clip,
                available_tools=request.available_tools,
                music_path=request.music_path,
            )
            if issues:
                raise InvalidModelOutput(
                    "planner output is still not executable after one repair: "
                    + "; ".join(issues[:6])
                )
        return plan

    async def _generate(self, request: PlannerRequest, *, extra_issues: list[str]) -> Plan:
        assert self._llm is not None
        try:
            return await self._llm.generate_json(
                system=self.SYSTEM_TEMPLATE.format(
                    verbs=describe_verbs(request.available_tools)
                ),
                user=_user_prompt(request, extra_issues),
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
        """Produce a *delta* plan (modify / fade / move) from an NL instruction."""
        if self._llm is None:
            return interpret_offline(
                instruction=instruction,
                timeline_state=timeline_state,
                target_project=target_project,
                target_timeline=target_timeline,
            )
        try:
            return await self._llm.generate_json(
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


# --- deterministic planner --------------------------------------------------------


@dataclass(frozen=True)
class _Tuning:
    """How the reviewer's feedback changes the next attempt."""

    snap_every_beat: bool = False
    rotate_clips: int = 0
    segment_scale: float = 1.0

    @classmethod
    def from_feedback(cls, feedback: list[str]) -> _Tuning:
        text = " ".join(feedback).lower()
        return cls(
            snap_every_beat="beat" in text,
            rotate_clips=1 if ("variety" in text or "same clip" in text) else 0,
            # "too short"/"coverage" -> longer segments; "too long"/"pacing" -> shorter.
            segment_scale=(
                0.75
                if ("pacing" in text or "too long" in text or "faster" in text)
                else 1.25
                if ("coverage" in text or "too short" in text)
                else 1.0
            ),
        )


@dataclass
class _Segment:
    start: float
    duration: float


def build_deterministic_plan(req: PlannerRequest) -> Plan:
    """Beat-synced plan with no model in the loop."""
    tuning = _Tuning.from_feedback(req.feedback)
    beats = sorted({round(float(b), 4) for b in (req.beat_times or []) if b is not None and b >= 0})
    total = _total_length(req)
    segment_target = _segment_target(req.user_prompt, req.music_bpm) * tuning.segment_scale
    segments = _segments(beats, segment_target, total, snap_every_beat=tuning.snap_every_beat)
    assignments = _assign_clips(segments, req.per_clip, rotate=tuning.rotate_clips)

    ops: list[PlanOp] = []
    appended: list[tuple[str, _Segment]] = []  # (symbol, segment) in plan order
    # Derive each shot's length from the NEXT shot's rounded start, so the numbers
    # that leave the planner tile the timeline exactly; rounding start and duration
    # independently leaves millisecond holes between shots.
    edges = [round(segment.start, 3) for segment, *_ in assignments] + [round(total, 3)]
    for index, (segment, clip, source_in, duration) in enumerate(assignments):
        symbol = f"<item:{index}>"
        start = edges[index]
        tiled = round(edges[index + 1] - start, 3)
        # A shot the source could not fill stays short (and the gap is real).
        emitted = tiled if abs(duration - segment.duration) < 1e-6 else round(duration, 3)
        ops.append(
            _op(
                PlanOpKind.APPEND_CLIP,
                {
                    "media_clip_id": clip.clip_id,
                    "timeline_track_index": VIDEO_TRACK,
                    "start_seconds": start,
                    "duration_seconds": emitted,
                    "source_in_seconds": round(source_in, 3),
                    "__symbolic_id__": symbol,
                },
                _cut_rationale(segment, beats),
            )
        )
        appended.append((symbol, _Segment(start, emitted)))

    music_symbol: str | None = None
    if req.music_path and req.can_use("append_clip") and total > 0:
        music_symbol = f"<item:{len(appended)}>"
        music_length = total
        if req.music_duration_seconds:
            music_length = min(total, req.music_duration_seconds)
        ops.append(
            _op(
                PlanOpKind.APPEND_CLIP,
                {
                    "media_clip_id": req.music_path,
                    "timeline_track_index": AUDIO_TRACK,
                    "start_seconds": 0.0,
                    "duration_seconds": round(music_length, 3),
                    "source_in_seconds": 0.0,
                    "__symbolic_id__": music_symbol,
                },
                "Lay the music bed under the whole cut.",
            )
        )

    ops.extend(_polish_ops(req, appended, music_symbol))

    return Plan(
        plan_id=f"plan_{uuid.uuid4().hex[:8]}",
        version=1,
        target_project=req.target_project,
        target_timeline=req.target_timeline,
        ops=ops,
        summary=_summary(req, appended, total, beats),
    )


def _total_length(req: PlannerRequest) -> float:
    """How long the finished cut should be."""
    requested = parse_target_duration(req.user_prompt)
    known = [c.duration_seconds for c in req.per_clip if c.duration_seconds > 0]
    if requested is not None:
        total = requested
    elif req.music_duration_seconds:
        total = req.music_duration_seconds
    elif known:
        total = min(sum(known), DEFAULT_TOTAL_SECONDS * 2)
    else:
        total = DEFAULT_TOTAL_SECONDS
    if req.music_duration_seconds:
        total = min(total, req.music_duration_seconds)
    return _clamp_total(total)


def _segment_target(prompt: str, bpm: float | None) -> float:
    """Average shot length implied by the brief (and the tempo, as a tiebreak)."""
    text = prompt.lower()
    if any(word in text for word in _FAST_WORDS):
        base = 1.5
    elif any(word in text for word in _SLOW_WORDS):
        base = 4.0
    else:
        base = 2.5
    if bpm and bpm > 0:
        bar = 4 * 60.0 / bpm  # one 4/4 bar
        base = max(MIN_SEGMENT_SECONDS, min(base, bar * 2))
    return base


def _segments(
    beats: list[float], segment_target: float, total: float, *, snap_every_beat: bool
) -> list[_Segment]:
    """Tile 0..total with cuts that land on beats wherever beats exist."""
    boundaries = _cut_boundaries(beats, segment_target, total, snap_every_beat=snap_every_beat)
    segments: list[_Segment] = []
    for start, end in pairwise(boundaries):
        duration = round(end - start, 4)
        if duration <= 0:
            continue
        if duration < MIN_SEGMENT_SECONDS and segments:
            # Absorb a sliver into the previous shot rather than dropping it,
            # which would leave a hole in the timeline.
            previous = segments[-1]
            segments[-1] = _Segment(
                start=previous.start, duration=round(previous.duration + duration, 4)
            )
            continue
        segments.append(_Segment(start=round(start, 4), duration=duration))
    if not segments and total > 0:
        segments.append(_Segment(start=0.0, duration=total))
    return segments


def _cut_boundaries(
    beats: list[float], segment_target: float, total: float, *, snap_every_beat: bool
) -> list[float]:
    usable = [b for b in beats if 0 < b < total]
    if not usable:
        # No beat grid: fall back to an even grid of the target shot length.
        count = max(1, round(total / max(segment_target, MIN_SEGMENT_SECONDS)))
        step = total / count
        return [round(i * step, 4) for i in range(count)] + [total]

    interval = statistics.median(
        [b - a for a, b in pairwise(usable)] or [segment_target]
    )
    beats_per_cut = 1 if snap_every_beat else max(1, round(segment_target / max(interval, 1e-6)))
    boundaries = [0.0]
    for index, beat in enumerate(usable):
        if index % beats_per_cut:
            continue
        if beat - boundaries[-1] >= MIN_SEGMENT_SECONDS:
            boundaries.append(beat)
    if total - boundaries[-1] < MIN_SEGMENT_SECONDS and len(boundaries) > 1:
        boundaries.pop()  # absorb a sliver into the previous shot
    boundaries.append(total)
    return boundaries


def _assign_clips(
    segments: list[_Segment], per_clip: list[PerClipMap], *, rotate: int
) -> list[tuple[_Segment, PerClipMap, float, float]]:
    """Round-robin clips over the segments, walking through each clip's footage."""
    if not per_clip:
        return []
    order = list(per_clip[rotate % len(per_clip):]) + list(per_clip[: rotate % len(per_clip)])
    cursors: dict[str, float] = {clip.clip_id: 0.0 for clip in order}
    out: list[tuple[_Segment, PerClipMap, float, float]] = []
    next_index = 0
    for segment in segments:
        chosen: PerClipMap | None = None
        source_in = 0.0
        duration = segment.duration
        for offset in range(len(order)):
            candidate = order[(next_index + offset) % len(order)]
            cursor = cursors[candidate.clip_id]
            available = candidate.duration_seconds
            if available <= 0:
                chosen, source_in = candidate, cursor  # unknown length: trust it
                next_index = (next_index + offset + 1) % len(order)
                break
            if cursor + segment.duration <= available + 1e-6:
                chosen, source_in = candidate, cursor
                next_index = (next_index + offset + 1) % len(order)
                break
            if available + 1e-6 >= segment.duration:
                chosen, source_in = candidate, 0.0  # wrap back to the head
                cursors[candidate.clip_id] = 0.0
                next_index = (next_index + offset + 1) % len(order)
                break
        if chosen is None:
            # Nothing is long enough: use the longest clip and shorten the shot.
            chosen = max(order, key=lambda c: c.duration_seconds)
            source_in = 0.0
            duration = max(MIN_SEGMENT_SECONDS, min(segment.duration, chosen.duration_seconds))
        source_in = _prefer_key_moment(chosen, source_in, duration)
        cursors[chosen.clip_id] = source_in + duration
        out.append((segment, chosen, source_in, duration))
    return out


def _prefer_key_moment(clip: PerClipMap, source_in: float, duration: float) -> float:
    """Start on an interesting moment when the vision pass found one that fits."""
    if not clip.key_moments or clip.duration_seconds <= 0:
        return source_in
    room = clip.duration_seconds - duration
    if room <= 0:
        return 0.0
    for moment in clip.key_moments:
        if source_in <= moment.position_seconds <= room:
            return float(moment.position_seconds)
    return min(source_in, room)


def _polish_ops(
    req: PlannerRequest, appended: list[tuple[str, _Segment]], music_symbol: str | None
) -> list[PlanOp]:
    """Top-and-tail fades (never one per cut — that dips to black on every edit)."""
    ops: list[PlanOp] = []
    if not req.can_use("add_fade") or not appended:
        return ops
    first_symbol, first_segment = appended[0]
    fade_in = min(MAX_FADE_SECONDS, first_segment.duration / 2)
    ops.append(
        _op(
            PlanOpKind.ADD_FADE,
            {
                "timeline_item_id": first_symbol,
                "fade_in_seconds": round(fade_in, 3),
                "fade_out_seconds": 0.0,
            },
            "Ease into the opening shot.",
        )
    )
    last_symbol, last_segment = appended[-1]
    fade_out = min(MAX_FADE_SECONDS, last_segment.duration / 2)
    ops.append(
        _op(
            PlanOpKind.ADD_FADE,
            {
                "timeline_item_id": last_symbol,
                "fade_in_seconds": 0.0,
                "fade_out_seconds": round(fade_out, 3),
            },
            "Fade the final shot out.",
        )
    )
    if music_symbol is not None:
        ops.append(
            _op(
                PlanOpKind.ADD_FADE,
                {
                    "timeline_item_id": music_symbol,
                    "fade_in_seconds": 0.0,
                    "fade_out_seconds": round(min(1.0, MAX_FADE_SECONDS * 2), 3),
                },
                "Fade the music out with the picture.",
            )
        )
    return ops


def _cut_rationale(segment: _Segment, beats: list[float]) -> str:
    if not beats:
        return f"Shot at {segment.start:.2f}s ({segment.duration:.2f}s) on an even grid."
    nearest = min(beats, key=lambda b: abs(b - segment.start))
    if abs(nearest - segment.start) <= 0.02:
        return f"Cut on the beat at {segment.start:.2f}s, holding {segment.duration:.2f}s."
    return f"Shot at {segment.start:.2f}s ({segment.duration:.2f}s); nearest beat {nearest:.2f}s."


def _summary(
    req: PlannerRequest, appended: list[tuple[str, _Segment]], total: float, beats: list[float]
) -> str:
    parts = [
        f"{len(appended)} shot(s) from {len(req.per_clip)} clip(s)",
        f"{total:.1f}s total",
    ]
    if beats:
        parts.append(f"cut to {len(beats)} detected beats")
        if req.music_bpm:
            parts.append(f"{req.music_bpm:.0f} BPM")
    else:
        parts.append("even grid (no beat data)")
    if req.music_path:
        parts.append("music on A1")
    return ", ".join(parts) + "."


def _op(kind: PlanOpKind, args: dict[str, Any], rationale: str) -> PlanOp:
    return PlanOp(id=f"op_{uuid.uuid4().hex[:8]}", kind=kind, args=args, rationale=rationale)


def _clamp_total(seconds: float) -> float:
    return max(MIN_SEGMENT_SECONDS, min(float(seconds), MAX_TOTAL_SECONDS))


# --- prompts ----------------------------------------------------------------------


def _user_prompt(req: PlannerRequest, extra_issues: list[str]) -> str:
    total = _total_length(req)
    parts: list[str] = [
        f"Brief: {req.user_prompt}",
        f"Target length: {total:.1f}s",
        f"Timeline fps: {req.target_fps}",
        f"Project: {req.target_project} / timeline: {req.target_timeline}",
    ]
    if req.music_path:
        parts.append(
            f"Music: {req.music_path} "
            f"({req.music_duration_seconds or 0:.1f}s"
            + (f", {req.music_bpm:.1f} BPM" if req.music_bpm else "")
            + ") — place it on track 2 (audio)."
        )
    beats = [b for b in (req.beat_times or []) if 0 <= b <= total]
    if beats:
        shown = beats if len(beats) <= 200 else beats[:: max(1, len(beats) // 200)]
        parts.append("Beat times (s), cut on these: " + ", ".join(f"{b:.3f}" for b in shown))
    parts.append("Clips:")
    for clip in req.per_clip:
        duration = (
            f"{clip.duration_seconds:.2f}s" if clip.duration_seconds > 0 else "unknown length"
        )
        moments = (
            "; key moments: "
            + ", ".join(f"{m.position_seconds:.1f}s {m.kind}" for m in clip.key_moments[:3])
            if clip.key_moments
            else ""
        )
        parts.append(
            f"- id={clip.clip_id} duration={duration} path={clip.source_path} "
            f"summary={clip.visual_summary or 'n/a'}{moments}"
        )
    if req.previous_plan is not None and (req.feedback or extra_issues):
        parts.append("Your previous plan was rejected:")
        parts.append(req.previous_plan.model_dump_json())
    problems = [*req.feedback, *extra_issues]
    if problems:
        parts.append("Fix these specific problems in the new plan:")
        parts.extend(f"- {problem}" for problem in problems)
    return "\n".join(parts)


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
        f"Current timeline state (JSON):\n{timeline_state!s}"
    )


# Backwards-compatible alias used by existing tests.
_build_deterministic_plan = build_deterministic_plan
