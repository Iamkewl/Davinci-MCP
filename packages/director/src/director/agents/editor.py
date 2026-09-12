"""Editor: executes a Plan via the MCP resolve-mcp server.

The Editor is the only component that mutates Resolve, so it is deliberately
defensive:

* **Bootstrap is idempotent.** A project/timeline that already exists is opened
  and made current instead of being silently skipped — otherwise a second run
  against live Resolve would append into whatever timeline happened to be open.
* **Media is reconciled first.** Plans speak in contextualizer clip ids
  (``clip_<hex>``) and raw source paths; every one of them is mapped to a real
  media-pool id before any append runs, and the music track is imported too.
* **Only tools the server actually offers are dispatched.** The live DaVinci
  backend hides operations Resolve cannot script (fades, retimes, transitions),
  so those ops are skipped with a warning rather than counted as failures.
* **Ids are re-bound as the timeline changes.** ``<item:N>`` symbols bind to the
  id observed after each append, and any ``id_remap`` a backend returns (live
  Resolve recreates items it moves) is applied to every later reference.
* **Nothing is mutated in place.** Plan ops are frozen pydantic models; arg
  resolution always builds fresh dicts, and tool-call records are constructed
  once with their final ok/error values.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..ingestion.gemini_client import GeminiClient
from ..mcp_client import ResolveClient
from ..schemas import (
    EventKind,
    OrchestratorEvent,
    PerClipMap,
    Plan,
    PlanOp,
    PlanOpKind,
    ToolCallRecord,
)
from ..settings import DirectorSettings
from ..store import EventLog, RunStore
from .base import Agent
from .logging_setup import get_logger

if TYPE_CHECKING:
    from ..llm.base import LLMClient

logger = get_logger("director.editor")


# Auto mode never plans destructive ops; the flag exists so a caller can forbid them outright.
DEFAULT_ALLOW_DESTRUCTIVE = True

# Symbolic timeline-item placeholder shared with the planner: "<item:N>".
_SYMBOLIC_ITEM_RE = re.compile(r"^<item:\d+>$")

_DESTRUCTIVE_KINDS = frozenset(
    {"delete_clip", "quit_app", "restart_app", "delete_timeline", "delete_media"}
)


@dataclass
class EditResult:
    """Outcome of an editor run: per-call successes, warnings, and the final state."""

    iteration: int
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    item_id_map: dict[str, str] = field(default_factory=dict)
    # Snapshot of the timeline after applying the plan (if available)
    final_state: dict[str, Any] | None = None

    @property
    def applied_count(self) -> int:
        return sum(1 for call in self.tool_calls if call.ok)


class Editor(Agent[EditResult]):
    """Translate PlanOps into MCP tool calls; verify each edit through its delta."""

    def __init__(
        self,
        *,
        settings: DirectorSettings,
        client: ResolveClient,
        run_store: RunStore,
        event_log: EventLog,
        gemini: GeminiClient | None = None,
        llm: LLMClient | None = None,
        allow_destructive: bool = DEFAULT_ALLOW_DESTRUCTIVE,
    ) -> None:
        super().__init__(gemini=gemini, llm=llm, settings=settings)
        self._client = client
        self._run_store = run_store
        self._event_log = event_log
        self._allow_destructive = allow_destructive
        self._available_tools: frozenset[str] | None = None

    # ---- capability discovery ----------------------------------------------------

    async def available_tools(self) -> frozenset[str]:
        """Tool names the connected server actually offers (cached per session)."""
        if self._available_tools is None:
            try:
                self._available_tools = frozenset(await self._client.list_tools())
            except Exception as exc:  # a server that cannot list tools cannot be filtered
                logger.warning("editor.list_tools_failed", error=str(exc))
                self._available_tools = frozenset()
        return self._available_tools

    async def run(
        self,
        *,
        run_id: str,
        plan: Plan,
        iteration: int,
        per_clip: list[PerClipMap] | None = None,
        extra_media: list[str] | None = None,
        target_fps: float = 24.0,
    ) -> EditResult:
        result = EditResult(iteration=iteration)
        tools = await self.available_tools()
        # Project + timeline must exist and be current before we touch items, and
        # every media source the plan references must be in the pool before appends.
        try:
            await self._ensure_project_and_timeline(
                plan.target_project, plan.target_timeline, target_fps, result
            )
            media_id_map, media_durations = await self._ensure_media_imported(per_clip, extra_media)
        except Exception as err:
            result.errors.append(str(err))
            return result

        # <item:N> binds by 0-based index among APPEND_CLIP ops in plan order.
        append_index = -1
        for op in plan.ops:
            tool_name = _tool_name_for_op(op)
            if self._kind_is_destructive(op) and not self._allow_destructive:
                result.warnings.append(f"{op.kind.value} skipped: destructive tools disabled")
                continue
            if tools and tool_name not in tools:
                result.warnings.append(
                    f"{tool_name} skipped: this backend does not support it "
                    "(Resolve's scripting API has no equivalent)"
                )
                continue
            auto_symbol: str | None = None
            if op.kind == PlanOpKind.APPEND_CLIP:
                append_index += 1
                auto_symbol = f"<item:{append_index}>"
            try:
                await self._dispatch(
                    run_id,
                    iteration,
                    op,
                    result,
                    media_id_map=media_id_map,
                    media_durations=media_durations,
                    auto_symbol=auto_symbol,
                )
            except Exception as exc:
                # One bad op shouldn't abandon the rest of the cut; the director
                # scores the run on what actually landed.
                result.errors.append(f"{tool_name}: {exc}")

        # Final state read-back.
        try:
            result.final_state = await self._client.call_tool("get_timeline_state", {})
        except Exception as exc:
            result.errors.append(f"final state read failed: {exc}")
        self._event_log.append(
            OrchestratorEvent(
                run_id=run_id,
                iteration=iteration,
                kind=EventKind.PLAN_APPLIED if not result.errors else EventKind.ERROR,
                payload={
                    "plan_id": plan.plan_id,
                    "tool_calls": [c.tool_name for c in result.tool_calls],
                    "applied": result.applied_count,
                    "errors": result.errors,
                    "warnings": result.warnings,
                },
            )
        )
        return result

    # ---- dispatch -----------------------------------------------------------------

    async def _dispatch(
        self,
        run_id: str,
        iteration: int,
        op: PlanOp,
        result: EditResult,
        *,
        media_id_map: dict[str, str],
        media_durations: dict[str, float],
        auto_symbol: str | None,
    ) -> None:
        tool_name = _tool_name_for_op(op)
        args = self._resolve_args(op, result, media_id_map)
        if op.kind == PlanOpKind.APPEND_CLIP and auto_symbol is not None:
            args.setdefault("__symbolic_id__", auto_symbol)
        raw_symbolic = args.pop("__symbolic_id__", None)
        symbolic = raw_symbolic if isinstance(raw_symbolic, str) else None
        if op.kind in (PlanOpKind.APPEND_CLIP, PlanOpKind.INSERT_CLIP):
            self._clamp_to_media(args, media_durations, result)
        try:
            response = await self._client.call_tool(tool_name, args)
        except Exception as exc:
            call = ToolCallRecord(
                run_id=run_id,
                iteration=iteration,
                tool_name=tool_name,
                arguments=args,
                ok=False,
                error=str(exc),
            )
            result.tool_calls.append(call)
            self._run_store.record_tool_call(call)
            raise
        call = ToolCallRecord(
            run_id=run_id,
            iteration=iteration,
            tool_name=tool_name,
            arguments=args,
            ok=True,
        )
        result.tool_calls.append(call)
        self._run_store.record_tool_call(call)
        if isinstance(response, dict):
            self._observe(run_id, iteration, op, response, result)
            self._apply_id_remap(response, result)
            if symbolic and op.kind == PlanOpKind.APPEND_CLIP:
                item_id = _bind_appended_item(
                    response,
                    media_clip_id=args.get("media_clip_id"),
                    start_seconds=args.get("start_seconds"),
                )
                if item_id:
                    result.item_id_map[symbolic] = item_id

    def _observe(
        self,
        run_id: str,
        iteration: int,
        op: PlanOp,
        response: dict[str, Any],
        result: EditResult,
    ) -> None:
        """Verify the mutation through its own delta — no extra round trip needed."""
        after = response.get("after")
        changed = response.get("changed_paths")
        if isinstance(changed, list) and not changed:
            result.warnings.append(f"{op.kind.value}: server reported no observable change")
        duration = after.get("duration_seconds") if isinstance(after, dict) else None
        self._event_log.append(
            OrchestratorEvent(
                run_id=run_id,
                iteration=iteration,
                kind=EventKind.TOOL_OBSERVED,
                payload={"plan_op": op.kind.value, "duration_seconds": duration},
            )
        )

    @staticmethod
    def _apply_id_remap(response: dict[str, Any], result: EditResult) -> None:
        """Follow ids the backend had to recreate (live Resolve move/ripple-insert)."""
        remap = response.get("id_remap")
        if not isinstance(remap, dict) or not remap:
            return
        for symbol, real_id in list(result.item_id_map.items()):
            replacement = remap.get(real_id)
            if isinstance(replacement, str):
                result.item_id_map[symbol] = replacement

    # ---- arg resolution ----------------------------------------------------------

    @staticmethod
    def _resolve_args(
        op: PlanOp, result: EditResult, media_id_map: dict[str, str]
    ) -> dict[str, Any]:
        """Rebuild the op's args as a fresh structure with ids resolved.

        The whole tree is rebuilt (nested dicts/lists included) because ops are
        frozen pydantic models and must never be mutated in place. Symbolic
        ``<item:N>`` strings at any depth map through ``item_id_map``; unresolved
        symbols stay literal so the server error surfaces honestly. Clip ids and
        source paths are rewritten through the media-id map.
        """
        args: dict[str, Any] = _substitute_symbols(dict(op.args), result.item_id_map)
        media_clip_id = args.get("media_clip_id")
        if isinstance(media_clip_id, str):
            args["media_clip_id"] = media_id_map.get(media_clip_id, media_clip_id)
        return args

    @staticmethod
    def _clamp_to_media(
        args: dict[str, Any], media_durations: dict[str, float], result: EditResult
    ) -> None:
        """Keep a planned source range inside the media, instead of failing the call."""
        media_id = args.get("media_clip_id")
        duration = args.get("duration_seconds")
        source_in = args.get("source_in_seconds", 0.0)
        if not isinstance(media_id, str) or not isinstance(duration, int | float):
            return
        available = media_durations.get(media_id, 0.0)
        if available <= 0 or not isinstance(source_in, int | float):
            return  # unknown duration: trust the plan, let the server judge
        room = available - float(source_in)
        if room <= 0:
            args["source_in_seconds"] = 0.0
            room = available
            result.warnings.append(
                f"source_in beyond media ({available:.2f}s); starting from 0 instead"
            )
        if float(duration) > room + 1e-6:
            args["duration_seconds"] = round(room, 3)
            result.warnings.append(
                f"clip shortened to {room:.2f}s to fit the source media ({available:.2f}s)"
            )

    @staticmethod
    def _kind_is_destructive(op: PlanOp) -> bool:
        return op.kind.value in _DESTRUCTIVE_KINDS

    # ---- bootstrap ----------------------------------------------------------------

    async def _ensure_project_and_timeline(
        self, project: str, timeline: str, target_fps: float, result: EditResult
    ) -> None:
        """Make ``project``/``timeline`` exist AND be current, or fail loudly.

        On a live Resolve the project and timeline usually already exist (a second
        run, a resume, a user's own project), so "create failed" must fall back to
        "open" rather than being swallowed — otherwise edits land somewhere else.
        """
        await self._ensure_project(project, target_fps, result)
        await self._ensure_timeline(timeline, target_fps)

    async def _ensure_project(self, project: str, target_fps: float, result: EditResult) -> None:
        try:
            await self._client.call_tool(
                "create_project",
                {
                    "name": project,
                    "fps": target_fps,
                    "drop_frame": False,
                    "width": 1920,
                    "height": 1080,
                },
            )
            return
        except Exception as create_error:
            try:
                await self._client.call_tool("open_project", {"name": project})
                return
            except Exception as open_error:
                current = await self._current_project_name()
                if current is not None:
                    result.warnings.append(
                        f"using the already-open project {current!r}: "
                        f"could not create or open {project!r} ({open_error})"
                    )
                    return
                msg = (
                    f"cannot create project {project!r} ({create_error}) "
                    f"nor open it ({open_error})"
                )
                raise RuntimeError(msg) from open_error

    async def _ensure_timeline(self, timeline: str, target_fps: float) -> None:
        try:
            await self._client.call_tool(
                "create_timeline", {"name": timeline, "fps": target_fps, "drop_frame": False}
            )
            return
        except Exception as create_error:
            try:
                await self._client.call_tool("set_current_timeline", {"name": timeline})
                return
            except Exception as select_error:
                msg = (
                    f"cannot create timeline {timeline!r} ({create_error}) "
                    f"nor select it ({select_error})"
                )
                raise RuntimeError(msg) from select_error

    async def _current_project_name(self) -> str | None:
        try:
            info = await self._client.call_tool("get_project_info", {})
        except Exception:
            return None
        name = info.get("name") if isinstance(info, dict) else None
        return name if isinstance(name, str) else None

    async def _ensure_media_imported(
        self,
        per_clip: list[PerClipMap] | None,
        extra_media: list[str] | None = None,
    ) -> tuple[dict[str, str], dict[str, float]]:
        """Make sure every planned source exists in the media pool.

        Returns ``(id_map, durations)`` where ``id_map`` maps BOTH contextualizer
        clip ids and raw source paths to real media-pool ids, and ``durations``
        maps pool ids to the pool's own duration (0.0 when unknown). Idempotent:
        paths already in the pool are never imported twice.
        """
        wanted: list[str] = [c.source_path for c in (per_clip or [])]
        wanted += [path for path in (extra_media or []) if path]
        if not wanted:
            return {}, {}
        pool = await self._pool_by_path()
        missing = [path for path in dict.fromkeys(wanted) if path not in pool]
        if missing:
            await self._client.call_tool("import_media", {"paths": missing})
            pool = await self._pool_by_path()
        id_map: dict[str, str] = {}
        durations: dict[str, float] = {}
        for path, (pool_id, duration) in pool.items():
            id_map[path] = pool_id
            durations[pool_id] = duration
        for clip_ctx in per_clip or []:
            entry = pool.get(clip_ctx.source_path)
            if entry is not None:
                id_map[clip_ctx.clip_id] = entry[0]
                # Prefer the contextualizer's probe when the pool doesn't know.
                if entry[1] <= 0 < clip_ctx.duration_seconds:
                    durations[entry[0]] = clip_ctx.duration_seconds
        return id_map, durations

    async def _pool_by_path(self) -> dict[str, tuple[str, float]]:
        pool = await self._client.call_tool("list_media_pool", {})
        clips = pool.get("clips") or []
        by_path: dict[str, tuple[str, float]] = {}
        if isinstance(clips, list):
            for clip in clips:
                if not isinstance(clip, dict):
                    continue
                path, clip_id = clip.get("path"), clip.get("id")
                if isinstance(path, str) and isinstance(clip_id, str):
                    raw_duration = clip.get("duration_seconds")
                    duration = float(raw_duration) if isinstance(raw_duration, int | float) else 0.0
                    by_path[path] = (clip_id, duration)
        return by_path


def _tool_name_for_op(op: PlanOp) -> str:
    # 1:1 verb mapping; the resolver remains the single source of truth.
    return {
        PlanOpKind.APPEND_CLIP: "append_clip",
        PlanOpKind.INSERT_CLIP: "insert_clip",
        PlanOpKind.MOVE_CLIP: "move_clip",
        PlanOpKind.DELETE_CLIP: "delete_clip",
        PlanOpKind.SET_TRANSFORM: "set_transform",
        PlanOpKind.SET_CROP: "set_crop",
        PlanOpKind.SET_OPACITY: "set_opacity",
        PlanOpKind.SET_COMPOSITE_MODE: "set_composite_mode",
        PlanOpKind.ADD_FADE: "add_fade",
        PlanOpKind.SET_SPEED: "set_speed",
        PlanOpKind.ADD_MARKER: "add_marker",
        PlanOpKind.ADD_TRANSITION: "add_transition",
    }[op.kind]


def _substitute_symbols(value: Any, item_id_map: dict[str, str]) -> Any:
    """Deep-walk ``value`` replacing "<item:N>" strings via ``item_id_map``.

    Dicts and lists are rebuilt at every level, so nested placeholders are
    resolved too while the source structure is left untouched.
    """
    if isinstance(value, str) and _SYMBOLIC_ITEM_RE.match(value):
        return item_id_map.get(value, value)
    if isinstance(value, dict):
        return {k: _substitute_symbols(v, item_id_map) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_symbols(v, item_id_map) for v in value]
    return value


def _bind_appended_item(
    response: dict[str, Any],
    *,
    media_clip_id: Any,
    start_seconds: Any,
) -> str | None:
    """Derive the timeline-item id created by a successful append_clip.

    The response is a StateDelta dump: ``{"before": {...timeline...}, "after":
    {...}, "changed_paths": [...]}``. Match preference:

    1. an after-item whose ``media_clip_id`` and ``start_seconds`` match the
       resolved request values (the backend may snap the position to a frame, so
       the comparison tolerates a frame's worth of rounding);
    2. an id-set difference (ids present in ``after`` but not ``before``);
    3. legacy fallback: the last item of the last non-empty track.
    """
    before_raw = response.get("before")
    after_raw = response.get("after")
    before = before_raw if isinstance(before_raw, dict) else {}
    after = after_raw if isinstance(after_raw, dict) else {}
    tracks_after = _tracks(after) or _tracks(response)
    tracks_before = _tracks(before)
    after_items = [item for track in tracks_after for item in track]
    before_items = [item for track in tracks_before for item in track]

    # Tier 1: match on the resolved request values.
    if isinstance(media_clip_id, str) and isinstance(start_seconds, int | float):
        wanted_start = float(start_seconds)
        for item in after_items:
            item_start = item.get("start_seconds")
            if item.get("media_clip_id") != media_clip_id or not isinstance(item_start, int | float):
                continue
            if math.isclose(float(item_start), wanted_start, rel_tol=0.0, abs_tol=0.05):
                item_id = item.get("id")
                if isinstance(item_id, str):
                    return item_id

    # Tier 2: ids added by this mutation (the newest wins).
    before_ids = {i["id"] for i in before_items if isinstance(i.get("id"), str)}
    new_ids: list[str] = []
    for item in after_items:
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id not in before_ids:
            new_ids.append(item_id)
    if new_ids:
        return new_ids[-1]

    # Tier 3: last non-empty track's last item.
    last_id: str | None = None
    for track in tracks_after:
        if track:
            candidate = track[-1].get("id")
            if isinstance(candidate, str):
                last_id = candidate
    return last_id


def _tracks(state: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Extract per-track item lists from a timeline-state-shaped dict."""
    tracks = state.get("tracks") or []
    if not isinstance(tracks, list):
        return []
    out: list[list[dict[str, Any]]] = []
    for track in tracks:
        items = track.get("items") if isinstance(track, dict) else None
        if isinstance(items, list):
            out.append([item for item in items if isinstance(item, dict)])
    return out


__all__ = ["EditResult", "Editor"]
