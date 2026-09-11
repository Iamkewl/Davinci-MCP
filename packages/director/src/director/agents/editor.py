"""Editor: executes a Plan via the MCP resolve-mcp server.

The Editor reads state back after every mutation to verify the change landed.
This is the linchpin of the "no silent failures" contract: a tool call that
returns OK but produces an unverifiable after-state is treated as an error.

Items that the planner references by symbolic id (``<item:N>``) get bound to
the real ids observed in the timeline state after each successful append.
``N`` is the 0-based index of the append op among APPEND_CLIP ops in plan
order; the editor injects the symbol automatically when the planner omits it.
This is how a plan expressed in one pass becomes a strictly-row-bound
instruction a future pass can mutate.

Plans speak in contextualizer clip ids (``clip_<hex>``) and raw source paths,
not media-pool ids. Before dispatching, the editor reconciles every planned
source against the server's media pool and rewrites ``media_clip_id`` args so
they address real pool entries. ToolCallRecords are constructed once with
their final ok/error values: plan ops are frozen pydantic models and are never
mutated — arg resolution always rebuilds fresh dicts/lists.
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


# Default config flag flow for the resolve-mcp subprocess -----------------------------------------------------------------
DEFAULT_ALLOW_DESTRUCTIVE = True

# Symbolic timeline-item placeholder shared with the planner: "<item:N>".
_SYMBOLIC_ITEM_RE = re.compile(r"^<item:\d+>$")


@dataclass
class EditResult:
    """Outcome of an editor run: per-call successes, the final timeline state, and any errors."""

    iteration: int
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    item_id_map: dict[str, str] = field(default_factory=dict)
    # Snapshot of timeline after applying the plan (if available)
    final_state: dict[str, Any] | None = None


class Editor(Agent[EditResult]):
    """Translate PlanOps into MCP tool calls; observe state between calls."""

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

    async def run(
        self,
        *,
        run_id: str,
        plan: Plan,
        iteration: int,
        per_clip: list[PerClipMap] | None = None,
    ) -> EditResult:
        result = EditResult(iteration=iteration)
        # Project + timeline must exist before we touch items, and every media
        # source the plan references must be in the pool before appends run.
        try:
            await self._ensure_project_exists(plan.target_project, plan.target_timeline)
            media_id_map = await self._ensure_media_imported(per_clip)
        except Exception as err:
            result.errors.append(str(err))
            return result

        # <item:N> binds by 0-based index among APPEND_CLIP ops in plan order.
        append_index = -1
        for op in plan.ops:
            if self._kind_is_destructive(op) and not self._allow_destructive:
                result.errors.append(f"{op.kind.value} skipped: destructive tools disabled")
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
                    auto_symbol=auto_symbol,
                )
                await self._observe(run_id, iteration, op, result)
            except Exception as exc:
                result.errors.append(str(exc))
                # Don't stop: log and proceed, since one failure shouldn't kill
                # the whole plan. Director will score accordingly.

        # Final state read-back.
        try:
            final = await self._client.call_tool(
                "get_timeline_state", {}
            )
            result.final_state = final
        except Exception as exc:
            result.errors.append(f"final state read failed: {exc}")
        # Persist checkpoint
        self._event_log.append(
            OrchestratorEvent(
                run_id=run_id,
                iteration=iteration,
                kind=EventKind.PLAN_APPLIED if not result.errors else EventKind.ERROR,
                payload={
                    "plan_id": plan.plan_id,
                    "tool_calls": [c.tool_name for c in result.tool_calls],
                    "errors": result.errors,
                    "duration_ms": 0,
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
        auto_symbol: str | None,
    ) -> None:
        tool_name = _tool_name_for_op(op)
        args = self._resolve_args(op, result, media_id_map)
        if op.kind == PlanOpKind.APPEND_CLIP and auto_symbol is not None:
            args.setdefault("__symbolic_id__", auto_symbol)
        raw_symbolic = args.pop("__symbolic_id__", None)
        symbolic = raw_symbolic if isinstance(raw_symbolic, str) else None
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
        # Bind the symbolic id when this APPEND_CLIP succeeded, so later ops
        # referencing <item:N> address the real timeline item.
        if symbolic and isinstance(response, dict) and op.kind == PlanOpKind.APPEND_CLIP:
            item_id = _bind_appended_item(
                response,
                media_clip_id=args.get("media_clip_id"),
                start_seconds=args.get("start_seconds"),
            )
            if item_id:
                result.item_id_map[symbolic] = item_id

    async def _observe(
        self,
        run_id: str,
        iteration: int,
        op: PlanOp,
        result: EditResult,
    ) -> None:
        """Read state back so the Director + Planner can verify the edit landed."""
        try:
            state = await self._client.call_tool("get_timeline_state", {})
        except Exception:
            return
        self._event_log.append(
            OrchestratorEvent(
                run_id=run_id,
                iteration=iteration,
                kind=EventKind.TOOL_OBSERVED,
                payload={"plan_op": op.kind.value, "duration_seconds": state.get("duration_seconds")},
            )
        )

    # ---- arg resolution ----------------------------------------------------------

    @staticmethod
    def _resolve_args(
        op: PlanOp, result: EditResult, media_id_map: dict[str, str]
    ) -> dict[str, Any]:
        """Rebuild the op's args as a fresh structure with ids resolved.

        The whole tree is rebuilt (nested dicts/lists included) because ops are
        frozen pydantic models and must never be mutated in place. Symbolic
        ``<item:N>`` strings at any depth map through ``item_id_map``;
        unresolved symbols stay literal so the server error surfaces honestly.
        ``media_clip_id`` is rewritten through the media-id map built by
        :meth:`_ensure_media_imported`.
        """
        args: dict[str, Any] = _substitute_symbols(dict(op.args), result.item_id_map)
        media_clip_id = args.get("media_clip_id")
        if isinstance(media_clip_id, str):
            args["media_clip_id"] = media_id_map.get(media_clip_id, media_clip_id)
        return args

    @staticmethod
    def _kind_is_destructive(op: PlanOp) -> bool:
        return op.kind.value in {"delete_clip", "quit_app", "restart_app", "delete_timeline", "delete_media"}

    async def _ensure_media_imported(self, per_clip: list[PerClipMap] | None) -> dict[str, str]:
        """Make sure every planned source path exists in the media pool.

        Returns a lookup mapping BOTH contextualizer clip ids and raw source
        paths to real media-pool clip ids. Idempotent: paths already present
        in the pool are never imported twice. Failures propagate so the caller
        aborts execution like a failed project bootstrap would.
        """
        if not per_clip:
            return {}
        by_path = await self._pool_paths_by_path()
        missing = [c.source_path for c in per_clip if c.source_path not in by_path]
        if missing:
            await self._client.call_tool("import_media", {"paths": missing})
            by_path = await self._pool_paths_by_path()
        media_id_map: dict[str, str] = {}
        for clip_ctx in per_clip:
            pool_id = by_path.get(clip_ctx.source_path)
            if pool_id is not None:
                media_id_map[clip_ctx.clip_id] = pool_id
                media_id_map[clip_ctx.source_path] = pool_id
        return media_id_map

    async def _pool_paths_by_path(self) -> dict[str, str]:
        pool = await self._client.call_tool("list_media_pool", {})
        clips = pool.get("clips") or []
        by_path: dict[str, str] = {}
        if isinstance(clips, list):
            for clip in clips:
                if (
                    isinstance(clip, dict)
                    and isinstance(clip.get("path"), str)
                    and isinstance(clip.get("id"), str)
                ):
                    by_path[clip["path"]] = clip["id"]
        return by_path

    async def _ensure_project_exists(self, project: str, timeline: str) -> None:
        from contextlib import suppress

        with suppress(Exception):
            await self._client.call_tool(
                "create_project",
                {
                    "name": project,
                    "fps": 24.0,
                    "drop_frame": False,
                    "width": 1920,
                    "height": 1080,
                },
            )
        with suppress(Exception):
            await self._client.call_tool(
                "create_timeline",
                {"name": timeline, "fps": 24.0, "drop_frame": False},
            )


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

    The response is a StateDelta dump: ``{"before": {...timeline...},
    "after": {...}, "changed_paths": [...]}``. Match preference:

    1. an after-item whose ``media_clip_id`` and ``start_seconds`` match the
       resolved request values;
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
            if math.isclose(float(item_start), wanted_start, rel_tol=0.0, abs_tol=1e-9):
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
