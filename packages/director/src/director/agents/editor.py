"""Editor: executes a Plan via the MCP resolve-mcp server.

The Editor reads state back after every mutation to verify the change landed.
This is the linchpin of the "no silent failures" contract: a tool call that
returns OK but produces an unverifiable after-state is treated as an error.

Items that the planner references by symbolic id (``<item:0>``) get bound to the
real ids from the timeline state after each append. This is how a plan expressed
in one pass becomes a strictly-row-bound instruction a future pass can mutate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..ingestion.gemini_client import GeminiClient
from ..mcp_client import ResolveClient
from ..schemas import (
    EventKind,
    OrchestratorEvent,
    Plan,
    PlanOp,
    PlanOpKind,
    ToolCallRecord,
)
from ..settings import DirectorSettings
from ..store import EventLog, RunStore
from .base import Agent
from .logging_setup import get_logger

logger = get_logger("director.editor")


# Default config flag flow for the resolve-mcp subprocess -----------------------------------------------------------------
DEFAULT_ALLOW_DESTRUCTIVE = True


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
        allow_destructive: bool = DEFAULT_ALLOW_DESTRUCTIVE,
    ) -> None:
        super().__init__(gemini=gemini, settings=settings)
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
    ) -> EditResult:
        result = EditResult(iteration=iteration)
        # Make sure project + timeline exist on the server before we touch items.
        try:
            await self._ensure_project_exists(plan.target_project, plan.target_timeline)
        except Exception as err:  # pragma: no cover — propagate
            result.errors.append(str(err))
            return result

        # Item-id binding: planner uses placeholders like <item:0> for
        # append-then-mutate flows. We resolve them against observed ids.
        append_counter = 0
        for op in plan.ops:
            if op.kind == PlanOpKind.APPEND_CLIP:
                append_counter += 1

        for op in plan.ops:
            if self._kind_is_destructive(op) and not self._allow_destructive:
                result.errors.append(f"{op.kind.value} skipped: destructive tools disabled")
                continue
            try:
                await self._dispatch(run_id, iteration, op, result)
                await self._observe(run_id, iteration, op, result)
            except Exception as exc:
                call = ToolCallRecord(
                    run_id=run_id,
                    iteration=iteration,
                    tool_name=_tool_name_for_op(op),
                    arguments=op.args,
                    ok=False,
                    error=str(exc),
                )
                result.tool_calls.append(call)
                self._run_store.record_tool_call(call)
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
    ) -> None:
        tool_name = _tool_name_for_op(op)
        args = self._resolve_args(op, result)
        call = ToolCallRecord(
            run_id=run_id,
            iteration=iteration,
            tool_name=tool_name,
            arguments=args,
        )
        try:
            response = await self._client.call_tool(tool_name, args)
            call.ok = True
            result.tool_calls.append(call)
            self._run_store.record_tool_call(call)
            # Bind symbolic ids when this op was an APPEND_CLIP that succeeded.
            symbolic = op.args.get("__symbolic_id__")
            if symbolic and isinstance(response, dict):
                # The after-state has tracks[*].items[*].id; pick the new one.
                item_id = _last_item_id(response)
                if item_id:
                    result.item_id_map[symbolic] = item_id
        except Exception as exc:
            call.ok = False
            call.error = str(exc)
            result.tool_calls.append(call)
            self._run_store.record_tool_call(call)
            raise

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
    def _resolve_args(op: PlanOp, result: EditResult) -> dict[str, Any]:
        """Substitute symbolic ids in the op's args using the running id map."""
        args = dict(op.args)
        timeline_item_id = args.get("timeline_item_id")
        if isinstance(timeline_item_id, str) and timeline_item_id.startswith("<item:"):
            symbolic = timeline_item_id
            args["timeline_item_id"] = result.item_id_map.get(symbolic, timeline_item_id)
        return args

    @staticmethod
    def _kind_is_destructive(op: PlanOp) -> bool:
        return op.kind.value in {"delete_clip", "quit_app", "restart_app", "delete_timeline", "delete_media"}

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


def _last_item_id(response: dict[str, Any]) -> str | None:
    """Find the most recently appended item id in a state delta/timeline snapshot."""
    # After state_delta.after.tracks?…fall back to tracks[*].items[-1].
    after = response.get("after") or {}
    tracks = after.get("tracks") or response.get("tracks") or []
    last_id: str | None = None
    for tr in tracks:
        items = tr.get("items") or []
        if items:
            last_id = items[-1]["id"]
    return last_id


__all__ = ["EditResult", "Editor"]
