"""Conversational REPL: take a natural-language instruction and apply it as a
delta plan to the currently-loaded timeline.

The REPL loads an existing run, reads the current timeline state, and for each
instruction:

1. Calls :meth:`Planner.interpret` for a delta plan (modify / fade / move only).
2. Re-scores the plan via :class:`Director` against the instruction.
3. Executes via :class:`Editor` over the MCP client.
4. Persists verdict + tool calls to the run store and emits JSONL events.

The interactive loop is both:

* callable from tests via :class:`InteractiveSession`,
* navigable from the CLI via ``director interactive``.

Commands supported:
* ``state`` — print current timeline.
* ``tools`` — print registered resolve-mcp tools.
* ``quit`` / ``exit`` — leave.
* anything else is treated as an instruction.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .agents import (
    Director,
    DirectorOutcome,
    Editor,
    EditResult,
    Planner,
)
from .agents.logging_setup import get_logger
from .mcp_client import ResolveClient
from .schemas import (
    DirectorVerdict,
    EventKind,
    OrchestratorEvent,
    Plan,
    RunMode,
    RunStatus,
)
from .settings import DirectorSettings
from .store import EventLog, RunStore

logger = get_logger("director.interactive")


@dataclass
class InterpretResult:
    instruction: str
    plan: Plan
    edit_result: EditResult
    verdict: DirectorOutcome


class InteractiveSession:
    """Manages the state for an interactive REPL session."""

    def __init__(
        self,
        *,
        settings: DirectorSettings,
        client: ResolveClient,
        run_store: RunStore,
        event_log: EventLog,
        planner: Planner,
        director: Director,
        editor: Editor,
        target_project: str,
        target_timeline: str,
        run_id: str | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._run_store = run_store
        self._event_log = event_log
        self._planner = planner
        self._director = director
        self._editor = editor
        self.target_project = target_project
        self.target_timeline = target_timeline
        if run_id is None:
            rec = run_store.create_run(
                mode=RunMode.INTERACTIVE,
                user_prompt="(interactive)",
                input_clips=[],
            )
            run_store.update_run(rec.model_copy(update={"status": RunStatus.RUNNING}))
            self.run_id = rec.run_id
        else:
            self.run_id = run_id

    # ---- state ----------------------------------------------------------------

    async def current_state(self) -> dict[str, Any]:
        return await self._client.call_tool("get_timeline_state", {})

    async def tools_summary(self) -> list[str]:
        return await self._client.list_tools()

    # ---- single-interpret -----------------------------------------------------

    async def interpret(self, instruction: str) -> InterpretResult:
        self._event_log.append(
            OrchestratorEvent(
                run_id=self.run_id,
                iteration=self._next_iteration(),
                kind=EventKind.CHECKPOINT,
                payload={"instruction": instruction},
            )
        )
        state = await self.current_state()
        plan = await self._planner.interpret(
            instruction=instruction,
            timeline_state=state,
            target_project=self.target_project,
            target_timeline=self.target_timeline,
        )
        self._event_log.append(
            OrchestratorEvent(
                run_id=self.run_id,
                iteration=plan.version,
                kind=EventKind.PLAN_COMPILED,
                payload={"plan_id": plan.plan_id, "ops": len(plan.ops)},
            )
        )

        director_outcome = await self._director.run(
            plan=plan,
            user_prompt=instruction,
            beat_count=0,
        )
        self._run_store.record_verdict(self.run_id, plan.version, director_outcome.evaluation)
        self._event_log.append(
            OrchestratorEvent(
                run_id=self.run_id,
                iteration=plan.version,
                kind=EventKind.DIRECTOR_VERDICT,
                payload=director_outcome.evaluation.model_dump(mode="json"),
            )
        )

        edit_result = await self._editor.run(
            run_id=self.run_id,
            plan=plan,
            iteration=plan.version,
        )

        return InterpretResult(
            instruction=instruction,
            plan=plan,
            edit_result=edit_result,
            verdict=director_outcome,
        )

    def _next_iteration(self) -> int:
        rec = self._run_store.get_run(self.run_id)
        return (rec.iterations + 1) if rec is not None else 1


# ---- REPL driver --------------------------------------------------------------


async def run_repl(
    *,
    session: InteractiveSession,
    input_reader: Callable[[], str | None],
    printer: Callable[[str], None],
) -> None:
    """Drive the REPL loop. Compatible with both async CLIs and tests."""

    printer("director interactive (commands: state, tools, quit)")
    while True:
        try:
            line = input_reader()
        except EOFError:
            line = "quit"
        if line is None:
            line = "quit"
        cmd = line.strip()
        if not cmd:
            continue
        if cmd in {"quit", "exit"}:
            printer("bye.")
            return
        if cmd == "state":
            printer(json.dumps(await session.current_state(), indent=2))
            continue
        if cmd == "tools":
            tools = await session.tools_summary()
            printer("\n".join(tools))
            continue
        # Otherwise treat it as an instruction.
        result = await session.interpret(cmd)
        printer(
            json.dumps(
                {
                    "instruction": result.instruction,
                    "plan_ops": [{"kind": op.kind.value, "rationale": op.rationale} for op in result.plan.ops],
                    "verdict": result.verdict.evaluation.verdict.value,
                    "overall": result.verdict.evaluation.overall,
                    "tool_calls": [
                        {"name": c.tool_name, "ok": c.ok, "error": c.error} for c in result.edit_result.tool_calls
                    ],
                    "errors": result.edit_result.errors,
                },
                indent=2,
            )
        )
        # If directing verdict was FAILED, surface and stop the loop.
        if result.verdict.evaluation.verdict == DirectorVerdict.FAILED:
            printer("director verdict = FAILED. stopping.")
            return


__all__ = ["InteractiveSession", "InterpretResult", "run_repl"]
