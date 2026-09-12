"""Conversational REPL: take a natural-language instruction and apply it as a
delta plan to the currently-loaded timeline.

For each instruction the session:

1. Calls :meth:`Planner.interpret` for a delta plan (modify / fade / move only).
   Without an LLM this goes through
   :mod:`director.agents.offline_interpreter`, which parses the common edit
   instructions and returns an EMPTY plan when it doesn't understand — better an
   honest "I didn't get that" than a confident wrong edit.
2. Re-scores the plan via :class:`Director` against the instruction.
3. Executes via :class:`Editor` **only if the verdict isn't FAILED and the plan
   actually contains ops** — a rejected plan is never applied.
4. Persists verdict + tool calls to the run store and emits JSONL events, and
   finalizes the run record on exit so the run doesn't sit in ``running`` forever
   (which would make it unresumable).

Commands supported:
* ``state`` — print the current timeline.
* ``tools`` — print the tools this backend actually offers.
* ``help`` — what the offline interpreter understands.
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
from .agents.director import PlanContext
from .agents.logging_setup import get_logger
from .agents.offline_interpreter import UNPARSED_SUMMARY_PREFIX
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

HELP_TEXT = """\
commands: state | tools | help | quit
offline instructions I understand (an LLM provider unlocks free-form direction):
  fade in the first clip 1s        fade out the last clip
  slow clip 2 to 0.5x              speed up clip 3
  set opacity of clip 1 to 60%     zoom clip 2 to 120%
  rotate clip 1 by 5               blend mode screen on clip 2
  mark clip 1 'hook' at 0.5s       move clip 3 to 12s
  delete clip 4                    ... 'all clips' works too\
"""


@dataclass
class InterpretResult:
    instruction: str
    plan: Plan
    edit_result: EditResult | None
    verdict: DirectorOutcome
    executed: bool = False
    note: str = ""


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
        self._applied_any = False
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
        """Timeline state, or a readable explanation instead of a traceback."""
        try:
            return await self._client.call_tool("get_timeline_state", {})
        except Exception as exc:
            return {"timeline": None, "message": str(exc)}

    async def tools_summary(self) -> list[str]:
        return await self._client.list_tools()

    def finalize(self) -> None:
        """Close the run record so it is inspectable and resumable afterwards."""
        record = self._run_store.get_run(self.run_id)
        if record is None or record.status != RunStatus.RUNNING:
            return
        status = (
            RunStatus.COMPLETED_APPROVED if self._applied_any else RunStatus.COMPLETED_WITH_WARNINGS
        )
        self._run_store.update_run(record.model_copy(update={"status": status}))

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
            context=PlanContext(available_tools=frozenset(await self.tools_summary())),
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

        if not plan.ops:
            note = plan.summary or "nothing to do"
            return InterpretResult(instruction, plan, None, director_outcome, False, note)
        if director_outcome.evaluation.verdict == DirectorVerdict.FAILED:
            return InterpretResult(
                instruction,
                plan,
                None,
                director_outcome,
                False,
                "director rejected this edit; nothing was applied",
            )

        edit_result = await self._editor.run(
            run_id=self.run_id,
            plan=plan,
            iteration=plan.version,
            bootstrap=False,  # refine what is open; never create a project here
        )
        self._applied_any = self._applied_any or edit_result.applied_count > 0
        return InterpretResult(instruction, plan, edit_result, director_outcome, True)

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

    printer(f"director interactive on timeline {session.target_timeline!r}")
    printer("commands: state, tools, help, quit")
    try:
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
            if cmd == "help":
                printer(HELP_TEXT)
                continue
            if cmd == "state":
                printer(json.dumps(await session.current_state(), indent=2))
                continue
            if cmd == "tools":
                printer("\n".join(await session.tools_summary()))
                continue

            result = await session.interpret(cmd)
            printer(json.dumps(_report(result), indent=2))
            # An instruction we could not parse is not a rejected edit — keep the
            # session alive. Only a real plan the director failed stops the loop.
            if result.plan.ops and result.verdict.evaluation.verdict == DirectorVerdict.FAILED:
                printer("director verdict = FAILED. stopping.")
                return
    finally:
        session.finalize()


def _report(result: InterpretResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "instruction": result.instruction,
        "applied": result.executed,
        "plan_ops": [
            {"kind": op.kind.value, "rationale": op.rationale} for op in result.plan.ops
        ],
        "verdict": result.verdict.evaluation.verdict.value,
        "overall": result.verdict.evaluation.overall,
    }
    if result.note:
        payload["note"] = result.note
        if result.note.startswith(UNPARSED_SUMMARY_PREFIX):
            payload["hint"] = "type 'help' to see the instructions offline mode understands"
    if result.edit_result is not None:
        payload["tool_calls"] = [
            {"name": c.tool_name, "ok": c.ok, "error": c.error}
            for c in result.edit_result.tool_calls
        ]
        payload["errors"] = result.edit_result.errors
        payload["warnings"] = result.edit_result.warnings
    return payload


__all__ = ["HELP_TEXT", "InteractiveSession", "InterpretResult", "run_repl"]
