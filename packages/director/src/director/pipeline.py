"""Top-level pipeline: glues Contextualizer → Planner ↔ Director → Editor.

The pipeline is the only thing the CLI talks to. Resumability: a run is
identified by ``run_id``; the store holds the iteration counter and the context
snapshot so a run can be re-executed later.

The plan/review loop is a real loop: the reviewer's issues and suggestions (plus
any structural problems found by :func:`director.plan_validation.validate_plan`)
are fed back into the next planner iteration, so "keep iterating so the planner
can act on the warnings" is something the code actually does.

Termination is honest:

* APPROVED → the plan executes once via the Editor.
* ACCEPTED_WITH_WARNINGS → keep iterating for a better plan; if the budget runs
  out, the best warned plan is executed (that verdict means "acceptable", and a
  run that reports ``completed_with_warnings`` while building nothing would be a
  lie). FAILED is never executed.
* Status reflects what actually landed: no ops applied → ``failed``; some ops
  applied with errors/warnings → ``completed_with_warnings``; everything applied
  cleanly on an APPROVED plan → ``completed_approved``.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .agents import (
    Contextualizer,
    Director,
    Editor,
    EditResult,
    InvalidModelOutput,
    Planner,
    PlannerRequest,
)
from .agents.director import PlanContext
from .agents.planner import effective_target_duration, parse_target_duration
from .ingestion.gemini_client import GeminiClient
from .mcp_client import ResolveClient
from .plan_validation import validate_plan
from .schemas import (
    DirectorEvaluation,
    DirectorVerdict,
    EventKind,
    OrchestratorEvent,
    PerClipMap,
    Plan,
    RunMode,
    RunStatus,
)
from .settings import DirectorSettings
from .store import EventLog, RunStore

if TYPE_CHECKING:
    from .llm.base import LLMClient

DEFAULT_PROJECT = "auto-reel"


@dataclass
class AutoResult:
    run_id: str
    status: RunStatus
    verdict: DirectorEvaluation | None
    iterations: int
    edit_result: EditResult | None
    plan: Plan | None
    target_timeline: str | None = None
    issues: list[str] = field(default_factory=list)

    @property
    def timeline_summary(self) -> dict[str, float | int]:
        """Item count and duration of the timeline this run produced."""
        state = self.edit_result.final_state if self.edit_result else None
        if not isinstance(state, dict):
            return {"items": 0, "duration_seconds": 0.0}
        items = 0
        for track in state.get("tracks") or []:
            if isinstance(track, dict) and isinstance(track.get("items"), list):
                items += len(track["items"])
        duration = state.get("duration_seconds")
        return {
            "items": items,
            "duration_seconds": float(duration) if isinstance(duration, int | float) else 0.0,
        }


class Orchestrator:
    """Wires the agents together around the MCP client + run store."""

    def __init__(
        self,
        *,
        settings: DirectorSettings,
        gemini: GeminiClient | None = None,
        llm: LLMClient | None = None,
        client: ResolveClient,
        run_store: RunStore,
        event_log: EventLog,
    ) -> None:
        self._settings = settings
        self._llm = llm if llm is not None else gemini
        self._client = client
        self._run_store = run_store
        self._event_log = event_log
        self._contextualizer = Contextualizer(llm=self._llm, settings=settings)
        self._planner = Planner(llm=self._llm, settings=settings)
        self._director = Director(llm=self._llm, settings=settings)
        self._editor = Editor(
            gemini=None,  # the editor never calls a model
            settings=settings,
            client=client,
            run_store=run_store,
            event_log=event_log,
            allow_destructive=False,  # auto mode never plans destructive ops
        )

    # ---- public API --------------------------------------------------------

    async def run_auto(
        self,
        *,
        clip_paths: list[str],
        music_path: str | None,
        user_prompt: str,
        target_project: str = DEFAULT_PROJECT,
        target_timeline: str | None = None,
        target_fps: float | None = None,
    ) -> AutoResult:
        """Run Contextualizer → (Planner ↔ Director)*N → Editor."""
        record = self._run_store.create_run(
            mode=RunMode.AUTO,
            user_prompt=user_prompt,
            input_clips=clip_paths,
            input_music=music_path,
        )
        # A fresh timeline per run: re-running must not append into an existing cut.
        timeline_name = target_timeline or f"Reel {record.run_id[:8]}"
        record_dict = record.model_copy(update={"status": RunStatus.RUNNING})
        self._run_store.update_run(record_dict)
        self._event_log.append(
            OrchestratorEvent(
                run_id=record.run_id,
                iteration=0,
                kind=EventKind.CHECKPOINT,
                payload={"stage": "started", "timeline": timeline_name},
            )
        )

        # 1. Contextualize (snapshot persisted so resume_run can re-execute later)
        ctx = await self._contextualizer.run(clip_paths, music_path)
        self._run_store.save_context(record.run_id, ctx.per_clip)
        music_bpm = ctx.music_analysis.bpm if ctx.music_analysis else None
        beats = ctx.music_analysis.beat_times.tolist() if ctx.music_analysis else []
        music_duration = ctx.music_analysis.duration_seconds if ctx.music_analysis else None
        fps = target_fps if target_fps is not None else _infer_fps(ctx.per_clip)
        tools = await self._editor.available_tools()
        requested_duration = parse_target_duration(user_prompt)
        # Score against what is achievable, not the raw number in the brief: the
        # planner caps the cut at the music's length, so a reviewer holding it to
        # a longer target would fail a correct plan on every iteration.
        target_duration = (
            effective_target_duration(user_prompt, ctx.per_clip, music_duration)
            if (requested_duration is not None or music_duration)
            else None
        )
        capped_note: list[str] = []
        if (
            requested_duration is not None
            and target_duration is not None
            and target_duration < requested_duration - 1e-6
        ):
            capped_note.append(
                f"requested {requested_duration:.0f}s but the music is only "
                f"{music_duration:.1f}s — the cut is capped at {target_duration:.1f}s"
            )

        verdict: DirectorEvaluation | None = None
        last_plan: Plan | None = None
        warned_plan: Plan | None = None
        edit_result: EditResult | None = None
        feedback: list[str] = []
        issues: list[str] = []
        previous_signature: str | None = None
        max_iterations = max(1, self._settings.max_planner_iterations)

        for iteration in range(1, max_iterations + 1):
            request = PlannerRequest(
                user_prompt=user_prompt,
                per_clip=ctx.per_clip,
                target_project=target_project,
                target_timeline=timeline_name,
                target_fps=fps,
                music_bpm=music_bpm,
                beat_times=beats,
                music_duration_seconds=music_duration,
                music_path=music_path,
                available_tools=tools,
                feedback=feedback,
                previous_plan=last_plan,
            )
            try:
                plan = await self._planner.run(request)
            except InvalidModelOutput as err:
                await self._record_error(record.run_id, iteration, str(err))
                issues = [str(err)]
                break
            self._run_store.save_plan(record.run_id, iteration, plan)
            validation_issues = validate_plan(
                plan,
                per_clip=ctx.per_clip,
                available_tools=tools,
                music_path=music_path,
            )
            self._event_log.append(
                OrchestratorEvent(
                    run_id=record.run_id,
                    iteration=iteration,
                    kind=EventKind.PLAN_COMPILED,
                    payload={
                        "plan_id": plan.plan_id,
                        "ops": len(plan.ops),
                        "validation_issues": validation_issues,
                    },
                )
            )

            context = PlanContext(
                beat_times=beats,
                per_clip=ctx.per_clip,
                target_duration_seconds=target_duration,
                music_duration_seconds=music_duration,
                music_path=music_path,
                available_tools=tools,
                validation_issues=validation_issues,
            )
            director_outcome = await self._director.run(
                plan=plan, user_prompt=user_prompt, context=context
            )
            verdict = director_outcome.evaluation
            self._run_store.record_verdict(record.run_id, iteration, verdict)
            self._event_log.append(
                OrchestratorEvent(
                    run_id=record.run_id,
                    iteration=iteration,
                    kind=EventKind.DIRECTOR_VERDICT,
                    payload=verdict.model_dump(mode="json"),
                )
            )
            record_dict = record_dict.model_copy(
                update={"iterations": iteration, "last_verdict": verdict.verdict}
            )
            self._run_store.update_run(record_dict)
            last_plan = plan
            issues = [*validation_issues, *verdict.issues]

            if verdict.verdict == DirectorVerdict.APPROVED and not validation_issues:
                edit_result = await self._execute(record.run_id, plan, iteration, ctx.per_clip,
                                                  music_path, fps)
                break
            if verdict.verdict == DirectorVerdict.FAILED:
                break  # a failed verdict is a hard stop; never execute it
            # ACCEPTED_WITH_WARNINGS (or APPROVED with structural problems): iterate
            # with concrete feedback, and remember the plan in case we run out of budget.
            repeated = previous_signature is not None and previous_signature == _signature(plan)
            previous_signature = _signature(plan)
            warned_plan = plan
            feedback = _merge_feedback(validation_issues, verdict)
            if (iteration == max_iterations or repeated) and warned_plan is not None:
                # Either the budget is gone or the planner has stopped changing its
                # answer; iterating again would just burn time. The verdict says the
                # plan is acceptable-with-warnings, so build it and report them.
                edit_result = await self._execute(
                    record.run_id, warned_plan, iteration, ctx.per_clip, music_path, fps
                )
                break

        status = self._terminal_status(verdict, edit_result)
        executed_plan = last_plan if edit_result is not None else None
        record_final = record_dict.model_copy(
            update={
                "status": status,
                "final_plan_id": executed_plan.plan_id if executed_plan else None,
            }
        )
        self._run_store.update_run(record_final)
        return AutoResult(
            run_id=record.run_id,
            status=status,
            verdict=verdict,
            iterations=record_final.iterations,
            edit_result=edit_result,
            plan=last_plan,
            target_timeline=timeline_name,
            issues=[*capped_note, *issues],
        )

    async def resume_run(self, *, run_id: str) -> AutoResult:
        """Resume a stored run at its last agreed state.

        * If the run has an executed plan, that plan is rebuilt on a fresh
          timeline named "<original> (resume N)", leaving the original cut alone.
        * If no plan was ever executed, the auto loop restarts from the run's
          original inputs.

        Refuses runs that are still in flight.
        """
        record = self._run_store.get_run(run_id)
        if record is None:
            raise ResumeError(f"unknown run_id: {run_id}")
        if record.status in (RunStatus.PENDING, RunStatus.RUNNING):
            raise ResumeError(f"run {run_id} is still {record.status.value}; not resumable")

        plan: Plan | None = None
        if record.final_plan_id:
            plan = self._run_store.load_plan(record.final_plan_id)
        if plan is None:
            return await self.run_auto(
                clip_paths=record.input_clips,
                music_path=record.input_music,
                user_prompt=record.user_prompt,
            )

        iteration = record.iterations + 1
        running = record.model_copy(update={"status": RunStatus.RUNNING})
        self._run_store.update_run(running)
        per_clip = self._run_store.load_context(run_id)
        # Rebuild onto a fresh timeline rather than replaying into the existing one:
        # appending the same cut twice would collide with what is already there, and
        # silently stacking duplicates on top of the user's edit is worse than a copy.
        resumed_timeline = f"{plan.target_timeline} (resume {iteration})"
        plan = plan.model_copy(update={"target_timeline": resumed_timeline})
        edit_result = await self._execute(
            run_id, plan, iteration, per_clip, record.input_music, _infer_fps(per_clip)
        )
        status = self._terminal_status(
            DirectorEvaluation(verdict=DirectorVerdict.APPROVED, overall=1.0), edit_result
        )
        final = running.model_copy(
            update={
                "status": status,
                "iterations": iteration,
                "final_plan_id": plan.plan_id,
            }
        )
        self._run_store.update_run(final)
        return AutoResult(
            run_id=run_id,
            status=status,
            verdict=None,
            iterations=iteration,
            edit_result=edit_result,
            plan=plan,
            target_timeline=resumed_timeline,
            issues=list(edit_result.errors),
        )

    # ---- helpers -----------------------------------------------------------

    async def _execute(
        self,
        run_id: str,
        plan: Plan,
        iteration: int,
        per_clip: list[PerClipMap],
        music_path: str | None,
        fps: float,
    ) -> EditResult:
        return await self._editor.run(
            run_id=run_id,
            plan=plan,
            iteration=iteration,
            per_clip=per_clip,
            extra_media=[music_path] if music_path else None,
            target_fps=fps,
        )

    async def _record_error(self, run_id: str, iteration: int, message: str) -> None:
        self._event_log.append(
            OrchestratorEvent(
                run_id=run_id,
                iteration=iteration,
                kind=EventKind.ERROR,
                payload={"error": message},
            )
        )

    @staticmethod
    def _terminal_status(
        verdict: DirectorEvaluation | None,
        edit_result: EditResult | None,
    ) -> RunStatus:
        if verdict is None or verdict.verdict == DirectorVerdict.FAILED:
            return RunStatus.FAILED
        if edit_result is None:
            # Nothing was executed: never call that "completed".
            return RunStatus.FAILED
        if edit_result.applied_count == 0:
            return RunStatus.FAILED
        if edit_result.errors or edit_result.warnings:
            return RunStatus.COMPLETED_WITH_WARNINGS
        if verdict.verdict == DirectorVerdict.ACCEPTED_WITH_WARNINGS:
            return RunStatus.COMPLETED_WITH_WARNINGS
        return RunStatus.COMPLETED_APPROVED


def _signature(plan: Plan) -> str:
    """Identity of a plan's *content*, so a repeated answer can be spotted."""
    return json.dumps(
        [[op.kind.value, sorted(op.args.items(), key=str)] for op in plan.ops],
        default=str,
        sort_keys=True,
    )


def _merge_feedback(validation_issues: list[str], verdict: DirectorEvaluation) -> list[str]:
    """What the next planner iteration is told to fix."""
    merged: list[str] = []
    for source in (validation_issues, verdict.issues, verdict.suggestions):
        for entry in source:
            if entry and entry not in merged:
                merged.append(entry)
    return merged


def _infer_fps(per_clip: list[PerClipMap]) -> float:
    """Timeline fps = the most common source fps, falling back to 24."""
    rates = Counter(round(c.fps, 3) for c in per_clip if c.fps and c.fps > 0)
    if not rates:
        return 24.0
    return float(rates.most_common(1)[0][0])


class ResumeError(RuntimeError):
    """Raised when a run cannot be resumed."""


__all__ = ["DEFAULT_PROJECT", "AutoResult", "Orchestrator", "ResumeError"]
