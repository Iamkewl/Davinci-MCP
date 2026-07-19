"""Top-level pipeline: glues Contextualizer → Planner ↔ Director → Editor.

The pipeline is the only thing the CLI talks to. Resumability: a run is
identified by ``run_id``; the store holds the iteration counter so a fresh run
starts at iteration 0 and a resumed run picks up at the next iteration.

Termination is honest:

* APPROVED → status = completed_approved
* ACCEPTED_WITH_WARNINGS → status = completed_with_warnings
* If we hit max_planner_iterations without APPROVED → status = failed and we
  surface the latest verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

from .agents import (
    Contextualizer,
    Director,
    Editor,
    EditResult,
    InvalidModelOutput,
    Planner,
    PlannerRequest,
)
from .ingestion.gemini_client import GeminiClient
from .mcp_client import ResolveClient
from .schemas import (
    DirectorEvaluation,
    DirectorVerdict,
    EventKind,
    OrchestratorEvent,
    Plan,
    RunMode,
    RunStatus,
)
from .settings import DirectorSettings
from .store import EventLog, RunStore


@dataclass
class AutoResult:
    run_id: str
    status: RunStatus
    verdict: DirectorEvaluation | None
    iterations: int
    edit_result: EditResult | None
    plan: Plan | None


class Orchestrator:
    """Wires the agents together around the MCP client + run store."""

    def __init__(
        self,
        *,
        settings: DirectorSettings,
        gemini: GeminiClient | None,
        client: ResolveClient,
        run_store: RunStore,
        event_log: EventLog,
    ) -> None:
        self._settings = settings
        self._gemini = gemini
        self._client = client
        self._run_store = run_store
        self._event_log = event_log
        self._contextualizer = Contextualizer(gemini=gemini, settings=settings)
        self._planner = Planner(gemini=gemini, settings=settings)
        self._director = Director(gemini=gemini, settings=settings)
        self._editor = Editor(
            gemini=None,  # editor doesn't call Gemini
            settings=settings,
            client=client,
            run_store=run_store,
            event_log=event_log,
            allow_destructive=True,  # auto mode does not use destructive ops anyway
        )

    # ---- public API --------------------------------------------------------

    async def run_auto(
        self,
        *,
        clip_paths: list[str],
        music_path: str | None,
        user_prompt: str,
        target_project: str = "auto-reel",
        target_timeline: str = "Timeline 1",
        target_fps: float = 24.0,
    ) -> AutoResult:
        """Run Contextualizer → (Planner ↔ Director)*N → Editor.

        Returns a summary record. Run state is persisted in ``run_store``
        regardless of verdict.
        """
        record = self._run_store.create_run(
            mode=RunMode.AUTO,
            user_prompt=user_prompt,
            input_clips=clip_paths,
            input_music=music_path,
        )
        record_dict = record.model_copy(update={"status": RunStatus.RUNNING})
        self._run_store.update_run(record_dict)
        self._event_log.append(
            OrchestratorEvent(
                run_id=record.run_id,
                iteration=0,
                kind=EventKind.CHECKPOINT,
                payload={"stage": "started"},
            )
        )

        # 1. Contextualize
        ctx = await self._contextualizer.run(clip_paths, music_path)
        music_bpm = ctx.music_analysis.bpm if ctx.music_analysis else None
        beats = (
            ctx.music_analysis.beat_times.tolist() if ctx.music_analysis else []
        )

        verdict: DirectorEvaluation | None = None
        last_plan: Plan | None = None
        edit_result: EditResult | None = None
        for iteration in range(1, self._settings.max_planner_iterations + 1):
            request = PlannerRequest(
                user_prompt=user_prompt,
                per_clip=ctx.per_clip,
                target_project=target_project,
                target_timeline=target_timeline,
                target_fps=target_fps,
                music_bpm=music_bpm,
                beat_times=beats,
                music_duration_seconds=(
                    ctx.music_analysis.duration_seconds if ctx.music_analysis else None
                ),
            )
            try:
                plan = await self._planner.run(request)
            except InvalidModelOutput as err:
                await self._record_error(record.run_id, iteration, str(err))
                break
            self._run_store.save_plan(record.run_id, iteration, plan)
            self._event_log.append(
                OrchestratorEvent(
                    run_id=record.run_id,
                    iteration=iteration,
                    kind=EventKind.PLAN_COMPILED,
                    payload={"plan_id": plan.plan_id, "ops": len(plan.ops)},
                )
            )

            director_outcome = await self._director.run(
                plan=plan,
                user_prompt=user_prompt,
                beat_count=len(beats),
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
            if verdict.verdict == DirectorVerdict.APPROVED:
                last_plan = plan
                # 3. Execute plan via Editor
                edit_result = await self._editor.run(
                    run_id=record.run_id,
                    plan=plan,
                    iteration=iteration,
                )
                break
            elif verdict.verdict == DirectorVerdict.FAILED:
                # Stop iterating; surface the verdict.
                break
            else:
                # ACCEPTED_WITH_WARNINGS: try once more with the verifier feedback
                last_plan = plan
                edit_result = await self._editor.run(
                    run_id=record.run_id,
                    plan=plan,
                    iteration=iteration,
                )
                # If the budget is exhausted on this iteration, we stop.
                if iteration == self._settings.max_planner_iterations:
                    break

        # Final status mapping
        status = self._terminal_status(verdict)
        record_final = record_dict.model_copy(
            update={
                "status": status,
                "final_plan_id": last_plan.plan_id if last_plan else None,
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
        )

    # ---- helpers -----------------------------------------------------------

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
    def _terminal_status(verdict: DirectorEvaluation | None) -> RunStatus:
        if verdict is None:
            return RunStatus.FAILED
        return {
            DirectorVerdict.APPROVED: RunStatus.COMPLETED_APPROVED,
            DirectorVerdict.ACCEPTED_WITH_WARNINGS: RunStatus.COMPLETED_WITH_WARNINGS,
            DirectorVerdict.FAILED: RunStatus.FAILED,
        }[verdict.verdict]


__all__ = ["AutoResult", "Orchestrator"]
