"""Tests for the run store + event log + deterministic helpers."""

from __future__ import annotations

import pathlib

from director.schemas import (
    DirectorEvaluation,
    DirectorVerdict,
    EventKind,
    OrchestratorEvent,
    Plan,
    PlanOp,
    PlanOpKind,
    RunMode,
    ToolCallRecord,
)
from director.store import EventLog, RunStore


def test_create_run(tmp_run: tuple[RunStore, EventLog, pathlib.Path]) -> None:
    store, _, _ = tmp_run
    rec = store.create_run(
        mode=RunMode.AUTO,
        user_prompt="hi",
        input_clips=["/x.mp4"],
    )
    assert rec.mode is RunMode.AUTO
    fetched = store.get_run(rec.run_id)
    assert fetched is not None
    assert fetched.status is not None


def test_save_and_load_plan(tmp_run: tuple[RunStore, EventLog, pathlib.Path]) -> None:
    store, _, _ = tmp_run
    rec = store.create_run(mode=RunMode.AUTO, user_prompt="x", input_clips=[])
    plan = Plan(
        plan_id="plan_xyz",
        version=1,
        target_project="p",
        target_timeline="t",
        ops=[
            PlanOp(
                id="op_1",
                kind=PlanOpKind.APPEND_CLIP,
                args={"media_clip_id": "clip_abc", "duration_seconds": 1.0},
            )
        ],
    )
    store.save_plan(rec.run_id, 1, plan)
    reloaded = store.load_plan("plan_xyz")
    assert reloaded is not None
    assert reloaded.ops[0].args["media_clip_id"] == "clip_abc"


def test_record_verdict(tmp_run: tuple[RunStore, EventLog, pathlib.Path]) -> None:
    store, _, _ = tmp_run
    rec = store.create_run(mode=RunMode.AUTO, user_prompt="x", input_clips=[])
    ev = DirectorEvaluation(
        verdict=DirectorVerdict.APPROVED,
        overall=0.9,
    )
    store.record_verdict(rec.run_id, 1, ev)
    verdicts = store.list_verdicts(rec.run_id)
    assert len(verdicts) == 1
    assert verdicts[0].verdict is DirectorVerdict.APPROVED


def test_record_tool_call(tmp_run: tuple[RunStore, EventLog, pathlib.Path]) -> None:
    store, _, _ = tmp_run
    rec = store.create_run(mode=RunMode.AUTO, user_prompt="x", input_clips=[])
    store.record_tool_call(
        ToolCallRecord(
            run_id=rec.run_id,
            iteration=1,
            tool_name="append_clip",
            arguments={"x": 1},
        )
    )
    calls = store.list_tool_calls(rec.run_id)
    assert len(calls) == 1
    assert calls[0].tool_name == "append_clip"


def test_event_log_append_and_read(tmp_path: pathlib.Path) -> None:
    log = EventLog(tmp_path / "e.jsonl")
    log.append(OrchestratorEvent(run_id="r1", iteration=1, kind=EventKind.CHECKPOINT, payload={"a": 1}))
    log.append(OrchestratorEvent(run_id="r2", iteration=1, kind=EventKind.CHECKPOINT, payload={"a": 2}))
    events = log.read_all()
    assert len(events) == 2
    by_id = [e for e in log.read_all(run_id="r1")]
    assert len(by_id) == 1
    assert by_id[0].payload["a"] == 1
