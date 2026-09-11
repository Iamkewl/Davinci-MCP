"""Tests for Orchestrator.resume_run and the `director resume` CLI command."""

from __future__ import annotations

import json
import pathlib

import pytest
from director.mcp_client import StubResolveClient
from director.pipeline import AutoResult, Orchestrator, ResumeError
from director.schemas import RunStatus
from director.settings import DirectorSettings
from director.store import EventLog, RunStore
from resolve_mcp.fake_backend import FakeResolveBackend
from typer.testing import CliRunner


def _orchestrator(
    store: RunStore, log: EventLog, backend: FakeResolveBackend
) -> Orchestrator:
    return Orchestrator(
        settings=DirectorSettings(gemini_api_key=None),
        llm=None,
        client=StubResolveClient(backend),
        run_store=store,
        event_log=log,
    )


def _item_count(backend: FakeResolveBackend) -> int:
    state = backend.get_timeline_state()
    return sum(len(tr.items) for tr in state.tracks)


async def test_resume_reexecutes_last_agreed_plan(
    tmp_run: tuple[RunStore, EventLog, pathlib.Path],
) -> None:
    store, log, _ = tmp_run
    backend = FakeResolveBackend()
    orch = _orchestrator(store, log, backend)

    first = await orch.run_auto(
        clip_paths=["/clips/a.mp4"],
        music_path=None,
        user_prompt="reel",
    )
    assert first.status is RunStatus.COMPLETED_APPROVED
    items_after_first = _item_count(backend)
    assert items_after_first >= 1

    resumed = await orch.resume_run(run_id=first.run_id)
    assert isinstance(resumed, AutoResult)
    assert resumed.run_id == first.run_id
    # The agreed plan was executed again against the same timeline.
    assert _item_count(backend) > items_after_first
    assert resumed.iterations == first.iterations + 1
    record = store.get_run(first.run_id)
    assert record is not None
    assert record.status is RunStatus.COMPLETED_APPROVED


async def test_resume_without_plan_restarts_full_loop(
    tmp_run: tuple[RunStore, EventLog, pathlib.Path],
) -> None:
    store, log, _ = tmp_run
    backend = FakeResolveBackend()
    orch = _orchestrator(store, log, backend)
    result = await orch.run_auto(
        clip_paths=["/clips/a.mp4"], music_path=None, user_prompt="x"
    )
    # Simulate a run that never reached an executed plan.
    rec = store.get_run(result.run_id)
    assert rec is not None

    stripped = rec.model_copy(update={"final_plan_id": None, "status": RunStatus.FAILED})
    store.update_run(stripped)

    restarted = await orch.resume_run(run_id=result.run_id)
    assert restarted.status in (
        RunStatus.COMPLETED_APPROVED,
        RunStatus.COMPLETED_WITH_WARNINGS,
        RunStatus.FAILED,
    )
    assert restarted.verdict is not None or restarted.edit_result is not None


async def test_resume_unknown_run_raises(tmp_run: tuple[RunStore, EventLog, pathlib.Path]) -> None:
    store, log, _ = tmp_run
    orch = _orchestrator(store, log, FakeResolveBackend())
    with pytest.raises(ResumeError):
        await orch.resume_run(run_id="run_missing")


async def test_resume_refuses_in_flight_run(
    tmp_run: tuple[RunStore, EventLog, pathlib.Path],
) -> None:
    store, log, _ = tmp_run
    orch = _orchestrator(store, log, FakeResolveBackend())
    result = await orch.run_auto(
        clip_paths=["/clips/a.mp4"], music_path=None, user_prompt="x"
    )
    rec = store.get_run(result.run_id)
    assert rec is not None
    store.update_run(rec.model_copy(update={"status": RunStatus.RUNNING}))
    with pytest.raises(ResumeError):
        await orch.resume_run(run_id=result.run_id)


def test_cli_resume_command(monkeypatch: pytest.MonkeyPatch, tmp_run: tuple[RunStore, EventLog, pathlib.Path]) -> None:
    """End-to-end CLI: auto into a persistent store, then resume via CliRunner."""
    from director.cli import app

    _store, _log, tmp = tmp_run
    backend = FakeResolveBackend()

    class StubDefault:
        def __init__(self, **_kw: object) -> None:
            self._client = StubResolveClient(backend)

        @classmethod
        def default(cls, **_kw: object) -> StubDefault:
            return cls()

        async def start(self) -> None:
            return None

        async def close(self) -> None:
            return None

        def __getattr__(self, name: str) -> object:
            return getattr(self._client, name)

    monkeypatch.setattr("director.cli.StdioResolveClient", StubDefault)
    monkeypatch.setenv("DIRECTOR_RUN_STORE_DIR", str(tmp))

    runner = CliRunner()
    clips_dir = tmp / "clips"
    clips_dir.mkdir(exist_ok=True)
    (clips_dir / "a.mp4").write_bytes(b"x")

    auto_result = runner.invoke(app, ["auto", str(clips_dir), "--fast"])
    assert auto_result.exit_code == 0, auto_result.output
    run_id = json.loads(auto_result.output)["run_id"]

    resume_result = runner.invoke(app, ["resume", run_id])
    assert resume_result.exit_code == 0, resume_result.output
    payload = json.loads(resume_result.output)
    assert payload["run_id"] == run_id
    assert payload["iterations"] >= 2


def test_cli_resume_unknown_run_fails_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_run: tuple[RunStore, EventLog, pathlib.Path]
) -> None:
    from director.cli import app

    _store, _log, tmp = tmp_run
    monkeypatch.setenv("DIRECTOR_RUN_STORE_DIR", str(tmp))
    runner = CliRunner()
    result = runner.invoke(app, ["resume", "run_nope"])
    assert result.exit_code == 1
