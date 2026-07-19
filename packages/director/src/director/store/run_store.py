"""Run store: SQLite + JSONL event log + resumable checkpoints.

Three concerns live here:

1. ``RunStore`` (SQLite) — fast key lookup of runs, plans, tool calls, verdicts.
2. ``EventLog`` (JSONL) — append-only streaming log of every agent decision.
3. ``checkpoint()`` — persist the orchestrator's last agreed state so ``--resume``
   can pick up from the right place.

Schema
------
- ``runs(run_id, mode, status, prompt, created_at, updated_at, ...)``
- ``plans(plan_id, run_id, iteration, plan_json, created_at)``
- ``verdicts(run_id, iteration, verdict_json, created_at)``
- ``tool_calls(run_id, iteration, tool_name, arguments_json, ok, error, recorded_at)``
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from ..schemas import (
    DirectorEvaluation,
    DirectorVerdict,
    OrchestratorEvent,
    Plan,
    RunMode,
    RunRecord,
    RunStatus,
    ToolCallRecord,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    user_prompt TEXT NOT NULL DEFAULT '',
    input_clips_json TEXT NOT NULL DEFAULT '[]',
    input_music TEXT,
    iterations INTEGER NOT NULL DEFAULT 0,
    last_verdict TEXT,
    final_plan_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS plans (
    plan_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    plan_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS verdicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    verdict_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    ok INTEGER NOT NULL,
    error TEXT,
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_plans_run ON plans(run_id);
CREATE INDEX IF NOT EXISTS idx_verdicts_run ON verdicts(run_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_run ON tool_calls(run_id);
"""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class RunStore:
    """SQLite-backed run store.

    Thread-safe via a single RLock. Single-process scope only by design; director
    avoids threading beyond a single orchestrator.
    """

    def __init__(self, path: pathlib.Path) -> None:
        self._path = pathlib.Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # --- Connection ---------------------------------------------------------

    @contextmanager
    def _cur(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            finally:
                cur.close()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- Runs ---------------------------------------------------------------

    def create_run(
        self,
        *,
        mode: RunMode,
        user_prompt: str,
        input_clips: list[str],
        input_music: str | None = None,
        run_id: str | None = None,
    ) -> RunRecord:
        rec = RunRecord(
            run_id=run_id or f"run_{uuid.uuid4().hex[:8]}",
            mode=mode,
            input_clips=input_clips,
            input_music=input_music,
            user_prompt=user_prompt,
            status=RunStatus.PENDING,
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )
        with self._cur() as cur:
            cur.execute(
                "INSERT INTO runs(run_id, mode, status, user_prompt, input_clips_json, "
                "input_music, iterations, last_verdict, final_plan_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 0, NULL, NULL, ?, ?)",
                (
                    rec.run_id,
                    rec.mode.value,
                    rec.status.value,
                    rec.user_prompt,
                    json.dumps(rec.input_clips),
                    rec.input_music,
                    rec.created_at,
                    rec.updated_at,
                ),
            )
        return rec

    def update_run(self, record: RunRecord) -> None:
        record_dict = record.model_copy(update={"updated_at": _now_iso()})
        with self._cur() as cur:
            cur.execute(
                "UPDATE runs SET status=?, iterations=?, last_verdict=?, final_plan_id=?, updated_at=? "
                "WHERE run_id=?",
                (
                    record_dict.status.value,
                    record_dict.iterations,
                    record_dict.last_verdict.value if record_dict.last_verdict else None,
                    record_dict.final_plan_id,
                    record_dict.updated_at,
                    record_dict.run_id,
                ),
            )

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._cur() as cur:
            cur.execute("SELECT * FROM runs WHERE run_id=?", (run_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return RunRecord(
                run_id=row["run_id"],
                mode=RunMode(row["mode"]),
                status=RunStatus(row["status"]),
                user_prompt=row["user_prompt"],
                input_clips=json.loads(row["input_clips_json"]),
                input_music=row["input_music"],
                iterations=row["iterations"],
                last_verdict=DirectorVerdict(row["last_verdict"]) if row["last_verdict"] else None,
                final_plan_id=row["final_plan_id"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    # --- Plans --------------------------------------------------------------

    def save_plan(self, run_id: str, iteration: int, plan: Plan) -> None:
        with self._cur() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO plans(plan_id, run_id, iteration, plan_json, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    plan.plan_id,
                    run_id,
                    iteration,
                    plan.model_dump_json(),
                    _now_iso(),
                ),
            )

    def load_plan(self, plan_id: str) -> Plan | None:
        with self._cur() as cur:
            cur.execute("SELECT plan_json FROM plans WHERE plan_id=?", (plan_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return Plan.model_validate_json(row["plan_json"])

    # --- Verdicts -----------------------------------------------------------

    def record_verdict(self, run_id: str, iteration: int, evaluation: DirectorEvaluation) -> None:
        with self._cur() as cur:
            cur.execute(
                "INSERT INTO verdicts(run_id, iteration, verdict_json, created_at) VALUES (?,?,?,?)",
                (
                    run_id,
                    iteration,
                    evaluation.model_dump_json(),
                    _now_iso(),
                ),
            )

    def list_verdicts(self, run_id: str) -> list[DirectorEvaluation]:
        with self._cur() as cur:
            cur.execute(
                "SELECT verdict_json FROM verdicts WHERE run_id=? ORDER BY iteration ASC",
                (run_id,),
            )
            return [DirectorEvaluation.model_validate_json(r["verdict_json"]) for r in cur.fetchall()]

    # --- Tool calls ---------------------------------------------------------

    def record_tool_call(self, record: ToolCallRecord) -> None:
        with self._cur() as cur:
            cur.execute(
                "INSERT INTO tool_calls(run_id, iteration, tool_name, arguments_json, ok, error, recorded_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (
                    record.run_id,
                    record.iteration,
                    record.tool_name,
                    json.dumps(record.arguments),
                    int(record.ok),
                    record.error,
                    record.recorded_at,
                ),
            )

    def list_tool_calls(self, run_id: str) -> list[ToolCallRecord]:
        with self._cur() as cur:
            cur.execute(
                "SELECT run_id, iteration, tool_name, arguments_json, ok, error, recorded_at "
                "FROM tool_calls WHERE run_id=? ORDER BY id ASC",
                (run_id,),
            )
            out: list[ToolCallRecord] = []
            for r in cur.fetchall():
                out.append(
                    ToolCallRecord(
                        run_id=r["run_id"],
                        iteration=r["iteration"],
                        tool_name=r["tool_name"],
                        arguments=json.loads(r["arguments_json"]),
                        ok=bool(r["ok"]),
                        error=r["error"],
                        recorded_at=r["recorded_at"],
                    )
                )
            return out


# --- Append-only JSONL event log ----------------------------------------------


class EventLog:
    """Append-only JSONL log for orchestrator events.

    Each event is one JSON object per line. We deliberately stay *append-only*
    so the log can be streamed, tailed, or grepped with standard tools.
    """

    def __init__(self, path: pathlib.Path) -> None:
        self._path = pathlib.Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, event: OrchestratorEvent) -> None:
        line = event.model_dump_json() + "\n"
        with self._lock, self._path.open("a", encoding="utf-8") as f:
            f.write(line)

    def read_all(self, *, run_id: str | None = None) -> list[OrchestratorEvent]:
        if not self._path.exists():
            return []
        out: list[OrchestratorEvent] = []
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj: dict[str, Any] = json.loads(line)
                if run_id is not None and obj.get("run_id") != run_id:
                    continue
                out.append(OrchestratorEvent.model_validate(obj))
        return out


__all__ = ["EventLog", "RunStore"]
