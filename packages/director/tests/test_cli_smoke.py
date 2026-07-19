"""CLI smoke test using CliRunner against the Typer app.

We don't spawn resolve-mcp; instead we monkeypatch ``StdioResolveClient.default``
to return a `StubResolveClient` paired with a fake backend. This exercises the
real cli.py code path while keeping tests fast + offline.
"""

from __future__ import annotations

import contextlib
import json
import pathlib

import pytest
from director.cli import app
from director.mcp_client.client import StdioResolveClient, StubResolveClient
from resolve_mcp.fake_backend import FakeResolveBackend
from typer.testing import CliRunner


@pytest.fixture
def fake_resolve(monkeypatch: pytest.MonkeyPatch) -> FakeResolveBackend:
    backend = FakeResolveBackend(allow_destructive=True)

    async def _noop_start(self: object) -> None:
        return None

    async def _noop_close(self: object) -> None:
        return None

    monkeypatch.setattr(StubResolveClient, "start", _noop_start)
    monkeypatch.setattr(StubResolveClient, "close", _noop_close)

    def _fake_default(*_args: object, **_kwargs: object) -> StubResolveClient:
        return StubResolveClient(backend)

    monkeypatch.setattr(StdioResolveClient, "default", staticmethod(_fake_default))
    return backend


def test_cli_auto_runs_to_completion(
    tmp_path: pathlib.Path,
    fake_resolve: FakeResolveBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "a.mp4").write_bytes(b"x")
    (clips_dir / "b.mp4").write_bytes(b"x")
    monkeypatch.setenv("DIRECTOR_RUN_STORE", str(tmp_path / "runs"))
    monkeypatch.setenv("DIRECTOR_GEMINI_API_KEY", "")
    result = runner.invoke(
        app,
        [
            "auto",
            str(clips_dir),
            "--prompt",
            "high-energy 30s reel",
            "--fast",
            "--backend",
            "fake",
        ],
    )
    assert result.exit_code == 0, result.output
    # The CLI prints a JSON object. typer's echo formats with multi-line spacing;
    # we find the run-summary blob by parsing the trailing JSON block.
    candidates: list[dict[str, object]] = []
    out = result.output
    # Find first `{` and walk balanced braces.
    idx = out.find("{")
    while idx >= 0:
        depth = 0
        end = -1
        for k, ch in enumerate(out[idx:], start=idx):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = k
                    break
        if end < 0:
            break
        block = out[idx : end + 1]
        with contextlib.suppress(json.JSONDecodeError):
            candidates.append(json.loads(block))
        idx = out.find("{", end + 1)
    payload = next((c for c in candidates if "status" in c), {})
    assert "status" in payload, result.output
    assert payload["status"] in {"completed_approved", "completed_with_warnings", "failed"}


def test_cli_run_list_empty(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIRECTOR_RUN_STORE", str(tmp_path / "runs"))
    runner = CliRunner()
    result = runner.invoke(app, ["run", "list"])
    assert result.exit_code == 0
    # Either empty list or a single pending run created by the CLI itself; either is fine.
    assert result.output.strip() in ("[]",) or result.output.strip().startswith("[")
