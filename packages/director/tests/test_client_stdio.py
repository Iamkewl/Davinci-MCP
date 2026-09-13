"""Stdio-client protocol honesty: isError must surface as ToolCallError."""

from __future__ import annotations

from typing import Any

import pytest
from director.mcp_client.client import StdioResolveClient, ToolCallError


class _FakeCallToolResult:
    def __init__(self, *, is_error: bool, text: str) -> None:
        self.isError = is_error
        self.content = [type("T", (), {"text": text})()]
        self.structuredContent = None


class _FakeSession:
    def __init__(self, result: _FakeCallToolResult) -> None:
        self._result = result

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return self._result


async def _exercise(is_error: bool, text: str) -> StdioResolveClient:
    from director.mcp_client.client import StdioResolveClient as C

    client = C(server_command=["dummy", "cmd"])
    client._session = _FakeSession(_FakeCallToolResult(is_error=is_error, text=text))
    return client


@pytest.mark.asyncio
async def test_iserror_raises() -> None:
    client = await _exercise(True, "boom: timeline missing")
    with pytest.raises(ToolCallError) as excinfo:
        await client.call_tool("append_clip", {"media_clip_id": "x"})
    assert "boom" in str(excinfo.value)
    assert excinfo.value.tool_name == "append_clip"


@pytest.mark.asyncio
async def test_iserror_false_returns_payload() -> None:
    client = await _exercise(False, '{"ok": true}')
    out = await client.call_tool("add_marker", {"timeline_item_id": "x"})
    assert out == {"ok": True}


class TestDavinciBackendAcceptance:
    """--backend davinci spawns the resolve-mcp child with --backend davinci."""

    def test_spawn_client_includes_backend_flag(self) -> None:
        import inspect

        from director import cli

        src = inspect.getsource(cli._spawn_client)
        assert "fake" in src and "davinci" in src

    def test_auto_backend_help_mentions_davinci(self) -> None:
        """Typer renders help through Rich, which wraps to the terminal width and
        will hyphen-break an option name on a narrow one — CI defaults to 80
        columns and this passed only on wide local terminals. Ask for a wide
        terminal and compare against text with the ANSI codes stripped."""
        import re

        from director.cli import app
        from typer.testing import CliRunner

        result = CliRunner().invoke(app, ["auto", "--help"], env={"COLUMNS": "200"})
        assert result.exit_code == 0
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--backend" in plain
        assert "davinci" in plain
