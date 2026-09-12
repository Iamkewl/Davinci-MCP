"""Server-wiring tests against both backends.

FastMCP doesn't make a synchronous in-process call surface easy; we test the
registrations, the advertised JSON schema, the destructive gate and settings
precedence by inspecting the built server and the CLI plumbing directly.
"""

from __future__ import annotations

from typing import Any

import pytest
from resolve_mcp.davinci_backend import DaVinciResolveBackend
from resolve_mcp.logging_setup import configure_logging, get_logger
from resolve_mcp.schemas import CompositeMode, MarkerColor
from resolve_mcp.server import build_server, select_backend, settings_from_argv

DESTRUCTIVE_TOOLS = {"quit_app", "restart_app", "delete_timeline", "delete_media"}
NON_DESTRUCTIVE_TOOLS = {
    "open_project",
    "create_project",
    "list_projects",
    "save_project",
    "get_project_info",
    "import_media",
    "list_media_pool",
    "create_bin",
    "create_timeline",
    "list_timelines",
    "set_current_timeline",
    "get_timeline_state",
    "append_clip",
    "insert_clip",
    "delete_clip",
    "move_clip",
    "set_transform",
    "set_crop",
    "set_composite_mode",
    "set_opacity",
    "add_fade",
    "set_speed",
    "add_marker",
    "add_transition",
    "add_render_job",
    "start_render",
    "get_render_status",
}
# Resolve's documented scripting API cannot do these, so the live backend hides them.
LIVE_UNSUPPORTED = {"add_fade", "set_speed", "add_transition", "restart_app"}


def _tools(server: Any) -> dict[str, Any]:
    return dict(server._tool_manager._tools)


def _schema(server: Any, name: str) -> dict[str, Any]:
    return dict(_tools(server)[name].parameters)


def test_fake_backend_registers_every_tool() -> None:
    server = build_server(select_backend("fake", allow_destructive=True), allow_destructive=True)
    names = set(_tools(server))
    assert names == NON_DESTRUCTIVE_TOOLS | DESTRUCTIVE_TOOLS


def test_server_without_destructive_omits_destructive_tools() -> None:
    server = build_server(select_backend("fake"), allow_destructive=False)
    names = set(_tools(server))
    assert names == NON_DESTRUCTIVE_TOOLS
    assert DESTRUCTIVE_TOOLS.isdisjoint(names)


def test_live_backend_hides_tools_resolve_cannot_do() -> None:
    # Constructing the live backend never needs Resolve to be present.
    backend = DaVinciResolveBackend()
    server = build_server(backend, allow_destructive=True)
    names = set(_tools(server))
    assert LIVE_UNSUPPORTED.isdisjoint(names), "unsupported tools must not be advertised"
    assert names == (NON_DESTRUCTIVE_TOOLS | DESTRUCTIVE_TOOLS) - LIVE_UNSUPPORTED
    assert set(backend.unsupported_tools()) == LIVE_UNSUPPORTED


def test_enum_parameters_are_advertised_with_their_values() -> None:
    """A client must be able to discover legal values without trial and error."""
    server = build_server(select_backend("fake"))
    composite = _schema(server, "set_composite_mode")
    assert set(composite["$defs"]["CompositeMode"]["enum"]) == {m.value for m in CompositeMode}
    marker = _schema(server, "add_marker")
    assert set(marker["$defs"]["MarkerColor"]["enum"]) == {c.value for c in MarkerColor}


def test_numeric_bounds_are_advertised() -> None:
    server = build_server(select_backend("fake"))
    append = _schema(server, "append_clip")["properties"]
    assert append["timeline_track_index"]["minimum"] == 1
    assert append["duration_seconds"]["exclusiveMinimum"] == 0.0
    assert append["start_seconds"]["minimum"] == 0.0
    opacity = _schema(server, "set_opacity")["properties"]["opacity"]
    assert (opacity["minimum"], opacity["maximum"]) == (0.0, 1.0)
    speed = _schema(server, "set_speed")["properties"]["speed"]
    assert speed["exclusiveMinimum"] == 0.0


def test_server_resources_are_registered() -> None:
    server = build_server(select_backend("fake"))
    resources = set(server._resource_manager._resources.keys())
    for needed in ("resolve://project", "resolve://media-pool", "resolve://timeline/current"):
        assert needed in resources


# --- settings precedence ------------------------------------------------------


def test_cli_flags_win_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESOLVE_MCP_BACKEND", "davinci")
    monkeypatch.setenv("RESOLVE_MCP_ALLOW_DESTRUCTIVE", "true")
    settings = settings_from_argv(["--backend", "fake", "--no-allow-destructive"])
    assert settings.backend == "fake"
    assert settings.allow_destructive is False


def test_env_applies_when_flags_are_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESOLVE_MCP_BACKEND", "davinci")
    monkeypatch.setenv("RESOLVE_MCP_ALLOW_DESTRUCTIVE", "true")
    monkeypatch.setenv("RESOLVE_MCP_LOG_LEVEL", "DEBUG")
    settings = settings_from_argv([])
    assert settings.backend == "davinci"
    assert settings.allow_destructive is True
    assert settings.log_level == "DEBUG"


def test_defaults_without_flags_or_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("RESOLVE_MCP_BACKEND", "RESOLVE_MCP_ALLOW_DESTRUCTIVE", "RESOLVE_MCP_LOG_LEVEL"):
        monkeypatch.delenv(var, raising=False)
    settings = settings_from_argv([])
    assert (settings.backend, settings.allow_destructive, settings.transport) == (
        "fake",
        False,
        "stdio",
    )


# --- stdio hygiene ------------------------------------------------------------


def test_logs_never_touch_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """stdout carries JSON-RPC frames; one stray log line breaks every MCP client."""
    configure_logging("INFO")
    get_logger("resolve_mcp.test").info("server.built", backend="FakeResolveBackend")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "server.built" in captured.err
