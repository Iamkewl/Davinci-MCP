"""Regression tests for the stdio client's child-environment handling."""

from __future__ import annotations

from director.mcp_client.client import _build_child_env


def test_child_env_inherits_parent_environment(monkeypatch) -> None:
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/home/u/.venvs/davinci-mcp")
    env = _build_child_env(None)
    assert env["UV_PROJECT_ENVIRONMENT"] == "/home/u/.venvs/davinci-mcp"
    assert "PATH" in env


def test_child_env_overrides_beat_parent(monkeypatch) -> None:
    monkeypatch.setenv("RESOLVE_MCP_LOG_LEVEL", "INFO")
    env = _build_child_env({"RESOLVE_MCP_LOG_LEVEL": "DEBUG"})
    assert env["RESOLVE_MCP_LOG_LEVEL"] == "DEBUG"


def test_child_env_none_extra_is_plain_copy(monkeypatch) -> None:
    monkeypatch.setenv("SOME_VAR", "1")
    env = _build_child_env(None)
    env["SOME_VAR"] = "mutated"
    import os

    assert os.environ["SOME_VAR"] == "1"
