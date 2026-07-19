"""Server-wiring tests against FakeResolveBackend.

FastMCP doesn't make a synchronous in-process call surface easy; we test the
registrations and the destructive gate by inspecting the tool manager directly.
"""

from __future__ import annotations

from resolve_mcp.server import build_server, select_backend

DESTRUCTIVE_TOOLS = {"quit_app", "restart_app", "delete_timeline", "delete_media"}
ALWAYS_PRESENT_TOOLS = {
    "create_project",
    "open_project",
    "import_media",
    "create_timeline",
    "append_clip",
    "set_transform",
    "set_crop",
    "add_marker",
    "add_render_job",
    "start_render",
}


def test_server_without_destructive_omits_destructive_tools() -> None:
    backend = select_backend("fake", allow_destructive=False)
    server = build_server(backend, allow_destructive=False)
    tool_names = set(server._tool_manager._tools.keys())
    assert DESTRUCTIVE_TOOLS.isdisjoint(tool_names)
    for must_present in ALWAYS_PRESENT_TOOLS:
        assert must_present in tool_names


def test_server_with_destructive_registers_all_four() -> None:
    backend = select_backend("fake", allow_destructive=True)
    server = build_server(backend, allow_destructive=True)
    tool_names = set(server._tool_manager._tools.keys())
    for d in DESTRUCTIVE_TOOLS:
        assert d in tool_names


def test_server_resources_are_registered() -> None:
    backend = select_backend("fake", allow_destructive=False)
    server = build_server(backend, allow_destructive=False)
    resources = set(server._resource_manager._resources.keys())
    for needed in ("resolve://project", "resolve://media-pool", "resolve://timeline/current"):
        assert needed in resources
