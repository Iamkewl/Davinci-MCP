"""MCP client: thin async wrapper that talks to ``resolve-mcp`` over stdio.

We use the MCP Python SDK's :class:`stdio_client` to spawn ``resolve-mcp`` as a
subprocess. The session exposes only the surface the Editor needs:

* ``call_tool(name, args)`` — returns the JSON-shaped payload the tool produced.
* ``read_resource(uri)`` — returns the resource text.
* ``close()`` — terminates the subprocess and the session.

For tests we substitute :class:`StubResolveClient` which talks directly to a
:class:`FakeResolveBackend` instance — no subprocess at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class ResolveClient(ABC):
    """Abstract client the Editor depends on."""

    @abstractmethod
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Invoke a tool and return its JSON-shaped result payload."""

    @abstractmethod
    async def read_resource(self, uri: str) -> str:
        """Return the resource at ``uri``."""

    @abstractmethod
    async def list_tools(self) -> list[str]:
        """Return the names of registered tools (used at startup)."""

    @abstractmethod
    async def close(self) -> None: ...


# --- Stdio client -----------------------------------------------------------------


@dataclass
class StdioResolveClient(ResolveClient):
    """Connect to resolve-mcp over stdio."""

    server_command: list[str]
    server_env: dict[str, str] | None = None

    _session: ClientSession | None = None
    _cm: Any = None
    _session_cm: Any = None

    @classmethod
    def default(
        cls,
        *,
        backend: str = "fake",
        allow_destructive: bool = False,
        log_level: str = "WARNING",
        uv_project: str | None = None,
    ) -> StdioResolveClient:
        """Default constructor: launches ``uv run resolve-mcp --backend X …``.

        ``uv_project`` lets tests/dev override the working directory of the
        workspace so the server can find the resolve-mcp package on disk.
        """
        cmd: list[str] = [
            "uv",
            "run",
            "--project",
            uv_project or ".",
            "resolve-mcp",
            "--backend",
            backend,
            "--log-level",
            log_level,
        ]
        if allow_destructive:
            cmd.append("--allow-destructive")
        return cls(server_command=cmd)

    async def start(self) -> None:
        """Open the stdio connection and the client session. Must be awaited first."""
        params = StdioServerParameters(
            command=self.server_command[0],
            args=self.server_command[1:],
            env=self.server_env,
        )
        self._cm = stdio_client(params)
        read, write = await self._cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()  # type: ignore[union-attr]

    async def close(self) -> None:
        if self._session_cm is not None:
            await self._session_cm.__aexit__(None, None, None)
            self._session_cm = None
        if self._cm is not None:
            await self._cm.__aexit__(None, None, None)
            self._cm = None
        self._session = None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._session is None:
            msg = "StdioResolveClient.start() was not awaited"
            raise RuntimeError(msg)
        result = await self._session.call_tool(name=name, arguments=arguments)
        # The SDK returns a CallToolResult containing a `content` list of TextContent
        # and optional `structuredContent`.
        if getattr(result, "structuredContent", None):
            return dict(result.structuredContent)
        # Fall back to parsing the text payload.
        for part in result.content or []:
            text = getattr(part, "text", None)
            if text:
                import json

                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"raw_text": text}
        return {}

    async def read_resource(self, uri: str) -> str:
        if self._session is None:
            msg = "StdioResolveClient.start() was not awaited"
            raise RuntimeError(msg)
        result = await self._session.read_resource(uri)
        # Concatenate text of returned contents.
        parts: list[str] = []
        for r in result.contents or []:
            text = getattr(r, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts)

    async def list_tools(self) -> list[str]:
        if self._session is None:
            msg = "StdioResolveClient.start() was not awaited"
            raise RuntimeError(msg)
        result = await self._session.list_tools()
        return [t.name for t in result.tools]


# --- In-process stub (used by director's auto-mode tests) ----------------------


@dataclass
class StubResolveClient(ResolveClient):
    """Drives a ``FakeResolveBackend`` directly without spawning a subprocess.

    The Editor deals only with this interface, so tests swap the impl freely.
    """

    backend: Any  # FakeResolveBackend

    async def start(self) -> None:
        """No-op for the stub. Exists so the CLI path doesn't need a separate branch."""
        return None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await _call_fake_tool(self.backend, name, arguments)

    async def read_resource(self, uri: str) -> str:
        from resolve_mcp.resources import (
            media_pool_resource,
            project_resource,
            timeline_resource,
        )

        if uri == "resolve://project":
            return project_resource(self.backend)
        if uri == "resolve://media-pool":
            return media_pool_resource(self.backend)
        if uri == "resolve://timeline/current":
            return timeline_resource(self.backend)
        msg = f"unknown resource URI: {uri}"
        raise KeyError(msg)

    async def list_tools(self) -> list[str]:
        return [
            "create_project",
            "import_media",
            "list_media_pool",
            "create_timeline",
            "append_clip",
            "get_timeline_state",
        ]

    async def close(self) -> None:  # nothing to do; backend lives in memory
        return None


# ---- Fake dispatcher -----------------------------------------------------------------


async def _call_fake_tool(be: Any, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Replays the resolve-mcp tool layer against a FakeResolveBackend WITHOUT
    needing the FastMCP server in the test process. We use the pure-Python
    functions in :mod:`resolve_mcp.tools` so behavior matches a real server.
    """
    from resolve_mcp.tools import (
        add_fade,
        add_marker,
        add_render_job,
        add_transition,
        append_clip,
        create_bin,
        create_project,
        create_timeline,
        delete_clip,
        delete_media,
        delete_timeline,
        get_project_info,
        get_render_status,
        get_timeline_state,
        import_media,
        insert_clip,
        list_media_pool,
        move_clip,
        open_project,
        quit_app,
        restart_app,
        save_project,
        set_composite_mode,
        set_crop,
        set_opacity,
        set_speed,
        set_transform,
        start_render,
    )

    table: dict[str, Any] = {
        "open_project": lambda a: open_project(be, a["name"]),
        "create_project": lambda a: create_project(
            be, name=a["name"], fps=a["fps"], drop_frame=a.get("drop_frame", False),
            width=a.get("width", 1920), height=a.get("height", 1080),
        ),
        "save_project": lambda a: save_project(be),
        "get_project_info": lambda a: get_project_info(be),
        "import_media": lambda a: import_media(be, paths=a["paths"], bin=a.get("bin")),
        "list_media_pool": lambda a: list_media_pool(be),
        "create_bin": lambda a: create_bin(be, a["name"]),
        "create_timeline": lambda a: create_timeline(be, name=a["name"], fps=a["fps"], drop_frame=a.get("drop_frame", False)),
        "get_timeline_state": lambda a: get_timeline_state(be),
        "append_clip": lambda a: append_clip(
            be,
            media_clip_id=a["media_clip_id"],
            timeline_track_index=a.get("timeline_track_index", 0),
            start_seconds=a.get("start_seconds", 0.0),
            duration_seconds=a["duration_seconds"],
            source_in_seconds=a.get("source_in_seconds", 0.0),
        ),
        "insert_clip": lambda a: insert_clip(
            be,
            media_clip_id=a["media_clip_id"],
            timeline_track_index=a["timeline_track_index"],
            timeline_position_seconds=a["timeline_position_seconds"],
            duration_seconds=a["duration_seconds"],
            source_in_seconds=a.get("source_in_seconds", 0.0),
        ),
        "delete_clip": lambda a: delete_clip(be, a["timeline_item_id"]),
        "move_clip": lambda a: move_clip(be, timeline_item_id=a["timeline_item_id"], new_position_seconds=a["new_position_seconds"]),
        "set_transform": lambda a: set_transform(
            be,
            timeline_item_id=a["timeline_item_id"],
            pan_x=a["pan_x"], pan_y=a["pan_y"],
            zoom_x=a["zoom_x"], zoom_y=a["zoom_y"],
            rotation=a["rotation"],
            anchor_x=a.get("anchor_x", 0.5),
            anchor_y=a.get("anchor_y", 0.5),
        ),
        "set_crop": lambda a: set_crop(be, timeline_item_id=a["timeline_item_id"], left=a["left"], right=a["right"], top=a["top"], bottom=a["bottom"]),
        "set_composite_mode": lambda a: set_composite_mode(be, timeline_item_id=a["timeline_item_id"], mode=a["mode"]),
        "set_opacity": lambda a: set_opacity(be, timeline_item_id=a["timeline_item_id"], opacity=a["opacity"]),
        "add_fade": lambda a: add_fade(be, timeline_item_id=a["timeline_item_id"], fade_in_seconds=a["fade_in_seconds"], fade_out_seconds=a["fade_out_seconds"]),
        "set_speed": lambda a: set_speed(be, timeline_item_id=a["timeline_item_id"], speed=a["speed"]),
        "add_marker": lambda a: add_marker(
            be,
            timeline_item_id=a["timeline_item_id"],
            position_seconds=a["position_seconds"],
            label=a["label"],
            color=a["color"],
            note=a.get("note", ""),
        ),
        "add_transition": lambda a: add_transition(
            be,
            timeline_item_id=a["timeline_item_id"],
            track_index=a["track_index"],
            style=a["style"],
            duration_seconds=a["duration_seconds"],
            alignment=a["alignment"],
        ),
        "add_render_job": lambda a: add_render_job(be, timeline_name=a["timeline_name"], format=a["format"], output_path=a["output_path"]),
        "start_render": lambda a: start_render(be, job_id=a["job_id"]),
        "get_render_status": lambda a: get_render_status(be, job_id=a["job_id"]),
        "quit_app": lambda a: quit_app(be, confirm=a["confirm"]),
        "restart_app": lambda a: restart_app(be, confirm=a["confirm"]),
        "delete_timeline": lambda a: delete_timeline(be, name=a["name"], confirm=a["confirm"]),
        "delete_media": lambda a: delete_media(be, media_clip_id=a["media_clip_id"], confirm=a["confirm"]),
    }
    try:
        fn = table[name]
    except KeyError as exc:
        raise KeyError(f"unknown tool: {name}") from exc
    return fn(args)


__all__ = [
    "ResolveClient",
    "StdioResolveClient",
    "StubResolveClient",
]


# ---- Async context helper ------------------------------------------------------------


@asynccontextmanager
async def open_stdio_client(
    *,
    backend: str = "fake",
    allow_destructive: bool = False,
    log_level: str = "WARNING",
    uv_project: str | None = None,
) -> Awaitable[Any]:
    """Async context manager wrapper used by the CLI."""
    client = StdioResolveClient.default(
        backend=backend,
        allow_destructive=allow_destructive,
        log_level=log_level,
        uv_project=uv_project,
    )
    await client.start()
    try:
        yield client
    finally:
        await client.close()
