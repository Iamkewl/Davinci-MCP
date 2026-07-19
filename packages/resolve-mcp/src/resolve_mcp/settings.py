"""Settings sourced from env vars / .env. Loaded by ``resolve_mcp.server``.

We centralize everything here so ``resolve-mcp`` itself stays dependency-light — only
``pydantic``, ``pydantic-settings``, the MCP SDK, and ``structlog``.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ResolveMCPSettings(BaseSettings):
    """Runtime settings for the resolve-mcp server process.

    All fields are env-driven so the server can be deployed unchanged in dev, CI,
    and prod. Defaults are safe (disable destructive, stdio transport).
    """

    model_config = SettingsConfigDict(
        env_prefix="RESOLVE_MCP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    transport: str = "stdio"
    allow_destructive: bool = False
    backend: str = "fake"  # "fake" | "davinci"
    log_level: str = "INFO"


__all__ = ["ResolveMCPSettings"]
