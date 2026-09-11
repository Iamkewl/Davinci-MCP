"""Director-side settings (env-driven, pydantic-settings-backed).

These are separate from resolve-mcp's settings so the two packages can be
deployed independently. The two packages share their env-prefix to keep
one combined .env.example usable.
"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class DirectorSettings(BaseSettings):
    """Director runtime settings."""

    model_config = SettingsConfigDict(
        env_prefix="DIRECTOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    gemini_api_key: str | None = None
    reasoning_model: str = "gemini-2.5-pro"
    vision_model: str = "gemini-2.5-flash"
    run_store_dir: str = "./runs"
    max_planner_iterations: int = 5
    director_min_overall: float = 0.6
    director_min_per_axis: float = 0.4
    # Provider-agnostic LLM layer (plan.md Phase 5). Model ids above are
    # provider-agnostic strings; for openai_compatible they are router ids,
    # e.g. anthropic/claude-sonnet-4.6 or google/gemini-3-flash.
    llm_provider: Literal["gemini", "openai_compatible", "none"] = "gemini"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str | None = None


__all__ = ["DirectorSettings"]

