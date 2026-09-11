"""Base agent abstraction.

Every agent (Contextualizer, Planner, Director, Editor) follows the pattern:

1. A request is built deterministically (a pydantic-validated system + user prompt).
2. The Gemini client is invoked, optionally twice: a structured-JSON call OR a
   streaming JSON call.
3. The output is validated as pydantic. If validation fails, the agent raises —
   no silent fix-ups. The orchestrator decides whether to retry or surface.

This module defines the abstract base. Each concrete agent lives in its own
file under :mod:`director.agents`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import ValidationError

from ..ingestion.gemini_client import GeminiClient
from ..settings import DirectorSettings
from .logging_setup import get_logger

if TYPE_CHECKING:
    from ..llm.base import LLMClient

logger = get_logger("director.agents")

OutputT = TypeVar("OutputT")


class Agent(ABC, Generic[OutputT]):
    """Common base for all orchestrator agents.

    Concrete subclasses implement :meth:`run`. They receive a provider-agnostic
    LLM client (``llm=``) and the director settings so the same base works in
    production and tests. ``gemini=`` remains accepted as a legacy alias.
    """

    def __init__(
        self,
        *,
        settings: DirectorSettings,
        gemini: GeminiClient | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self._llm = llm if llm is not None else gemini
        self._settings = settings

    @property
    def settings(self) -> DirectorSettings:
        return self._settings

    @property
    def llm(self) -> Any:
        """The configured LLM client, or None in offline mode."""
        return self._llm

    @property
    def gemini(self) -> Any:
        """Legacy alias for :attr:`llm`."""
        return self._llm

    @abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> OutputT:  # pragma: no cover
        ...


class InvalidModelOutput(Exception):
    """Raised when an agent's structured output cannot be pydantic-validated."""


def raise_or_rethrow_validation(err: ValidationError, *, context: str) -> None:
    """Surface a pydantic validation error with the agent context."""
    msg = f"{context}: validation failed: {err}"
    raise InvalidModelOutput(msg) from err


__all__ = ["Agent", "InvalidModelOutput", "raise_or_rethrow_validation"]
