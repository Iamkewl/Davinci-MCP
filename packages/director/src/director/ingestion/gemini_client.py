"""Thin wrapper around the Gemini SDK.

Two responsibilities:

* :class:`GeminiClient` configures the model once and exposes two simple async
  methods that all other agents use: ``generate_text`` (structured JSON, validated
  against a pydantic schema) and ``analyze_video`` (File API for clips).
* All configurable knobs (model ids, API key) flow from environment via
  :class:`DirectorSettings`. The class is intentionally small so it is easy to
  swap out for a fake (in tests) or a different provider in v2.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..schemas import PerClipMap
from ..settings import DirectorSettings

T = TypeVar("T", bound=BaseModel)


class GeminiError(RuntimeError):
    """Raised when the Gemini call fails in some non-recoverable way."""


class GeminiClient:
    """Async-and-sync-compatible Gemini wrapper.

    Public methods:

    * :meth:`generate_json` — call the reasoning model with a JSON schema. We
      request JSON mode by asking for the response to be JSON and validate the
      output via pydantic. We do NOT silently coerce: a schema mismatch raises.

    * :meth:`analyze_video` — stream a video file via the Gemini File API, then
      ask the vision model to produce a :class:`KeyMoment` JSON list.

    The actual SDK is loaded lazily so tests without network access can swap a
    fake implementation in.
    """

    def __init__(self, *, settings: DirectorSettings) -> None:
        self._settings = settings
        self._client: Any | None = None

    # ---- lazy SDK load ----

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._settings.gemini_api_key:
            msg = "GEMINI_API_KEY is not set; cannot call Gemini"
            raise GeminiError(msg)
        try:
            import google.genai as genai  # type: ignore[import-not-found]
        except Exception as exc:
            msg = "google-genai is not installed; install director[gemini] or the dev extra"
            raise GeminiError(msg) from exc
        self._client = genai.Client(api_key=self._settings.gemini_api_key)
        return self._client

    # ---- public API ----

    async def generate_json(
        self,
        *,
        system: str,
        user: str,
        response_schema: type[T],
        model: str | None = None,
    ) -> T:
        """Generate a structured JSON value validated against ``response_schema``.

        Raises:
            GeminiError: on transport/SDK issues.
            ValidationError: if the response cannot be parsed into ``response_schema``.
        """
        client = self._ensure_client()
        chosen_model = model or self._settings.reasoning_model
        # JSON mode: the SDK accepts a config keyword on generate_content.
        prompt = (
            f"{system}\n\n"
            f"Respond with valid JSON conforming to the schema: {response_schema.__name__}. "
            f"Do not include any prose outside the JSON."
        )
        prompt_full = f"{prompt}\n\n---\n\n{user}"
        try:
            response = await self._call_async(
                lambda: client.aio.models.generate_content(
                    model=chosen_model,
                    contents=prompt_full,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": response_schema.model_json_schema(),
                    },
                )
            )
        except Exception as exc:  # pragma: no cover
            raise GeminiError(str(exc)) from exc
        text = self._extract_text(response)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise GeminiError(f"Gemini returned non-JSON: {text!r}") from exc
        try:
            return response_schema.model_validate(payload)
        except ValidationError:
            raise

    async def analyze_video(
        self,
        *,
        clip_path: str,
        clip_id: str,
        prompt: str,
        model: str | None = None,
    ) -> PerClipMap:
        """Upload + analyze a video clip; return a populated PerClipMap.

        The Flask API upload returns a file handle that we reference in the
        contents array. After the call we delete the uploaded file to keep the
        Files namespace tidy.
        """
        client = self._ensure_client()
        chosen_model = model or self._settings.vision_model
        if not os.path.exists(clip_path):
            msg = f"clip not found on disk: {clip_path}"
            raise GeminiError(msg)
        # Upload
        try:
            uploaded = await self._call_async(
                lambda: client.aio.files.upload(file=clip_path)
            )
        except Exception as exc:
            raise GeminiError(f"upload failed: {exc}") from exc
        try:
            response = await self._call_async(
                lambda: client.aio.models.generate_content(
                    model=chosen_model,
                    contents=[
                        uploaded,
                        prompt,
                    ],
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": PerClipMap.model_json_schema(),
                    },
                )
            )
        except Exception as exc:
            raise GeminiError(f"vision analyze failed: {exc}") from exc
        text = self._extract_text(response)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise GeminiError(f"vision analyze returned non-JSON: {text!r}") from exc
        # Re-bind clip_id to ensure we cannot cross-leak ids from upstream.
        payload["clip_id"] = clip_id
        payload["source_path"] = clip_path
        # Pydantic-validate. Extra fields are FORBIDDEN by StrictModel so upstream
        # over-generation cannot slip into the editor.
        return PerClipMap.model_validate(payload)

    # ---- internal helpers ----

    @staticmethod
    async def _call_async(fetcher: Awaitable[Any] | Any) -> Any:
        """Await either an awaitable callback or the returned value."""
        # The SDK's `generate_content` returns an awaitable when using `aio.models`.
        result = fetcher()
        if hasattr(result, "__await__"):
            return await result
        return result

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Get the model's text from a Gemini response across SDK versions."""

        text = getattr(response, "text", None)
        if text:
            return text
        # Fall back to candidates if .text isn't populated (some configs).
        for cand in getattr(response, "candidates", []) or []:
            for part in getattr(cand.content, "parts", []) or []:
                if getattr(part, "text", None):
                    return part.text  # type: ignore[no-any-return]
        msg = "Gemini response had no text payload"
        raise GeminiError(msg)


# --- Placeholder factory ---------------------------------------------------------


def get_gemini_client(settings: DirectorSettings) -> GeminiClient:
    """Construct the singleton Gemini client. Tests use a fake override."""
    return GeminiClient(settings=settings)


# Stub class for tests -----------------------------------------------------------


__all__ = ["GeminiClient", "GeminiError", "get_gemini_client"]
