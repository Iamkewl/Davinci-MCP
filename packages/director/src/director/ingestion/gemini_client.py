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

import asyncio
import contextlib
import json
import os
import time
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..errors import ProviderError
from ..schemas import PerClipMap
from ..settings import DirectorSettings

T = TypeVar("T", bound=BaseModel)


class GeminiError(ProviderError):
    """Raised when the Gemini call fails in some non-recoverable way."""


#: How long to wait for an uploaded video to finish processing before giving up.
UPLOAD_ACTIVE_TIMEOUT_SECONDS = 180.0
UPLOAD_POLL_SECONDS = 2.0


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
            import google.genai as genai
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
                        # The SDK converts a pydantic class into Gemini's own schema
                        # dialect. Handing it model_json_schema() instead ships
                        # $defs/$ref, which the API rejects for every nested model.
                        "response_schema": response_schema,
                    },
                )
            )
        except Exception as exc:
            raise GeminiError(self._explain(exc, chosen_model)) from exc
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
        # Upload, then wait: a freshly uploaded video is PROCESSING and cannot be
        # referenced until the File API reports ACTIVE.
        try:
            uploaded = await self._call_async(
                lambda: client.aio.files.upload(file=clip_path)
            )
        except Exception as exc:
            raise GeminiError(f"upload failed: {exc}") from exc
        try:
            uploaded = await self._await_active(client, uploaded)
            response = await self._call_async(
                lambda: client.aio.models.generate_content(
                    model=chosen_model,
                    contents=[uploaded, prompt],
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": PerClipMap,
                    },
                )
            )
        except GeminiError:
            await self._delete_uploaded(client, uploaded)
            raise
        except Exception as exc:
            await self._delete_uploaded(client, uploaded)
            raise GeminiError(f"vision analyze failed: {self._explain(exc, chosen_model)}") from exc
        await self._delete_uploaded(client, uploaded)
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

    async def aclose(self) -> None:
        """No-op: the google-genai client holds no per-instance resources."""
        return None

    # ---- internal helpers ----

    async def _await_active(self, client: Any, uploaded: Any) -> Any:
        """Poll the File API until the upload is usable (or fails/expires)."""
        name = getattr(uploaded, "name", None)
        state = _state_name(uploaded)
        if not name or state == "ACTIVE":
            return uploaded
        deadline = time.monotonic() + UPLOAD_ACTIVE_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if state == "FAILED":
                raise GeminiError(f"Gemini could not process {name}: upload state FAILED")
            if state == "ACTIVE":
                return uploaded
            await asyncio.sleep(UPLOAD_POLL_SECONDS)
            try:
                uploaded = await self._call_async(lambda: client.aio.files.get(name=name))
            except Exception as exc:
                raise GeminiError(f"could not read upload state for {name}: {exc}") from exc
            state = _state_name(uploaded)
        raise GeminiError(
            f"upload {name} was still {state} after {UPLOAD_ACTIVE_TIMEOUT_SECONDS:.0f}s"
        )

    async def _delete_uploaded(self, client: Any, uploaded: Any) -> None:
        """Best-effort cleanup so the Files namespace does not fill up."""
        name = getattr(uploaded, "name", None)
        if not name:
            return
        with contextlib.suppress(Exception):
            await self._call_async(lambda: client.aio.files.delete(name=name))

    def _explain(self, exc: Exception, model: str) -> str:
        """Turn an SDK error into something the user can act on."""
        text = str(exc)
        if "not found" in text.lower() or "404" in text:
            return (
                f"model {model!r} is not available to this API key ({text}). "
                "Set DIRECTOR_REASONING_MODEL / DIRECTOR_VISION_MODEL to a model your "
                "key can use."
            )
        if "api key" in text.lower() or "permission" in text.lower() or "401" in text:
            return f"Gemini rejected the credentials: {text}"
        if "quota" in text.lower() or "429" in text or "resource_exhausted" in text.lower():
            return f"Gemini quota/rate limit hit: {text}"
        return text

    @staticmethod
    async def _call_async(fetcher: Callable[[], Any]) -> Any:
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
            return str(text)
        # Fall back to candidates if .text isn't populated (some configs).
        for cand in getattr(response, "candidates", []) or []:
            for part in getattr(cand.content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    return str(part_text)
        msg = "Gemini response had no text payload"
        raise GeminiError(msg)


def _state_name(uploaded: Any) -> str:
    """File state across SDK shapes: an enum, an object with .name, or a string."""
    state = getattr(uploaded, "state", None)
    if state is None:
        return "UNKNOWN"
    return str(getattr(state, "name", state)).upper()


# --- Placeholder factory ---------------------------------------------------------


def get_gemini_client(settings: DirectorSettings) -> GeminiClient:
    """Construct the singleton Gemini client. Tests use a fake override."""
    return GeminiClient(settings=settings)


# Stub class for tests -----------------------------------------------------------


__all__ = ["GeminiClient", "GeminiError", "get_gemini_client"]
