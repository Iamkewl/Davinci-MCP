"""OpenAI-compatible chat-completions adapter (OpenRouter, vLLM, …).

Speaks the :class:`director.llm.base.LLMClient` protocol against any endpoint
implementing ``POST /chat/completions``. Structured output strategy:

1. Try ``response_format={"type": "json_schema", ...}``.
2. On provider rejection, retry once with ``{"type": "json_object"}``.
3. Parse + pydantic-validate; on failure, ONE repair retry feeding the validator
   message back; then raise :class:`InvalidModelOutput`.

Video analysis has no upload equivalent in OpenAI-compatible APIs, so clips are
reduced to <=8 evenly-spaced ffmpeg keyframes sent as base64 image parts. If
ffmpeg is absent (or extraction fails) we degrade honestly to the same
placeholder PerClipMap the offline path produces, with a logged warning.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..agents.base import InvalidModelOutput
from ..schemas import PerClipMap
from ..settings import DirectorSettings
from .base import LLMError, T

logger = logging.getLogger(__name__)

_MAX_KEYFRAMES = 8


class OpenAICompatClient:
    """Any OpenAI-compatible /chat/completions endpoint, OpenRouter by default."""

    def __init__(
        self,
        *,
        settings: DirectorSettings,
        _client_override: Any | None = None,
    ) -> None:
        self._settings = settings
        self._override = _client_override
        self._client: Any | None = None

    # ---- key / client resolution -------------------------------------------

    def _resolve_api_key(self) -> str | None:
        return (
            self._settings.llm_api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )

    def _ensure_client(self) -> Any:
        if self._override is not None:
            return self._override
        if self._client is not None:
            return self._client
        key = self._resolve_api_key()
        if not key:
            msg = (
                "no API key for openai_compatible provider; set DIRECTOR_LLM_API_KEY "
                "(or OPENROUTER_API_KEY / OPENAI_API_KEY)"
            )
            raise LLMError(msg)
        try:
            from openai import AsyncOpenAI
        except Exception as exc:  # pragma: no cover - dependency always installed
            msg = "openai package is not installed"
            raise LLMError(msg) from exc
        self._client = AsyncOpenAI(base_url=self._settings.llm_base_url, api_key=key)
        return self._client

    # ---- structured generation ----------------------------------------------

    async def generate_json(self, *, system: str, user: str, response_schema: type[T]) -> T:
        client = self._ensure_client()
        model = self._settings.reasoning_model
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            content = await self._complete(
                client, model, messages,
                {"type": "json_schema", "json_schema": {
                    "name": response_schema.__name__,
                    "schema": response_schema.model_json_schema(),
                    "strict": False,
                }},
            )
        except Exception as exc:
            logger.debug("json_schema response_format rejected; falling back: %s", exc)
            instruction = (
                f"{user}\n\nRespond with ONLY a JSON object conforming to the "
                f"{response_schema.__name__} schema. No prose."
            )
            messages[1] = {"role": "user", "content": instruction}
            content = await self._complete(client, model, messages, {"type": "json_object"})
        return await self._validate_with_repair(
            client, model, messages, content, response_schema,
        )

    @staticmethod
    async def _complete(
        client: Any,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, Any],
    ) -> str:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=response_format,
        )
        return OpenAICompatClient._extract_content(response)

    @staticmethod
    def _extract_content(response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMError("completion returned no choices")
        content = getattr(choices[0].message, "content", None)
        if not content:
            raise LLMError("completion returned empty content")
        return str(content)

    async def _validate_with_repair(
        self,
        client: Any,
        model: str,
        messages: list[dict[str, str]],
        content: str,
        schema: type[T],
    ) -> T:
        problem = ""
        for attempt in range(2):
            try:
                payload = json.loads(content)
                return schema.model_validate(payload)
            except json.JSONDecodeError as exc:
                problem = f"output was not valid JSON: {exc}"
            except ValidationError as exc:
                problem = f"schema validation failed: {exc}"
            if attempt == 0:
                repair_messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            f"Your previous output had a problem: {problem}\n"
                            f"Return corrected JSON only, conforming to the "
                            f"{schema.__name__} schema. No prose."
                        ),
                    },
                ]
                response_format = {"type": "json_object"}
                try:
                    content = await self._complete(
                        client, model, repair_messages, response_format,
                    )
                except Exception as exc:
                    raise InvalidModelOutput(f"repair call failed: {exc}") from exc
        raise InvalidModelOutput(f"{schema.__name__}: {problem}")

    # ---- video analysis ------------------------------------------------------

    async def analyze_video(self, *, clip_path: str, clip_id: str, prompt: str) -> PerClipMap:
        def _placeholder() -> PerClipMap:
            return PerClipMap(clip_id=clip_id, source_path=clip_path)

        try:
            client = self._ensure_client()
            frames = await self._extract_keyframes(clip_path, _MAX_KEYFRAMES)
            if not frames:
                logger.warning("no keyframes extracted for %s; using placeholder map", clip_path)
                return _placeholder()
            parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            parts.extend(
                {"type": "image_url", "image_url": {"url": frame}} for frame in frames
            )
            response = await client.chat.completions.create(
                model=self._settings.vision_model,
                messages=[{"role": "user", "content": parts}],
                response_format={"type": "json_object"},
            )
            payload = json.loads(self._extract_content(response))
            allowed = set(type(self).allowed_fields())
            filtered = {k: v for k, v in payload.items() if k in allowed}
            filtered["clip_id"] = clip_id
            filtered["source_path"] = clip_path
            return PerClipMap.model_validate(filtered)
        except Exception as exc:
            logger.warning("vision analyze failed for %s: %s", clip_path, exc)
            return _placeholder()

    @staticmethod
    def allowed_fields() -> list[str]:
        return list(PerClipMap.model_fields.keys())

    async def _extract_keyframes(self, clip_path: str, count: int) -> list[str]:
        """Return up to ``count`` base64 JPEG data URLs sampled across the clip."""
        if not Path(clip_path).is_file():
            return []
        if shutil.which("ffmpeg") is None:
            logger.warning("ffmpeg not found; cannot extract keyframes from %s", clip_path)
            return []
        timestamps = self._sample_timestamps(clip_path, count)
        frames: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            for index, ts in enumerate(timestamps):
                out = Path(tmp) / f"frame_{index:02d}.jpg"
                proc = await self._run_ffmpeg(clip_path, ts, out)
                if proc.returncode != 0 or not out.is_file():
                    continue
                data = out.read_bytes()
                frames.append(
                    "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
                )
        return frames

    @staticmethod
    def _sample_timestamps(clip_path: str, count: int) -> list[float]:
        duration: float | None = None
        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            try:
                proc = subprocess.run(
                    [ffprobe, "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", clip_path],
                    capture_output=True, text=True, check=True, timeout=30,
                )
                duration = float(proc.stdout.strip())
            except Exception:
                duration = None
        if not duration or duration <= 0:
            return [0.0]
        step = max(duration / count, 0.1)
        return [min(i * step, max(duration - 0.1, 0.0)) for i in range(count)]

    @staticmethod
    async def _run_ffmpeg(clip_path: str, timestamp: float, out: Path) -> subprocess.CompletedProcess[bytes]:
        ffmpeg = shutil.which("ffmpeg")
        assert ffmpeg is not None  # guarded by caller
        return subprocess.run(
            [ffmpeg, "-v", "error", "-ss", f"{timestamp:.3f}", "-i", clip_path,
             "-frames:v", "1", "-q:v", "4", "-y", str(out)],
            capture_output=True, timeout=60,
        )

    # ---- lifecycle ------------------------------------------------------------

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None) if self._client else None
        if close is not None:
            await close()
        self._client = None


__all__ = ["OpenAICompatClient"]
