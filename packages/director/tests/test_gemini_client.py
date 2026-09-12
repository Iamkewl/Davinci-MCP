"""Gemini adapter behaviour that only shows up against the real API.

These are the three things that made the live path fail before: handing Gemini a
JSON Schema it cannot parse, using an uploaded video before it finished
processing, and leaving uploads behind. A stub SDK stands in for the network.
"""

from __future__ import annotations

import types
from typing import Any

import pytest
from director.ingestion import gemini_client as gc
from director.ingestion.gemini_client import GeminiClient, GeminiError
from director.schemas import PerClipMap
from director.settings import DirectorSettings


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text
        self.candidates: list[Any] = []


class _File:
    def __init__(self, name: str, states: list[str]) -> None:
        self.name = name
        self._states = states
        self.state = types.SimpleNamespace(name=states[0])

    def advance(self) -> None:
        if len(self._states) > 1:
            self._states.pop(0)
            self.state = types.SimpleNamespace(name=self._states[0])


class _StubSDK:
    """Records what the client asked the SDK to do."""

    def __init__(self, *, payload: str, upload_states: list[str] | None = None) -> None:
        self.payload = payload
        self.generate_calls: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        self.get_calls = 0
        self._file = _File("files/abc", upload_states or ["ACTIVE"])
        sdk = self

        class _Models:
            async def generate_content(self, **kwargs: Any) -> _Response:
                sdk.generate_calls.append(kwargs)
                return _Response(sdk.payload)

        class _Files:
            async def upload(self, **kwargs: Any) -> _File:
                return sdk._file

            async def get(self, **kwargs: Any) -> _File:
                sdk.get_calls += 1
                sdk._file.advance()
                return sdk._file

            async def delete(self, *, name: str) -> None:
                sdk.deleted.append(name)

        self.aio = types.SimpleNamespace(models=_Models(), files=_Files())


def _client(monkeypatch: pytest.MonkeyPatch, sdk: _StubSDK) -> GeminiClient:
    client = GeminiClient(settings=DirectorSettings(gemini_api_key="test-key"))
    monkeypatch.setattr(client, "_ensure_client", lambda: sdk)
    return client


async def test_generate_json_passes_the_pydantic_class_not_json_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini rejects $defs/$ref, which model_json_schema() emits for nested models."""
    sdk = _StubSDK(payload='{"clip_id": "c", "source_path": "/m/a.mp4"}')
    client = _client(monkeypatch, sdk)

    await client.generate_json(system="s", user="u", response_schema=PerClipMap)

    config = sdk.generate_calls[0]["config"]
    assert config["response_schema"] is PerClipMap
    assert config["response_mime_type"] == "application/json"


async def test_analyze_video_waits_for_the_upload_to_become_active(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    clip = tmp_path / "a.mp4"
    clip.write_bytes(b"")
    monkeypatch.setattr(gc, "UPLOAD_POLL_SECONDS", 0.0)
    sdk = _StubSDK(
        payload='{"clip_id": "x", "source_path": "/x", "visual_summary": "a shot"}',
        upload_states=["PROCESSING", "PROCESSING", "ACTIVE"],
    )
    client = _client(monkeypatch, sdk)

    result = await client.analyze_video(clip_path=str(clip), clip_id="clip_1", prompt="p")

    assert sdk.get_calls >= 2, "the client must poll until the file is usable"
    assert result.clip_id == "clip_1"
    assert result.source_path == str(clip)
    assert sdk.deleted == ["files/abc"], "the upload should be cleaned up"


async def test_analyze_video_reports_a_failed_upload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    clip = tmp_path / "a.mp4"
    clip.write_bytes(b"")
    monkeypatch.setattr(gc, "UPLOAD_POLL_SECONDS", 0.0)
    sdk = _StubSDK(payload="{}", upload_states=["PROCESSING", "FAILED"])
    client = _client(monkeypatch, sdk)

    with pytest.raises(GeminiError, match="FAILED"):
        await client.analyze_video(clip_path=str(clip), clip_id="c", prompt="p")
    assert sdk.deleted == ["files/abc"]


async def test_missing_key_is_reported_before_any_call() -> None:
    client = GeminiClient(settings=DirectorSettings(gemini_api_key=None))
    with pytest.raises(GeminiError, match="GEMINI_API_KEY"):
        await client.generate_json(system="s", user="u", response_schema=PerClipMap)


async def test_unknown_model_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = _StubSDK(payload="{}")

    async def _boom(**kwargs: Any) -> None:
        raise RuntimeError("404 models/does-not-exist is not found for API version v1beta")

    monkeypatch.setattr(sdk.aio.models, "generate_content", _boom)
    client = _client(monkeypatch, sdk)
    settings = DirectorSettings(gemini_api_key="k", reasoning_model="does-not-exist")
    client._settings = settings

    with pytest.raises(GeminiError, match="DIRECTOR_REASONING_MODEL"):
        await client.generate_json(system="s", user="u", response_schema=PerClipMap)


async def test_non_json_response_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch, _StubSDK(payload="I'm afraid I can't do that"))
    with pytest.raises(GeminiError, match="non-JSON"):
        await client.generate_json(system="s", user="u", response_schema=PerClipMap)


def test_sdk_is_the_supported_generation() -> None:
    """google-genai 0.x had a different Files/schema surface; pin the 1.x line."""
    import google.genai

    assert int(google.genai.__version__.split(".")[0]) >= 1
