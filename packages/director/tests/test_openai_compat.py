"""Offline tests for OpenAICompatClient against a hand-stubbed transport.

No network: the AsyncOpenAI client is injected via ``_client_override``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from director.llm.base import LLMError
from director.llm.openai_compat import OpenAICompatClient
from director.schemas import PerClipMap, Plan
from director.settings import DirectorSettings


class StubCompletions:
    """Records create() calls; replays a scripted list of outcomes."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _response(payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
    )


def _client(script: list[Any], **kw: Any) -> tuple[OpenAICompatClient, StubCompletions]:
    stub = StubCompletions(script)
    settings = DirectorSettings(
        llm_provider="openai_compatible",
        reasoning_model="anthropic/claude-sonnet-4.6",
        vision_model="google/gemini-3-flash",
    )
    override = SimpleNamespace(chat=SimpleNamespace(completions=stub))
    client = OpenAICompatClient(settings=settings, _client_override=override, **kw)
    return client, stub


async def test_generate_json_happy_path_uses_json_schema_and_model_id() -> None:
    plan_payload = {"plan_id": "p1", "target_project": "proj", "target_timeline": "tl", "ops": []}
    client, stub = _client([_response(plan_payload)])
    got = await client.generate_json(system="s", user="u", response_schema=Plan)
    assert isinstance(got, Plan)
    call = stub.calls[0]
    assert call["model"] == "anthropic/claude-sonnet-4.6"
    assert call["response_format"]["type"] == "json_schema"


async def test_generate_json_falls_back_to_json_object_on_rejection() -> None:
    payload = {"plan_id": "p2", "target_project": "proj", "target_timeline": "tl", "ops": []}
    client, stub = _client([TypeError("unexpected keyword 'response_format'"), _response(payload)])
    got = await client.generate_json(system="s", user="u", response_schema=Plan)
    assert got.plan_id == "p2"
    assert stub.calls[1]["response_format"] == {"type": "json_object"}


async def test_generate_json_repairs_invalid_payload_once() -> None:
    bad = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="not json at all"))]
    )
    good_payload = {"plan_id": "p3", "target_project": "proj", "target_timeline": "tl", "ops": []}
    client, stub = _client([bad, _response(good_payload)])
    got = await client.generate_json(system="s", user="u", response_schema=Plan)
    assert got.plan_id == "p3"
    repair_messages = stub.calls[1]["messages"]
    assert any("not valid JSON" in m["content"] for m in repair_messages if m["role"] == "user")


async def test_generate_json_raises_invalid_model_output_after_repair_fails() -> None:
    from director.agents.base import InvalidModelOutput

    bad = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="still not json"))]
    )
    client, _ = _client([bad, bad])
    with pytest.raises(InvalidModelOutput):
        await client.generate_json(system="s", user="u", response_schema=Plan)


async def test_no_api_key_raises_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = DirectorSettings(
        llm_provider="openai_compatible", llm_api_key=None,
    )
    client = OpenAICompatClient(settings=settings)
    with pytest.raises(LLMError):
        await client.generate_json(system="s", user="u", response_schema=Plan)


async def test_analyze_video_without_ffmpeg_returns_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("director.llm.openai_compat.shutil.which", lambda _: None)
    client, _ = _client([])
    got = await client.analyze_video(clip_path="/does/not/matter.mp4", clip_id="clip_x", prompt="p")
    assert got.clip_id == "clip_x"
    assert got.source_path == "/does/not/matter.mp4"
    assert got.visual_summary == ""


async def test_analyze_video_sends_keyframes_and_filters_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "visual_summary": "a sunset",
        "dominant_shot_type": "wide",
        "rogue_field": "must be filtered (extra=forbid)",
    }
    client, stub = _client([_response(payload)])

    async def _fake_extract(self: Any, path: str, count: int) -> list[str]:
        return ["data:image/jpeg;base64,QUJD"]

    monkeypatch.setattr(OpenAICompatClient, "_extract_keyframes", _fake_extract)
    got = await client.analyze_video(clip_path="/clips/a.mp4", clip_id="clip_a", prompt="describe")
    assert got.visual_summary == "a sunset"
    assert got.dominant_shot_type == "wide"
    assert got.clip_id == "clip_a"
    parts = stub.calls[0]["messages"][0]["content"]
    assert parts[0]["type"] == "text"
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    model_used = stub.calls[0]["model"]
    assert model_used == "google/gemini-3-flash"


async def test_analyze_video_placeholder_when_vision_call_fails() -> None:
    client, _ = _client([RuntimeError("provider down")])
    client._extract_keyframes = lambda path, count: [  # type: ignore[method-assign]
        "data:image/jpeg;base64,QUJD"
    ]
    got = await client.analyze_video(clip_path="/clips/a.mp4", clip_id="clip_b", prompt="p")
    assert isinstance(got, PerClipMap)
    assert got.visual_summary == ""


def test_key_resolution_prefers_settings_then_env(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = DirectorSettings(
        llm_provider="openai_compatible", llm_api_key="direct-key"
    )
    client = OpenAICompatClient(settings=settings)
    assert client._resolve_api_key() == "direct-key"
    settings2 = DirectorSettings(llm_provider="openai_compatible", llm_api_key=None)
    client2 = OpenAICompatClient(settings=settings2)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    assert client2._resolve_api_key() == "router-key"
