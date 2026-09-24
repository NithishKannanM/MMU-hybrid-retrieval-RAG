"""OllamaGenerator over `httpx.MockTransport` (no network, no model) plus the
registry's alias resolution and import-isolation guarantees.

The one test that matters most here is `test_stream_assembles_line_split_across_chunks`
— an NDJSON line split across two transport chunks is the single most common bug in
hand-rolled Ollama streaming, and it is silent unless something specifically forces the
split to land mid-line and checks the assembled text.
"""

from __future__ import annotations

import json
import sys

import httpx
import pytest

from mmu.config import Settings
from mmu.generate.anthropic import AnthropicGenerator
from mmu.generate.base import GeneratorUnavailable, Message
from mmu.generate.ollama import OllamaGenerator
from mmu.generate.registry import available_aliases, build_generator

# ------------------------------------------------------------------------------ helpers


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ndjson_handler(*chunks: bytes):
    async def handler(request: httpx.Request) -> httpx.Response:
        async def body():
            for c in chunks:
                yield c

        return httpx.Response(200, content=body())

    return handler


PROMPT = [Message(role="user", content="When must a breach be reported?")]


# ------------------------------------------------------------------------------ stream


async def test_stream_assembles_normal_response() -> None:
    handler = _ndjson_handler(
        b'{"message":{"content":"Notify within"},"done":false}\n',
        b'{"message":{"content":" 72 hours."},"done":true}\n',
    )
    gen = OllamaGenerator(host="http://ollama", model="qwen", num_ctx=4096, client=_client(handler))

    deltas = [d async for d in gen.stream(PROMPT, temperature=0.0, max_tokens=64)]

    assert "".join(d.text for d in deltas) == "Notify within 72 hours."


async def test_stream_assembles_line_split_across_chunks() -> None:
    """The partial-line buffer test. The first transport chunk ends *inside* a JSON
    object (mid string value, mid-key even) — a naive `chunk.splitlines()` approach
    would either drop the rest of that line or hand `json.loads` a truncated,
    unparseable fragment. The line is only valid once both chunks are concatenated."""
    full_line = json.dumps({"message": {"content": "Notify within 72 hours."}, "done": True}) + "\n"
    split_at = full_line.index('"content": "Notify') + len('"content": "Notify')
    first, second = full_line[:split_at].encode(), full_line[split_at:].encode()
    assert first and second  # sanity: both halves are non-empty

    handler = _ndjson_handler(first, second)
    gen = OllamaGenerator(host="http://ollama", model="qwen", num_ctx=4096, client=_client(handler))

    deltas = [d async for d in gen.stream(PROMPT, temperature=0.0, max_tokens=64)]

    assert "".join(d.text for d in deltas) == "Notify within 72 hours."


async def test_stream_multiple_lines_split_across_chunk_boundary() -> None:
    """A second, harder split: the boundary lands between the closing `}` of one line
    and the opening `{` of the next, plus a further split mid-way through that next
    line, to prove the carried-over buffer keeps working across repeated splits."""
    line1 = json.dumps({"message": {"content": "Part one. "}, "done": False})
    line2 = json.dumps({"message": {"content": "Part two."}, "done": True})
    text = line1 + "\n" + line2 + "\n"
    # Build three chunks: up through line1's newline, then half of line2, then the rest.
    cut1 = len(line1) + 1
    cut2 = cut1 + (len(line2) // 2)
    parts = [text[:cut1].encode(), text[cut1:cut2].encode(), text[cut2:].encode()]

    handler = _ndjson_handler(*parts)
    gen = OllamaGenerator(host="http://ollama", model="qwen", num_ctx=4096, client=_client(handler))

    deltas = [d async for d in gen.stream(PROMPT, temperature=0.0, max_tokens=64)]

    assert "".join(d.text for d in deltas) == "Part one. Part two."


async def test_stream_connection_error_raises_generator_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    gen = OllamaGenerator(host="http://ollama", model="qwen", num_ctx=4096, client=_client(handler))

    with pytest.raises(GeneratorUnavailable, match="ollama serve"):
        async for _ in gen.stream(PROMPT, temperature=0.0, max_tokens=64):
            pass


# ---------------------------------------------------------------------------- complete


async def test_complete_returns_full_text() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {"content": "Notify within 72 hours."},
                "done": True,
                "prompt_eval_count": 120,
                "eval_count": 8,
            },
        )

    gen = OllamaGenerator(host="http://ollama", model="qwen", num_ctx=4096, client=_client(handler))
    completion = await gen.complete(PROMPT, temperature=0.0, max_tokens=64)

    assert completion.text == "Notify within 72 hours."
    assert completion.model == "qwen"
    assert completion.usage is not None
    assert completion.usage.prompt_tokens == 120
    assert completion.usage.completion_tokens == 8


async def test_num_ctx_and_temperature_present_in_request_body() -> None:
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "ok"}, "done": True})

    gen = OllamaGenerator(host="http://ollama", model="qwen", num_ctx=4096, client=_client(handler))
    await gen.complete(PROMPT, temperature=0.3, max_tokens=99)

    body = captured["body"]
    assert body["stream"] is False
    assert body["options"]["num_ctx"] == 4096
    assert body["options"]["temperature"] == 0.3
    assert body["model"] == "qwen"


async def test_non_200_raises_generator_unavailable_naming_ollama_pull() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404, json={"error": "model 'qwen2.5:7b-instruct-q4_K_M' not found, try pulling it first"}
        )

    gen = OllamaGenerator(
        host="http://ollama", model="qwen2.5:7b-instruct-q4_K_M", num_ctx=4096, client=_client(handler)
    )

    with pytest.raises(GeneratorUnavailable, match="ollama pull"):
        await gen.complete(PROMPT, temperature=0.0, max_tokens=64)


async def test_connection_error_raises_generator_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    gen = OllamaGenerator(host="http://ollama", model="qwen", num_ctx=4096, client=_client(handler))

    with pytest.raises(GeneratorUnavailable, match="ollama serve"):
        await gen.complete(PROMPT, temperature=0.0, max_tokens=64)


async def test_generic_server_error_raises_generator_unavailable() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    gen = OllamaGenerator(host="http://ollama", model="qwen", num_ctx=4096, client=_client(handler))

    with pytest.raises(GeneratorUnavailable, match="500"):
        await gen.complete(PROMPT, temperature=0.0, max_tokens=64)


def test_name_returns_model_id() -> None:
    gen = OllamaGenerator(host="http://ollama", model="llama3.2:3b", num_ctx=4096)
    assert gen.name == "llama3.2:3b"


# -------------------------------------------------------------------------- anthropic


async def test_anthropic_missing_key_raises_generator_unavailable() -> None:
    gen = AnthropicGenerator(model="claude-sonnet-5", api_key=None)
    with pytest.raises(GeneratorUnavailable, match="MMU_ANTHROPIC_API_KEY"):
        await gen.complete(PROMPT, temperature=0.0, max_tokens=64)


async def test_anthropic_missing_package_raises_generator_unavailable(monkeypatch) -> None:
    # Simulate the package being absent regardless of whether it happens to be
    # installed in this environment (it is an optional extra, never pulled by a plain
    # `uv sync`) so this test is deterministic either way.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError("simulated: anthropic is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    gen = AnthropicGenerator(model="claude-sonnet-5", api_key="fake-key-for-test")
    with pytest.raises(GeneratorUnavailable, match="uv sync --extra anthropic"):
        await gen.complete(PROMPT, temperature=0.0, max_tokens=64)


# ------------------------------------------------------------------------------ registry


def _settings(**overrides) -> Settings:
    return Settings(**overrides)


def test_alias_resolves_to_configured_ollama_models() -> None:
    settings = _settings(
        generator_backend="ollama",
        ollama_model="qwen2.5:7b-instruct-q4_K_M",
        ollama_fast_model="llama3.2:3b",
    )

    default_gen = build_generator(settings, alias="default")
    fast_gen = build_generator(settings, alias="fast")
    none_gen = build_generator(settings)  # alias=None behaves like "default"

    assert isinstance(default_gen, OllamaGenerator)
    assert default_gen.name == "qwen2.5:7b-instruct-q4_K_M"
    assert fast_gen.name == "llama3.2:3b"
    assert none_gen.name == default_gen.name


def test_unknown_alias_raises_keyerror_listing_known_aliases() -> None:
    settings = _settings(generator_backend="ollama")
    with pytest.raises(KeyError, match="default"):
        build_generator(settings, alias="claude-opus-5")  # a raw model id, not an alias


def test_anthropic_backend_default_alias_resolves() -> None:
    settings = _settings(generator_backend="anthropic", anthropic_model="claude-sonnet-5")
    gen = build_generator(settings, alias="default")
    assert isinstance(gen, AnthropicGenerator)
    assert gen.name == "claude-sonnet-5"


def test_anthropic_backend_unknown_alias_raises_keyerror() -> None:
    settings = _settings(generator_backend="anthropic")
    with pytest.raises(KeyError, match="fast"):
        build_generator(settings, alias="fast")


def test_selecting_ollama_never_imports_anthropic_module() -> None:
    sys.modules.pop("mmu.generate.anthropic", None)
    settings = _settings(generator_backend="ollama")

    build_generator(settings, alias="default")

    assert "mmu.generate.anthropic" not in sys.modules


def test_selecting_anthropic_never_imports_ollama_module() -> None:
    sys.modules.pop("mmu.generate.ollama", None)
    settings = _settings(generator_backend="anthropic", anthropic_model="claude-sonnet-5")

    build_generator(settings, alias="default")

    assert "mmu.generate.ollama" not in sys.modules


def test_available_aliases_reports_local_ollama_aliases() -> None:
    settings = _settings(generator_backend="ollama")
    aliases = available_aliases(settings)
    names = {a["alias"] for a in aliases}
    assert names == {"default", "fast"}
    assert all(a["local"] for a in aliases)


def test_available_aliases_reports_anthropic_as_not_local() -> None:
    settings = _settings(generator_backend="anthropic")
    aliases = available_aliases(settings)
    assert aliases == [
        {"alias": "default", "backend": "anthropic", "local": False, "model": settings.anthropic_model}
    ]
    # Never the key itself.
    assert all("api_key" not in a and "key" not in str(a).lower() for a in aliases)
