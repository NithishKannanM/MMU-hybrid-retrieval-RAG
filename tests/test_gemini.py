"""GeminiGenerator over ``httpx.MockTransport`` — no network, no key, no quota.

Two tests here carry the weight. ``test_rotates_to_next_key_on_429`` is the reason the
backend holds a list of keys at all: Gemini's free tier is per-key, a benchmark burst
trips it, and rotation is what turns that from a failed run into a slower one.
``test_empty_text_with_max_tokens_finish_raises`` covers the thinking-budget trap, where
a 200 OK carries no answer and every downstream metric would score the silence as a
faithfulness failure rather than a configuration error.
"""

from __future__ import annotations

import json
import sys

import httpx
import pytest

from mmu.config import Settings, _discover_gemini_keys
from mmu.generate.base import GeneratorUnavailable, Message
from mmu.generate.gemini import GeminiGenerator
from mmu.generate.registry import available_aliases, build_generator

PROMPT = [
    Message(role="system", content="Answer with [S1] markers."),
    Message(role="user", content="When must a breach be reported?"),
]


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ok(text: str, *, prompt_tokens: int = 11, completion_tokens: int = 5) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": prompt_tokens,
                "candidatesTokenCount": completion_tokens,
            },
        },
    )


def _sse(*frames: bytes):
    async def handler(request: httpx.Request) -> httpx.Response:
        async def body():
            for f in frames:
                yield f

        return httpx.Response(200, content=body())

    return handler


def _frame(text: str) -> str:
    return "data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": text}]}}]}) + "\n\n"


# ------------------------------------------------------------------------------ complete


@pytest.mark.anyio
async def test_complete_returns_text_and_usage() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return _ok("Within 72 hours [S1].")

    gen = GeminiGenerator(model="gemini-2.5-flash", api_keys=["k1"], client=_client(handler))
    out = await gen.complete(PROMPT, temperature=0.0, max_tokens=256)
    assert out.text == "Within 72 hours [S1]."
    assert out.model == "gemini-2.5-flash"
    assert out.usage is not None
    assert out.usage.prompt_tokens == 11
    assert out.usage.completion_tokens == 5


@pytest.mark.anyio
async def test_system_prompt_goes_to_system_instruction_and_role_is_mapped() -> None:
    """Gemini names the assistant role ``model`` and takes the system prompt in its own
    field; an unmapped ``assistant`` turn is rejected as an invalid enum."""
    seen: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return _ok("ok")

    msgs = [*PROMPT, Message(role="assistant", content="prior turn")]
    gen = GeminiGenerator(model="gemini-2.5-flash", api_keys=["k1"], client=_client(handler))
    await gen.complete(msgs, temperature=0.0, max_tokens=256)

    assert seen["systemInstruction"]["parts"][0]["text"] == "Answer with [S1] markers."
    roles = [c["role"] for c in seen["contents"]]
    assert roles == ["user", "model"]
    assert all(r != "assistant" for r in roles)


@pytest.mark.anyio
async def test_thinking_budget_is_zero_and_generation_config_is_sent() -> None:
    """The trap this guards: reasoning tokens bill against maxOutputTokens, so a 1024
    budget can be spent entirely on thinking and return an empty answer."""
    seen: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return _ok("ok")

    gen = GeminiGenerator(model="gemini-2.5-flash", api_keys=["k1"], client=_client(handler))
    await gen.complete(PROMPT, temperature=0.0, max_tokens=1024)

    cfg = seen["generationConfig"]
    assert cfg["thinkingConfig"]["thinkingBudget"] == 0
    assert cfg["temperature"] == 0.0
    assert cfg["maxOutputTokens"] == 1024


@pytest.mark.anyio
async def test_api_key_is_sent_as_header_not_query_string() -> None:
    """A key in the URL lands in server logs and proxy history; the header does not."""
    seen: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["url"] = str(request.url)
        return _ok("ok")

    gen = GeminiGenerator(model="gemini-2.5-flash", api_keys=["secret-key"], client=_client(handler))
    await gen.complete(PROMPT, temperature=0.0, max_tokens=64)
    assert seen["key"] == "secret-key"
    assert "secret-key" not in seen["url"]


# ------------------------------------------------------------------------------ rotation


@pytest.mark.anyio
async def test_rotates_to_next_key_on_429() -> None:
    """The whole point of holding several keys: a per-key quota is a signal to move on,
    not a failed run."""
    tried: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        key = request.headers["x-goog-api-key"]
        tried.append(key)
        if key in ("k1", "k2"):
            return httpx.Response(429, json={"error": {"message": "quota exceeded"}})
        return _ok("answered on the third key")

    gen = GeminiGenerator(
        model="gemini-2.5-flash", api_keys=["k1", "k2", "k3"], client=_client(handler)
    )
    out = await gen.complete(PROMPT, temperature=0.0, max_tokens=64)
    assert out.text == "answered on the third key"
    assert tried == ["k1", "k2", "k3"]


@pytest.mark.anyio
async def test_all_keys_rate_limited_raises_naming_the_count() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "quota exceeded"}})

    gen = GeminiGenerator(
        model="gemini-2.5-flash", api_keys=["k1", "k2"], client=_client(handler)
    )
    with pytest.raises(GeneratorUnavailable) as e:
        await gen.complete(PROMPT, temperature=0.0, max_tokens=64)
    assert "2 configured key(s)" in str(e.value)


@pytest.mark.anyio
async def test_cursor_advances_across_calls_so_load_spreads() -> None:
    """Without a persisted cursor every run would hammer key 1 first and exhaust it
    while the others sat idle."""
    first_of_each_call: list[str] = []
    seen_this_call: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_this_call.append(request.headers["x-goog-api-key"])
        return _ok("ok")

    gen = GeminiGenerator(
        model="gemini-2.5-flash", api_keys=["k1", "k2", "k3"], client=_client(handler)
    )
    for _ in range(3):
        seen_this_call.clear()
        await gen.complete(PROMPT, temperature=0.0, max_tokens=64)
        first_of_each_call.append(seen_this_call[0])
    assert first_of_each_call == ["k1", "k2", "k3"]


def test_duplicate_keys_are_collapsed() -> None:
    """A copy-paste that repeats a key would otherwise inflate the apparent quota and
    make rotation retry the same exhausted key."""
    gen = GeminiGenerator(model="gemini-2.5-flash", api_keys=["k1", "k1", "k2", ""])
    assert gen.key_count == 2


@pytest.mark.anyio
async def test_no_keys_raises_naming_the_env_var() -> None:
    gen = GeminiGenerator(model="gemini-2.5-flash", api_keys=[])
    with pytest.raises(GeneratorUnavailable) as e:
        await gen.complete(PROMPT, temperature=0.0, max_tokens=64)
    assert "MMU_GEMINI_API_KEYS" in str(e.value)


# ------------------------------------------------------------------ thinkingConfig fallback


@pytest.mark.anyio
async def test_retries_without_thinking_config_on_400() -> None:
    """Measured against the live API: the ``-lite`` tiers reject ``thinkingConfig`` with
    a bare 400 naming no field. Sending it unconditionally would make them unusable."""
    bodies: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "thinkingConfig" in body["generationConfig"]:
            return httpx.Response(400, json={"error": {"message": "Request contains an invalid argument."}})
        return _ok("ok")

    gen = GeminiGenerator(model="gemini-3.5-flash-lite", api_keys=["k1"], client=_client(handler))
    out = await gen.complete(PROMPT, temperature=0.0, max_tokens=64)
    assert out.text == "ok"
    assert len(bodies) == 2
    assert "thinkingConfig" in bodies[0]["generationConfig"]
    assert "thinkingConfig" not in bodies[1]["generationConfig"]


@pytest.mark.anyio
async def test_thinking_refusal_is_remembered_so_later_calls_send_one_request() -> None:
    bodies: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "thinkingConfig" in body["generationConfig"]:
            return httpx.Response(400, json={"error": {"message": "invalid argument"}})
        return _ok("ok")

    gen = GeminiGenerator(model="gemini-3.5-flash-lite", api_keys=["k1"], client=_client(handler))
    await gen.complete(PROMPT, temperature=0.0, max_tokens=64)
    bodies.clear()
    await gen.complete(PROMPT, temperature=0.0, max_tokens=64)
    assert len(bodies) == 1
    assert "thinkingConfig" not in bodies[0]["generationConfig"]


@pytest.mark.anyio
async def test_unrelated_400_is_not_retried() -> None:
    """The retry is narrow on purpose: a genuinely malformed request must fail once and
    loudly, not twice and confusingly."""
    calls: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(400, json={"error": {"message": "temperature must be <= 2.0"}})

    gen = GeminiGenerator(model="gemini-3.1-flash-lite", api_keys=["k1"], client=_client(handler))
    with pytest.raises(GeneratorUnavailable):
        # First 400 carries thinkingConfig so one retry is legitimate; the retry's 400
        # does not, and must not loop.
        await gen.complete(PROMPT, temperature=0.0, max_tokens=64)
    assert len(calls) == 2


# -------------------------------------------------------------------------- failure shapes


@pytest.mark.anyio
async def test_empty_text_with_max_tokens_finish_raises() -> None:
    """A 200 OK carrying no text: the thinking-budget trap. Downstream this would score
    as a total faithfulness failure, so it must fail loudly as a config error instead."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]},
        )

    gen = GeminiGenerator(model="gemini-2.5-flash", api_keys=["k1"], client=_client(handler))
    with pytest.raises(GeneratorUnavailable) as e:
        await gen.complete(PROMPT, temperature=0.0, max_tokens=8)
    assert "MAX_TOKENS" in str(e.value)


@pytest.mark.anyio
async def test_bad_key_raises_without_rotating() -> None:
    """403 means the key is wrong, not throttled — burning the other keys on it would
    turn one configuration mistake into N identical failures."""
    tried: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        tried.append(request.headers["x-goog-api-key"])
        return httpx.Response(403, json={"error": {"message": "API key not valid"}})

    gen = GeminiGenerator(
        model="gemini-2.5-flash", api_keys=["k1", "k2", "k3"], client=_client(handler)
    )
    with pytest.raises(GeneratorUnavailable) as e:
        await gen.complete(PROMPT, temperature=0.0, max_tokens=64)
    assert tried == ["k1"]
    assert "rejected the API key" in str(e.value)


@pytest.mark.anyio
async def test_unknown_model_raises_naming_the_model() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "models/nope is not found"}})

    gen = GeminiGenerator(model="nope", api_keys=["k1"], client=_client(handler))
    with pytest.raises(GeneratorUnavailable) as e:
        await gen.complete(PROMPT, temperature=0.0, max_tokens=64)
    assert "nope" in str(e.value)


@pytest.mark.anyio
async def test_connection_error_raises_generator_unavailable() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    gen = GeminiGenerator(model="gemini-2.5-flash", api_keys=["k1"], client=_client(handler))
    with pytest.raises(GeneratorUnavailable):
        await gen.complete(PROMPT, temperature=0.0, max_tokens=64)


# -------------------------------------------------------------------------------- stream


@pytest.mark.anyio
async def test_stream_assembles_sse_frames() -> None:
    handler = _sse(_frame("Within ").encode(), _frame("72 hours [S1].").encode())
    gen = GeminiGenerator(model="gemini-2.5-flash", api_keys=["k1"], client=_client(handler))
    got = "".join([d.text async for d in gen.stream(PROMPT, temperature=0.0, max_tokens=64)])
    assert got == "Within 72 hours [S1]."


@pytest.mark.anyio
async def test_stream_assembles_frame_split_across_chunks() -> None:
    """The bug this exists to prevent: transport chunk boundaries have no relationship
    to line boundaries, so a frame routinely arrives in two pieces."""
    whole = _frame("Within 72 hours [S1].")
    cut = len(whole) // 2
    handler = _sse(whole[:cut].encode(), whole[cut:].encode())
    gen = GeminiGenerator(model="gemini-2.5-flash", api_keys=["k1"], client=_client(handler))
    got = "".join([d.text async for d in gen.stream(PROMPT, temperature=0.0, max_tokens=64)])
    assert got == "Within 72 hours [S1]."


@pytest.mark.anyio
async def test_stream_rotates_on_429() -> None:
    calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        key = request.headers["x-goog-api-key"]
        calls.append(key)
        if key == "k1":
            return httpx.Response(429, json={"error": {"message": "quota"}})

        async def body():
            yield _frame("ok").encode()

        return httpx.Response(200, content=body())

    gen = GeminiGenerator(
        model="gemini-2.5-flash", api_keys=["k1", "k2"], client=_client(handler)
    )
    got = "".join([d.text async for d in gen.stream(PROMPT, temperature=0.0, max_tokens=64)])
    assert got == "ok"
    assert calls == ["k1", "k2"]


# ------------------------------------------------------------------------------ registry


def test_gemini_aliases_resolve_to_configured_models() -> None:
    s = Settings(generator_backend="gemini", gemini_api_keys=("k1",))
    assert build_generator(s, "default").name == s.gemini_model
    assert build_generator(s, "fast").name == s.gemini_fast_model


def test_gemini_unknown_alias_raises_keyerror_listing_known_aliases() -> None:
    s = Settings(generator_backend="gemini", gemini_api_keys=("k1",))
    with pytest.raises(KeyError) as e:
        build_generator(s, "turbo")
    assert "default" in str(e.value) and "fast" in str(e.value)


def test_selecting_gemini_never_imports_ollama_or_anthropic(monkeypatch) -> None:
    """Same deferred-import guarantee the other two backends hold each other to.

    ``monkeypatch.delitem`` rather than ``sys.modules.pop``: popping permanently would
    let a later import rebuild the module as a *different* object, so every subsequent
    ``isinstance(gen, OllamaGenerator)`` in the suite compares against a stale class and
    fails for reasons that have nothing to do with the test that broke it.
    """
    for mod in ("mmu.generate.ollama", "mmu.generate.anthropic"):
        monkeypatch.delitem(sys.modules, mod, raising=False)
    s = Settings(generator_backend="gemini", gemini_api_keys=("k1",))
    build_generator(s, "default")
    assert "mmu.generate.ollama" not in sys.modules
    assert "mmu.generate.anthropic" not in sys.modules


def test_available_aliases_reports_gemini_as_not_local_and_leaks_no_key() -> None:
    s = Settings(generator_backend="gemini", gemini_api_keys=("super-secret",))
    rows = available_aliases(s)
    assert {r["alias"] for r in rows} == {"default", "fast"}
    assert all(r["local"] is False for r in rows)
    assert "super-secret" not in json.dumps(rows)


def test_unknown_backend_error_lists_gemini() -> None:
    s = Settings(generator_backend="llamafile")
    with pytest.raises(ValueError) as e:
        build_generator(s)
    assert "gemini" in str(e.value)


# ------------------------------------------------------------------- key discovery


def test_comma_separated_keys_are_split() -> None:
    s = Settings(generator_backend="gemini", gemini_api_keys="a, b ,c")
    assert s.gemini_api_keys == ("a", "b", "c")


@pytest.fixture
def no_ambient_keys(monkeypatch, tmp_path):
    """Isolate discovery from the developer's own environment and real ``.env``.

    Without this the assertions below depend on whoever runs the suite — and a failure
    diff prints their live API keys into the terminal.
    """
    for name in [n for n in list(__import__("os").environ) if n.startswith("GEMINI_API_KEY")]:
        monkeypatch.delenv(name, raising=False)
    return tmp_path / "empty.env"


def test_discovers_numbered_bare_keys_in_numeric_order(monkeypatch, no_ambient_keys) -> None:
    """``GEMINI_API_KEY_10`` must sort after ``_9``, not between ``_1`` and ``_2`` —
    rotation order has to be reproducible across processes."""
    for n in (1, 2, 9, 10):
        monkeypatch.setenv(f"GEMINI_API_KEY_{n}", f"key{n}")
    assert _discover_gemini_keys(no_ambient_keys) == ("key1", "key2", "key9", "key10")


def test_discovers_unnumbered_bare_key(monkeypatch, no_ambient_keys) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "solo")
    assert _discover_gemini_keys(no_ambient_keys) == ("solo",)


def test_reads_numbered_keys_from_an_env_file(no_ambient_keys) -> None:
    """The shape of a real .env: a comment, `export` prefixes, quoted values."""
    no_ambient_keys.write_text(
        "#Gemini api\n"
        "GEMINI_API_KEY_1=alpha\n"
        "export GEMINI_API_KEY_2='beta'\n"
        'GEMINI_API_KEY_3="gamma"\n'
        "UNRELATED=ignored\n",
        encoding="utf-8",
    )
    assert _discover_gemini_keys(no_ambient_keys) == ("alpha", "beta", "gamma")


def test_missing_env_file_is_not_an_error(no_ambient_keys) -> None:
    assert _discover_gemini_keys(no_ambient_keys) == ()
