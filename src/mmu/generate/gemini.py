"""Gemini backend — async HTTP against Google's Generative Language API via ``httpx``.

**No optional extra, unlike ``anthropic.py``.** This module speaks the REST API directly
with ``httpx``, which is already a hard dependency (``ollama.py`` uses it), so there is
no ``google-generativeai`` SDK to install and nothing for ``registry.py`` to import
defensively. The cost is that the request/response shapes below are hand-rolled; the
benefit is that selecting this backend adds zero packages to the environment.

**Key rotation is the reason this class holds a list, not a string.** Gemini's free tier
is rate-limited *per key*, and a benchmark run issues a burst of requests back to back —
exactly the shape that trips a per-minute quota. Given N keys, a 429 is not a failure but
a signal to move to the next key and retry the same request; only when every key has
returned 429 for one request is the run genuinely out of quota. Keys are advanced
round-robin and the rotation offset persists across calls, so load spreads instead of
always hammering key 1 first.

**The thinking-budget trap.** Gemini 2.5+ models reason before answering, and those
reasoning tokens are billed against ``maxOutputTokens``. With a RAG-sized budget
(``max_tokens=1024``) the model can spend the entire allowance thinking and return a
candidate with ``finishReason: MAX_TOKENS`` and **no text part at all** — a silent empty
answer, which this project's metrics would score as a total faithfulness failure and
which looks exactly like a broken prompt. Requests therefore set
``generationConfig.thinkingConfig.thinkingBudget = 0``. A benchmark wants the model's
direct answer, reproducibly, not a variable-length hidden monologue.

**...which not every model accepts.** The ``-lite`` tiers (``gemini-3.5-flash-lite``,
``gemini-flash-lite-latest``) reject ``thinkingConfig`` outright with a bare
``400 Request contains an invalid argument`` — measured, not guessed — while the same
request without it returns 200. Sending the field unconditionally would make those
models permanently unusable behind an error message that names no field. So a 400 on a
request that carried ``thinkingConfig`` is retried once without it, and the answer is
remembered per instance: later calls skip straight to the shape that works. The guard
stays where it is supported and silently steps aside where it is not.

Same network-I/O rule as the other backends: both methods are awaits on a socket, so
neither is ever wrapped in ``asyncio.to_thread``. See ``generate/base.py``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from itertools import islice
from typing import Any, Sequence

import httpx

from mmu.generate.base import Completion, GeneratorUnavailable, Message, StreamDelta, Usage

#: Default host for the Generative Language API.
DEFAULT_HOST = "https://generativelanguage.googleapis.com"

#: API version path segment. v1beta is where `systemInstruction` and `thinkingConfig`
#: live; v1 does not expose them.
API_VERSION = "v1beta"

#: Documented in config.py. Named here too because every error below has to tell a user
#: with no keys exactly which name to set.
_KEYS_ENV_VAR = "MMU_GEMINI_API_KEYS"

#: Status codes worth burning another key on. 429 is the per-key quota; 500/503 are
#: Google-side blips that commonly clear on an immediate retry against another key.
_ROTATE_ON = frozenset({429, 500, 503})


class GeminiGenerator:
    """Generator backend for Google's Generative Language API (``:generateContent``)."""

    def __init__(
        self,
        *,
        model: str,
        api_keys: Sequence[str],
        host: str = DEFAULT_HOST,
        timeout: float = 300.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        #: Order is preserved and duplicates dropped — a copy-paste that repeats one key
        #: would otherwise inflate the apparent quota and make rotation a no-op.
        self._keys: tuple[str, ...] = tuple(dict.fromkeys(k for k in api_keys if k))
        self._host = host.rstrip("/")
        self._timeout = timeout
        #: Tests inject a client wired to ``httpx.MockTransport``; production code
        #: leaves this ``None`` and gets a fresh client per call, closed when done.
        self._client = client
        #: Round-robin cursor, persisted across calls. See the module docstring.
        self._cursor = 0
        #: Tri-state: None = not yet learned, True/False = this model's answer. Set on
        #: the first 400 that follows a request carrying thinkingConfig.
        self._thinking_ok: bool | None = None

    @property
    def name(self) -> str:
        return self._model

    @property
    def key_count(self) -> int:
        """How many distinct keys rotation has to work with. Reported by ``/health``;
        never the keys themselves."""
        return len(self._keys)

    # ---------------------------------------------------------------- request plumbing

    def _keys_or_raise(self) -> tuple[str, ...]:
        if not self._keys:
            raise GeneratorUnavailable(
                f"no Gemini API key configured. Set {_KEYS_ENV_VAR} to a comma-separated "
                f"list of keys (or GEMINI_API_KEY / GEMINI_API_KEY_1..N in .env) to use "
                f"the gemini generator backend."
            )
        return self._keys

    def _key_order(self) -> list[str]:
        """The keys to try for one request, starting at the rotation cursor."""
        keys = self._keys_or_raise()
        n = len(keys)
        start = self._cursor % n
        doubled = keys + keys
        return list(islice(doubled, start, start + n))

    def _advance(self) -> None:
        if self._keys:
            self._cursor = (self._cursor + 1) % len(self._keys)

    def _url(self, *, stream: bool) -> str:
        verb = "streamGenerateContent" if stream else "generateContent"
        base = f"{self._host}/{API_VERSION}/models/{self._model}:{verb}"
        return f"{base}?alt=sse" if stream else base

    @staticmethod
    def _split(messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
        """Gemini takes the system prompt as a separate ``systemInstruction`` field and
        names the assistant role ``model``, not ``assistant``. Mapping the role is not
        cosmetic: an unmapped ``assistant`` turn is rejected as an invalid enum, so a
        multi-turn caller would break only once it sent a second turn."""
        system_parts = [m.content for m in messages if m.role == "system"]
        turns = [
            {
                "role": "model" if m.role == "assistant" else "user",
                "parts": [{"text": m.content}],
            }
            for m in messages
            if m.role != "system"
        ]
        return "\n\n".join(system_parts), turns

    def _body(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_tokens: int,
        thinking: bool | None = None,
    ) -> dict[str, Any]:
        system, turns = self._split(messages)
        config: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        use_thinking = self._thinking_ok is not False if thinking is None else thinking
        if use_thinking:
            # See the module docstring — without this the whole output budget can be
            # spent thinking and the answer comes back empty.
            config["thinkingConfig"] = {"thinkingBudget": 0}
        body: dict[str, Any] = {"contents": turns, "generationConfig": config}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    @staticmethod
    def _sent_thinking(body: dict[str, Any]) -> bool:
        return "thinkingConfig" in body.get("generationConfig", {})

    def _retry_without_thinking(self, resp: httpx.Response, body: dict[str, Any]) -> bool:
        """``True`` when this 400 is the lite tier refusing ``thinkingConfig``.

        Narrow on purpose: only a 400, only when the request actually carried the field.
        A 400 for any other reason (malformed content, bad temperature) still fails
        loudly rather than being retried into a second, equally wrong request.
        """
        if resp.status_code != 400 or not self._sent_thinking(body):
            return False
        self._thinking_ok = False
        body["generationConfig"].pop("thinkingConfig", None)
        return True

    def _borrow_client(self) -> tuple[httpx.AsyncClient, bool]:
        if self._client is not None:
            return self._client, False
        return httpx.AsyncClient(timeout=self._timeout), True

    @staticmethod
    def _detail(resp: httpx.Response) -> str:
        try:
            payload = resp.json()
        except Exception:
            return resp.text[:400]
        err = payload.get("error", payload) if isinstance(payload, dict) else payload
        if isinstance(err, dict):
            return str(err.get("message", err))[:400]
        return str(err)[:400]

    def _fail(self, resp: httpx.Response, *, attempts: int) -> None:
        detail = self._detail(resp)
        if resp.status_code == 429:
            raise GeneratorUnavailable(
                f"Gemini rate-limited every one of the {attempts} configured key(s) for "
                f"model {self._model!r}. The free tier is per-key and per-minute; wait "
                f"and retry, or add more keys to {_KEYS_ENV_VAR}. Google said: {detail}"
            )
        if resp.status_code in (400, 401, 403):
            raise GeneratorUnavailable(
                f"Gemini rejected the API key for model {self._model!r} "
                f"(HTTP {resp.status_code}). Check the values in {_KEYS_ENV_VAR}. "
                f"Google said: {detail}"
            )
        if resp.status_code == 404:
            raise GeneratorUnavailable(
                f"Gemini model {self._model!r} does not exist or does not support "
                f"generateContent. Google said: {detail}"
            )
        raise GeneratorUnavailable(
            f"Gemini returned HTTP {resp.status_code} for model {self._model!r}: {detail}"
        )

    @staticmethod
    def _text_of(payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates") or []
        if not candidates:
            return ""
        parts = (candidates[0].get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict))

    # ------------------------------------------------------------------------ complete

    async def complete(
        self, messages: Sequence[Message], *, temperature: float, max_tokens: int
    ) -> Completion:
        body = self._body(messages, temperature=temperature, max_tokens=max_tokens)
        keys = self._key_order()
        client, owns = self._borrow_client()
        last: httpx.Response | None = None
        try:
            for key in keys:
                resp = await self._post(client, body, key=key)
                if resp.status_code == 400 and self._retry_without_thinking(resp, body):
                    # Same key, same request minus the field this model rejects.
                    resp = await self._post(client, body, key=key)
                if resp.status_code == 200:
                    self._advance()
                    data = resp.json()
                    return self._completion(data)
                last = resp
                if resp.status_code not in _ROTATE_ON:
                    self._fail(resp, attempts=len(keys))
                self._advance()
            assert last is not None  # the loop body always assigns before exhausting
            self._fail(last, attempts=len(keys))
            raise AssertionError("unreachable")  # pragma: no cover
        finally:
            if owns:
                await client.aclose()

    def _completion(self, data: dict[str, Any]) -> Completion:
        text = self._text_of(data)
        meta = data.get("usageMetadata") or {}
        usage = Usage(
            prompt_tokens=meta.get("promptTokenCount"),
            completion_tokens=meta.get("candidatesTokenCount"),
            model=self._model,
        )
        if not text:
            finish = ((data.get("candidates") or [{}])[0]).get("finishReason")
            if finish and finish != "STOP":
                raise GeneratorUnavailable(
                    f"Gemini returned no text for model {self._model!r} "
                    f"(finishReason={finish}). If this is MAX_TOKENS, raise "
                    f"MMU_MAX_TOKENS; if SAFETY, the prompt was blocked."
                )
        return Completion(text=text, model=self._model, usage=usage)

    async def _post(
        self, client: httpx.AsyncClient, body: dict[str, Any], *, key: str
    ) -> httpx.Response:
        try:
            return await client.post(
                self._url(stream=False),
                json=body,
                headers={"x-goog-api-key": key},
                timeout=self._timeout,
            )
        except httpx.ConnectError as e:
            raise GeneratorUnavailable(f"cannot reach Gemini at {self._host} ({e})") from e
        except httpx.TimeoutException as e:
            raise GeneratorUnavailable(
                f"Gemini request to {self._host} timed out after {self._timeout}s"
            ) from e

    # -------------------------------------------------------------------------- stream

    def stream(
        self, messages: Sequence[Message], *, temperature: float, max_tokens: int
    ) -> AsyncIterator[StreamDelta]:
        # Not `async def` — matches the Protocol in generate/base.py, same as the other
        # two backends: the iterator is returned synchronously.
        body = self._body(messages, temperature=temperature, max_tokens=max_tokens)
        return self._stream_body(body)

    async def _stream_body(self, body: dict[str, Any]) -> AsyncIterator[StreamDelta]:
        keys = self._key_order()
        client, owns = self._borrow_client()
        try:
            for i, key in enumerate(keys):
                # At most two attempts per key: the second only to drop a thinkingConfig
                # this model rejects (see _retry_without_thinking).
                for _ in range(2):
                    retry_same_key = False
                    opened = False
                    try:
                        async with client.stream(
                            "POST",
                            self._url(stream=True),
                            json=body,
                            headers={"x-goog-api-key": key},
                            timeout=self._timeout,
                        ) as resp:
                            if resp.status_code != 200:
                                await resp.aread()
                                if self._retry_without_thinking(resp, body):
                                    retry_same_key = True
                                else:
                                    self._advance()
                                    # Rotate while a key remains; the last one reports.
                                    if resp.status_code in _ROTATE_ON and i < len(keys) - 1:
                                        break
                                    self._fail(resp, attempts=len(keys))
                            else:
                                opened = True
                                async for delta in self._parse_sse(resp):
                                    yield delta
                    except httpx.ConnectError as e:
                        raise GeneratorUnavailable(
                            f"cannot reach Gemini at {self._host} ({e})"
                        ) from e
                    except httpx.TimeoutException as e:
                        raise GeneratorUnavailable(
                            f"Gemini stream to {self._host} timed out after {self._timeout}s"
                        ) from e
                    if opened:
                        self._advance()
                        return
                    if not retry_same_key:
                        break
        finally:
            if owns:
                await client.aclose()

    @classmethod
    async def _parse_sse(cls, resp: httpx.Response) -> AsyncIterator[StreamDelta]:
        """Assemble SSE ``data:`` frames from a text stream.

        The same carry-over-buffer discipline as ``ollama.py``'s NDJSON parser, and for
        the same reason: transport chunk boundaries have no relationship to line
        boundaries, so splitting each chunk in isolation truncates JSON at random. SSE
        adds one wrinkle — a frame is terminated by a blank line and its payload is
        prefixed with ``data: `` — so lines are accumulated until the blank separator.
        """
        buffer = ""
        async for text in resp.aiter_text():
            buffer += text
            lines = buffer.split("\n")
            buffer = lines.pop()  # trailing partial line, kept for the next chunk
            for raw in lines:
                line = raw.strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if not payload or payload == "[DONE]":
                    continue
                piece = cls._text_of(json.loads(payload))
                if piece:
                    yield StreamDelta(text=piece)
        tail = buffer.strip()
        if tail.startswith("data:"):
            payload = tail[len("data:") :].strip()
            if payload and payload != "[DONE]":
                piece = cls._text_of(json.loads(payload))
                if piece:
                    yield StreamDelta(text=piece)
