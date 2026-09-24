"""Ollama backend — async HTTP against a local ``ollama serve``, via ``httpx.AsyncClient``.

**This is network I/O and is therefore never wrapped in ``asyncio.to_thread``.** Both
``complete`` and ``stream`` await an ``httpx.AsyncClient`` call over a socket; the event
loop can wait on that socket for free. Offloading it to a worker thread would burn a
thread per request to babysit an await that costs nothing, which is exactly the
inversion `generate/base.py` warns about — the thread-pool offload in this codebase is
reserved for the one genuinely CPU-bound call, `HybridRetriever.search`.

**The context-window trap this module exists to not fall into.** Ollama defaults
``num_ctx`` to 2048 tokens unless a request says otherwise, silently — there is no
error, no truncation warning, nothing in the response that flags it. A RAG prompt
carrying ``top_k=8`` legal chunks at ~384 tokens each is 3k+ tokens of context alone,
before the system prompt and the question. Ollama does not refuse that; it keeps only
the *last* ``num_ctx`` tokens of the prompt, which means the truncation eats the system
prompt and the earliest (highest-ranked) sources first and leaves the model answering
from a fragment of the middle-to-late context. The failure this produces looks exactly
like a retrieval bug — wrong or missing citations, an answer that ignores a source that
was clearly retrieved — and nothing about it points at the prompt. Every request below
sets ``options.num_ctx`` explicitly from ``settings.ollama_num_ctx`` for this reason;
never rely on the server-side default.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Sequence

import httpx

from mmu.generate.base import Completion, GeneratorUnavailable, Message, StreamDelta, Usage


class OllamaGenerator:
    """Generator backend for a local Ollama server, speaking ``POST /api/chat``."""

    def __init__(
        self,
        *,
        host: str,
        model: str,
        num_ctx: int,
        timeout: float = 300.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._num_ctx = num_ctx
        self._timeout = timeout
        #: Tests inject a client wired to ``httpx.MockTransport``; production code
        #: leaves this ``None`` and gets a fresh client per call, closed when done.
        self._client = client

    @property
    def name(self) -> str:
        return self._model

    def _borrow_client(self) -> tuple[httpx.AsyncClient, bool]:
        if self._client is not None:
            return self._client, False
        return httpx.AsyncClient(timeout=self._timeout), True

    def _body(self, messages: Sequence[Message], *, temperature: float, max_tokens: int, stream: bool) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "options": {
                # See the module docstring — this is the one line standing between a
                # normal RAG prompt and a silent left-truncation.
                "num_ctx": self._num_ctx,
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

    def _fail_unavailable(self, resp: httpx.Response) -> None:
        try:
            detail = resp.json().get("error", resp.text)
        except Exception:
            detail = resp.text
        detail_l = str(detail).lower()
        if resp.status_code == 404 or "not found" in detail_l:
            raise GeneratorUnavailable(
                f"Ollama model {self._model!r} is not pulled on the server at "
                f"{self._host} (it will not appear in `ollama list`). Run "
                f"`ollama pull {self._model}` and retry. Server said: {detail}"
            )
        raise GeneratorUnavailable(
            f"Ollama at {self._host} returned HTTP {resp.status_code} for model "
            f"{self._model!r}: {detail}"
        )

    async def complete(
        self, messages: Sequence[Message], *, temperature: float, max_tokens: int
    ) -> Completion:
        body = self._body(messages, temperature=temperature, max_tokens=max_tokens, stream=False)
        client, owns = self._borrow_client()
        try:
            resp = await self._post(client, body)
            if resp.status_code != 200:
                self._fail_unavailable(resp)
            data = resp.json()
        finally:
            if owns:
                await client.aclose()
        text = data.get("message", {}).get("content", "")
        usage = Usage(
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            model=self._model,
        )
        return Completion(text=text, model=self._model, usage=usage)

    def stream(
        self, messages: Sequence[Message], *, temperature: float, max_tokens: int
    ) -> AsyncIterator[StreamDelta]:
        # Not `async def` — matches the Protocol in generate/base.py: this returns the
        # async iterator directly, so `async for d in gen.stream(...)` needs no extra
        # await and a GeneratorUnavailable raised before the first byte propagates from
        # the *call*, not from the first `anext()`.
        body = self._body(messages, temperature=temperature, max_tokens=max_tokens, stream=True)
        return self._stream_body(body)

    async def _stream_body(self, body: dict[str, Any]) -> AsyncIterator[StreamDelta]:
        client, owns = self._borrow_client()
        try:
            async with client.stream("POST", f"{self._host}/api/chat", json=body, timeout=self._timeout) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    self._fail_unavailable(resp)
                async for delta in self._parse_ndjson(resp):
                    yield delta
        except httpx.ConnectError as e:
            raise GeneratorUnavailable(
                f"cannot reach Ollama at {self._host} — is `ollama serve` running? ({e})"
            ) from e
        except httpx.TimeoutException as e:
            raise GeneratorUnavailable(
                f"Ollama request to {self._host} timed out after {self._timeout}s"
            ) from e
        finally:
            if owns:
                await client.aclose()

    @staticmethod
    async def _parse_ndjson(resp: httpx.Response) -> AsyncIterator[StreamDelta]:
        """Assemble NDJSON lines from a text stream whose chunk boundaries have no
        relationship to line boundaries.

        **The bug this exists to prevent.** ``for line in chunk.splitlines()`` over each
        transport chunk independently is the naive approach, and it is wrong: TCP (and
        httpx's own internal buffering) chunks bytes on no particular boundary, so a
        single NDJSON line is routinely split across two chunks. Splitting each chunk in
        isolation either drops the second half of that line or hands ``json.loads`` a
        truncated object that raises. The fix is a single carry-over buffer: every chunk
        is appended to it, every *complete* line (i.e. every element but the last after
        splitting on ``\\n``) is parsed and emitted, and the trailing partial segment —
        which may be empty, or may be half a JSON object — is kept for the next chunk.
        """
        buffer = ""
        async for text in resp.aiter_text():
            buffer += text
            lines = buffer.split("\n")
            buffer = lines.pop()  # last element: either "" or an incomplete line
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                piece = obj.get("message", {}).get("content", "")
                if piece:
                    yield StreamDelta(text=piece)
                if obj.get("done"):
                    return
        # A final line with no trailing newline lands here instead of in the loop above.
        tail = buffer.strip()
        if tail:
            obj = json.loads(tail)
            piece = obj.get("message", {}).get("content", "")
            if piece:
                yield StreamDelta(text=piece)

    async def _post(self, client: httpx.AsyncClient, body: dict[str, Any]) -> httpx.Response:
        try:
            return await client.post(f"{self._host}/api/chat", json=body, timeout=self._timeout)
        except httpx.ConnectError as e:
            raise GeneratorUnavailable(
                f"cannot reach Ollama at {self._host} — is `ollama serve` running? ({e})"
            ) from e
        except httpx.TimeoutException as e:
            raise GeneratorUnavailable(
                f"Ollama request to {self._host} timed out after {self._timeout}s"
            ) from e
