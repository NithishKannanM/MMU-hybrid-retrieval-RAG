"""Anthropic backend — the cloud opt-in behind the same :class:`Generator` Protocol.

**Lazy import, always.** ``anthropic`` is an optional extra
(``uv sync --extra anthropic``) and is not installed by a plain ``uv sync`` — this
environment has neither the package nor an API key. Importing it at module load would
make ``import mmu.generate.anthropic`` fail on every box that never opted in, which
would in turn force ``registry.py`` to import it eagerly to find out whether it's usable
— exactly the coupling `registry.py`'s deferred-import design exists to avoid (selecting
the ollama backend must never import this module's third-party dependency at all). The
import happens inside :meth:`AnthropicGenerator._client_or_raise`, on first real use.

Same network-I/O rule as ``ollama.py``: both methods are ``await``s on an
``httpx``-backed SDK call, so neither is ever wrapped in ``asyncio.to_thread``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Sequence

from mmu.generate.base import Completion, GeneratorUnavailable, Message, StreamDelta, Usage

#: The env var Settings actually reads (env_prefix="MMU_" in config.py) — named
#: explicitly in every error below rather than the bare `ANTHROPIC_API_KEY` the
#: underlying SDK also recognizes, because a user who only sets the bare name will not
#: see it picked up by `Settings.anthropic_api_key` and needs the concrete fix.
_API_KEY_ENV_VAR = "MMU_ANTHROPIC_API_KEY"
_EXTRA = "anthropic"


class AnthropicGenerator:
    """Generator backend for the Anthropic Messages API."""

    def __init__(self, *, model: str, api_key: str | None, timeout: float = 300.0) -> None:
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._client: Any = None  # constructed lazily; see module docstring

    @property
    def name(self) -> str:
        return self._model

    def _client_or_raise(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise GeneratorUnavailable(
                f"no Anthropic API key configured. Set {_API_KEY_ENV_VAR} (this is the "
                f"'{_EXTRA}' generator backend, installed via `uv sync --extra {_EXTRA}`) "
                f"to use it."
            )
        try:
            import anthropic  # deferred — see module docstring
        except ImportError as e:
            raise GeneratorUnavailable(
                f"the 'anthropic' package is not installed. Install the optional extra "
                f"with `uv sync --extra {_EXTRA}` and set {_API_KEY_ENV_VAR} to use the "
                f"anthropic generator backend."
            ) from e
        self._client = anthropic.AsyncAnthropic(api_key=self._api_key, timeout=self._timeout)
        return self._client

    @staticmethod
    def _split(messages: Sequence[Message]) -> tuple[str, list[dict[str, str]]]:
        """The Messages API takes `system` as a separate top-level field, not a turn in
        `messages` — `build_messages` always puts exactly one system Message first, so
        this just peels it off; any others (there should be none) are joined in, rather
        than silently dropped, so a caller mistake is visible in the request instead of
        vanishing."""
        system_parts = [m.content for m in messages if m.role == "system"]
        turns = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        return "\n\n".join(system_parts), turns

    async def complete(
        self, messages: Sequence[Message], *, temperature: float, max_tokens: int
    ) -> Completion:
        client = self._client_or_raise()
        system, turns = self._split(messages)
        try:
            resp = await client.messages.create(
                model=self._model,
                system=system,
                messages=turns,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except GeneratorUnavailable:
            raise
        except Exception as e:
            # Collapse every anthropic-SDK exception (auth, rate limit, connection,
            # bad request, ...) into the one typed failure this codebase's callers
            # branch on — see generate/base.py's GeneratorUnavailable docstring. The
            # distinction between those causes is in the wrapped message, not the type.
            raise GeneratorUnavailable(f"Anthropic request failed: {e}") from e
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        usage = Usage(
            prompt_tokens=getattr(resp.usage, "input_tokens", None),
            completion_tokens=getattr(resp.usage, "output_tokens", None),
            model=self._model,
        )
        return Completion(text=text, model=self._model, usage=usage)

    def stream(
        self, messages: Sequence[Message], *, temperature: float, max_tokens: int
    ) -> AsyncIterator[StreamDelta]:
        # Not `async def` — see the Protocol note in ollama.py; the iterator object is
        # returned synchronously and errors surface from `anext()`/the `async for`.
        return self._stream_body(messages, temperature=temperature, max_tokens=max_tokens)

    async def _stream_body(
        self, messages: Sequence[Message], *, temperature: float, max_tokens: int
    ) -> AsyncIterator[StreamDelta]:
        client = self._client_or_raise()
        system, turns = self._split(messages)
        try:
            async with client.messages.stream(
                model=self._model,
                system=system,
                messages=turns,
                temperature=temperature,
                max_tokens=max_tokens,
            ) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield StreamDelta(text=text)
        except GeneratorUnavailable:
            raise
        except Exception as e:
            raise GeneratorUnavailable(f"Anthropic stream failed: {e}") from e
