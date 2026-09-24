"""The generator contract.

**Both methods are async, and neither is ever offloaded to a thread.** Every backend
here is network I/O — an ``httpx.AsyncClient`` call to Ollama on localhost or to the
Anthropic API. Wrapping async network I/O in ``asyncio.to_thread`` is the classic
inversion of the fix applied in :mod:`mmu.api.deps`: it burns a worker thread per
request to wait on a socket that the event loop was already able to wait on for free.
The thread-pool offload in this codebase exists for exactly one thing, CPU-bound
retrieval, and this module is the note explaining why it stops here.

``GeneratorUnavailable`` is a typed failure because the two callers need to react
differently: ``POST /answer`` turns it into a 503 with the reason, while
``POST /answer/stream`` emits an ``error`` event — and neither may fall back to a
different model once tokens are already on the user's screen.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol, Sequence, runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class StreamDelta:
    """One incremental piece of generated text."""

    text: str


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model: str = ""


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    usage: Usage | None = None


class GeneratorUnavailable(RuntimeError):
    """The backend cannot serve this request: not running, no model pulled, no key."""


@runtime_checkable
class Generator(Protocol):
    @property
    def name(self) -> str:
        """The concrete model identifier, for the report header and Answer.model."""
        ...

    async def complete(
        self, messages: Sequence[Message], *, temperature: float, max_tokens: int
    ) -> Completion: ...

    def stream(
        self, messages: Sequence[Message], *, temperature: float, max_tokens: int
    ) -> AsyncIterator[StreamDelta]:
        """Not ``async def`` — returns the async iterator directly, so callers write
        ``async for d in gen.stream(...)`` without an extra await."""
        ...
