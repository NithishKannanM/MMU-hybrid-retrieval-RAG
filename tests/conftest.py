"""Shared fakes and fixtures. Nothing here downloads a model or opens a socket.

The default ``uv run pytest`` deselects ``-m live``, so every test that runs by default
is served by the doubles below. Two of them are load-bearing rather than convenient:

* :class:`RecordingEmbedder` is what makes the query/document asymmetry testable. The
  bug it exists to catch is not "the prefix function is wrong" — that is easy — but
  "the indexer and the searcher called the embedder with the same ``kind``", which is
  invisible in the output and only shows up as slightly worse rankings.
* :class:`FakeGenerator` streams in scripted pieces *including mid-word splits*, because
  the real Ollama client assembles NDJSON across TCP chunk boundaries and a fake that
  always yields whole tokens would let that bug through.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import httpx
import pytest

from mmu.core.models import Chunk, RetrievedChunk
from mmu.embed.base import Encoding, InputKind
from mmu.embed.hashing import HashingEmbedder
from mmu.generate.base import Completion, GeneratorUnavailable, Message, StreamDelta, Usage


# --------------------------------------------------------------------------- embedders


@dataclass
class RecordingEmbedder:
    """Wraps an embedder and records every ``(texts, kind)`` call it receives."""

    inner: object = field(default_factory=HashingEmbedder)
    calls: list[tuple[tuple[str, ...], InputKind]] = field(default_factory=list)

    def encode(self, texts: Sequence[str], *, kind: InputKind) -> Encoding:
        self.calls.append((tuple(texts), kind))
        return self.inner.encode(texts, kind=kind)

    @property
    def dim(self) -> int:
        return self.inner.dim

    @property
    def model_name(self) -> str:
        return self.inner.model_name

    def kinds(self) -> list[InputKind]:
        return [kind for _, kind in self.calls]


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def recording_embedder() -> RecordingEmbedder:
    return RecordingEmbedder()


# -------------------------------------------------------------------------- generators


@dataclass
class FakeGenerator:
    """Scripted generator. Implements :class:`mmu.generate.base.Generator`."""

    reply: str = "The controller must notify within 72 hours [S1]."
    model: str = "fake-model"
    #: Raise on the next call, to exercise the 503 path and the SSE ``error`` event.
    unavailable: bool = False
    #: Raise *after* this many stream deltas, to exercise mid-stream failure — the case
    #: where a partial answer is already on the user's screen and a silent fallback to
    #: another model would splice two systems' output together.
    fail_after: int | None = None
    prompts: list[tuple[Message, ...]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.model

    async def complete(
        self, messages: Sequence[Message], *, temperature: float, max_tokens: int
    ) -> Completion:
        self.prompts.append(tuple(messages))
        if self.unavailable:
            raise GeneratorUnavailable("fake generator is unavailable")
        return Completion(text=self.reply, model=self.model, usage=Usage(model=self.model))

    def stream(
        self, messages: Sequence[Message], *, temperature: float, max_tokens: int
    ) -> AsyncIterator[StreamDelta]:
        self.prompts.append(tuple(messages))

        async def it() -> AsyncIterator[StreamDelta]:
            if self.unavailable:
                raise GeneratorUnavailable("fake generator is unavailable")
            # Deliberately uneven pieces, some splitting a word, mirroring how tokens
            # actually arrive over the wire.
            pieces = [self.reply[i : i + 7] for i in range(0, len(self.reply), 7)]
            for n, piece in enumerate(pieces):
                if self.fail_after is not None and n >= self.fail_after:
                    raise GeneratorUnavailable("fake generator died mid-stream")
                yield StreamDelta(text=piece)

        return it()


@pytest.fixture
def generator() -> FakeGenerator:
    return FakeGenerator()


# ------------------------------------------------------------------------------ corpus


#: Three tiny documents that mirror the real benchmark shape: a regulation, a national
#: law and a company policy, where the answer to the seed question exists in NONE of
#: them alone. Used by retrieval, API and eval tests.
MINI_CORPUS: dict[str, str] = {
    "gdpr.txt": (
        "Article 33 Notification of a personal data breach to the supervisory authority.\n\n"
        "In the case of a personal data breach, the controller shall without undue delay "
        "and, where feasible, not later than 72 hours after having become aware of it, "
        "notify the personal data breach to the competent supervisory authority.\n\n"
        "The processor shall notify the controller without undue delay after becoming "
        "aware of a personal data breach.\n"
    ),
    "national-law.txt": (
        "Section 40 Competent supervisory authority.\n\n"
        "The competent supervisory authority for controllers established in this Member "
        "State is the Federal Commissioner for Data Protection.\n\n"
        "A controller shall address every notification under Article 33 to that "
        "supervisory authority in the official language.\n"
    ),
    "company-policy.txt": (
        "Security incident response policy.\n\n"
        "Any suspected security incident must be reported to the incident channel "
        "immediately and escalated to the Data Protection Officer within four hours of "
        "detection, well before any statutory deadline.\n\n"
        "The Data Protection Officer owns all external notifications.\n"
    ),
}


@pytest.fixture
def mini_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, text in MINI_CORPUS.items():
        (corpus / name).write_text(text, encoding="utf-8")
    return corpus


# ----------------------------------------------------------------------------- helpers


def make_chunk(
    doc_id: str, index: int, text: str, *, filename: str | None = None, page: int | None = None
) -> Chunk:
    """A Chunk whose offsets are internally consistent, for tests that do not ingest."""
    return Chunk(
        chunk_id=f"{doc_id}:{index}",
        doc_id=doc_id,
        filename=filename or f"{doc_id}.txt",
        chunk_index=index,
        text=text,
        char_start=0,
        char_end=len(text),
        page_from=page,
        page_to=page,
    )


def make_retrieved(chunks: Sequence[Chunk]) -> list[RetrievedChunk]:
    """Wrap chunks in fused order with plausible ranks, for metric tests."""
    return [
        RetrievedChunk(chunk=c, rank=i + 1, rrf_score=1.0 / (60 + i + 1), dense_rank=i + 1)
        for i, c in enumerate(chunks)
    ]


@pytest.fixture
def chunk_factory():
    return make_chunk


# --------------------------------------------------------------------------- app client


@asynccontextmanager
async def app_client(app) -> AsyncIterator[httpx.AsyncClient]:
    """An httpx client with the app's **lifespan actually running**.

    ``httpx.ASGITransport`` does not emit lifespan events, so an app tested through it
    bare never runs its startup: ``app.state.retriever`` is unset, every dependency
    resolves to None, and the whole suite reports 503 on endpoints that work fine in
    production. Entering ``lifespan_context`` explicitly means these tests exercise the
    real wiring — including the injected-subsystem path and the semaphore construction —
    rather than a half-initialized app.
    """
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://mmu.test") as client:
            yield client
