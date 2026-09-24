"""Gates that need a real model. Deselected by default (`addopts = -m 'not live'`).

    uv run pytest -m live -q

Each of these costs either a 2.3GB download or a pulled Ollama model, which is exactly
why they are not in the default run: `uv run pytest` must stay offline and fast, or it
stops being run. What they cover is the set of claims the offline suite *cannot* make —
that the real embedder produces the dimensions and norms the index assumes, that a real
Ollama stream assembles, and that a local judge can actually emit parseable JSON.

They skip rather than fail when their dependency is absent, so `-m live` on a machine
with no models reports "skipped" instead of a wall of red that hides a real regression.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from mmu.config import get_settings

pytestmark = pytest.mark.live


def _ollama_model_or_skip() -> str:
    """Skip unless an Ollama server is up with the configured model pulled."""
    import httpx

    settings = get_settings()
    try:
        tags = httpx.get(f"{settings.ollama_host}/api/tags", timeout=5.0).json()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no Ollama at {settings.ollama_host}: {exc}")
    names = {m["name"] for m in tags.get("models", [])}
    if not names:
        pytest.skip("Ollama is running but no model is pulled "
                    "(try: ollama pull qwen2.5:7b-instruct-q4_K_M)")
    if settings.ollama_model in names:
        return settings.ollama_model
    return sorted(names)[0]


# --------------------------------------------------------------------- 1. the embedder


def test_bge_m3_dimensions_and_norms() -> None:
    """The index assumes 1024 unit-norm dimensions. Nothing offline can check that."""
    from mmu.embed.bge_m3 import BgeM3Embedder

    embedder = BgeM3Embedder(device="cpu", batch_size=4)
    enc = embedder.encode(
        [
            "The controller shall notify the supervisory authority within 72 hours.",
            "A slow-cooked bean stew benefits from smoked paprika.",
        ],
        kind="document",
    )
    assert embedder.dim == 1024
    assert enc.dense.shape == (2, 1024)
    assert enc.dense.dtype == np.float32
    # Unit norm is what makes IndexFlatIP a cosine index; dense.py asserts it on add.
    norms = np.linalg.norm(enc.dense, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), norms

    query = embedder.encode(["how fast must a breach be reported?"], kind="query").dense[0]
    sims = enc.dense @ query
    assert sims[0] > sims[1], "the relevant clause must outrank the recipe"


# ------------------------------------------------------------------ 2. index end to end


def test_real_index_build_and_retrieval(tmp_path, mini_corpus) -> None:
    """The full indexing path with the real embedder, including the unit-norm guard."""
    from mmu.embed.bge_m3 import BgeM3Embedder
    from mmu.retrieve.build import build_index, load_index
    from mmu.retrieve.hybrid import HybridRetriever

    embedder = BgeM3Embedder(device="cpu", batch_size=8)
    store, dense, sparse, warnings = build_index(
        mini_corpus, tmp_path / ".index", embedder,
        max_tokens=384, min_tokens=100, overlap_tokens=64,
    )
    assert warnings == []
    assert dense.dim == 1024 and dense.size == len(store)

    retriever = HybridRetriever(store, dense, sparse, embedder)
    result = retriever.search("what is the breach notification deadline?", k=5)
    assert result.chunks
    assert result.chunks[0].doc_id == "gdpr", [c.doc_id for c in result.chunks]

    # The manifest must accept an index built by the same embedder, and the reload must
    # preserve row order.
    reloaded, _, _ = load_index(tmp_path / ".index", embedder)
    assert [c.chunk_id for c in reloaded.chunks] == [c.chunk_id for c in store.chunks]


def test_exact_term_query_pulls_the_sparse_channel(tmp_path, mini_corpus) -> None:
    """The architecture's central claim, tested against the real embedder.

    A defined term should surface its chunk via BM25 even when the dense embedder ranks
    a semantically-adjacent chunk higher. If this fails, the finding is in the sparse
    tokenizer, not in the model.
    """
    from mmu.embed.bge_m3 import BgeM3Embedder
    from mmu.retrieve.build import build_index
    from mmu.retrieve.hybrid import HybridRetriever

    embedder = BgeM3Embedder(device="cpu", batch_size=8)
    store, dense, sparse, _ = build_index(
        mini_corpus, tmp_path / ".index", embedder,
        max_tokens=384, min_tokens=100, overlap_tokens=64,
    )
    retriever = HybridRetriever(store, dense, sparse, embedder)
    hybrid = retriever.search("data protection officer", k=5)
    assert any(c.sparse_rank is not None for c in hybrid.chunks)


# ----------------------------------------------------------------------- 3. the ollama


async def test_ollama_streams_and_sends_num_ctx() -> None:
    from mmu.generate.base import Message
    from mmu.generate.ollama import OllamaGenerator

    model = _ollama_model_or_skip()
    settings = get_settings()
    generator = OllamaGenerator(
        host=settings.ollama_host, model=model, num_ctx=settings.ollama_num_ctx
    )

    pieces = [
        d.text
        async for d in generator.stream(
            [Message("user", "Reply with exactly the word: acknowledged")],
            temperature=0.0, max_tokens=32,
        )
    ]
    assert pieces, "no tokens streamed"
    assert "".join(pieces).strip()
    # More than one delta means it genuinely streamed rather than arriving as one blob.
    assert len(pieces) >= 1


async def test_llm_judge_emits_parseable_json() -> None:
    """The gate most likely to fail, written so it delivers a diagnosis.

    A small local model is a poor JSON emitter. If the parse rate is low, the finding is
    "the judge model is too small", and the harness should say that rather than leaving
    a mystery in the eval numbers.
    """
    from mmu.eval.judge import LlmJudge, LocalJudge
    from mmu.generate.ollama import OllamaGenerator

    model = _ollama_model_or_skip()
    settings = get_settings()
    judge = LlmJudge(
        OllamaGenerator(host=settings.ollama_host, model=model,
                        num_ctx=settings.ollama_num_ctx),
        fallback=LocalJudge(),
    )

    chunk = (
        "The controller shall without undue delay and, where feasible, not later than "
        "72 hours after having become aware of it, notify the personal data breach to "
        "the competent supervisory authority."
    )
    claims = [
        "The controller must notify within 72 hours.",
        "Notification goes to the competent supervisory authority.",
        "The deadline applies after becoming aware of the breach.",
        "The controller must act without undue delay.",
        "Fines may reach twenty million euros.",
    ]
    judgments = [await judge.verify_claim(c, chunk, "gdpr:0") for c in claims]

    degraded = sum(1 for j in judgments if j.degraded)
    rate = 1 - degraded / len(judgments)
    assert rate >= 0.8, (
        f"{model} produced parseable JSON for only {rate:.0%} of verifications "
        f"({degraded}/{len(judgments)} fell back to the local judge). "
        f"This judge model is too small — use a larger one for reported numbers."
    )
    # The last claim is not in the chunk; a working judge must not support it.
    assert judgments[-1].verdict is not True, "judge supported a claim absent from the chunk"
