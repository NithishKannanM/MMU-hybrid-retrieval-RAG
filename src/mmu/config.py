"""Settings, from ``MMU_*`` environment variables or a ``.env`` file.

Every knob has a default that works offline on this box. The two defaults that are
decisions rather than conveniences are ``embed_device`` and ``ollama_num_ctx``; both
carry their reasoning below, because both fail *silently* when set wrong.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _discover_gemini_keys(env_file: Path | None = None) -> tuple[str, ...]:
    """Collect Gemini keys written under Google's own names rather than MMU's.

    ``model_config`` sets ``env_prefix="MMU_"`` with ``extra="ignore"``, so a ``.env``
    holding the names Google's console hands you — ``GEMINI_API_KEY``, or
    ``GEMINI_API_KEY_1..N`` when you hold several to widen a per-key free-tier quota —
    is silently ignored: not an error, just no keys, and the backend then reports "no
    key configured" while the user is looking straight at seven of them. ``anthropic.py``
    documents the same trap for ``ANTHROPIC_API_KEY``; this function is the one place
    that resolves it instead of only warning about it.

    ``MMU_GEMINI_API_KEYS`` (comma-separated) remains the canonical spelling and wins
    outright — this runs only when that is unset. Numbered keys are returned in numeric
    order so rotation is reproducible across processes, which matters because a report
    that names "key 3" has to mean the same key on the next run.

    ``env_file`` is a parameter rather than a constant so a test can point it at a
    fixture instead of the developer's real ``.env``; reading the repo file by default
    made the discovery order depend on whoever ran the suite.
    """
    found: dict[int, str] = {}
    unnumbered: list[str] = []

    def absorb(name: str, value: str) -> None:
        value = value.strip().strip("'\"")
        if not value:
            return
        suffix = name[len("GEMINI_API_KEY") :]
        if suffix.startswith("_") and suffix[1:].isdigit():
            found[int(suffix[1:])] = value
        elif not suffix:
            unnumbered.append(value)

    for name, value in os.environ.items():
        if name.startswith("GEMINI_API_KEY"):
            absorb(name, value)

    # The .env file is read directly: pydantic-settings applies env_prefix when it
    # parses that file, so these bare names never reach Settings through it.
    env_file = _REPO_ROOT / ".env" if env_file is None else env_file
    try:
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip().removeprefix("export ").strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            name = name.strip()
            if name.startswith("GEMINI_API_KEY") and name not in os.environ:
                absorb(name, value)
    except OSError:
        pass

    ordered = [found[i] for i in sorted(found)] + unnumbered
    return tuple(dict.fromkeys(ordered))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MMU_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- paths ---------------------------------------------------------------------
    data_dir: Path = _REPO_ROOT / "data"
    reports_dir: Path = _REPO_ROOT / "reports"

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "corpus"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / ".index"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / ".cache"

    @property
    def questions_path(self) -> Path:
        return self.data_dir / "questions.jsonl"

    # --- embedding -----------------------------------------------------------------
    embed_model: str = "BAAI/bge-m3"
    #: **CPU by default, deliberately.** BGE-M3 is XLM-RoBERTa-large: 568M params,
    #: ~2.3GB fp32. This box has 6GB of VRAM shared with Ollama, and a 7B q4 generator at
    #: num_ctx=4096 is ~4.7GB of weights plus ~0.8GB of KV cache. Putting the embedder on
    #: the same card makes Ollama evict and reload the generator on every request — a
    #: multi-second cold-load penalty per query, which would show up in the benchmark as
    #: a latency problem with no visible cause. Indexing on CPU is a one-time cost;
    #: per-query encoding on CPU is tens of milliseconds, well inside the retrieval
    #: budget. Set MMU_EMBED_DEVICE=cuda only when the generator is remote.
    embed_device: str = "cpu"
    embed_batch_size: int = 16

    # --- chunking ------------------------------------------------------------------
    #: Sized against the *generator's* context window, not the embedder's. BGE-M3 would
    #: happily take 8192, but the retrieved context has to fit `ollama_num_ctx` with the
    #: prompt scaffolding: k=8 x 512 x 1.3 is ~5300 tokens against a 4096 window, which
    #: `context_budget_warning` correctly rejects. Something had to give, and it is chunk
    #: size rather than k: cross-document reasoning needs three or more *documents*
    #: represented in the context, so cutting k directly damages the headline metric,
    #: while 384 tokens (~1500 characters) still comfortably holds a legal clause.
    chunk_max_tokens: int = 384
    chunk_min_tokens: int = 100
    chunk_overlap_tokens: int = 64

    # --- retrieval -----------------------------------------------------------------
    top_k: int = 8
    #: Per-channel over-fetch before fusion. Fusing two lists that are already cut to
    #: the final size is nearly a no-op — the union effect is where hybrid earns its keep.
    #: None means max(4 * top_k, 20).
    depth: int | None = None
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0

    # --- reranking -----------------------------------------------------------------
    #: Cross-encoder used when a reranking arm is selected. Measured headroom at
    #: k=8/depth=32 on the seed corpus is recall 0.469 -> 0.817, so this is the one
    #: retrieval change with a demonstrated ceiling worth paying for; see
    #: `scripts/rerank_headroom.py`.
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_device: str = "cpu"
    rerank_batch_size: int = 16
    #: Truncation length for each (query, chunk) pair. Chunks are 384 tokens by default
    #: and the query is short, so 512 holds a whole pair; raising it costs quadratic
    #: attention for text the pair does not contain.
    rerank_max_length: int = 512

    # --- context selection ----------------------------------------------------------
    #: MMR's relevance/novelty trade-off. 1.0 is exactly top-k, 0.0 ignores relevance.
    #: **Not tuned.** Tuning it against the same 30 questions used to evaluate it would
    #: be fitting the test set, so it sits at the midpoint until there is a held-out
    #: set to tune against.
    select_mmr_lambda: float = 0.5
    #: Hard cap on chunks per document for the quota selector. None derives
    #: ceil(k / documents in the pool) — 1 per document at k=3 over this 3-document
    #: corpus, which is what the measured Cov=0.50 failures call for.
    select_max_per_doc: int | None = None

    #: Bounds concurrent retrievals. NOT redundant with asyncio.to_thread: to_thread
    #: keeps the event loop responsive but does not bound parallelism, so N in-flight
    #: requests become N threads contending for the same cores. See api/deps.py.
    search_concurrency: int = Field(default_factory=lambda: os.cpu_count() or 4)

    # --- generation ----------------------------------------------------------------
    generator_backend: str = "ollama"          # "ollama" | "anthropic" | "gemini"
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b-instruct-q4_K_M"
    ollama_fast_model: str = "llama3.2:3b"
    #: **Ollama silently defaults to a 2048-token context.** A RAG prompt carrying 8
    #: legal chunks blows past that and is LEFT-truncated, which removes the system
    #: prompt and the first sources. The model then answers from a fragment and the
    #: failure presents as broken retrieval, not as a truncated prompt. 8192 would cost
    #: ~1GB more KV cache and spill past free VRAM on a 6GB card; 4096 is the compromise.
    #: See `validate_context_budget` — raising top_k without raising this makes every
    #: metric silently measure truncation.
    ollama_num_ctx: int = 4096
    anthropic_model: str = "claude-sonnet-5"
    anthropic_api_key: str | None = None

    #: Pinned deliberately, not `gemini-flash-latest`. A moving alias silently changes
    #: the model under a benchmark, which makes two runs incomparable while every number
    #: still looks valid. Both carry a 1,048,576-token input window — 256x
    #: `ollama_num_ctx` — so the context-budget trap that governs the local backend does
    #: not bind here; `context_budget_warning` is checked only for ollama.
    #:
    #: These two are chosen from what `:generateContent` actually served on 2026-10-13,
    #: which is *not* what `ListModels` advertises: `gemini-2.5-flash` and
    #: `gemini-2.5-pro` are both listed and both answer 404 "no longer available to new
    #: users", while the full-flash tier (`gemini-3.5`/`3.6`/`3.7`/`3.8-flash` and
    #: `gemini-flash-latest`) answered 503 "experiencing high demand" on every attempt.
    #: Pick a stronger model here when that tier frees up — the backend needs no change,
    #: and `gemini.py` already copes with the lite tier's refusal of `thinkingConfig`.
    gemini_model: str = "gemini-3.1-flash-lite"
    gemini_fast_model: str = "gemini-3.5-flash-lite"
    gemini_host: str = "https://generativelanguage.googleapis.com"
    #: Several keys are a quota strategy, not redundancy: Gemini's free tier is limited
    #: per key and per minute, and a benchmark issues its requests in a burst. See
    #: `_discover_gemini_keys` for why bare GEMINI_API_KEY_N names are picked up too.
    gemini_api_keys: tuple[str, ...] = Field(default_factory=_discover_gemini_keys)
    temperature: float = 0.0                   # a benchmark must be reproducible
    max_tokens: int = 1024
    request_timeout_s: float = 300.0

    # --- evaluation ----------------------------------------------------------------
    judge_backend: str = "local"               # "local" | "ollama" | "anthropic"
    judge_model: str | None = None             # defaults to the generator's model
    #: Quality weights for LpQU. Printed in every report header — a tradeoff you are
    #: meant to argue with requires seeing the weights. XDR carries weight equal to AF
    #: because this is a cross-document benchmark.
    weight_af: float = 0.4
    weight_cp: float = 0.2
    weight_xdr: float = 0.4
    #: LpQU's divide-by-zero clamp. Caps LpQU at t/eps, which correctly makes a fast
    #: wrong answer score worse than a slow correct one. NOT a CLI flag: LpQU is not
    #: comparable across runs with different eps, so it is printed, never varied.
    lpqu_epsilon: float = 0.01

    @field_validator("gemini_api_keys", mode="before")
    @classmethod
    def _split_keys(cls, v: object) -> object:
        """Accept ``MMU_GEMINI_API_KEYS=a,b,c``.

        Without this, pydantic-settings parses a value destined for a tuple field as
        JSON, so the natural comma-separated spelling raises a parse error that names
        JSON — which is a confusing thing to be told when you were setting an API key.
        """
        if isinstance(v, str):
            return tuple(part.strip() for part in v.split(",") if part.strip())
        return v

    @field_validator("embed_device")
    @classmethod
    def _known_device(cls, v: str) -> str:
        if v not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"embed_device must be cpu|cuda|mps, got {v!r}")
        return v

    def resolved_depth(self, top_k: int | None = None) -> int:
        k = top_k if top_k is not None else self.top_k
        return self.depth if self.depth is not None else max(4 * k, 20)

    def context_budget_warning(self, top_k: int | None = None) -> str | None:
        """Return a warning when the retrieved context cannot fit the generator window.

        ``k`` chunks of ``chunk_max_tokens`` plus the prompt scaffolding, with a 1.3
        factor because the ``len // 4`` token estimate under-counts legal text (long
        words, many numerals). Returns None when it fits. The caller warns loudly rather
        than raising: a too-small window still produces answers, it just produces them
        from a truncated prompt, and a silent wrong number is the thing to prevent.
        """
        if self.generator_backend != "ollama":
            return None
        k = top_k if top_k is not None else self.top_k
        needed = int(k * self.chunk_max_tokens * 1.3)
        if needed < self.ollama_num_ctx:
            return None
        return (
            f"context budget: k={k} x {self.chunk_max_tokens} tokens x 1.3 = ~{needed} "
            f"tokens, but MMU_OLLAMA_NUM_CTX={self.ollama_num_ctx}. Ollama LEFT-truncates, "
            f"dropping the system prompt and the first sources — every metric would then "
            f"measure truncation. Raise MMU_OLLAMA_NUM_CTX or lower MMU_TOP_K."
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
