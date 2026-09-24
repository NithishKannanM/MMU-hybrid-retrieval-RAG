# MMU — Modular Memory Unit

A hybrid-retrieval RAG framework, benchmarked on legal/compliance documents.

BGE-M3 → FAISS `IndexFlatIP` + BM25Okapi → Reciprocal Rank Fusion → grounded generation,
served over FastAPI, and measured by five metrics designed to make the tradeoffs
arguable rather than asserted.

The interesting engineering here is in *why* it combines what it combines. Most of that
reasoning lives in the module docstrings, which are worth reading before the code.

## Why these pieces

**BGE-M3** produces dense, sparse and ColBERT-style multi-vector representations from a
single forward pass. Most hybrid systems maintain a separate dense encoder and a separate
sparse encoder; BGE-M3 collapses that into one model, which is what makes running two
retrieval channels affordable. MMU uses the dense head and pairs it with BM25 (see
`src/mmu/embed/base.py` for why `sentence-transformers` and not `FlagEmbedding`, and what
that costs).

**FAISS `IndexFlatIP`** is exact, not approximate — no HNSW, no IVF. Zero recall loss from
approximation, at O(n·d) per query. That is the right trade at benchmark scale, and it is
the first thing to swap when scaling. `src/mmu/retrieve/dense.py` names the upgrade path
and the one line that changes.

**BM25Okapi** does the work dense embeddings are bad at: exact term matching. In legal
text a defined term like "data controller" carries precise meaning, and an embedding will
blur it into a semantically-similar-but-legally-wrong neighbour. Everything this channel
is for lives in its tokenizer, which is why `src/mmu/retrieve/sparse.py` keeps `Art. 33`
and `data-controller` intact as single tokens.

**RRF** fuses by rank, not score: `score(d) = Σ 1/(k + rank_i(d))`. Cosine similarity and
BM25 scores live on uncalibrated, incomparable scales, and every normalization scheme that
would let you add them is itself a tuning knob and a bug source. RRF sidesteps calibration
entirely by only ever looking at rank position.

## Results

Measured on the seed corpus: 3 documents, 378 chunks (gdpr 271, national-law 100,
company-policy 7). `k=8`, `depth=32`, `rrf_k=60`, BGE-M3 dense + BM25Okapi.

Retrieval (measured):

- Coverage (required docs present in context) = **1.000** on all 3 cross-document
  questions
- Context Precision = **0.811** (n=5; the unanswerable question has no relevant set
  and is correctly excluded from the average)
- The answer text was present in the retrieved context for **6 of 6** questions
- `company-policy` is 7 chunks of 378 (1.9% of the corpus) and reached the top-8 on
  all three cross-document questions

The result worth leading with is `single-002` ("What distinguishes a data
controller from a data processor?"): the chunk containing the literal answer
(`determines the purposes`) sits at **dense rank 12 and sparse rank 12**. Neither
channel's own top-8 contains it. RRF fused it to **rank 6**, into the context:

```
both channels @12:             1/(60+12) + 1/(60+12) = 0.02778
one channel @1, other absent:  1/(60+1)  + 0          = 0.01639   -> 69% lower
```

Consensus across two mediocre rankings beats one channel's confident favourite.
This is why `core/fusion.py` gives an absent document **exactly zero** rather than a
sentinel rank: with the rejected `n+1` sentinel (`depth=32`, so sentinel rank 33)
the second line becomes `1/(60+1) + 1/(60+33) = 0.02715`, collapsing a 69% margin
to **2.3%**.

Honest caveat: 31% of retrieved context slots (15 of 48) hold chunks whose dense
rank was worse than 8, but "promoted" is not "good" — BM25 also promotes noise
(for `single-002` its own #1 hit is a term-dense recital about supervisory-authority
cooperation that is definitionally useless).

Test suite: **226 passed, 6 deselected, 0.93s, zero model loads** — the offline
gate downloads nothing.

Generation-side metrics (Answer Faithfulness, Cite, Syn, XDR, all CUR variants,
`marker_compliance`) have **not** been measured yet — the generator model was still
downloading. The numbers above are retrieval-side only.

## Quickstart

```sh
uv sync                  # torch is ~2.5GB on first run
uv run pytest -q         # the gate: fully offline, downloads nothing
```

Then drop at least three documents (PDF / txt / md) into `data/corpus/`. The seed
questions are written against files named `gdpr.*`, `national-law.*` and
`company-policy.*`; either use those names or run `mmu index build` and copy the real
`doc_id` slugs it prints into `data/questions.jsonl`.

```sh
uv run mmu index build --verbose   # downloads BGE-M3 (~2.3GB) once; prints the doc_id table
uv run mmu questions validate      # HARD GATE — fix typos here, not by debugging a 0.0 later
uv run uvicorn mmu.api.app:app --port 8099
```

### Seeing the hybrid actually work

```sh
curl -s -X POST localhost:8099/search -H 'content-type: application/json' \
  -d '{"query":"personal data breach notification deadline","k":8}' \
  | jq '.chunks[] | {doc_id, rank, dense_rank, sparse_rank, rrf_score}'
```

What to actually look for is **rank promotion**: a chunk whose `dense_rank` is far
worse than its final `rank` is one BM25 pulled up. At `depth=32` over the 378-chunk
seed corpus, all 48 retrieved chunks across the 6 seed questions were found by both
channels — `dense_rank: null` essentially never occurs at this scale. Exclusive
discovery does happen at larger corpus sizes, once the two channels' candidate pools
stop overlapping, but not here.

### Streaming

```sh
curl -N -X POST localhost:8099/answer/stream -H 'content-type: application/json' \
  -d '{"query":"If we suffer a breach in Germany, who do we notify and by when?"}'
```

`-N` is required; without it curl buffers and streaming looks broken. Events arrive as
`retrieval` → `token`* → `usage` → `done`, or `error`.

## Evaluation

```sh
uv run mmu eval run --judge local --out reports/local.json     # deterministic, free, seconds
ollama pull qwen2.5:7b-instruct-q4_K_M
uv run mmu eval run --judge ollama --out reports/ollama.json
```

`llama3.2:3b` works as the `fast` alias and is a much smaller download. Setting
`MMU_OLLAMA_MODEL=llama3.2:3b` makes **both** the generator and the LLM judge
resolve to it, because `mmu.cli._build_judge` calls `build_generator(settings)`
with no alias.

The ablation is the experiment that justifies the architecture:

```sh
make ablate
```

If hybrid does not beat dense-only on Context Precision for exact-defined-term questions,
the sparse tokenizer is wrong — and that is the finding, not a failure.

### Reading the numbers honestly

Five metrics, with the caveats that make them usable:

| metric | what it asks | needs an LLM judge? |
|---|---|---|
| **Answer Faithfulness** | is the answer grounded in the retrieved context, or hallucinated past it | **yes** for a reportable number |
| **Context Precision** | of what was retrieved, how much was actually relevant | no, when labels exist |
| **Cross-Document Reasoning** | can it synthesize across GDPR + national law + policy at once | only the `Syn` factor |
| **LpQU** | quality normalized against latency, so tradeoffs are arguable | no |
| **Context Utilization** | how much retrieved context the model actually drew on | no |

Three things the report will not let you get wrong:

- **XDR is printed as four columns** (`Cov`, `Cite`, `Syn`, XDR), never as the product
  alone. A zero from low `Cov` is a retrieval bug; a zero from low `Cite` is a generation
  bug. They demand opposite fixes.
- **The local judge is negation-blind.** "must notify within 72 hours" and "must **not**
  notify within 72 hours" score ~0.95 against the same chunk. In a compliance corpus that
  is precisely the failure that matters, so `--judge local` is a CI regression tripwire,
  not a number to quote.
- **CUR is a diagnostic, not a target.** `CUR = 1.0` at `k=8` probably means `k` is too
  small. Shrinking `k` to raise it will reduce faithfulness and cross-document reasoning.

## Layout

```
src/mmu/core/       models + RRF (the cross-package contract)
src/mmu/ingest/     bytes → text → overlapping span chunks
src/mmu/embed/      Embedder protocol, BGE-M3, the prefix registry, a hashing test double
src/mmu/retrieve/   FAISS dense, BM25 sparse, the chunk store, the hybrid retriever
src/mmu/generate/   Generator protocol, Ollama + Anthropic, the grounded prompt
src/mmu/api/        FastAPI app, the thread-pool offload, SSE
src/mmu/eval/       questions, judges, the five metrics, the runner and report
```

## Notes on the environment

- Python 3.14. `torch >= 2.9` and `faiss-cpu >= 1.14.3` are the floors where cp314 wheels
  exist; below them `uv` attempts a source build.
- `MMU_EMBED_DEVICE` defaults to `cpu`. On a 6GB card, putting BGE-M3 on the GPU makes
  Ollama evict and reload the generator on every request.
- `MMU_OLLAMA_NUM_CTX` defaults to 4096 because Ollama's own default is 2048 and it
  **left-truncates** past it, silently removing the system prompt and the first sources.
  Raise it if you raise `k`.

## License

Code is MIT — see `LICENSE`. The corpus documents in `data/corpus/` have their own
terms; see `data/corpus/SOURCES.md`.
