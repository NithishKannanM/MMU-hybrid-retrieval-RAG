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

Test suite: **293 passed, 6 deselected, 1.3s, zero model loads** — the offline
gate downloads nothing.

### Generation (measured)

Generator `qwen2.5:7b-instruct-q4_K_M` (Ollama, `num_ctx=4096`, `temperature=0.0`,
CPU-only — `ollama ps` reports `size_vram: 0`), scored by the deterministic
`--judge local`. Six seed questions, `k=8`. **Interleaved**: all three arms answer each
question back to back inside one process, arm order rotated per question
(`mmu eval ablate`, 18 generations, `reports/ablate/`).

```
run                  AF     CP    Cov   Cite    Syn    XDR  CURcit  CURatt    p50    LpQU
hybrid            0.875  0.811  1.000  0.833  1.000  0.833   0.438   0.000  61.76   74.01
dense-only        0.667  0.800  0.833  0.833  1.000  0.750   0.312   0.000  63.33   92.47
sparse-only       0.800  0.691  1.000  0.667  0.667  0.500   0.354   0.000  56.39   82.75
```

`marker_compliance` is 1.000 on all three arms — over 5, 4 and 4 defined answers
respectively, the rest being refusals where the metric is undefined (see below). No
column in this table is flagged unreliable.

**Read almost nothing into the differences between those rows.** At n=6 the 95%
noise band on any proportion is ±0.40, and no gap in that table comes close. The
ablation is reported because the *shape* of the comparison is the experiment.

The sharpest evidence for that band is the project's own repeated runs. The identical
hybrid configuration — same `k=8`, same `rrf_k=60`, same channels, same judge,
`temperature=0.0` — has scored corpus AF **0.697** and **0.875** on two separate
invocations, and **3 of its 6 answers differ in text between them**. Greedy decoding is
not reproducible across sessions on this setup. Anything this table shows at a 0.1 scale
is weather.

#### What the pairing settled

One claim did not survive contact with the paired design, and killing it is the most
useful thing this harness has done. The earlier sequential ablation showed hybrid at a
p50 of **18.79s** against 61.77s and 55.20s — a 3x gap, far too large to look like
noise, and it was reported here as the one surviving signal. It was not a signal.
Interleaved:

```
                    p50 (sequential)   p50 (interleaved)   paired sign test
hybrid                       18.79               61.76    —
dense-only                   61.77               63.33    3 up / 3 down, p=1.000
sparse-only                  55.20               56.39    3 up / 3 down, p=1.000
```

The three arms now land within 11% of each other, and the per-question differences
split exactly evenly in both directions. The mechanism was never mysterious in
hindsight: **retrieval is 0.0–1.4% of wall clock** (hybrid: 1.92s of 137.7s;
sparse-only: 0.01s), so a p50 difference between runs is almost entirely the
generator's, and the generator is on CPU where throughput follows whatever else the
host is doing. The tell had been sitting in the data — a `llama3.2:3b` run managed only
12.2 chars/s against the fast hybrid run's 20.5, and a 3B being slower than a 7B is
backwards.

Both latency tests had **full power** (`floor 0.031`: a consistent effect would have
been detected) and found nothing. That is a genuine null, and the report says so in
those words — it distinguishes it from the AF comparisons, which ran on 2–4 untied
pairs where the best attainable p is 0.125–0.25 and *could not* have concluded anything.
Collapsing "we tested and found nothing" into "we couldn't have tested" is the error the
floor column exists to prevent.

So: **no latency effect, and that is now a measurement rather than a caveat.** `p50`
and `LpQU` stay in the table because they are real costs, but they are properties of
this host, not of the retrieval strategy.

#### What survives

- **Coverage 1.000 for hybrid and sparse-only, 0.833 for dense-only — and it is
  reproducible per question.** Dense-only scores 0.5 on `xdr-002` in *both* the
  sequential and the interleaved runs, identically; hybrid and sparse-only score 1.0 on
  every cross-document question in all of them. This is the one finding that needs no
  statistics, because `Cov` is computed from retrieval alone: no generator, no judge, no
  sampling. Dense retrieval by itself misses a required document that BM25 finds, which
  is the union effect the architecture exists for.
- **Hybrid leads on AF — but only after a scoring bug was removed, and still not
  significantly.** Before the fix described below, sparse-only posted the *highest* AF
  of the three (0.967) because it was being paid 1.00 for refusing a question it should
  have answered. Corrected, the order is hybrid 0.875 / sparse-only 0.800 / dense-only
  0.667. That is the architecture's expected ordering, and it is still 0.21 wide inside
  a ±0.40 band with a paired sign test of 2-up/2-down, p=1.000. **It is not a result.**
  What it is: the previous ranking was an artifact, and the corrected one no longer
  contradicts the design.

Two honest caveats about that table's own columns:

**`CURatt` is 0.000 because it cannot be measured here, not because the answers
cite nothing they used.** `LocalJudge` deliberately returns `span=None` rather than
inventing a supporting span (`eval/judge.py:170`), and `CUR_attr`/`CUR_tok` count
chunks with a *verified span*. Under `--judge local` they are structurally zero, so
the `CUR_cited - CUR_attr` "citation-theater gap" is unreadable — it reports total
theater when the truth is no measurement. Those columns need `--judge ollama` or
`--judge anthropic` to mean anything.

**`Syn` is a lexical fallback, not a judgment.** Under `--judge local`,
`synthesis_score_fallback` checks whether the expected keys are *present* in the answer
(normalized substring, or `rapidfuzz token_set_ratio >= 85`), then snaps the fraction to
{0, 0.5, 1}. `Syn` 1.000 therefore means the facts were mentioned, not that the
conclusion depended on combining them — a model that lists all three without connecting
them scores the same. The gap between local `Syn` and LLM `Syn` on the same answer is
itself the concatenation-vs-synthesis measurement.

A smaller model was run on the same configuration for contrast (`llama3.2:3b`,
`--generator fast`): corpus AF 0.861, but `marker_compliance` 0.667 — **below the
floor**, because it dropped markers entirely on one question. Its higher AF is both
inside the noise band and resting on citations the report declines to trust. (That run
predates the two scoring fixes below and has not been repeated.)

### Four scoring bugs this run surfaced

Every one was invisible until a real corpus and a real generator existed at the same
time, which is the lesson `LEARNING.txt` records: a green suite can mean "nothing is
here yet." Two of them were introduced by the fixes to the other two — written in the
same session, each hiding inside the carve-out that fixed its predecessor.

**1. A correct refusal scored 0.0 on Answer Faithfulness.** `questions.jsonl` states
the contract — "a correct refusal must score 1.0" — and AF implements it as a
carve-out at `m == 0` claims. But the judge decomposes "The provided context does
not contain this information." (verbatim the wording `generate/prompt.py` suggests)
into one unsupported claim, so `m == 1`, the carve-out never fired, and the model
was scored maximally unfaithful for doing exactly what it was asked. Corpus AF read
0.531 instead of 0.789. A refusal that cites nothing is now treated as claim-free;
the "cites nothing" condition is what stops a refusal phrase appended to a real
answer from laundering it to 1.0, which `prompt.py` calls the worse failure.

**2. `--verbose` crashed the run at the last question.** Context Precision is `None`,
not 0.0, for a question with no relevant set; the CLI formatted it with `:.2f` and
raised `TypeError` — after all six answers had been generated. The fixture had one
question, with a relevant set, so no test could see it.

**3. A correct refusal dragged `marker_compliance` below its own floor — and that
floor gates other columns.** A refusal cites nothing by design, so its per-answer
compliance was 0.0 on a metric about citing. Averaged in, that caps the attainable
corpus rate at 0.833 with one unanswerable question in six, so a *single* further
answer without markers lands at 0.667 and trips `MARKER_COMPLIANCE_FLOOR` — at which
point `report.py` declares `Cite`, `XDR` and every CUR column unreliable **because the
model refused correctly**. Measured: the three-arm ablation read 0.833 / 0.667 / 0.667
and flagged two of three arms. Compliance is now `None` for a refusal — *undefined*,
exactly as `cp` is undefined for a question with no relevant set — and the rate is
taken over the defined answers, printed with its own `n`:

```
marker_compliance: 1.00  [OK]  over 5/6 answer(s), 1 correct refusal(s) excluded as undefined
```

**4. An *unjustified* refusal scored 1.00 on Answer Faithfulness, which made corpus AF
gameable to 1.000 by refusing everything.** This one was created by fix 1. Its carve-out
keyed on "does this text look like a refusal" and never asked whether refusing was the
right answer. `single-002` ("What distinguishes a data controller from a data
processor?") is `difficulty: easy`, `required_doc_ids: ["gdpr"]` — and under dense-only
and sparse-only the generator refused it **with gdpr in its context at CP 0.74 and
1.00**, scoring 1.00 for it, four times across two independent runs. It was inflating
precisely the arms that had failed: sparse-only's AF of 0.967 made it the apparent
winner.

`answer_faithfulness` now takes `refusal_is_correct` rather than `is_refusal` — a
judgment, not a text-shape test — and the rule is three-way, because the obvious
two-way version is also wrong:

| situation | AF | why |
|---|---|---|
| question is unanswerable | **1.0** | the contract: refusing is the correct output |
| answerable, required docs **not** retrieved | **1.0** | abstaining beats inventing; the failure is retrieval's, and `Cov`/`CP` already carry it |
| answerable, required docs **present** | **0.0** | it declined with the material in hand |

The middle row is the reason this is not simply "refusing an answerable question scores
0.0". Charging the generator for a retrieval failure would stop AF being a generation
metric, and would punish a model for correctly declining to hallucinate.

Fix 3 and fix 4 deliberately use *different* conditions. `marker_compliance` is
undefined for **any** pure refusal, including an unjustified one, while AF scores the
unjustified one 0.0. An unjustified refusal is a real failure, but it is an *answering*
failure — charging it to the rate that gates `Cite`/`XDR`/`CUR` as well would bill it
twice and recreate bug 3, where a gating metric trips for a reason unrelated to how well
the model cites. The failure lands in the metric that measures it.

Both fixes were verified by reverting each and confirming the new tests fail
(`assert 0.0 is None`; `Obtained: 1.0 / Expected: 0.0`), and the corrected re-run
reproduced the previous one **answer-for-answer** — the only deltas in the whole report
are the two mis-scored refusals, so nothing here is confounded with generation noise.

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
with no alias. A 3B is a poor judge — `tests/test_live.py` has a gate that measures
its JSON parse rate and says so — and it fell below the marker-compliance floor as a
generator, so keep it off any reported number.

### Backends

`MMU_GENERATOR_BACKEND` selects one of three, all behind the same `Generator`
Protocol in `generate/base.py`:

| backend | needs | notes |
|---|---|---|
| `ollama` (default) | a local `ollama serve` | the only one that runs offline |
| `anthropic` | `uv sync --extra anthropic` + `MMU_ANTHROPIC_API_KEY` | lazily imported; a plain `uv sync` never installs the SDK |
| `gemini` | a key only | speaks REST over `httpx`, which is already a dependency, so there is no extra to install |

`registry.py` accepts an **alias** (`default`/`fast`), never a raw model id, because
the same parameter is exposed to HTTP callers through `AnswerRequest` — a raw id on
the wire could route a request past the local default the benchmark is built around.

The Gemini backend rotates across several keys: the free tier is limited *per key*
and per minute, and a benchmark issues its requests in a burst, so a 429 advances to
the next key rather than failing the run. It reads `MMU_GEMINI_API_KEYS=a,b,c` and
also picks up Google's own bare `GEMINI_API_KEY` / `GEMINI_API_KEY_1..N` names,
which `env_prefix="MMU_"` would otherwise ignore silently.

Two things it learned the hard way, both measured on 2026-10-13 rather than read
from docs. `ListModels` advertises models that `:generateContent` refuses — both
`gemini-2.5-flash` and `gemini-2.5-pro` return 404 "no longer available to new
users" — so the defaults are pinned to ids that actually served. And the `-lite`
tiers reject `generationConfig.thinkingConfig` with a bare 400 naming no field,
while succeeding without it; since Gemini bills reasoning tokens against
`maxOutputTokens` and can otherwise return an empty answer that scores as a total
faithfulness failure, the field is sent, and dropped-and-retried once per model if
refused.

### The ablation is interleaved, and that is not a detail

The ablation is the experiment that justifies the architecture:

```sh
make ablate        # == mmu eval ablate --judge local --out-dir reports/ablate
```

If hybrid does not beat dense-only on Context Precision for exact-defined-term questions,
the sparse tokenizer is wrong — and that is the finding, not a failure.

`eval ablate` runs every arm **question-major inside one process**: hybrid, dense-only
and sparse-only all answer `xdr-001`, then all answer `xdr-002`, and so on. It replaced
three sequential `eval run` calls differenced by `eval compare`, because that design
cannot support a claim and this project proved it on its own data. Across five
sequential runs of the same `qwen2.5:7b-instruct-q4_K_M`, generator throughput ranged
from 6.7 to 20.5 characters per second while retrieval accounted for **0.0–1.4% of wall
clock**. The "3x faster p50" that fell out of that comparison was host state, and the
tell was that a 3B model in the same series came out *slower* than the fastest 7B run.

Interleaving makes each question its own control: both arms answer it seconds apart on
the same loaded model, at the same temperature, under the same thermal and contention
state. Two further details matter and are tested:

- **The arm order rotates per question.** Within a question the leading arm pays any
  cold-start cost and later arms inherit a warm cache, so a fixed order would simply
  relabel that bias as an arm effect. With three arms over six questions each arm leads
  exactly twice, and `QuestionRun.arm_index` records where each run actually landed so
  the balance can be checked instead of trusted.
- **The paired latency column is `retrieval_s + generate_s`, not `latency_s`.**
  `latency_s` includes judging, and `CachingJudge` is keyed by `question_id`, which the
  arms share — so whichever arm ran second got identical verdicts for free and would
  look faster for a reason unrelated to retrieval.

The report then prints an **exact two-sided sign test** on the per-question differences
(`metrics.paired_sign_test`). A sign test, not a paired t-test: these are bounded
proportions over six questions, visibly non-normal and frequently tied at 1.0, and a
t-test on six such values reports a precision it does not have. The price is power, and
the report states it rather than hiding it — it prints the **best p-value the design
could possibly reach** next to the one it got, because at this n that floor is often
above 0.05, and a test that *cannot* reach significance has to say so instead of
returning a non-significant p that reads as "no difference". At n=6 the floor is 0.031
and only unanimity reaches it; at n=5 it is 0.063, and the comparison is unwinnable
before it starts.

`eval compare` refuses to compute any of this over runs that were not interleaved, and
says why where it would otherwise print their latency columns. That guard is the whole
lesson of the mistake, encoded.

### Reading the numbers honestly

Five metrics, with the caveats that make them usable:

| metric | what it asks | needs an LLM judge? |
|---|---|---|
| **Answer Faithfulness** | is the answer grounded in the retrieved context, or hallucinated past it | **yes** for a reportable number |
| **Context Precision** | of what was retrieved, how much was actually relevant | no, when labels exist |
| **Cross-Document Reasoning** | can it synthesize across GDPR + national law + policy at once | only the `Syn` factor |
| **LpQU** | quality normalized against latency, so tradeoffs are arguable | no |
| **Context Utilization** | how much retrieved context the model actually drew on | no |

Four things the report will not let you get wrong:

- **XDR is printed as four columns** (`Cov`, `Cite`, `Syn`, XDR), never as the product
  alone. A zero from low `Cov` is a retrieval bug; a zero from low `Cite` is a generation
  bug. They demand opposite fixes.
- **The local judge is negation-blind.** "must notify within 72 hours" and "must **not**
  notify within 72 hours" score ~0.95 against the same chunk. In a compliance corpus that
  is precisely the failure that matters, so `--judge local` is a CI regression tripwire,
  not a number to quote.
- **CUR is a diagnostic, not a target.** `CUR = 1.0` at `k=8` probably means `k` is too
  small. Shrinking `k` to raise it will reduce faithfulness and cross-document reasoning.
- **`p50` and `LpQU` do not measure retrieval, and `eval compare` says so.** Retrieval is
  under 2% of wall clock here; the rest is the generator. Latency is only comparable
  between configurations that were run *interleaved* (`mmu eval ablate`), and `compare`
  refuses to pair separately-executed runs rather than print a p-value about host drift.

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
