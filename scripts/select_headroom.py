"""Compare context selectors on retrieval alone, before spending any generations.

    uv run python scripts/select_headroom.py

Reranks each question's candidate pool **once** and then applies every selector to that
same reranked pool, so all policies are compared on identical candidates and the whole
sweep costs 30 cross-encoder passes rather than 30 per selector. No generator, no judge.

It reports the two things the diversity experiment turns on:

- **doc coverage** — the fraction of each question's `required_doc_ids` present in the
  chosen context. This is `Cov`, the metric reranking broke (0.929 -> 0.583), and it is
  computable from retrieval alone, which is exactly why it is worth checking here first.
- **recall@k / MRR** — so that a coverage gain bought by discarding relevant passages
  shows up rather than hiding.

`scripts/rerank_headroom.py` is the same idea one stage earlier.
"""

from __future__ import annotations

import sys

from mmu.config import get_settings
from mmu.embed.bge_m3 import BgeM3Embedder
from mmu.eval.metrics import count_relevant_chunks, rank_metrics
from mmu.eval.questions import load_questions
from mmu.retrieve.build import load_index
from mmu.retrieve.hybrid import HybridRetriever
from mmu.retrieve.registry import build_reranker, build_selector
from mmu.retrieve.rerank import rerank as rerank_result
from mmu.retrieve.select import TopKSelector


def _coverage(chunks, required: tuple[str, ...]) -> float | None:
    """`Cov` for one question: required documents present in the chosen context."""
    if not required:
        return None
    present = {c.doc_id for c in chunks}
    return sum(1 for d in required if d in present) / len(required)


def main() -> int:
    settings = get_settings()
    questions = load_questions(settings.questions_path)
    scored = [q for q in questions if q.relevant_doc_ids]

    embedder = BgeM3Embedder(
        settings.embed_model, device=settings.embed_device,
        batch_size=settings.embed_batch_size,
    )
    store, dense, sparse = load_index(settings.index_dir, embedder)
    retriever = HybridRetriever(store, dense, sparse, embedder)
    corpus = [(c.doc_id, c.text) for c in store.chunks]

    reranker = build_reranker(settings, "default")
    selectors = {
        "top-k": TopKSelector(),
        "mmr": build_selector(settings, "mmr", store=store, dense=dense),
        "quota": build_selector(settings, "quota", store=store, dense=dense),
    }
    k, depth = 3, 32
    print(
        f"{len(scored)} scored questions | pool depth={depth} | context k={k} | "
        f"reranker={reranker.name}\nSelectors applied to the SAME reranked pool per "
        f"question.\n"
    )

    # arm -> metric -> values
    agg: dict[str, dict[str, list[float]]] = {
        name: {"cov": [], "recall": [], "mrr": []} for name in ("k8-reference", *selectors)
    }
    xdoc: dict[str, list[float]] = {name: [] for name in agg}

    head = f"{'question':<18}{'req':>4}" + "".join(
        f"{n[:9]:>11}" for n in ("k8-ref", *selectors)
    )
    print(head + "   <- doc coverage")
    print("-" * len(head))

    for q in scored:
        n_rel = count_relevant_chunks(
            corpus, relevant_doc_ids=q.relevant_doc_ids, must_contain=q.must_contain
        )
        kw = dict(
            relevant_doc_ids=q.relevant_doc_ids, must_contain=q.must_contain,
            n_relevant_total=n_rel,
        )
        pool = retriever.search(
            q.question, k=depth, depth=depth, channels=["dense", "sparse"],
            rrf_k=settings.rrf_k,
        )
        # The k=8 reference is the unreranked pool truncated to 8, matching the
        # production arm (search(k=8, depth=32) and the top 8 of a depth-32 fusion are
        # the same list).
        ref = TopKSelector().select(pool.chunks, 8)
        reranked = rerank_result(pool, reranker, k=depth)

        row = {"k8-reference": ref}
        for name, selector in selectors.items():
            row[name] = selector.select(reranked.chunks, k)

        cells = ""
        for name, chunks in row.items():
            cov = _coverage(chunks, q.required_doc_ids)
            m = rank_metrics(chunks, k=len(chunks), **kw)
            if cov is not None:
                agg[name]["cov"].append(cov)
                if len(q.required_doc_ids) >= 2:
                    xdoc[name].append(cov)
            agg[name]["recall"].append(m.recall or 0.0)
            agg[name]["mrr"].append(m.mrr)
            cells += f"{'-':>11}" if cov is None else f"{cov:>11.2f}"
        print(f"{q.id:<18}{len(q.required_doc_ids):>4}{cells}")

    print()
    print(f"{'arm':<16}{'Cov(all)':>10}{'Cov(xdoc)':>11}{'recall':>9}{'MRR':>8}{'n_xdoc':>8}")
    print("-" * 62)
    for name, metrics in agg.items():
        cov = metrics["cov"]
        print(
            f"{name:<16}{sum(cov) / len(cov):>10.3f}"
            f"{(sum(xdoc[name]) / len(xdoc[name]) if xdoc[name] else 0):>11.3f}"
            f"{sum(metrics['recall']) / len(metrics['recall']):>9.3f}"
            f"{sum(metrics['mrr']) / len(metrics['mrr']):>8.3f}"
            f"{len(xdoc[name]):>8d}"
        )
    print()
    print("Cov(xdoc) is the number reranking broke: 0.929 at k=8 -> 0.583 at k=3.")
    print("A selector that does not move it will not move XDR either, and is not worth")
    print("120 generations to confirm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
