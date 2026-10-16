"""Measure how much a reranker could possibly win — and, with --apply, how much it did.

    uv run python scripts/rerank_headroom.py              # ceiling only, no model
    uv run python scripts/rerank_headroom.py --apply       # also run the cross-encoder

A reranker sits between over-fetch and truncation: retrieval fuses `depth`
candidates, the reranker reorders them, and the top `k` survive. So its ceiling is
**recall@depth** and its baseline is **recall@k** — and both must be measured from the
*same* fusion, because RRF's output depends on the size of the candidate pool and
prefixes of a depth-128 ranking are not the top of a depth-32 ranking.

This is a script rather than a test for the same reason as `index_smoke.py`: it loads
the real BGE-M3 and the real built index, so `uv run pytest` must not depend on it.

Why it exists at all: the obvious metrics cannot see a reranker. `coverage` is a set
intersection over `doc_id`, so reordering the same chunks cannot move it — and hybrid
already scores 1.000 on it. Running this *first* is what established that the headroom
is real and where it lives, rather than discovering after a day of plumbing that every
number came back identical.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from mmu.config import get_settings
from mmu.embed.bge_m3 import BgeM3Embedder
from mmu.eval.metrics import count_relevant_chunks, rank_metrics
from mmu.eval.questions import load_questions
from mmu.retrieve.build import load_index
from mmu.retrieve.hybrid import HybridRetriever
from mmu.retrieve.registry import build_reranker
from mmu.retrieve.rerank import rerank as rerank_result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="also run the real cross-encoder over each pool and report the recall it "
             "actually achieved against the ceiling (downloads ~2.3GB on first use)",
    )
    parser.add_argument("--rerank-alias", default="default")
    args = parser.parse_args(argv)

    settings = get_settings()
    questions = [q for q in load_questions(settings.questions_path) if q.relevant_doc_ids]
    if not questions:
        print("no questions with a relevant set — nothing to measure", file=sys.stderr)
        return 1

    embedder = BgeM3Embedder(
        settings.embed_model,
        device=settings.embed_device,
        batch_size=settings.embed_batch_size,
    )
    store, dense, sparse = load_index(settings.index_dir, embedder)
    retriever = HybridRetriever(store, dense, sparse, embedder)
    corpus = [(c.doc_id, c.text) for c in store.chunks]

    reranker = build_reranker(settings, args.rerank_alias) if args.apply else None
    if reranker is not None:
        print(f"applying reranker: {reranker.name}\n")

    k, depth = settings.top_k, 32
    small_k = 3

    print(
        f"{len(questions)} questions with a relevant set | {len(store)} chunks | "
        f"k={k}, depth={depth}\n"
        f"One fusion per question at depth={depth}; prefixes of it scored at "
        f"k={small_k}/{k}/{depth}.\n"
    )
    head = (
        f"{'question':<18}{'Nrel':>5}{f'R@{small_k}':>7}{f'R@{k}':>7}"
        f"{f'R@{depth}':>7}{'1st':>5}{f'MRR@{k}':>8}"
    )
    if reranker is not None:
        head += f"{'|':>3}{f'rrR@{small_k}':>8}{f'rrR@{k}':>8}{f'rrMRR@{k}':>10}"
    head += "  headroom"
    print(head)
    print("-" * len(head))

    agg: dict[int, list[float]] = {small_k: [], k: [], depth: []}
    rr_agg: dict[str, list[float]] = {"r3": [], "r8": [], "mrr8": []}
    rescuable: list[tuple[str, int]] = []
    partial: list[str] = []
    saturated: list[str] = []

    for q in questions:
        n_rel = count_relevant_chunks(
            corpus, relevant_doc_ids=q.relevant_doc_ids, must_contain=q.must_contain
        )
        kw = dict(
            relevant_doc_ids=q.relevant_doc_ids,
            must_contain=q.must_contain,
            n_relevant_total=n_rel,
        )
        result = retriever.search(
            q.question, k=depth, depth=depth,
            channels=["dense", "sparse"], rrf_k=settings.rrf_k,
        )
        m = {cut: rank_metrics(result.chunks, k=cut, **kw) for cut in agg}
        for cut in agg:
            agg[cut].append(m[cut].recall or 0.0)

        first = m[depth].first_relevant_rank
        gain = (m[depth].recall or 0.0) - (m[k].recall or 0.0)
        if m[k].recall == 0.0 and (m[depth].recall or 0.0) > 0.0:
            note = f"RESCUABLE (1st relevant at rank {first})"
            rescuable.append((q.id, first or 0))
        elif gain > 0:
            note = f"+{gain:.3f} recall reachable by reordering"
            partial.append(q.id)
        else:
            note = f"none — k={k} already holds all of it"
            saturated.append(q.id)

        row = (
            f"{q.id:<18}{n_rel:>5}{m[small_k].recall:>7.3f}{m[k].recall:>7.3f}"
            f"{m[depth].recall:>7.3f}{(first if first else '-'):>5}"
            f"{m[k].mrr:>8.3f}"
        )
        if reranker is not None:
            # The reranker reorders the SAME pool, so this is strictly a measurement of
            # ordering: nothing enters or leaves, only the top-k selection changes.
            reordered = rerank_result(result, reranker, k=depth)
            rr = {cut: rank_metrics(reordered.chunks, k=cut, **kw) for cut in (small_k, k)}
            rr_agg["r3"].append(rr[small_k].recall or 0.0)
            rr_agg["r8"].append(rr[k].recall or 0.0)
            rr_agg["mrr8"].append(rr[k].mrr)
            row += (
                f"{'|':>3}{rr[small_k].recall:>8.3f}{rr[k].recall:>8.3f}"
                f"{rr[k].mrr:>10.3f}"
            )
        print(row + f"  {note}")

    n = len(questions)
    print()
    print(f"  mean recall@{small_k:<3} {sum(agg[small_k]) / n:.3f}   what a reranker must "
          f"preserve if k shrinks to {small_k}")
    print(f"  mean recall@{k:<3} {sum(agg[k]) / n:.3f}   production baseline today")
    print(f"  mean recall@{depth:<3} {sum(agg[depth]) / n:.3f}   the reranker's CEILING "
          f"at this depth")
    print()
    ceiling = sum(agg[depth]) / n
    base = sum(agg[k]) / n
    print(f"  headroom: {ceiling - base:+.3f} absolute "
          f"({(ceiling / base - 1) * 100:+.0f}% relative), available purely by "
          f"reordering\n            candidates retrieval ALREADY fetched.")
    print()
    print(f"  nothing relevant at k={k}, but present in the pool : {len(rescuable)}")
    for qid, rank in rescuable:
        print(f"      {qid:<18} first relevant chunk at rank {rank}")
    print(f"  partial at k={k} with more reachable               : {len(partial)}")
    print(f"  already saturated at k={k}                         : {len(saturated)}")
    print()
    if reranker is not None:
        print()
        print("  WITH THE REAL CROSS-ENCODER (same pool, reordered):")
        print(f"    mean recall@{small_k:<3} {sum(rr_agg['r3']) / n:.3f}   "
              f"(was {sum(agg[small_k]) / n:.3f})")
        print(f"    mean recall@{k:<3} {sum(rr_agg['r8']) / n:.3f}   "
              f"(was {base:.3f}, ceiling {ceiling:.3f})")
        print(f"    mean MRR@{k:<6} {sum(rr_agg['mrr8']) / n:.3f}")
        achieved = sum(rr_agg['r8']) / n
        if ceiling > base:
            print(f"    captured {(achieved - base) / (ceiling - base) * 100:.0f}% of the "
                  f"available headroom at k={k}")
        print()
    print("A reranker cannot recover anything outside the pool. Raising `depth` is the")
    print("lever for that, and it is free at retrieval time but costs the reranker a")
    print("linear number of cross-encoder forward passes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
