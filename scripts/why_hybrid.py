"""Demo: why MMU runs two retrieval channels instead of one.

Uses the REAL BGE-M3 and the REAL BM25 index from the project — no fakes — on a
corpus built to contain the exact trap legal text is full of: a pair of chunks
that are semantically adjacent and legally opposite.

Reports whatever actually happens, including the margin between the right answer
and its confusable twin. The margin is the interesting number: a channel that
ranks correctly by 0.01 is one paraphrase away from ranking incorrectly.
"""

from __future__ import annotations

import sys

import numpy as np

from mmu.core.fusion import rrf_scores
from mmu.embed.bge_m3 import BgeM3Embedder
from mmu.retrieve.sparse import Bm25Index

# A miniature compliance corpus. Note chunks 0 and 1: near-identical sentence
# shape, opposite legal meaning. That is the whole problem, in two lines.
CHUNKS = [
    "The data controller shall determine the purposes and means of the processing "
    "of personal data and remains accountable for compliance.",                      # 0 CONTROLLER
    "The data processor shall process personal data only on documented instructions "
    "from the controller and remains accountable to it.",                            # 1 PROCESSOR
    "Each party shall implement appropriate technical and organisational measures "
    "to ensure a level of security appropriate to the risk.",                        # 2 generic
    "In the case of a personal data breach, notification shall be made without "
    "undue delay and not later than 72 hours after becoming aware of it.",           # 3 breach
    "The supervisory authority shall be competent for the performance of the tasks "
    "assigned to it in accordance with this Regulation.",                            # 4 authority
    "A slow-cooked bean stew benefits from smoked paprika and a great deal of "
    "patience on the part of the cook.",                                             # 5 noise
]

LABEL = {0: "CONTROLLER", 1: "PROCESSOR ", 2: "generic   ", 3: "breach    ",
         4: "authority ", 5: "noise     "}

# (query, the row that is actually correct, the row it is most confusable with)
QUERIES = [
    ("What are the obligations of the data processor?", 1, 0),
    ("Who is the data controller and what do they decide?", 0, 1),
    ("data processor documented instructions", 1, 0),
]


def bar(value: float, lo: float, hi: float, width: int = 22) -> str:
    if hi <= lo:
        return " " * width
    n = int(round((value - lo) / (hi - lo) * width))
    return "#" * max(0, min(width, n))


def main() -> int:
    print("loading BAAI/bge-m3 on cpu ...", flush=True)
    embedder = BgeM3Embedder(device="cpu", batch_size=8)
    doc_vecs = embedder.encode(CHUNKS, kind="document").dense
    bm25 = Bm25Index(CHUNKS)
    print(f"  dim={embedder.dim}  chunks={len(CHUNKS)}\n")

    for query, correct, confusable in QUERIES:
        print("=" * 78)
        print(f"QUERY: {query!r}")
        print(f"  correct answer = chunk {correct} ({LABEL[correct].strip()}), "
              f"its legal opposite = chunk {confusable} ({LABEL[confusable].strip()})")
        print("=" * 78)

        # ---- dense channel -------------------------------------------------
        q = embedder.encode([query], kind="query").dense[0]
        cos = doc_vecs @ q
        dense_order = np.argsort(-cos)
        lo, hi = float(cos.min()), float(cos.max())

        print("\n  DENSE (BGE-M3 cosine)")
        for rank, row in enumerate(dense_order[:4], 1):
            mark = " <-- CORRECT" if row == correct else (
                   " <-- legal opposite" if row == confusable else "")
            print(f"    {rank}. [{LABEL[row]}] {cos[row]:.4f}  "
                  f"{bar(float(cos[row]), lo, hi)}{mark}")
        d_margin = float(cos[correct] - cos[confusable])
        print(f"    margin(correct - opposite) = {d_margin:+.4f}")

        # ---- sparse channel ------------------------------------------------
        hits = bm25.search(query, k=len(CHUNKS))
        scores = {row: s for row, s in hits}
        print("\n  SPARSE (BM25Okapi)")
        if hits:
            shi = max(scores.values())
            for rank, (row, s) in enumerate(hits[:4], 1):
                mark = " <-- CORRECT" if row == correct else (
                       " <-- legal opposite" if row == confusable else "")
                print(f"    {rank}. [{LABEL[row]}] {s:.4f}  "
                      f"{bar(s, 0.0, shi)}{mark}")
            s_margin = scores.get(correct, 0.0) - scores.get(confusable, 0.0)
            print(f"    margin(correct - opposite) = {s_margin:+.4f}")
        else:
            print("    (no lexical signal at all)")
            s_margin = 0.0

        # ---- fusion --------------------------------------------------------
        fused = rrf_scores([
            (1.0, [int(r) for r in dense_order]),
            (1.0, [row for row, _ in hits]),
        ])
        order = sorted(fused, key=lambda r: -fused[r])
        print("\n  HYBRID (RRF, k=60)")
        for rank, row in enumerate(order[:4], 1):
            d = list(dense_order).index(row) + 1
            s = next((i + 1 for i, (r, _) in enumerate(hits) if r == row), None)
            mark = " <-- CORRECT" if row == correct else (
                   " <-- legal opposite" if row == confusable else "")
            prov = f"dense#{d}" + (f" + sparse#{s}" if s else " only")
            print(f"    {rank}. [{LABEL[row]}] {fused[row]:.5f}  ({prov}){mark}")

        # ---- verdict -------------------------------------------------------
        dense_top = int(dense_order[0])
        sparse_top = hits[0][0] if hits else None
        fused_top = order[0]
        print("\n  VERDICT")
        print(f"    dense  picked {LABEL[dense_top].strip():11s} "
              f"{'RIGHT' if dense_top == correct else 'WRONG'}"
              f"   (margin {d_margin:+.4f})")
        print(f"    sparse picked "
              f"{LABEL[sparse_top].strip() if sparse_top is not None else 'nothing':11s} "
              f"{'RIGHT' if sparse_top == correct else 'WRONG'}"
              f"   (margin {s_margin:+.4f})")
        print(f"    hybrid picked {LABEL[fused_top].strip():11s} "
              f"{'RIGHT' if fused_top == correct else 'WRONG'}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
