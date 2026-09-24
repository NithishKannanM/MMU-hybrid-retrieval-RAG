"""The sparse channel: BM25Okapi over the same chunks, in the same row space.

**Why a lexical channel at all.** Dense embeddings are good at meaning and bad at exact
terms. In legal and compliance text a defined term is not a synonym of its neighbours:
"data controller" and "data processor" are semantically adjacent and legally opposite,
and an embedding will happily rank the wrong one first. BM25 scores the literal token —
term frequency, inverse document frequency, document-length normalization — so the chunk
that actually contains the defined term surfaces even when the embedder blurred it.

**The tokenizer is the channel.** Everything BM25 is here to do lives in what counts as
a token, which is why :func:`tokenize` is a separate pure function with its own tests
rather than a lambda inside the index. Two rules earn their keep on legal text:

* ``Art. 33`` must not become ``art`` + ``33``. Article references are the most precise
  retrieval signal a legal corpus has, and splitting on the period throws it away.
* ``data-controller`` must not become ``data`` + ``controller``, or a hyphenated defined
  term collapses into the two words it is defined *against*.

Both are handled by keeping intra-word periods and hyphens and splitting on everything
else. Note what this does *not* do: "Art. 33" is two tokens (``art``, ``33``), because
the period is followed by a space and is therefore sentence punctuation, not part of the
token. That is fine — both terms still match. The rule earns its keep on forms with no
space, like ``13.2`` or ``data-controller``, which would otherwise fragment into the
very words they are defined against. Case is folded; BM25 has no notion of proper nouns
and "Controller" at the start of a sentence is the same term as "controller" inside one.

**A measured caveat about IDF at small N.** ``BM25Okapi`` assigns an IDF of exactly 0.0
to a term occurring in half or more of the corpus, so such a term scores 0 in every
document and contributes nothing. This channel indexes *chunks*, not documents, so N is
in the hundreds even for a handful of files and ordinary defined terms stay well under
the threshold. But a genuinely ubiquitous word ("data", "processing" in a privacy
corpus) can cross it, and a query made only of such words returns an empty sparse
result. That is the correct outcome — those terms carry no discriminative lexical
signal and dense retrieval is what should answer — but it is a real property worth
knowing rather than discovering as a mystery.

Row indices are shared with the dense channel, so both hand ``rrf_fuse`` ids drawn from
one space and no tagging or offsetting is needed at the fusion step.
"""

from __future__ import annotations

import re
from typing import Sequence

#: A token is a run of letters/digits, optionally continued through a single internal
#: hyphen, period or apostrophe. The trailing period of a sentence is not internal, so
#: "Art. 33." yields "art." only if a digit follows — which it does in a citation.
_TOKEN = re.compile(r"[a-z0-9]+(?:[-.'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase, punctuation-stripped tokens that preserve legal term shapes."""
    return _TOKEN.findall(text.lower())


class Bm25Index:
    """BM25Okapi over chunk texts, addressed by row. Blocking and CPU-bound.

    ``get_scores`` is a numpy loop over the whole corpus, which is why this — like the
    FAISS scan and the query encode — lives inside the synchronous
    :meth:`mmu.retrieve.hybrid.HybridRetriever.search` and is offloaded exactly once,
    in :mod:`mmu.api.deps`.
    """

    def __init__(self, texts: Sequence[str]) -> None:
        self._size = len(texts)
        self._index = None
        if self._size:
            from rank_bm25 import BM25Okapi

            corpus = [tokenize(t) for t in texts]
            # rank_bm25 divides by the average document length; an all-empty corpus
            # (every chunk pure punctuation) would make that zero.
            if any(corpus):
                self._index = BM25Okapi(corpus)

    @property
    def size(self) -> int:
        return self._size

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` ``(row, score)`` pairs, best first.

        Rows scoring exactly zero are dropped. BM25 gives every document a score, and a
        zero means the query contributed no evidence for it — either no query term occurs
        in it, or every query term has an IDF of zero (see the module docstring).
        Including such a row would hand RRF a rank position implying evidence the channel
        does not have, which is the same mistake as the rejected ``n+1`` sentinel in the
        fusion step.
        """
        if self._index is None or k <= 0:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        import numpy as np

        scores = np.asarray(self._index.get_scores(tokens), dtype=np.float64)
        top = np.argsort(-scores, kind="stable")[: min(k, self._size)]
        return [(int(r), float(scores[r])) for r in top if scores[r] > 0.0]
