"""Turns a retrieved-context question into a *parseable* one — the seam the whole
deterministic side of the eval harness rests on.

The eval harness has five metrics; the spec for this project counts only one of them
(Answer Faithfulness) as needing real entailment judgment for its headline number, and
even that one relies on markers to locate the source. The other deterministic pieces —
XDR's ``Cite`` factor and all three ``CUR`` variants — all reduce to "which ``[Sn]``
markers appear in the answer text, in what order, pointing at which chunk." None of that
is possible unless the prompt makes the model's citation behavior visible as text rather
than as reasoning nobody can inspect. That is what this module buys: a system prompt
that *mandates* the marker format, and parsing that is honest about a model's failure
modes (inventing out-of-range markers, dropping the format entirely, refusing without
saying so) rather than crashing or silently mis-scoring on them.

**Why refusal gets a dedicated detector instead of being scored as "zero markers."** A
model that correctly says "the context does not answer this" produces an answer with no
claims and, correctly, no citations — indistinguishable at the marker level from a model
that just dropped the citation format while still asserting things. Answer Faithfulness
needs to tell those apart: a correct refusal is maximally faithful (there is nothing
unsupported in it), while an uncited assertion is a formatting failure that should zero
out the deterministic metrics for that answer. ``is_refusal`` is the seam that lets the
harness make that distinction; see its docstring for why the pattern list stays narrow.
"""

from __future__ import annotations

import re
from typing import Sequence

from mmu.core.models import Citation, RetrievedChunk
from mmu.generate.base import Message

#: Restricts the model to the numbered context, mandates an inline marker on every
#: factual statement, allows multi-source citation, and — the clause that makes a
#: correct refusal scoreable as maximally faithful rather than as a failure — instructs
#: an explicit, uncited refusal when the context does not answer the question. The
#: suggested refusal wording below is deliberately the same phrasing `is_refusal` looks
#: for, so a compliant model's refusal is recognized rather than merely correct in spirit.
SYSTEM_PROMPT = (
    "You are a compliance research assistant. Answer strictly from the numbered "
    "sources given in the context below — never from prior knowledge, training data, "
    "or an assumption about what a document probably says. Do not use any information "
    "that is not present in the numbered sources.\n\n"
    "Every factual statement in your answer must end with an inline bracketed source "
    'marker, for example "...not later than 72 hours [S3]." A statement drawing on '
    'more than one source must cite them all, for example "...applies to both '
    'controllers and processors [S1][S4]." Cite only source numbers that appear in the '
    "context below — never invent a number that was not given to you.\n\n"
    "If the numbered sources do not contain the answer, say so explicitly — for "
    'example: "The provided context does not contain this information." — and cite '
    "nothing. A clear refusal is the correct response when the context does not "
    "support one; do not guess, speculate, or answer from outside knowledge in order "
    "to avoid saying so."
)


def _format_source(marker: int, chunk: RetrievedChunk) -> str:
    """Render one numbered context block. The location suffix (page or chunk index) is
    for a human skimming the prompt during debugging; the model only needs the number."""
    if chunk.chunk.page_from is not None:
        where = f"p.{chunk.chunk.page_from}"
    else:
        where = f"chunk {chunk.chunk.chunk_index}"
    return f"[S{marker}] ({chunk.doc_id}, {where})\n{chunk.text}"


def build_messages(query: str, chunks: Sequence[RetrievedChunk]) -> list[Message]:
    """Build the system + user turns, numbering ``chunks`` ``[S1]…[Sk]`` in the order
    given — i.e. fused rank order, since that is the order retrieval hands them over in.

    The marker number is the 1-based position in ``chunks``, not ``chunk.rank`` — they
    should coincide for a normal retrieval result, but making the *sequence* position
    authoritative is what lets ``resolve_citations`` index back into this same list with
    no dependency on a caller having preserved rank metadata correctly.
    """
    context = "\n\n".join(_format_source(i, c) for i, c in enumerate(chunks, start=1))
    if not context:
        context = "(no sources retrieved)"
    user = f"Context:\n{context}\n\nQuestion: {query}"
    return [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=user),
    ]


#: Matches an ``[Sn]`` marker. Deliberately simple — digits only, no whitespace inside
#: the brackets — because a looser pattern (e.g. allowing "[S3, S4]") would need to
#: define its own sub-grammar and every downstream consumer would have to agree on it.
#: A model that wants to cite two sources is instructed to write two markers.
MARKER_RE = re.compile(r"\[S(\d+)\]")


def parse_markers(text: str, k: int) -> list[int]:
    """Distinct 1-based marker numbers in ``text``, first-appearance order.

    Markers outside ``[1, k]`` are dropped, not raised on. A model given ``k`` sources
    will sometimes cite ``[S12]`` — this is not an off-by-one bug to fix, it is the
    model inventing a number past what it was given, and the only correct response to
    that at parse time is to ignore it: indexing into ``chunks`` with it would crash the
    caller, and silently clamping it into range would attribute a claim to the wrong
    source. ``k`` itself is a caller-supplied bound (normally ``len(chunks)``) rather
    than inferred from the text, so a run over an answer with zero real sources still
    parses safely.
    """
    seen: set[int] = set()
    ordered: list[int] = []
    for match in MARKER_RE.finditer(text):
        n = int(match.group(1))
        if n < 1 or n > k:
            continue
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def resolve_citations(text: str, chunks: Sequence[RetrievedChunk]) -> list[Citation]:
    """Map every well-formed, in-range marker in ``text`` to a :class:`Citation`,
    in first-appearance order. Delegates the in-range filtering to `parse_markers`
    rather than re-deriving it, so the two functions can never disagree about which
    markers are valid."""
    citations: list[Citation] = []
    for n in parse_markers(text, len(chunks)):
        rc = chunks[n - 1]
        citations.append(
            Citation(
                marker=n,
                chunk_id=rc.chunk_id,
                doc_id=rc.doc_id,
                filename=rc.chunk.filename,
                page_from=rc.chunk.page_from,
            )
        )
    return citations


def marker_compliance(text: str, k: int) -> float:
    """1.0 iff ``text`` contains at least one well-formed, in-range marker, else 0.0.

    This is the run-level tripwire the eval report checks before trusting `Cite`, XDR,
    or any CUR variant — see :class:`mmu.core.models.Answer.marker_compliance`. An
    answer whose only marker is out-of-range (e.g. a model given 4 sources that cites
    only ``[S9]``) scores 0.0 here, same as an answer with no markers at all: neither
    one leaves anything for the deterministic metrics to attribute a claim to.
    """
    return 1.0 if parse_markers(text, k) else 0.0


#: Conservative on purpose. Each pattern requires the noun ("context"/"sources"/
#: "documents") and a negated verb of *containing information* to sit right next to each
#: other. A negation elsewhere in the answer — e.g. "the policy does not require
#: notification within 24 hours, only 72 [S1]" — does not match any of these, because
#: the negation there is about the *content* of a cited claim, not about whether the
#: context itself lacks the answer. A false positive here is the worse failure: it would
#: award an unsupported real answer a faithfulness score of 1.0 by mistaking it for a
#: refusal, exactly backwards from what the refusal carve-out is for.
REFUSAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"context does not (contain|state|mention|address|specify|provide|include)",
        r"context[s]? do(es)? not (contain|state|mention|address|specify|provide|include)",
        r"cannot be answered (from|based on|using) the (provided |given )?context",
        r"no (relevant )?information (is )?(provided|available|found) in the "
        r"(provided |given )?(context|sources|documents)",
        r"(sources|documents) do not (contain|state|mention|address|specify|provide|include)",
        r"context (provided |given )?(does not|doesn't) (answer|address) this",
    )
]


def is_refusal(text: str) -> bool:
    """True when ``text`` is an explicit "the context does not answer this" refusal.

    Used by the AF metric to award ``m=0`` refusals a score of 1.0 instead of 0.0 — see
    the module docstring. The pattern list is intentionally short and literal rather
    than semantic (no LLM call here): a false negative just costs one refusal scored as
    an unremarkable empty answer, which is recoverable; a false positive silently
    launders a real, unsupported answer into a perfect faithfulness score, which is not.
    """
    return any(pattern.search(text) for pattern in REFUSAL_PATTERNS)
