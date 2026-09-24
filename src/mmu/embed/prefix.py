"""Query/document prefix policy — the seam for retrieval-instruction asymmetry.

**The risk this module exists to prevent.** The original spec claimed BGE-M3 is
asymmetrically instruction-tuned and needs a query prefix distinct from its document
side. That claim is true of ``bge-large-en-v1.5`` (and its base/small siblings) — it is
**not** true of ``bge-m3``. BAAI's own guidance is that BGE-M3 needs no retrieval
instruction, and FlagEmbedding's ``BGEM3FlagModel`` defaults
``query_instruction_for_retrieval`` to ``None``. Inventing a query prefix for bge-m3
"to be safe" would not be a safe default — it would *cause* the exact silent ranking
degradation this whole package exists to prevent, because a wrong prefix changes the
embedding without ever raising.

**Why `policy_for` raises on an unknown model instead of defaulting to empty.** A wrong
guess is worse than a loud failure: defaulting silently to empty/empty means that
swapping in a model which *does* need an instruction (e.g. moving to
``bge-large-en-v1.5`` from ``bge-m3``) produces no error at all — retrieval quality
just gets quantifiably worse and nobody is told. Raising forces the one-line fix
(register the model) at the point someone would otherwise ship a silent regression.
"""

from __future__ import annotations

from dataclasses import dataclass

from mmu.embed.base import InputKind


class UnknownEmbeddingModel(KeyError):
    """Raised by :func:`policy_for` for a model with no registered prefix policy.

    A ``KeyError`` subclass rather than a bare ``ValueError``: callers that already
    handle "missing key" style errors elsewhere in the stack get this for free, while
    ``str(err)`` still carries the actionable message (KeyError's repr quirk is worked
    around in :func:`policy_for` by passing the full message as the single arg).
    """


@dataclass(frozen=True)
class PrefixPolicy:
    """The text prepended to a query vs. a document before encoding.

    Empty string is a valid, explicit choice (BGE-M3's actual behavior) — it is not a
    placeholder for "not yet configured". See :data:`POLICIES` for the registry that
    makes that distinction real, and :func:`policy_for` for why an unregistered model
    fails loudly instead of falling back here.
    """

    query: str = ""
    document: str = ""

    def apply(self, text: str, kind: InputKind) -> str:
        """Prepend the side-appropriate prefix. Never mutates ``text`` in place (str is
        immutable anyway, but the point is this returns a new string, not a reference)."""
        prefix = self.query if kind == "query" else self.document
        return prefix + text


#: Model name -> PrefixPolicy. Keyed on the exact string that will be passed as
#: ``Settings.embed_model`` / ``SentenceTransformer`` model id, so lookups are exact,
#: not pattern-matched — a near-miss model name should fail loudly via `policy_for`
#: rather than silently matching the wrong entry.
POLICIES: dict[str, PrefixPolicy] = {
    # Empty by DECISION, not by omission. See the module docstring: the spec's premise
    # ("BGE-M3 needs an asymmetric query instruction") describes bge-large-en-v1.5, not
    # bge-m3. BAAI's guidance and FlagEmbedding's own default
    # (query_instruction_for_retrieval=None) agree BGE-M3 takes no retrieval
    # instruction. If you are here because retrieval quality looks off, the fix is
    # almost certainly not to add a prefix — re-read the risk note above first.
    "BAAI/bge-m3": PrefixPolicy(query="", document=""),
    # bge-*-en-v1.5 IS the asymmetric family the original spec had in mind.
    "BAAI/bge-large-en-v1.5": PrefixPolicy(
        query="Represent this sentence for searching relevant passages: ", document=""
    ),
    "BAAI/bge-base-en-v1.5": PrefixPolicy(
        query="Represent this sentence for searching relevant passages: ", document=""
    ),
    "BAAI/bge-small-en-v1.5": PrefixPolicy(
        query="Represent this sentence for searching relevant passages: ", document=""
    ),
    # The deterministic test double (mmu.embed.hashing.HashingEmbedder). Empty/empty so
    # it resolves through the same policy_for() path as a real model rather than needing
    # a special case in every caller.
    "hashing-test-embedder": PrefixPolicy(query="", document=""),
}


def policy_for(model_name: str) -> PrefixPolicy:
    """Look up the :class:`PrefixPolicy` for ``model_name``.

    Raises :class:`UnknownEmbeddingModel` (a ``KeyError``) on an unregistered model
    rather than defaulting to empty prefixes — see the module docstring for why a loud
    failure here is the correct trade against a silent one.
    """
    try:
        return POLICIES[model_name]
    except KeyError:
        known = ", ".join(sorted(POLICIES))
        raise UnknownEmbeddingModel(
            f"no PrefixPolicy registered for embedding model {model_name!r}. "
            f"Register it in mmu.embed.prefix.POLICIES rather than defaulting to empty "
            f"prefixes — a wrong (including missing) prefix silently degrades ranking "
            f"instead of raising, which is the failure this module exists to prevent. "
            f"Known models: {known}"
        ) from None
