"""Reranker factory — the one place a reranker *alias* becomes a live object.

Mirrors :mod:`mmu.generate.registry` in both respects that matter:

**Deferred import.** Selecting no reranker must never import
:mod:`mmu.retrieve.rerank` (which lazy-imports ``sentence_transformers`` and therefore
torch on first use). An offline test run touches neither.

**Aliases only, never a raw model id.** :func:`build_reranker` takes ``alias: str |
None``. The reasoning is the same security boundary documented at length in the
generator registry: this parameter is the kind of thing that ends up exposed to an HTTP
caller, and accepting a raw model string would let a request name arbitrary weights to
download and execute. The table below is the only thing that can turn a request into a
model id; extending it means editing this file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mmu.config import Settings
    from mmu.retrieve.rerank import Reranker

#: alias -> the Settings attribute holding the model id it resolves to. ``"none"`` is a
#: first-class alias rather than a magic absence so that an arm can say "deliberately no
#: reranker" and have that spelling validated like any other.
_ALIASES: dict[str, str] = {
    "default": "rerank_model",
    "bge": "rerank_model",
}

#: Aliases that resolve to no reranker at all.
_DISABLED: frozenset[str] = frozenset({"none", "off"})


def available_rerankers() -> tuple[str, ...]:
    return tuple(sorted(_ALIASES)) + tuple(sorted(_DISABLED))


#: alias -> a factory taking (settings, store, dense). Same alias-only discipline as
#: the reranker table above, and the same reason: an arm names a strategy, never
#: arbitrary parameters smuggled through on the wire.
_SELECTORS: frozenset[str] = frozenset({"top-k", "mmr", "quota"})


def available_selectors() -> tuple[str, ...]:
    return tuple(sorted(_SELECTORS)) + tuple(sorted(_DISABLED))


def build_selector(
    settings: Settings,
    alias: str | None = None,
    *,
    store=None,
    dense=None,
):
    """Resolve ``alias`` to a context :class:`~mmu.retrieve.select.Selector`, or None.

    ``mmr`` needs chunk-to-chunk similarity and therefore the store and dense index;
    asking for it without them is an error rather than a silent fallback to a lexical
    measure, because two arms differing in their notion of redundancy while sharing a
    label would make the comparison meaningless.
    """
    if alias is None:
        return None
    key = alias.strip().lower()
    if key in _DISABLED:
        return None
    if key not in _SELECTORS:
        raise ValueError(
            f"unknown selector alias {alias!r}; valid aliases are "
            f"{', '.join(available_selectors())}"
        )

    from mmu.retrieve.select import DocQuotaSelector, MMRSelector, TopKSelector, embedding_similarity

    if key == "top-k":
        return TopKSelector()
    if key == "quota":
        return DocQuotaSelector(max_per_doc=settings.select_max_per_doc)
    if store is None or dense is None:
        raise ValueError(
            "selector 'mmr' needs the chunk store and dense index to measure "
            "redundancy between candidates; pass store= and dense="
        )
    return MMRSelector(
        similarity=embedding_similarity(store, dense),
        lambda_=settings.select_mmr_lambda,
    )


def build_reranker(settings: Settings, alias: str | None = None) -> Reranker | None:
    """Resolve ``alias`` to a reranker, or ``None`` for an unreranked arm.

    ``None`` is returned only for an *explicitly* disabled alias or no alias at all. An
    unrecognized alias raises rather than silently falling back to no reranking: an arm
    labelled as reranked that quietly did not rerank would make the whole comparison a
    measurement of nothing, and that failure would be invisible in the report.
    """
    if alias is None:
        return None
    key = alias.strip().lower()
    if key in _DISABLED:
        return None
    if key not in _ALIASES:
        raise ValueError(
            f"unknown reranker alias {alias!r}; valid aliases are "
            f"{', '.join(available_rerankers())}"
        )

    from mmu.retrieve.rerank import CrossEncoderReranker

    return CrossEncoderReranker(
        getattr(settings, _ALIASES[key]),
        device=settings.rerank_device,
        batch_size=settings.rerank_batch_size,
        max_length=settings.rerank_max_length,
    )
