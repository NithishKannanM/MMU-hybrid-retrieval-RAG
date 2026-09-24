"""BGE-M3 via ``sentence-transformers`` — dense head only.

Ported from ``projects/local_infra/src/local_infra/embed/st.py``. Kept from that
version: the lazy ``_load()`` (importing this module must not import torch — only
calling it does), the configurable batch size, and L2 normalization. Changed for MMU:
this now implements :class:`mmu.embed.base.Embedder` exactly (``encode`` takes ``kind``
as a required keyword-only argument — see that module for why), applies the
model-keyed :class:`mmu.embed.prefix.PrefixPolicy` to every text before encoding, and
is explicit that only the dense head is available (see the library tradeoff recorded in
``embed/base.py`` and reiterated in ``__init__`` below).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from mmu.embed.base import Encoding, InputKind
from mmu.embed.prefix import policy_for


class BgeM3Embedder:
    """BGE-M3 (or any sentence-transformers bi-encoder), lazily loaded on first use.

    Implements :class:`mmu.embed.base.Embedder`.

    ``device`` defaults to ``"cpu"`` and is set by the caller, not decided here — see
    ``mmu.config.Settings.embed_device`` for the 6GB-VRAM-contention reasoning (BGE-M3
    sharing a card with the generator forces repeated model eviction/reload); this class
    just does what it is told.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        *,
        batch_size: int = 16,
        return_sparse: bool = False,
        return_colbert: bool = False,
    ) -> None:
        if return_sparse or return_colbert:
            # Recorded at the point someone would hit it, not just in a docstring: the
            # sparse and ColBERT heads are extra layers that sentence-transformers does
            # not expose for BGE-M3. Getting them requires FlagEmbedding's
            # BGEM3FlagModel, which MMU deliberately does not depend on (it pulls the
            # whole training stack — datasets, accelerate, peft, ir-datasets — to do
            # inference, and declares no requires-python, i.e. unvetted for py3.14). See
            # embed/base.py's "library tradeoff, recorded" section for the full context.
            raise NotImplementedError(
                "sentence-transformers only exposes BGE-M3's dense head. "
                "return_sparse/return_colbert require swapping this class for "
                "FlagEmbedding's BGEM3FlagModel — see mmu.embed.base and the "
                "'Embedder library tradeoff' note for why that swap was not made by "
                "default."
            )
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None  # lazy: importing sentence-transformers pulls in torch.

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name, device=self._device)

    def encode(self, texts: Sequence[str], *, kind: InputKind) -> Encoding:
        self._load()
        policy = policy_for(self._model_name)
        prefixed = [policy.apply(text, kind) for text in texts]
        vecs = self._model.encode(
            prefixed,
            batch_size=self._batch_size,
            # Always True: normalizing makes inner product equal cosine similarity,
            # which is what makes faiss.IndexFlatIP (see mmu.retrieve.dense) a cosine
            # index rather than a magnitude-sensitive one. This is not a tunable knob —
            # turning it off would silently make long chunks outrank relevant short
            # ones by norm alone, with no error anywhere in the pipeline.
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        dense = np.asarray(vecs, dtype=np.float32)
        return Encoding(dense=dense, sparse=None, colbert=None)

    @property
    def dim(self) -> int:
        self._load()
        return int(self._model.get_sentence_embedding_dimension())

    @property
    def model_name(self) -> str:
        return self._model_name
