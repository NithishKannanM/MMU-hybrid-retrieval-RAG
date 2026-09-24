"""ChunkStore: the row mapping both channels share, and the manifest guard."""

from __future__ import annotations

import pytest

from mmu.core.models import Chunk
from mmu.retrieve.store import ChunkStore, Manifest


def _chunk(doc_id: str, index: int, text: str = "text") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}:{index}",
        doc_id=doc_id,
        filename=f"{doc_id}.txt",
        chunk_index=index,
        text=text,
        char_start=0,
        char_end=len(text),
    )


def _manifest(**over) -> Manifest:
    base = dict(
        embed_model="BAAI/bge-m3",
        dim=1024,
        chunk_max_tokens=512,
        chunk_min_tokens=100,
        chunk_overlap_tokens=64,
        n_chunks=2,
        doc_ids=["gdpr"],
        built_at="2026-09-21T00:00:00+00:00",
    )
    return Manifest(**{**base, **over})


def test_row_addressing_round_trips() -> None:
    chunks = [_chunk("gdpr", 0), _chunk("gdpr", 1), _chunk("policy", 0)]
    store = ChunkStore(chunks)
    assert len(store) == 3
    assert store.row(1).chunk_id == "gdpr:1"
    assert store.row_of("policy:0") == 2
    assert store.doc_ids() == ["gdpr", "policy"]
    assert store.counts_by_doc() == {"gdpr": 2, "policy": 1}


def test_duplicate_chunk_id_is_refused() -> None:
    """Two chunks sharing an id make row lookup ambiguous and corrupt every
    per-document metric. It means doc_id slugs collided upstream."""
    with pytest.raises(ValueError, match="duplicate chunk_id"):
        ChunkStore([_chunk("gdpr", 0), _chunk("gdpr", 0)])


def test_text_of_doc_joins_only_that_document() -> None:
    store = ChunkStore([_chunk("a", 0, "alpha"), _chunk("b", 0, "beta"), _chunk("a", 1, "gamma")])
    assert store.text_of_doc("a") == "alpha\ngamma"


def test_save_load_round_trip(tmp_path) -> None:
    chunks = [_chunk("gdpr", 0, "first"), _chunk("gdpr", 1, "second")]
    ChunkStore(chunks, _manifest()).save(tmp_path)

    loaded = ChunkStore.load(tmp_path)
    assert [c.chunk_id for c in loaded.chunks] == ["gdpr:0", "gdpr:1"]
    assert loaded.chunks[0].text == "first"
    assert loaded.manifest is not None
    assert loaded.manifest.embed_model == "BAAI/bge-m3"


def test_load_without_an_index_says_what_to_run(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="mmu index build"):
        ChunkStore.load(tmp_path)


def test_manifest_mismatch_names_the_diverging_field() -> None:
    """An index built with one model and queried with another produces no error —
    just quietly wrong neighbours. The manifest is what turns that into a message."""
    built = _manifest()
    assert built.mismatch(_manifest()) is None

    problem = built.mismatch(_manifest(embed_model="BAAI/bge-large-en-v1.5"))
    assert problem is not None and "embed_model" in problem and "mmu index build" in problem

    assert "chunk_max_tokens" in (built.mismatch(_manifest(chunk_max_tokens=350)) or "")
    assert "dim" in (built.mismatch(_manifest(dim=768)) or "")
