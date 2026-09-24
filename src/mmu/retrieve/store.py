"""The chunk store: the row <-> chunk mapping both channels address, persisted.

The FAISS index holds vectors addressed by row position and nothing else; BM25 holds
tokenized texts addressed by the same row. This module owns the one list that turns a
row back into a :class:`mmu.core.models.Chunk`, plus the manifest that records *what
was indexed and how*.

**The manifest exists to catch a silent mismatch.** An index built with one embedding
model, chunk size or prefix policy and then queried with another produces no error --
just quietly wrong neighbours. Recording the model name, dimension, chunk parameters and
corpus fingerprint means :func:`ChunkStore.load` can refuse an index that does not match
the current settings, and say which field diverged.

Storage is JSONL plus a small JSON manifest rather than a database: the whole artifact
is rebuilt from ``data/corpus/`` by one command, it is a few hundred rows at benchmark
scale, and keeping it as plain text means a failing metric can be debugged by reading
the chunks with ``jq``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from mmu.core.models import Chunk

CHUNKS_FILE = "chunks.jsonl"
INDEX_FILE = "faiss.bin"
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class Manifest:
    """What the on-disk index was built from. Compared on load, not just recorded."""

    embed_model: str
    dim: int
    chunk_max_tokens: int
    chunk_min_tokens: int
    chunk_overlap_tokens: int
    n_chunks: int
    doc_ids: list[str]
    built_at: str

    def mismatch(self, other: "Manifest") -> str | None:
        """Return the first field that would make ``other``'s queries meaningless."""
        for field in ("embed_model", "dim", "chunk_max_tokens", "chunk_overlap_tokens"):
            mine, theirs = getattr(self, field), getattr(other, field)
            if mine != theirs:
                return (
                    f"index was built with {field}={mine!r} but the current settings say "
                    f"{theirs!r}. Rebuild with `mmu index build`."
                )
        return None


class ChunkStore:
    """Chunks in row order, plus their manifest. Row i is chunks[i]."""

    def __init__(self, chunks: list[Chunk], manifest: Manifest | None = None) -> None:
        self._chunks = chunks
        self._manifest = manifest
        self._by_id = {c.chunk_id: i for i, c in enumerate(chunks)}
        if len(self._by_id) != len(chunks):
            # Two chunks sharing an id would make row lookup ambiguous and corrupt every
            # per-document metric. It means doc_id slugging collided upstream.
            raise ValueError("duplicate chunk_id in store; doc_id slugs likely collided")

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    @property
    def manifest(self) -> Manifest | None:
        return self._manifest

    @property
    def texts(self) -> list[str]:
        return [c.text for c in self._chunks]

    def row(self, index: int) -> Chunk:
        return self._chunks[index]

    def row_of(self, chunk_id: str) -> int:
        return self._by_id[chunk_id]

    def doc_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for c in self._chunks:
            seen.setdefault(c.doc_id, None)
        return list(seen)

    def counts_by_doc(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self._chunks:
            out[c.doc_id] = out.get(c.doc_id, 0) + 1
        return out

    def text_of_doc(self, doc_id: str) -> str:
        """Every chunk of one document, joined. Used by `mmu questions validate` to
        check a `must_contain` span exists verbatim in the source."""
        return "\n".join(c.text for c in self._chunks if c.doc_id == doc_id)

    # ------------------------------------------------------------------ persistence

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        with (index_dir / CHUNKS_FILE).open("w", encoding="utf-8") as fh:
            for c in self._chunks:
                fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        if self._manifest is not None:
            (index_dir / MANIFEST_FILE).write_text(
                json.dumps(asdict(self._manifest), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    @classmethod
    def load(cls, index_dir: Path) -> "ChunkStore":
        chunks_path = index_dir / CHUNKS_FILE
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"no index at {index_dir}. Run `mmu index build` first."
            )
        chunks = [
            Chunk(**json.loads(line))
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest = None
        manifest_path = index_dir / MANIFEST_FILE
        if manifest_path.exists():
            manifest = Manifest(**json.loads(manifest_path.read_text(encoding="utf-8")))
        return cls(chunks, manifest)
