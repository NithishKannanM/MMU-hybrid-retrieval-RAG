"""Request models.

Requests are pydantic (they validate untrusted input); responses stay dataclasses from
:mod:`mmu.core.models`, which FastAPI serializes directly. That asymmetry is
deliberate — duplicating every response shape as a pydantic model would create two
definitions of the same contract that drift.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Channel = Literal["dense", "sparse"]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    k: int = Field(8, ge=1, le=50)
    #: Per-channel over-fetch before fusion. None means max(4k, 20).
    depth: int | None = Field(None, ge=1, le=500)
    #: **The ablation switch.** `["dense"]` is the experiment that justifies the whole
    #: hybrid design, so it is a first-class parameter rather than something reproduced
    #: by setting a weight to zero — a zeroed weight still pays the channel's latency
    #: and still influences the over-fetch, which makes the comparison dishonest.
    channels: list[Channel] = Field(default_factory=lambda: ["dense", "sparse"])
    dense_weight: float = Field(1.0, ge=0.0)
    sparse_weight: float = Field(1.0, ge=0.0)
    rrf_k: int = Field(60, ge=1)


class AnswerRequest(SearchRequest):
    #: An **alias** ("default"/"fast"), never a raw model id. A request must not be able
    #: to name a cloud model and route around the locally configured default.
    generator: str | None = None
    #: 0.0 by default: a benchmark that is not reproducible is not a benchmark.
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1, le=8192)
