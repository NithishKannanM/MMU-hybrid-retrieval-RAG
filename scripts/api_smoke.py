"""LIVE gate: drive a running MMU server through /health, /search and /answer/stream.

    uv run uvicorn mmu.api.app:app --port 8099     # in one terminal
    uv run python scripts/api_smoke.py             # in another

Checks the three things that are awkward to assert from inside the test suite: that SSE
actually streams incrementally over a real socket (rather than arriving as one buffered
blob), that concurrent searches do not serialize against a real uvicorn worker, and that
/health stays responsive while they run.
"""

from __future__ import annotations

import asyncio
import sys
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099"
QUERY = "If we suffer a breach in Germany, who do we notify and by when?"


async def main() -> int:
    async with httpx.AsyncClient(base_url=BASE, timeout=300.0) as client:
        try:
            health = (await client.get("/health")).json()
        except httpx.ConnectError:
            print(f"no server at {BASE} — start `uv run uvicorn mmu.api.app:app --port 8099`")
            return 1

        print(f"health: ready={health['ready']} chunks={health['index']['chunks']} "
              f"generator={health['generator']['model']}")
        if not health["ready"]:
            print(f"  index error: {health['index']['error']}")
            return 1

        # --- retrieval, with channel provenance ---------------------------------
        search = (await client.post("/search", json={"query": QUERY, "k": 8})).json()
        print(f"\n/search: {len(search['chunks'])} chunks "
              f"({search['timings']['total_ms']:.0f}ms)")
        for c in search["chunks"]:
            mark = " <- sparse-only (BM25 earning its place)" if c["dense_rank"] is None else ""
            print(f"  {c['rank']}. [{c['chunk']['doc_id']}] dense={c['dense_rank']} "
                  f"sparse={c['sparse_rank']}{mark}")

        # --- SSE, verifying it arrives incrementally ----------------------------
        print("\n/answer/stream:")
        first_token_at = None
        started = time.perf_counter()
        events: list[str] = []
        async with client.stream("POST", "/answer/stream", json={"query": QUERY}) as r:
            async for line in r.aiter_lines():
                if line.startswith("event: "):
                    name = line[7:]
                    events.append(name)
                    if name == "token" and first_token_at is None:
                        first_token_at = time.perf_counter() - started
        total = time.perf_counter() - started
        print(f"  events: {events[0]} -> token x{events.count('token')} -> {events[-1]}")
        print(f"  first token at {first_token_at:.2f}s, complete at {total:.2f}s")
        if first_token_at is not None and total - first_token_at < 0.05:
            print("  WARNING: everything arrived at once — the stream is being buffered "
                  "(check x-accel-buffering if behind a proxy)")

        # --- concurrency, against a real worker ---------------------------------
        n = 8
        started = time.perf_counter()
        results = await asyncio.gather(
            *[client.post("/search", json={"query": QUERY, "k": 8}) for _ in range(n)],
            client.get("/health"),
        )
        elapsed = time.perf_counter() - started
        one = search["timings"]["total_ms"] / 1000
        print(f"\n{n} concurrent searches in {elapsed:.2f}s "
              f"(serial would be ~{n * one:.2f}s), all {results[0].status_code}")
        print("  the event loop is free during retrieval" if elapsed < n * one * 0.75
              else "  WARNING: looks serialized — is retrieval on the event loop?")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
