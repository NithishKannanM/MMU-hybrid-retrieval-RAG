.PHONY: sync test test-live index validate serve eval ablate clean

sync:            ## install deps (torch ~2.5GB on first run)
	uv sync

test:            ## the real gate: fully offline, downloads nothing
	uv run pytest -q

test-live:       ## needs BGE-M3 downloaded and an Ollama model pulled
	uv run pytest -m live -q

index:           ## build the index from data/corpus/ ; prints the doc_id table
	uv run mmu index build --verbose

validate:        ## hard gate on data/questions.jsonl against the built index
	uv run mmu questions validate

serve:
	uv run uvicorn mmu.api.app:app --port 8099

eval:            ## offline deterministic run
	uv run mmu eval run --judge local --out reports/local.json

ablate:          ## the experiment that justifies the architecture (interleaved + paired)
# One process, question-major: every arm answers the same question back to back, so the
# differences are paired and `eval ablate` can print an exact sign test over them. This
# target used to be three sequential `eval run` calls differenced by `eval compare`; that
# version reported a 3x latency win for hybrid that was entirely generator variance
# between the runs (see README, "Generation (measured)"). Do not restore it.
	uv run mmu eval ablate --judge local --out-dir reports/ablate --verbose

headroom:        ## measure what a reranker could win, BEFORE running one (retrieval only)
# Cheap: loads BGE-M3 and the built index, embeds 30 queries, runs no generator and no
# judge. Reports recall@k against recall@depth, which is the reranker's ceiling. Run this
# first -- `coverage` is a set intersection over doc_id and cannot see a reranker at all,
# so the obvious metrics would have reported a byte-identical result either way.
	uv run python scripts/rerank_headroom.py

ablate-rerank:   ## the reranking experiment (3 arms x every question, interleaved + paired)
# hybrid-k8 (control) / hybrid-rerank-k8 / hybrid-rerank-k3, all over the same 32-candidate
# pool. Two reranked arms rather than one so that quality and prompt length are separated:
# a single "rerank and shrink" arm coming back faster and worse could not say which change
# did which. This downloads BAAI/bge-reranker-v2-m3 (~2.3GB) on first run and generates
# 3 answers per question on CPU -- budget hours, not minutes.
	uv run mmu eval ablate --rerank --judge local --out-dir reports/rerank --verbose

clean:
	rm -rf data/.index data/.cache reports/*.json
