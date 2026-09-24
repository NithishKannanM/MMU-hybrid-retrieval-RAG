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

ablate:          ## the experiment that justifies the architecture
	uv run mmu eval run --channels dense  --judge local --out reports/dense-only.json
	uv run mmu eval run --channels sparse --judge local --out reports/sparse-only.json
	uv run mmu eval run --judge local --out reports/hybrid.json
	uv run mmu eval compare reports/hybrid.json reports/dense-only.json reports/sparse-only.json

clean:
	rm -rf data/.index data/.cache reports/*.json
