PY ?= python
DATA := data
FASTAPI_DOCS := $(DATA)/fastapi/docs/en/docs
FASTAPI_SRC := $(DATA)/fastapi/docs_src
SQUAD := $(DATA)/squad/dataset/dev-v1.1.json
EMBEDDER ?= fastembed:BAAI/bge-small-en-v1.5
RERANKER ?= fastembed:BAAI/bge-reranker-base

.PHONY: data install test lint dev ingest ingest-local bench bench-fastapi bench-squad bench-fusion up

data:
	./scripts/fetch_data.sh

install:
	cd backend && $(PY) -m pip install -e ".[dev]"
	cd frontend && npm ci

test:
	cd backend && $(PY) -m pytest -q
	cd frontend && npm test

lint:
	cd backend && ruff check . && ruff format --check .
	cd frontend && npm run typecheck

up: data
	docker compose up --build -d

# Index the FastAPI docs inside the running compose stack.
ingest: data
	docker compose exec api python -m nexusrag.ingest /corpus/fastapi/docs/en/docs \
		--base-url https://fastapi.tiangolo.com --url-style mkdocs --include-root /corpus/fastapi/docs_src
	docker compose restart api  # the API loads the index at startup

# Same, without Docker (in-memory vectors under backend/var).
ingest-local: data
	cd backend && $(PY) -m nexusrag.ingest ../$(FASTAPI_DOCS) \
		--base-url https://fastapi.tiangolo.com --url-style mkdocs --include-root ../$(FASTAPI_SRC)

dev:
	cd backend && uvicorn nexusrag.api:app --reload --port 8000 & cd frontend && npm run dev

bench: bench-fastapi bench-squad bench-fusion

bench-fastapi: data
	cd backend && $(PY) eval/run_retrieval.py fastapi --docs ../$(FASTAPI_DOCS) --include-root ../$(FASTAPI_SRC) \
		--embedder $(EMBEDDER) --reranker $(RERANKER) --tag bge

bench-squad: data
	cd backend && $(PY) eval/run_retrieval.py squad --squad-file ../$(SQUAD) --embedder $(EMBEDDER) --tag bge
	cd backend && $(PY) eval/run_retrieval.py squad --squad-file ../$(SQUAD) --embedder $(EMBEDDER) \
		--reranker $(RERANKER) --squad-split test --sample 400 --tag bge-rerank-test400

bench-fusion: data
	cd backend && $(PY) eval/tune_fusion.py --squad-file ../$(SQUAD) --docs ../$(FASTAPI_DOCS) \
		--include-root ../$(FASTAPI_SRC) --embedder $(EMBEDDER) --out ../bench/results/fusion-tuning-bge.json
