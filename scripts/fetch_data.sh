#!/usr/bin/env bash
# Fetch the demo corpus and the evaluation data at pinned commits, into ./data.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data

FASTAPI_COMMIT=ce6ee83e5411b98582e996e47edf7870418a68f4
SQUAD_COMMIT=eee5fdbf62f8613a7812b03419e6b29617b74fd1

if [ ! -d data/fastapi ]; then
  git clone --filter=blob:none --no-checkout https://github.com/fastapi/fastapi data/fastapi
  git -C data/fastapi sparse-checkout set docs/en/docs docs_src
  git -C data/fastapi checkout --quiet "$FASTAPI_COMMIT"
fi

if [ ! -f data/squad/dataset/dev-v1.1.json ]; then
  git clone --filter=blob:none --no-checkout https://github.com/rajpurkar/SQuAD-explorer data/squad
  git -C data/squad sparse-checkout set --no-cone dataset/dev-v1.1.json
  git -C data/squad checkout --quiet "$SQUAD_COMMIT"
fi

echo "FastAPI docs: $(find data/fastapi/docs/en/docs -name '*.md' | wc -l) pages; SQuAD dev: $(du -h data/squad/dataset/dev-v1.1.json | cut -f1)"
