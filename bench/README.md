# Benchmarks

Everything here was measured on a 2-vCPU cloud VM (x86_64, no GPU), Python 3.13, embeddings and re-ranking on CPU via ONNX Runtime. Raw output is in [`results/`](results); every number can be regenerated with the commands at the end.

## Datasets

| | Questions | Passages | Relevance | Who wrote the questions |
|---|---|---|---|---|
| **SQuAD v1.1 dev** | 10,570 | 2,067 Wikipedia paragraphs | the paragraph the question was written about | crowd workers (Rajpurkar et al., 2016) |
| **FastAPI docs** | 80 | 2,411 chunks of 155 pages | the page(s) that answer it | written for this project with AI assistance and spot-checked; see `backend/eval/fastapi_qa.jsonl` |

SQuAD is the main benchmark: the labels are independent of this project and there are enough questions for tight confidence intervals. Its questions were written while looking at the paragraph, so they share many words with it, which favours keyword search. The FastAPI set is the project's real domain (developer documentation), but with 80 questions most differences on it are within noise; it is reported as a sanity check, not as evidence.

Metrics: **Hit@k** (a relevant passage/page is in the top k), **MRR@10** (mean of 1/rank of the first relevant result, 0 if not in the top 10), **nDCG@10**. Confidence intervals are 95% paired bootstrap intervals over questions (2,000 resamples).

## 1. Equal-weight hybrid search did not beat keyword search

The common recipe is BM25 + vector search merged with Reciprocal Rank Fusion (k=60, equal weights). On all 10,570 SQuAD questions with `BAAI/bge-small-en-v1.5` embeddings ([`squad-bge.json`](results/squad-bge.json)):

| Retriever | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|
| BM25 | 0.792 | 0.939 | 0.963 | 0.856 |
| Vectors (bge-small) | 0.667 | 0.888 | 0.935 | 0.762 |
| Hybrid, equal-weight RRF | 0.781 | 0.950 | 0.976 | 0.854 |

RRF found more relevant passages somewhere in the top 10 but ranked them lower: MRR@10 was −0.001 against BM25 (95% CI −0.006 to +0.004). RRF throws away score magnitudes, so a confident BM25 match and a marginal one count the same.

With a much weaker embedder the effect is large. Using `wordllama` (a 16 MB static-embedding model that ships inside its pip wheel), equal-weight RRF scored **7.9 MRR points below BM25 alone** on held-out questions (−0.079, CI −0.087 to −0.072; [`fusion-tuning-wordllama.json`](results/fusion-tuning-wordllama.json)): averaging with a weak ranker drags a strong one down. The tuned weight for that model was 0, meaning "don't use the vectors", which is the honest answer for it.

## 2. Tuned fusion on held-out questions

`eval/tune_fusion.py` splits SQuAD by article: questions about even-numbered articles (5,397) are used to choose a fusion method and weight; questions about odd-numbered articles (5,173) are only used to report results. Two methods were swept:

- weighted RRF: `1/(k + rank_bm25) + w/(k + rank_vector)`, w ∈ {0.2 … 2.0}
- weighted score fusion: min-max normalise each list's scores, then `(1−w)·bm25 + w·vector`, w ∈ {0.1 … 0.8}

Best on the tuning half: weighted score fusion with w = 0.4. On the **5,173 held-out questions** ([`fusion-tuning-bge.json`](results/fusion-tuning-bge.json)):

| Retriever | Hit@1 | Hit@5 | Hit@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| BM25 | 0.818 | 0.944 | 0.965 | 0.873 | 0.896 |
| Vectors (bge-small) | 0.689 | 0.899 | 0.940 | 0.780 | 0.819 |
| Hybrid, equal-weight RRF | 0.803 | 0.954 | 0.977 | 0.868 | 0.895 |
| Hybrid, RRF w=0.2 (best RRF) | 0.820 | 0.955 | 0.975 | 0.880 | 0.903 |
| **Hybrid, score fusion w=0.4 (default)** | **0.846** | **0.964** | **0.985** | **0.899** | **0.920** |

Against BM25, the default improves MRR@10 by **+0.026 (CI +0.022 to +0.031)**, Hit@1 by +0.029 (CI +0.021 to +0.036), and cuts questions with no relevant passage in the top 10 from 3.5% to 1.5%. Against vectors alone, MRR@10 is +0.119 (CI +0.111 to +0.127).

Applied unchanged to the FastAPI questions (page-level relevance), the same setting gave MRR@10 0.849 vs 0.840 for vectors and 0.804 for BM25, with Hit@10 = 1.000. The differences against vectors are within noise at 80 questions (+0.010, CI −0.044 to +0.064).

The weight is embedder-specific. That is why it is a config value with a script to retune it, rather than a constant.

## 3. Re-ranking

Re-ranking is implemented (`BAAI/bge-reranker-base` via ONNX, Cohere Rerank, or Claude) and tested, but on this CPU-only machine it was only benchmarked on the 80 FastAPI questions, on top of equal-weight RRF ([`fastapi-bge.json`](results/fastapi-bge.json)):

| | Hit@1 | Hit@5 | Hit@10 | MRR@10 | p50 latency |
|---|---|---|---|---|---|
| Hybrid, equal-weight RRF | 0.738 | 0.975 | 0.975 | 0.844 | 24 ms |
| + bge-reranker-base on the top 20 | 0.738 | 0.950 | 0.988 | 0.839 | ~10 s |

On this set the re-ranker did not improve ranking (it helped the identifier questions, MRR@10 0.963 vs 0.938, and hurt the natural-language ones, 0.716 vs 0.750, 40 questions each) and it cost about 10 seconds per query on 2 CPU cores, measured while other work was running on the machine. A SQuAD re-ranking run on 400 held-out questions was started but stopped before it finished, so there is no large-sample re-ranking result yet. The command to run it is at the end of this file. Until it is measured, re-ranking is **off** by default (`NEXUS_RERANKER=none`).

## 4. Smaller findings

**Code-aware tokenization.** Splitting identifiers (`HTTPException` → `httpexception`, `http`, `exception`) changed nothing on SQuAD (prose; MRR@10 0.856 both ways) and helped on the FastAPI natural-language questions (MRR@10 0.714 vs 0.681 for a plain tokenizer). It slightly hurt the identifier-only FastAPI questions (0.892 vs 0.904). Each subset is 40 questions, so treat both directions as anecdotal.

**Embedding model matters more than fusion.** On FastAPI, swapping wordllama for bge-small moved vector-only MRR@10 from 0.682 to 0.840 ([`fastapi-wordllama.json`](results/fastapi-wordllama.json) vs [`fastapi-bge.json`](results/fastapi-bge.json)). No fusion setting made the weak model useful.

**Latency (this 2-vCPU VM, p50 per query).**

| Stage | p50 |
|---|---|
| BM25 over 2,411 chunks (scipy sparse) | 0.24 ms |
| Query embedding (bge-small, ONNX, CPU) + exact vector search | ~16–24 ms |
| Fusion | < 0.1 ms |
| Re-ranking 20 candidates (bge-reranker-base, CPU) | ~10 s |
| Indexing 2,411 chunks (embedding included) | ~10 minutes, single-process |

The re-ranker dominates latency on CPU; on a GPU, or with a hosted re-ranker (`NEXUS_RERANKER=cohere`), it is a different trade-off.

## What is not measured here

- **Answer quality.** `eval/run_answers.py` measures faithfulness (share of an answer's claims supported by its cited sources), citation validity and abstention, with Claude as the judge. It needs an Anthropic API key and was not run for this write-up, so no answer-quality numbers are claimed.
- **Qdrant HNSW and pgvector HNSW at scale.** Both stores are tested for correctness against exact search (`tests/test_stores_and_retriever.py`, pgvector in CI), but the corpora here are small enough that exact search was used for the benchmarks.

## Reproduce

```bash
make data                   # FastAPI docs @ ce6ee83, SQuAD dev @ eee5fdb
cd backend && pip install -e ".[dev]"
python eval/run_retrieval.py squad --squad-file ../data/squad/dataset/dev-v1.1.json \
    --embedder fastembed:BAAI/bge-small-en-v1.5 --tag bge
python eval/tune_fusion.py --squad-file ../data/squad/dataset/dev-v1.1.json \
    --docs ../data/fastapi/docs/en/docs --include-root ../data/fastapi/docs_src \
    --embedder fastembed:BAAI/bge-small-en-v1.5 --out ../bench/results/fusion-tuning-bge.json
python eval/run_retrieval.py squad --squad-file ../data/squad/dataset/dev-v1.1.json --squad-split test \
    --sample 400 --embedder fastembed:BAAI/bge-small-en-v1.5 \
    --reranker fastembed:BAAI/bge-reranker-base --tag bge-rerank-test400
python eval/run_retrieval.py fastapi --docs ../data/fastapi/docs/en/docs \
    --include-root ../data/fastapi/docs_src --embedder fastembed:BAAI/bge-small-en-v1.5 \
    --reranker fastembed:BAAI/bge-reranker-base --tag bge
```
