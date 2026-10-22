"""Retrieval benchmark: BM25 vs dense vs hybrid (vs re-ranked) on labelled questions.

    # SQuAD v1.1 dev: 10,570 crowd-written questions over 2,067 Wikipedia paragraphs
    python eval/run_retrieval.py squad --squad-file data/squad/dataset/dev-v1.1.json

    # FastAPI docs: 80 hand-labelled developer questions over the FastAPI documentation
    python eval/run_retrieval.py fastapi --docs data/fastapi/docs/en/docs --include-root data/fastapi/docs_src

    # add a re-ranker row (needs the model or API key)
    python eval/run_retrieval.py fastapi ... --reranker cross-encoder:BAAI/bge-reranker-base

Results go to bench/results/<dataset>[-<tag>].json and a Markdown table on stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nexusrag.bm25 import BM25Index  # noqa: E402
from nexusrag.chunking import Chunk  # noqa: E402
from nexusrag.embeddings import make_embedder  # noqa: E402
from nexusrag.index import CorpusIndex  # noqa: E402
from nexusrag.ingest import collect  # noqa: E402
from nexusrag.rerank import NoReranker  # noqa: E402
from nexusrag.retriever import HybridRetriever  # noqa: E402
from nexusrag.stores.memory import MemoryStore  # noqa: E402
from nexusrag.textproc import plain_tokens  # noqa: E402

KS = (1, 3, 5, 10, 20)


@dataclass
class Query:
    id: str
    text: str
    gold: set[str]
    type: str = ""


# -- datasets --------------------------------------------------------------------
def load_squad(path: Path) -> tuple[list[Chunk], list[Query], str]:
    data = json.loads(path.read_text())
    chunks, queries = [], []
    for art in data["data"]:
        title = art["title"].replace("_", " ")
        for p in art["paragraphs"]:
            cid = hashlib.sha1(p["context"].encode()).hexdigest()[:16]
            chunks.append(Chunk(id=cid, doc_id=title, title=title, heading_path=[], text=p["context"], ord=len(chunks)))
            for qa in p["qas"]:
                queries.append(Query(qa["id"], qa["question"], {cid}))
    return chunks, queries, "chunk"


def load_fastapi(docs: Path, include_root: Path | None, qa_path: Path) -> tuple[list[Chunk], list[Query], str]:
    chunks = collect(docs, "https://fastapi.tiangolo.com", "mkdocs", include_root=include_root)
    queries = []
    for line in qa_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            queries.append(Query(r["id"], r["q"], set(r["gold"]), r.get("type", "")))
    return chunks, queries, "doc"


# -- metrics ---------------------------------------------------------------------
def ranked_keys(hits, level: str) -> list[str]:
    out, seen = [], set()
    for h in hits:
        key = h.chunk.id if level == "chunk" else h.chunk.doc_id
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def per_query_metrics(ranked: list[str], gold: set[str]) -> dict[str, float]:
    m: dict[str, float] = {}
    first = next((i for i, k in enumerate(ranked, start=1) if k in gold), None)
    for k in KS:
        m[f"recall@{k}"] = len(gold & set(ranked[:k])) / len(gold) if gold else 0.0
        m[f"hit@{k}"] = 1.0 if first is not None and first <= k else 0.0
    m["mrr@10"] = 1.0 / first if first is not None and first <= 10 else 0.0
    dcg = sum(1.0 / np.log2(i + 1) for i, k in enumerate(ranked[:10], start=1) if k in gold)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(gold), 10) + 1))
    m["ndcg@10"] = dcg / idcg if idcg else 0.0
    return m


def paired_bootstrap(a: list[float], b: list[float], n: int = 2000, seed: int = 7) -> tuple[float, float, float]:
    """Mean of (a - b) with a 95% percentile bootstrap interval."""
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    means = []
    for _ in range(n):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        means.append(sum(sample) / len(sample))
    means.sort()
    return sum(diffs) / len(diffs), means[int(0.025 * n)], means[int(0.975 * n)]


# -- runner ----------------------------------------------------------------------
def run_config(retriever, queries, level, mode, limit_k=20, rerank=False):
    rows, lat = [], []
    for q in queries:
        res = retriever.search(q.text, k=limit_k, mode=mode, rerank=rerank)
        lat.append(res.timings_ms["total"])
        rows.append(per_query_metrics(ranked_keys(res.hits, level), q.gold))
    agg = {k: round(statistics.fmean(r[k] for r in rows), 4) for k in rows[0]}
    lat_sorted = sorted(lat)
    agg["latency_ms_p50"] = round(lat_sorted[len(lat) // 2], 3)
    agg["latency_ms_p95"] = round(lat_sorted[int(len(lat) * 0.95) - 1], 3)
    return agg, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["squad", "fastapi"])
    ap.add_argument("--squad-file", type=Path)
    ap.add_argument("--docs", type=Path)
    ap.add_argument("--include-root", type=Path)
    ap.add_argument("--qa", type=Path, default=ROOT / "eval" / "fastapi_qa.jsonl")
    ap.add_argument("--embedder", default="wordllama")
    ap.add_argument("--reranker", default="none", help="e.g. cross-encoder:BAAI/bge-reranker-base, cohere, claude")
    ap.add_argument("--limit", type=int, help="evaluate only the first N questions")
    ap.add_argument("--sample", type=int, help="evaluate a random sample of N questions (seeded)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument(
        "--squad-split",
        choices=["all", "tune", "test"],
        default="all",
        help="tune = questions on even-numbered articles, test = odd (the split eval/tune_fusion.py uses)",
    )
    ap.add_argument("--rerank-candidates", type=int, default=20)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, default=ROOT.parent / "bench" / "results")
    args = ap.parse_args()

    if args.dataset == "squad":
        chunks, queries, level = load_squad(args.squad_file)
        if args.squad_split != "all":
            titles = list(dict.fromkeys(c.doc_id for c in chunks))
            art = {c.id: titles.index(c.doc_id) for c in chunks}
            want = 0 if args.squad_split == "tune" else 1
            queries = [q for q in queries if art[next(iter(q.gold))] % 2 == want]
    else:
        chunks, queries, level = load_fastapi(args.docs, args.include_root, args.qa)
    if args.limit:
        queries = queries[: args.limit]
    if args.sample and args.sample < len(queries):
        queries = random.Random(args.seed).sample(queries, args.sample)

    embedder = make_embedder(args.embedder)
    t0 = time.perf_counter()
    index = CorpusIndex(embedder, MemoryStore(embedder.dim))
    index.add(chunks)
    build_s = time.perf_counter() - t0

    retriever = HybridRetriever(index, NoReranker(), rerank_candidates=args.rerank_candidates)
    code_bm25 = index.bm25
    plain_bm25 = BM25Index(tokenizer=plain_tokens).fit([c.index_text for c in index.chunks])

    configs: dict[str, dict] = {}
    per_query: dict[str, list[dict]] = {}

    def record(name, agg, rows):
        configs[name] = agg
        per_query[name] = rows
        print(
            f"  {name:<28} mrr@10={agg['mrr@10']:.3f} hit@5={agg['hit@5']:.3f} p50={agg['latency_ms_p50']}ms",
            flush=True,
        )

    print(f"{args.dataset}: {len(chunks)} chunks, {len(queries)} questions, index built in {build_s:.1f}s")
    index.bm25 = plain_bm25
    record("bm25 (plain tokenizer)", *run_config(retriever, queries, level, "bm25"))
    index.bm25 = code_bm25
    record("bm25 (code-aware tokenizer)", *run_config(retriever, queries, level, "bm25"))
    record(f"dense ({embedder.name})", *run_config(retriever, queries, level, "dense"))
    retriever.fusion = "rrf"
    record("hybrid (RRF, k=60)", *run_config(retriever, queries, level, "hybrid"))
    retriever.fusion, retriever.dense_weight = "weighted", 0.4
    record("hybrid (weighted, 0.4)", *run_config(retriever, queries, level, "hybrid"))

    # Sensitivity of the weighted alternative (reported, not used to pick the default).
    sweep = {}
    retriever.fusion = "weighted"
    for w in (0.2, 0.4, 0.6, 0.8):
        retriever.dense_weight = w
        agg, _ = run_config(retriever, queries, level, "hybrid")
        sweep[str(w)] = {"mrr@10": agg["mrr@10"], "hit@5": agg["hit@5"]}
    retriever.fusion, retriever.dense_weight = "weighted", 0.4  # the shipped default
    print("  weighted-fusion sweep (dense weight -> mrr@10):", {k: v["mrr@10"] for k, v in sweep.items()})

    if args.reranker != "none":
        from types import SimpleNamespace

        from nexusrag.rerank import make_reranker

        retriever.reranker = make_reranker(
            SimpleNamespace(
                reranker=args.reranker,
                cohere_api_key=__import__("os").environ.get("COHERE_API_KEY"),
                anthropic_api_key=__import__("os").environ.get("ANTHROPIC_API_KEY"),
                rerank_model=__import__("os").environ.get("NEXUS_RERANK_MODEL", "claude-haiku-5-5"),
            )
        )
        record(
            f"hybrid + rerank ({retriever.reranker.name})",
            *run_config(retriever, queries, level, "hybrid", rerank=True),
        )

    # Paired comparison: the shipped hybrid default vs the better single retriever.
    single = [n for n in configs if n.startswith(("bm25 (code", "dense"))]
    best_single = max(single, key=lambda n: configs[n]["mrr@10"])
    comparisons = {}
    for metric in ("mrr@10", "hit@5"):
        a = [r[metric] for r in per_query["hybrid (weighted, 0.4)"]]
        b = [r[metric] for r in per_query[best_single]]
        mean, lo, hi = paired_bootstrap(a, b)
        comparisons[metric] = {"vs": best_single, "diff": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]}

    by_type = {}
    if any(q.type for q in queries):
        for t in sorted({q.type for q in queries}):
            idx = [i for i, q in enumerate(queries) if q.type == t]
            by_type[t] = {
                name: {m: round(statistics.fmean(rows[i][m] for i in idx), 4) for m in ("mrr@10", "hit@5")}
                for name, rows in per_query.items()
            }

    result = {
        "dataset": args.dataset,
        "chunks": len(chunks),
        "questions": len(queries),
        "relevance_level": level,
        "embedder": embedder.name,
        "index_build_seconds": round(build_s, 2),
        "configs": configs,
        "weighted_fusion_sweep": sweep,
        "hybrid_vs_best_single": comparisons,
        "by_question_type": by_type,
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "processor_count": __import__("os").cpu_count(),
            "numpy": np.__version__,
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    name = args.dataset + (f"-{args.tag}" if args.tag else "")
    (args.out / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.dataset == "fastapi":  # small enough to commit and read by hand
        with (args.out / f"{name}-per-query.jsonl").open("w") as f:
            for i, q in enumerate(queries):
                row = {"id": q.id, "type": q.type, "q": q.text, "gold": sorted(q.gold)}
                row["mrr@10"] = {n: rows[i]["mrr@10"] for n, rows in per_query.items()}
                f.write(json.dumps(row) + "\n")

    print()
    print("| Retriever | Hit@1 | Hit@5 | Hit@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |")
    print("|---|---|---|---|---|---|---|---|")
    for n, a in configs.items():
        print(
            f"| {n} | {a['hit@1']:.3f} | {a['hit@5']:.3f} | {a['hit@10']:.3f} | {a['mrr@10']:.3f} | "
            f"{a['ndcg@10']:.3f} | {a['latency_ms_p50']} | {a['latency_ms_p95']} |"
        )
    c = comparisons["mrr@10"]
    print(f"\nhybrid - {c['vs']}: MRR@10 {c['diff']:+.4f} (95% CI {c['ci95'][0]:+.4f} to {c['ci95'][1]:+.4f})")


if __name__ == "__main__":
    main()
