"""Choose the fusion method and weight on held-out data, then report them on unseen data.

SQuAD articles are split in two: questions about even-numbered articles tune,
questions about odd-numbered articles test (retrieval is always over the full
corpus). Two fusion methods are swept:

- ``rrf``:      weighted Reciprocal Rank Fusion, score = 1/(k+r_bm25) + w/(k+r_dense)
- ``weighted``: convex combination of min-max-normalised scores, (1-w)*bm25 + w*dense

The best (method, weight) on the tune half is then reported on the test half and
applied, untouched, to the FastAPI docs questions.

    python eval/tune_fusion.py --squad-file data/squad/dataset/dev-v1.1.json \\
        --docs data/fastapi/docs/en/docs --include-root data/fastapi/docs_src
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from run_retrieval import load_fastapi, load_squad, paired_bootstrap, per_query_metrics  # noqa: E402

from nexusrag.embeddings import make_embedder  # noqa: E402
from nexusrag.fusion import rrf, weighted  # noqa: E402
from nexusrag.index import CorpusIndex  # noqa: E402
from nexusrag.stores.memory import MemoryStore  # noqa: E402

GRID = {
    "rrf": [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0],
    "weighted": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
}
METRICS = ("hit@1", "hit@5", "hit@10", "mrr@10", "ndcg@10")


def candidate_lists(index: CorpusIndex, queries, n: int = 50):
    """Retrieve once per query; every fusion setting reuses these lists."""
    out = []
    for q in queries:
        kw = index.keyword_search(q.text, n)
        de = index.dense_search(index.embedder.embed_query(q.text), n)
        out.append((kw, de))
    return out


def fuse(kw, de, method: str, w: float) -> list[str]:
    if method == "bm25":
        return [c for c, _ in kw]
    if method == "dense":
        return [c for c, _ in de]
    if method == "rrf":
        return [c for c, _ in rrf([[c for c, _ in kw], [c for c, _ in de]], k=60, weights=[1.0, w])]
    return [c for c, _ in weighted(kw, de, dense_weight=w)]


def evaluate(index, queries, lists, level, method, w):
    rows = []
    for q, (kw, de) in zip(queries, lists):
        ranked, seen = [], set()
        for cid in fuse(kw, de, method, w):
            key = cid if level == "chunk" else index.get(cid).doc_id
            if key not in seen:
                seen.add(key)
                ranked.append(key)
        rows.append(per_query_metrics(ranked, q.gold))
    return rows


def summarise(rows):
    return {m: round(statistics.fmean(r[m] for r in rows), 4) for m in METRICS}


def compare(rows, base_rows):
    out = {}
    for m in ("mrr@10", "hit@1", "hit@5"):
        d, lo, hi = paired_bootstrap([r[m] for r in rows], [r[m] for r in base_rows])
        out[m] = {"diff": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    return out


def report(index, queries, lists, level, settings: dict[str, tuple[str, float]]):
    rows = {name: evaluate(index, queries, lists, level, m, w) for name, (m, w) in settings.items()}
    summary = {name: summarise(r) for name, r in rows.items()}
    vs_bm25 = {name: compare(r, rows["bm25"]) for name, r in rows.items() if name != "bm25"}
    vs_dense = {name: compare(r, rows["dense"]) for name, r in rows.items() if name not in ("bm25", "dense")}
    return summary, vs_bm25, vs_dense


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--squad-file", type=Path, required=True)
    ap.add_argument("--docs", type=Path, required=True)
    ap.add_argument("--include-root", type=Path)
    ap.add_argument("--embedder", default="fastembed:BAAI/bge-small-en-v1.5")
    ap.add_argument("--out", type=Path, default=ROOT.parent / "bench" / "results" / "fusion-tuning.json")
    args = ap.parse_args()
    embedder = make_embedder(args.embedder)

    # --- SQuAD: tune on even articles, test on odd ---------------------------
    chunks, queries, level = load_squad(args.squad_file)
    titles: list[str] = []
    for c in chunks:
        if c.doc_id not in titles:
            titles.append(c.doc_id)
    art_idx = {t: i for i, t in enumerate(titles)}
    chunk_art = {c.id: art_idx[c.doc_id] for c in chunks}
    tune_q = [q for q in queries if chunk_art[next(iter(q.gold))] % 2 == 0]
    test_q = [q for q in queries if chunk_art[next(iter(q.gold))] % 2 == 1]

    index = CorpusIndex(embedder, MemoryStore(embedder.dim))
    index.add(chunks)
    tune_lists = candidate_lists(index, tune_q)
    test_lists = candidate_lists(index, test_q)

    sweep: dict[str, dict] = {}
    for method, grid in GRID.items():
        for w in grid:
            s = summarise(evaluate(index, tune_q, tune_lists, level, method, w))
            sweep[f"{method}:{w}"] = s
            print(f"tune {method:<8} w={w:<4} mrr@10={s['mrr@10']:.4f} hit@5={s['hit@5']:.4f}", flush=True)
    best_rrf = max(GRID["rrf"], key=lambda w: sweep[f"rrf:{w}"]["mrr@10"])
    best_weighted = max(GRID["weighted"], key=lambda w: sweep[f"weighted:{w}"]["mrr@10"])
    chosen = max([("rrf", best_rrf), ("weighted", best_weighted)], key=lambda mw: sweep[f"{mw[0]}:{mw[1]}"]["mrr@10"])
    print(f"best rrf weight {best_rrf}, best weighted weight {best_weighted}, chosen {chosen}", flush=True)

    settings = {
        "bm25": ("bm25", 0.0),
        "dense": ("dense", 0.0),
        "rrf_equal": ("rrf", 1.0),
        f"rrf_w{best_rrf}": ("rrf", best_rrf),
        f"weighted_w{best_weighted}": ("weighted", best_weighted),
    }
    squad_summary, squad_vs_bm25, squad_vs_dense = report(index, test_q, test_lists, level, settings)
    print("squad test:", json.dumps(squad_summary), flush=True)

    # --- FastAPI docs: same settings, no retuning ----------------------------
    f_chunks, f_queries, f_level = load_fastapi(args.docs, args.include_root, ROOT / "eval" / "fastapi_qa.jsonl")
    f_index = CorpusIndex(embedder, MemoryStore(embedder.dim))
    f_index.add(f_chunks)
    f_lists = candidate_lists(f_index, f_queries)
    f_summary, f_vs_bm25, f_vs_dense = report(f_index, f_queries, f_lists, f_level, settings)
    print("fastapi:", json.dumps(f_summary), flush=True)

    result = {
        "embedder": embedder.name,
        "rrf_k": 60,
        "split": "SQuAD v1.1 dev; tune = questions on even-numbered articles, test = odd-numbered",
        "tune_questions": len(tune_q),
        "test_questions": len(test_q),
        "sweep_on_tune": sweep,
        "best_rrf_dense_weight": best_rrf,
        "best_weighted_dense_weight": best_weighted,
        "chosen": {"method": chosen[0], "dense_weight": chosen[1]},
        "squad_test": squad_summary,
        "squad_test_vs_bm25": squad_vs_bm25,
        "squad_test_vs_dense": squad_vs_dense,
        "fastapi_untuned_transfer": f_summary,
        "fastapi_vs_bm25": f_vs_bm25,
        "fastapi_vs_dense": f_vs_dense,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
