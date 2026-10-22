"""Ingest a folder of Markdown / text files.

    python -m nexusrag.ingest fastapi/docs/en/docs --base-url https://fastapi.tiangolo.com \\
        --url-style mkdocs --include-root fastapi/docs_src
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from .chunking import Chunk, chunk_markdown, chunk_plain
from .config import get_settings

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}


def doc_url(rel_path: str, base_url: str | None, style: str) -> str | None:
    if not base_url:
        return None
    base = base_url.rstrip("/")
    if style == "mkdocs":
        p = rel_path[: -len(".md")] if rel_path.endswith(".md") else rel_path
        if p == "index":
            return f"{base}/"
        if p.endswith("/index"):
            p = p[: -len("/index")]
        return f"{base}/{p}/"
    return f"{base}/{rel_path}"


def make_include_resolver(md_path: Path, include_root: Path | None):
    """Resolve mkdocs ``{* path *}`` snippet includes. Paths are tried relative to
    the Markdown file and to each parent of ``include_root``; the resolved file
    must live inside ``include_root`` so a document cannot pull in arbitrary files."""
    if include_root is None:
        return None
    allowed = include_root.resolve()

    def resolve(ref: str) -> str | None:
        bases = [md_path.parent, *list(md_path.parents)[:6]]
        for base in bases:
            cand = (base / ref).resolve()
            if cand.is_file() and cand.is_relative_to(allowed) and cand.stat().st_size < 200_000:
                return cand.read_text(encoding="utf-8", errors="replace")
        return None

    return resolve


def chunk_file(
    path: Path, doc_id: str, url: str | None, max_tokens: int = 400, include_root: Path | None = None
) -> list[Chunk]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in (".md", ".markdown"):
        return chunk_markdown(
            text,
            doc_id,
            base_url=url,
            max_tokens=max_tokens,
            include_resolver=make_include_resolver(path, include_root),
        )
    return chunk_plain(text, doc_id, title=path.stem, url=url, max_tokens=max_tokens)


def collect(
    root: Path,
    base_url: str | None = None,
    style: str = "plain",
    max_tokens: int = 400,
    include_root: Path | None = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES)
    for p in files:
        rel = p.relative_to(root).as_posix()
        if any(part.startswith(("_", ".")) for part in Path(rel).parts):
            continue  # drafts, partials, hidden files
        chunks.extend(chunk_file(p, rel, doc_url(rel, base_url, style), max_tokens, include_root))
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--base-url")
    ap.add_argument("--url-style", choices=["plain", "mkdocs"], default="plain")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument(
        "--include-root",
        type=Path,
        help="inline mkdocs {* file *} snippet includes found under this folder",
    )
    ap.add_argument("--reset", action="store_true", help="drop the existing index first")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from .embeddings import make_embedder
    from .index import CorpusIndex
    from .stores import make_store

    s = get_settings()
    if args.reset:
        for f in ("chunks.jsonl", "embedder.txt", "vectors.npz"):
            (s.data_dir / f).unlink(missing_ok=True)
    embedder = make_embedder(s.embedder)
    store = make_store(s, embedder.dim)
    if args.reset:
        store.reset()
    index = CorpusIndex(embedder, store, data_dir=s.data_dir)

    t0 = time.perf_counter()
    chunks = collect(args.root, args.base_url, args.url_style, args.max_tokens, args.include_root)
    t1 = time.perf_counter()
    index.add(chunks)
    t2 = time.perf_counter()
    stats = index.stats()
    print(
        f"ingested {stats['documents']} documents -> {stats['chunks']} chunks "
        f"(chunking {t1 - t0:.2f}s, embedding+indexing {t2 - t1:.2f}s) "
        f"into {stats['vector_store']} with {stats['embedder']}"
    )


if __name__ == "__main__":
    main()
