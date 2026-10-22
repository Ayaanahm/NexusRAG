"""Heading-aware chunking for Markdown documentation.

Chunks never cross a heading, never split a fenced code block, and carry their
heading path ("Security > OAuth2 with JWT > Hash the password"). The heading path
is prepended to the text that gets indexed, because a chunk that says "Now
update the dependency" is unfindable without knowing which page and section it
belongs to.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .textproc import estimate_tokens

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*(?:\{\s*#([\w\-]+)\s*\})?\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.S)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
# mkdocs-material syntax: snippet includes `{* path *}` and `/// note | Title` blocks.
_INCLUDE_RE = re.compile(r"^\s*\{\*\s*(\S+).*\*\}\s*$")
_FENCE_LANG = {".py": "python", ".ts": "typescript", ".js": "javascript", ".sh": "bash", ".toml": "toml"}
_ADMONITION_OPEN_RE = re.compile(r"^///\s*(\w+)(?:\s*\|\s*(.*))?\s*$")
_ADMONITION_CLOSE_RE = re.compile(r"^///\s*$")


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    heading_path: list[str]
    text: str
    url: str | None = None
    ord: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def index_text(self) -> str:
        """What BM25 and the embedder see: location + content."""
        path = " > ".join(p for p in [self.title, *self.heading_path] if p)
        return f"{path}\n\n{self.text}" if path else self.text

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        return cls(**d)


IncludeResolver = Callable[[str], "str | None"]


def _clean_lines(markdown: str, include_resolver: IncludeResolver | None = None) -> list[str]:
    text = _FRONT_MATTER_RE.sub("", markdown)
    text = _HTML_COMMENT_RE.sub("", text)
    out = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if not in_fence:
            m = _INCLUDE_RE.match(line)
            if m:
                # Inline the referenced code so it is searchable, as it is on the rendered page.
                code = include_resolver(m.group(1)) if include_resolver else None
                if code:
                    lang = _FENCE_LANG.get(Path(m.group(1)).suffix, "")
                    out.extend([f"```{lang}", *code.rstrip("\n").splitlines(), "```"])
                continue
            if _ADMONITION_CLOSE_RE.match(line):
                continue
            m = _ADMONITION_OPEN_RE.match(line)
            if m:
                kind, title = m.group(1).capitalize(), (m.group(2) or "").strip()
                out.append(f"{kind}: {title}" if title else f"{kind}:")
                continue
        out.append(line)
    return out


def _blocks(lines: list[str]):
    """Yield ("heading", level, title, anchor) or ("block", text) items.
    Blocks are paragraphs, list runs or whole fenced code blocks."""
    buf: list[str] = []
    in_fence = False

    def flush():
        if buf and any(s.strip() for s in buf):
            yield ("block", "\n".join(buf).strip("\n"))
        buf.clear()

    for line in lines:
        if _FENCE_RE.match(line):
            if not in_fence:
                yield from flush()
            buf.append(line)
            in_fence = not in_fence
            if not in_fence:
                yield from flush()
            continue
        if in_fence:
            buf.append(line)
            continue
        m = _HEADING_RE.match(line)
        if m:
            yield from flush()
            title = re.sub(r"`([^`]*)`", r"\1", m.group(2)).strip()
            yield ("heading", len(m.group(1)), title, m.group(3))
            continue
        if not line.strip():
            yield from flush()
            continue
        buf.append(line)
    yield from flush()


def _split_oversized(block: str, max_tokens: int) -> list[str]:
    """A single block bigger than the budget (long code listing, giant table)
    is split on line boundaries rather than mid-line."""
    if estimate_tokens(block) <= max_tokens:
        return [block]
    parts, cur = [], []
    for line in block.splitlines():
        if cur and estimate_tokens("\n".join([*cur, line])) > max_tokens:
            parts.append("\n".join(cur))
            cur = []
        cur.append(line)
    if cur:
        parts.append("\n".join(cur))
    return parts


def chunk_markdown(
    markdown: str,
    doc_id: str,
    base_url: str | None = None,
    max_tokens: int = 400,
    overlap_blocks: int = 1,
    include_resolver: IncludeResolver | None = None,
) -> list[Chunk]:
    """Split one Markdown document into chunks of at most ``max_tokens``
    (estimated), carrying ``overlap_blocks`` trailing blocks into the next chunk
    of the same section so a sentence that refers back still has its referent."""
    title = ""
    stack: list[tuple[int, str, str | None]] = []  # (level, title, anchor)
    chunks: list[Chunk] = []
    section_blocks: list[str] = []

    def emit_section():
        if not section_blocks:
            return
        path = [t for _, t, _ in stack if t != title]
        anchor = next((a for _, _, a in reversed(stack) if a), None)
        url = f"{base_url}#{anchor}" if base_url and anchor and len(stack) > 1 else base_url
        pieces: list[str] = []
        for b in section_blocks:
            pieces.extend(_split_oversized(b, max_tokens))
        cur: list[str] = []
        for p in pieces:
            if cur and estimate_tokens("\n\n".join([*cur, p])) > max_tokens:
                _add("\n\n".join(cur), path, url)
                keep = cur[-overlap_blocks:] if overlap_blocks else []
                # Only overlap when the carried block leaves room for new content.
                cur = keep if estimate_tokens("\n\n".join([*keep, p])) <= max_tokens else []
            cur.append(p)
        if cur:
            _add("\n\n".join(cur), path, url)
        section_blocks.clear()

    def _add(text: str, path: list[str], url: str | None):
        ordinal = len(chunks)
        cid = hashlib.sha1(f"{doc_id}\x00{ordinal}\x00{text}".encode()).hexdigest()[:16]
        chunks.append(
            Chunk(id=cid, doc_id=doc_id, title=title, heading_path=list(path), text=text, url=url, ord=ordinal)
        )

    for item in _blocks(_clean_lines(markdown, include_resolver)):
        if item[0] == "heading":
            _, level, htitle, anchor = item
            emit_section()
            if level == 1 and not title:
                title = htitle
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, htitle, anchor))
        else:
            section_blocks.append(item[1])
    emit_section()
    return chunks


def chunk_plain(text: str, doc_id: str, title: str = "", url: str | None = None, max_tokens: int = 400) -> list[Chunk]:
    """Chunk plain text on paragraph boundaries (no heading structure)."""
    md = "\n\n".join(p for p in re.split(r"\n\s*\n", text) if p.strip())
    chunks = chunk_markdown(md, doc_id, base_url=url, max_tokens=max_tokens)
    for c in chunks:
        c.title = title
    return chunks
