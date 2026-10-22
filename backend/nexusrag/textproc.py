"""Tokenization for keyword search over technical text.

Developer docs mix prose with identifiers: ``HTTPException``, ``include_router``,
``app.add_middleware``, ``status_code=201``. A plain word tokenizer turns
``HTTPException`` into one token that never matches a query for "http exception",
and drops the ``201`` that a user pastes from an error. ``code_tokens`` keeps the
whole identifier *and* its parts, so both phrasings match.
"""

from __future__ import annotations

import re
from functools import lru_cache

import snowballstemmer

_STEMMER = snowballstemmer.stemmer("english")

# Words that carry no signal in queries like "how do I ...".
STOPWORDS = frozenset(
    """
    a an and are as at be but by can do does for from how i if in into is it its me
    my of on or so that the their then there these this to use using was what when
    where which who why will with you your we our should would could get
    """.split()
)

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*|\d+")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|[A-Z]+|\d+")


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    return _STEMMER.stemWord(word)


def _split_identifier(token: str) -> list[str]:
    """``app.include_router`` -> ``app``, ``include``, ``router``;
    ``HTTPException`` -> ``http``, ``exception``."""
    parts: list[str] = []
    for dotted in token.split("."):
        for snake in dotted.split("_"):
            if snake:
                parts.extend(p.lower() for p in _CAMEL_RE.findall(snake))
    return parts


def code_tokens(text: str) -> list[str]:
    """Tokens for BM25: stemmed words, plus whole identifiers and their parts."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        lower = raw.lower()
        is_identifier = "_" in raw or "." in raw or (raw[1:] != raw[1:].lower() and not raw.isupper())
        if is_identifier:
            out.append(lower)  # keep the exact identifier as its own token
            out.extend(_stem(p) for p in _split_identifier(raw) if p not in STOPWORDS)
        elif lower not in STOPWORDS:
            out.append(_stem(lower))
    return out


def plain_tokens(text: str) -> list[str]:
    """Baseline tokenizer (lowercase words, stopwords removed, stemmed). Used in the
    benchmark to measure what the identifier handling adds."""
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def estimate_tokens(text: str) -> int:
    """Cheap token estimate for context budgeting (~4 characters per token for
    English and code). Deliberately conservative: budgets are a ceiling."""
    return max(1, (len(text) + 3) // 4)
