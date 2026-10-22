import hashlib

import numpy as np
import pytest

from nexusrag.chunking import chunk_markdown
from nexusrag.textproc import code_tokens


class HashEmbedder:
    """Deterministic bag-of-tokens embedder so tests need no model download."""

    name = "hash-64"
    dim = 64

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for t in code_tokens(text):
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text):
        return self._vec(text)


DOCS = {
    "cors.md": """# CORS

Browsers block requests from a frontend on another origin.

## Use CORSMiddleware { #use-corsmiddleware }

Add `CORSMiddleware` with `allow_origins` to let the frontend call the API.

```python
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"])
```
""",
    "errors.md": """# Handling Errors

## Use HTTPException { #use-httpexception }

Raise `HTTPException` with a `status_code` such as 404 when an item is not found.

## Add custom headers

You can pass `headers` to `HTTPException`.
""",
    "background.md": """# Background Tasks

Use `BackgroundTasks.add_task` to send an email after returning the response.
""",
}


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def sample_chunks():
    out = []
    for name, md in DOCS.items():
        out.extend(chunk_markdown(md, name, base_url=f"https://docs.example.com/{name[:-3]}/"))
    return out
