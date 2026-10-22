from pathlib import Path

from nexusrag.chunking import chunk_markdown
from nexusrag.ingest import doc_url, make_include_resolver


def test_heading_path_title_and_anchor_url():
    md = "# Guide\n\nIntro text.\n\n## Install { #install }\n\nRun pip install.\n\n### Extras\n\nMore.\n"
    chunks = chunk_markdown(md, "guide.md", base_url="https://x.dev/guide/")
    assert [c.heading_path for c in chunks] == [[], ["Install"], ["Install", "Extras"]]
    assert chunks[0].title == "Guide"
    assert chunks[1].url == "https://x.dev/guide/#install"
    assert chunks[2].url == "https://x.dev/guide/#install"  # nearest heading with an anchor
    assert chunks[1].index_text.startswith("Guide > Install")


def test_code_fence_is_never_split_and_hash_inside_is_not_a_heading():
    code = "\n".join(f"# comment {i}\nx = {i}" for i in range(5))
    md = f"# Doc\n\n```python\n{code}\n```\n"
    chunks = chunk_markdown(md, "d.md")
    assert len(chunks) == 1
    assert chunks[0].text.startswith("```python") and chunks[0].text.endswith("```")


def test_long_sections_are_split_under_budget_with_overlap():
    paras = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(12))
    chunks = chunk_markdown(f"# Doc\n\n{paras}\n", "d.md", max_tokens=200)
    assert len(chunks) > 3
    assert all(len(c.text) // 4 <= 200 for c in chunks)
    # consecutive chunks share their boundary paragraph
    assert chunks[1].text.split("\n\n")[0] == chunks[0].text.split("\n\n")[-1]


def test_admonitions_and_front_matter_are_cleaned():
    md = "---\ntitle: x\n---\n# Doc\n\n/// tip | Pro tip\n\nUse it.\n\n///\n"
    text = chunk_markdown(md, "d.md")[0].text
    assert "Tip: Pro tip" in text and "///" not in text and "title: x" not in text


def test_chunk_ids_are_stable():
    md = "# Doc\n\nHello.\n"
    assert chunk_markdown(md, "d.md")[0].id == chunk_markdown(md, "d.md")[0].id


def test_includes_are_inlined_only_from_inside_the_include_root(tmp_path: Path):
    docs = tmp_path / "docs"
    src = tmp_path / "src"
    docs.mkdir()
    src.mkdir()
    (src / "example.py").write_text("def handler():\n    return 1\n")
    (tmp_path / "secret.txt").write_text("TOP SECRET")
    md_path = docs / "page.md"
    md = "# Page\n\n{* ../src/example.py hl[1] *}\n\n{* ../secret.txt *}\n"
    md_path.write_text(md)
    chunks = chunk_markdown(md, "page.md", include_resolver=make_include_resolver(md_path, src))
    text = "\n".join(c.text for c in chunks)
    assert "def handler" in text and "```python" in text
    assert "TOP SECRET" not in text


def test_mkdocs_urls():
    assert doc_url("index.md", "https://x.dev", "mkdocs") == "https://x.dev/"
    assert doc_url("tutorial/index.md", "https://x.dev", "mkdocs") == "https://x.dev/tutorial/"
    assert doc_url("tutorial/cors.md", "https://x.dev/", "mkdocs") == "https://x.dev/tutorial/cors/"
