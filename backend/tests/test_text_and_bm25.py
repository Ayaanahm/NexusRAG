from nexusrag.bm25 import BM25Index
from nexusrag.textproc import code_tokens, plain_tokens


def test_identifiers_keep_whole_token_and_parts():
    toks = code_tokens("raise HTTPException via app.include_router")
    assert "httpexception" in toks
    assert "http" in toks and "except" in toks  # stemmed part
    assert "app.include_router" in toks and "router" in toks


def test_snake_case_and_numbers():
    toks = code_tokens("status_code=201")
    assert {"status_code", "status", "code", "201"} <= set(toks)


def test_plain_tokenizer_does_not_split_identifiers():
    assert "http" not in plain_tokens("HTTPException")


def test_stopwords_removed():
    assert code_tokens("how do I use the") == []


def test_bm25_ranks_matching_document_first():
    docs = [
        "Add CORSMiddleware to allow other origins",
        "Raise HTTPException with status_code 404",
        "Background tasks run after the response",
    ]
    idx = BM25Index().fit(docs)
    assert idx.search("http exception 404", 3)[0][0] == 1
    assert idx.search("cors middleware", 3)[0][0] == 0


def test_bm25_unknown_terms_return_nothing():
    idx = BM25Index().fit(["alpha beta", "gamma delta"])
    assert idx.search("zzz", 5) == []


def test_bm25_rare_term_outweighs_common_term():
    docs = ["common common rare", "common common common", "common other words"]
    idx = BM25Index().fit(docs)
    assert idx.search("common rare", 3)[0][0] == 0
