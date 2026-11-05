"""Azure code paths, tested against the *real* SDKs with fake HTTP transports: no Azure account, no network.

What this proves: the requests the openai and azure-search-documents SDKs put on the wire (URL, auth header,
JSON body) are the ones we intend, and our parsing of Azure-shaped responses (streamed chunks with an empty
first chunk and a usage-only last chunk, search results with @search.score) is right. What it cannot prove:
that Azure accepts them. That is what `make azure-check` does against the real service.
"""

import io
import json
import time

import httpx2
import numpy as np
import pytest
import requests
from azure.core.credentials import AccessToken
from azure.core.pipeline.transport import RequestsTransport

from rag.azure_auth import openai_client
from rag.azure_search import (
    FIELDS,
    AzureSearchRetriever,
    index_definition,
    keyword_query,
    to_documents,
)
from rag.config import Settings
from rag.embed import QUERY_PREFIX, AzureOpenAIEmbedder
from rag.llm import AzureOpenAILLM, Usage
from rag.pricing import cost_usd, load_prices, price_for
from rag.retrieve import POOL, RERANK_POOL

AOAI = "https://aoai-test.openai.azure.com"
SEARCH = "https://srch-test.search.windows.net"


class FakeCredential:
    """Stands in for DefaultAzureCredential: always the same token, records the scopes asked for."""

    def __init__(self):
        self.scopes: list[str] = []

    def get_token(self, *scopes, **kwargs):
        self.scopes += scopes
        return AccessToken("fake-entra-token", int(time.time()) + 3600)


def azure_settings(**kw) -> Settings:
    base = dict(
        _env_file=None,
        llm_provider="azure_openai",
        search_provider="azure_search",
        azure_openai_endpoint=AOAI,
        azure_search_endpoint=SEARCH,
        azure_chat_deployment="gpt-4o-mini",
    )
    return Settings(**(base | kw))


# --- Azure OpenAI over the v1 API --------------------------------------------------------------------------------


def sse(*chunks: dict) -> bytes:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode() + b"data: [DONE]\n\n"


def chunk(content=None, usage=None, choices=True) -> dict:
    c = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-4o-mini-2024-07-18"}
    c["choices"] = [{"index": 0, "delta": {"content": content}, "finish_reason": None}] if choices else []
    if usage:
        c["usage"] = usage
    return c


def mock_openai(handler):
    cred = FakeCredential()
    client = openai_client(AOAI, credential=cred, http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    return client, cred


def test_chat_streams_tokens_and_records_usage_keyless():
    seen = {}

    def handler(req: httpx2.Request) -> httpx2.Response:
        seen["url"], seen["headers"], seen["body"] = str(req.url), req.headers, json.loads(req.content)
        body = sse(
            chunk(choices=False),  # Azure sends prompt-filter results first, with no choices
            chunk("Phase 2"),
            chunk(" [NCT00000001]."),
            chunk(choices=False, usage={"prompt_tokens": 812, "completion_tokens": 9, "total_tokens": 821}),
        )
        return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client, cred = mock_openai(handler)
    llm = AzureOpenAILLM(azure_settings(), client=client)
    text = "".join(llm.stream([{"role": "user", "content": "Which phase?"}]))

    assert text == "Phase 2 [NCT00000001]."
    assert llm.last_usage == Usage(812, 9, "gpt-4o-mini-2024-07-18")
    assert seen["url"] == f"{AOAI}/openai/v1/chat/completions"  # v1 API: no api-version query parameter
    assert seen["headers"]["authorization"] == "Bearer fake-entra-token"
    assert "api-key" not in seen["headers"]
    assert cred.scopes == ["https://cognitiveservices.azure.com/.default"]
    b = seen["body"]
    assert b["model"] == "gpt-4o-mini"  # the deployment name
    assert b["stream"] is True and b["stream_options"] == {"include_usage": True}
    assert b["max_completion_tokens"] == 256 and "max_tokens" not in b
    assert b["temperature"] == 0.0


def test_reasoning_models_get_no_temperature():
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return httpx2.Response(200, content=sse(chunk("ok")), headers={"content-type": "text/event-stream"})

    client, _ = mock_openai(handler)
    llm = AzureOpenAILLM(azure_settings(llm_temperature=None), client=client)
    assert "".join(llm.stream([{"role": "user", "content": "x"}])) == "ok"
    assert "temperature" not in bodies[0]
    assert llm.last_usage is None  # no usage chunk → no made-up numbers


def test_embeddings_keep_order_count_tokens_and_shorten():
    seen = {}

    def handler(req):
        seen["url"], seen["body"] = str(req.url), json.loads(req.content)
        n = len(seen["body"]["input"])
        data = [{"object": "embedding", "index": i, "embedding": [float(i + 1), 0.0, 0.0]} for i in range(n)]
        data.reverse()  # the API does not promise order: we must sort by index
        body = {
            "object": "list",
            "data": data,
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 7, "total_tokens": 7},
        }
        return httpx2.Response(200, json=body)

    client, _ = mock_openai(handler)
    emb = AzureOpenAIEmbedder(azure_settings(embed_provider="azure_openai", azure_openai_embed_dimensions=512), client)
    v = emb.encode(["a", "b", "c"], batch_size=2)

    assert seen["url"] == f"{AOAI}/openai/v1/embeddings"
    assert seen["body"]["model"] == "text-embedding-3-small" and seen["body"]["dimensions"] == 512
    assert v.shape == (3, 3) and np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert np.allclose(v[:, 0], 1.0)  # each row is its own (normalized) vector, in input order
    assert emb.tokens == 14 and emb.model == "text-embedding-3-small"  # 2 requests × 7 tokens
    assert emb.query_prefix == ""  # OpenAI embeddings take raw text, unlike bge


# --- Azure AI Search ------------------------------------------------------------------------------------------------


class FakeSearchAdapter(requests.adapters.BaseAdapter):
    """Answers like the Azure AI Search REST API and records every request."""

    def __init__(self, docs: list[dict]):
        super().__init__()
        self.docs, self.requests = docs, []

    def send(self, request, **kwargs):
        from urllib3 import HTTPResponse

        body = json.loads(request.body) if request.body else None
        self.requests.append({"method": request.method, "url": request.url, "headers": request.headers, "body": body})
        if "/docs/$count" in request.url:
            content, ctype = str(len(self.docs)).encode(), "text/plain"
        else:
            top = body.get("top", 50)
            value = [{"@search.score": 10.0 - i, **d} for i, d in enumerate(self.docs[:top])]
            content, ctype = json.dumps({"value": value}).encode(), "application/json; odata.metadata=none"
        raw = HTTPResponse(
            body=io.BytesIO(content),
            headers={"Content-Type": ctype, "Content-Length": str(len(content))},
            status=200,
            preload_content=False,
        )
        return requests.adapters.HTTPAdapter().build_response(request, raw)

    def close(self):
        pass


DOCS = [
    {
        "chunk_id": f"NCT0000000{i}#design-0",
        "nct_id": f"NCT0000000{i}",
        "title": f"T{i}",
        "section": "Design",
        "text": f"text {i}",
    }
    for i in range(1, 6)
]


class FakeEmbedder:
    def __init__(self, dims=384, prefix=QUERY_PREFIX, name="BAAI/bge-small-en-v1.5"):
        self.dims, self.query_prefix, self.name, self.calls = dims, prefix, name, []

    def encode(self, texts, batch_size=32):
        self.calls += texts
        return np.ones((len(texts), self.dims), dtype=np.float32) / np.sqrt(self.dims)


class ReverseReranker:
    """Prefers the *last* candidate, so a test can see that reranking changed the order."""

    def __init__(self):
        self.seen = None

    def score(self, query, docs):
        self.seen = docs
        return np.arange(len(docs), dtype=np.float32)


@pytest.fixture
def search():
    from azure.search.documents import SearchClient

    adapter = FakeSearchAdapter(DOCS)
    session = requests.Session()
    session.mount("https://", adapter)
    client = SearchClient(SEARCH, "chunks-section", FakeCredential(), transport=RequestsTransport(session=session))

    def make(**kw):
        emb = kw.pop("embedder", FakeEmbedder())
        r = AzureSearchRetriever(azure_settings(**kw), client=client, embedder=emb, reranker=ReverseReranker())
        return r, adapter, emb

    return make


def last_search(adapter) -> dict:
    return [r for r in adapter.requests if r["method"] == "POST"][-1]


def test_keyword_mode_is_plain_bm25_over_text(search):
    r, adapter, emb = search()
    hits = r.search('What phase is the trial "Metformin-XR in Type 2"?', "bm25", k=3)
    req = last_search(adapter)
    assert req["url"].startswith(f"{SEARCH}/indexes('chunks-section')/docs/search.post.search")
    assert req["headers"]["Authorization"] == "Bearer fake-entra-token" and "api-key" not in req["headers"]
    b = req["body"]
    assert b["search"] == "What phase is the trial Metformin XR in Type 2"  # no phrase / NOT operators
    assert b["top"] == 3 and b["searchMode"] == "any" and "vectorQueries" not in b
    assert b["select"] == ",".join(FIELDS)  # vectors are never downloaded
    assert [h.chunk["nct_id"] for h in hits] == ["NCT00000001", "NCT00000002", "NCT00000003"]
    assert hits[0].scores == {"bm25": 10.0} and set(hits[0].chunk) == set(FIELDS)
    assert emb.calls == []  # keyword search needs no embedding call


def test_dense_mode_sends_one_vector_query_with_the_bge_prefix(search):
    r, adapter, emb = search()
    r.search("elderly patients", "dense", k=2)
    b = last_search(adapter)["body"]
    assert "search" not in b and b["top"] == 2
    (vq,) = b["vectorQueries"]
    assert vq["kind"] == "vector" and vq["fields"] == "vec_bge" and vq["k"] == 2 and len(vq["vector"]) == 384
    assert emb.calls == [QUERY_PREFIX + "elderly patients"]


def test_hybrid_lets_azure_fuse_text_and_vectors(search):
    r, adapter, _ = search()
    hits = r.search("metformin enrollment", "hybrid", k=5)
    b = last_search(adapter)["body"]
    assert b["search"] == "metformin enrollment" and b["top"] == 5
    assert b["vectorQueries"][0]["k"] == POOL  # the vector side contributes a wide pool, as the local RRF does
    assert set(hits[0].scores) == {"rrf"}


def test_rerank_reorders_azure_candidates_with_the_local_cross_encoder(search):
    r, adapter, _ = search()
    hits = r.search("metformin enrollment", "rerank", k=2)
    assert last_search(adapter)["body"]["top"] == RERANK_POOL
    assert r.reranker.seen == [d["text"] for d in DOCS]  # all candidates reached the reranker
    assert [h.chunk["nct_id"] for h in hits] == ["NCT00000005", "NCT00000004"]  # the reranker's order wins
    assert hits[0].scores == {"rrf": 6.0, "rerank": 4.0}


def test_te3_queries_use_their_own_field_and_no_prefix(search):
    emb = FakeEmbedder(dims=512, prefix="", name="azure_openai:text-embedding-3-small")
    r, adapter, _ = search(embed_provider="azure_openai", embedder=emb)
    r.search("elderly patients", "dense", k=2)
    vq = last_search(adapter)["body"]["vectorQueries"][0]
    assert vq["fields"] == "vec_te3" and len(vq["vector"]) == 512
    assert emb.calls == ["elderly patients"]
    assert r.name == "azure_search:chunks-section(azure_openai:text-embedding-3-small)"


def test_n_chunks_comes_from_the_service(search):
    r, adapter, _ = search()
    assert r.n_chunks == 5 and r.n_chunks == 5
    assert sum("/docs/$count" in x["url"] for x in adapter.requests) == 1  # cached after the first call


def test_keyword_query_strips_search_syntax():
    assert keyword_query('"a b" -c +d (e|f) g* h~ COVID-19 Phase 2/3') == "a b c d e f g h COVID 19 Phase 2 3"


def test_index_schema_matches_the_local_bm25_setup():
    d = index_definition("chunks-section", 384, 512).as_dict()
    f = {x["name"]: x for x in d["fields"]}
    assert [n for n, x in f.items() if x.get("searchable") and "dimensions" not in x] == ["text"]
    assert f["id"]["key"] is True
    assert f["vec_bge"]["dimensions"] == 384 and f["vec_te3"]["dimensions"] == 512
    assert f["vec_bge"]["stored"] is False and f["vec_bge"]["retrievable"] is False
    assert d["similarity"] == {"@odata.type": "#Microsoft.Azure.Search.BM25Similarity", "k1": 1.2, "b": 0.75}
    assert d["vectorSearch"]["algorithms"][0]["hnswParameters"]["metric"] == "cosine"
    assert "vec_te3" not in {x["name"] for x in index_definition("x", 384, None).as_dict()["fields"]}


def test_documents_get_safe_keys_and_both_vectors():
    chunks = [{f: f"{f}{i}" for f in FIELDS} | {"chunk_id": f"NCT1#design-{i}"} for i in range(3)]
    docs = to_documents(chunks, np.zeros((3, 4)), np.ones((3, 2)))
    assert [d["id"] for d in docs] == ["0", "1", "2"]  # '#' is not allowed in Azure keys
    assert docs[1]["chunk_id"] == "NCT1#design-1" and docs[1]["vec_te3"] == [1.0, 1.0]
    assert "vec_te3" not in to_documents(chunks, np.zeros((3, 4)), None)[0]


# --- pricing and settings -------------------------------------------------------------------------------------------


def test_prices_match_the_longest_model_name():
    table = {"usd_per_1m_tokens": {"gpt-4.1": {"input": 2, "output": 8}, "gpt-4.1-mini": {"input": 0.4, "output": 1.6}}}
    assert price_for("gpt-4.1-mini-2025-04-14", table) == {"input": 0.4, "output": 1.6}
    assert price_for("gpt-4.1-2025-04-14", table)["input"] == 2
    assert price_for("gpt-4.1-minimal", table) is None  # only "<name>-<version date>" counts
    assert price_for("gpt-4.1-mini-tts", table) is None  # a different model that shares the prefix
    assert cost_usd("unknown-model", 1000, 10, table) is None  # never a guessed price
    assert cost_usd("gpt-4.1-mini", 1_000_000, 500_000, table) == pytest.approx(0.4 + 0.8)


def test_pricing_file_is_complete_and_honest():
    p = load_prices()
    # an official Microsoft source only: the price page or the Retail Prices API behind it
    assert p["source"].startswith(("https://azure.microsoft.com/", "https://prices.azure.com/")) and p["checked"]
    assert isinstance(p["verified"], bool)
    assert all("input" in v for v in p["usd_per_1m_tokens"].values())
    for model in ("gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14", "text-embedding-3-small"):
        assert price_for(model) is not None, model


@pytest.mark.parametrize(
    ("kw", "msg"),
    [
        ({"llm_provider": "azure_openai"}, "AZURE_OPENAI_ENDPOINT"),
        ({"search_provider": "azure_search"}, "AZURE_SEARCH_ENDPOINT"),
        ({"embed_provider": "azure_openai", "azure_openai_endpoint": AOAI}, "needs SEARCH_PROVIDER=azure_search"),
    ],
)
def test_settings_fail_early_on_incomplete_azure_config(kw, msg):
    with pytest.raises(ValueError, match=msg):
        Settings(_env_file=None, **kw)


def test_default_settings_are_fully_local():
    s = Settings(_env_file=None)
    assert (s.search_provider, s.embed_provider, s.llm_provider) == ("local", "local", "ollama")
