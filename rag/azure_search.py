"""The same four retrieval modes on Azure AI Search, so the eval can put them next to the local stack.

local (rag/retrieve.py)                     Azure AI Search (this file)
bm25    own 40-line BM25, k1=1.2, b=0.75    full-text search, BM25 with the same k1 and b, standard.lucene analyzer
dense   bge vectors in Qdrant (HNSW)        the same bge vectors in a vector field (HNSW), or text-embedding-3-small
hybrid  own RRF, k=60                       Azure's built-in RRF over the text and the vector query
rerank  hybrid top 20 → local cross-encoder Azure hybrid top 20 → the same local cross-encoder

Why rerank keeps a local cross-encoder: Azure's own reranker (semantic ranker) is not available on the Free tier,
and with the same reranker on both stacks the comparison isolates what Azure's first stage finds.

One index holds both embeddings (`vec_bge`, `vec_te3`). EMBED_PROVIDER picks which vector field a query uses, so
"bge vs text-embedding-3-small" is a config switch on the same documents.
"""

import re
import time

import numpy as np

from rag.config import Settings
from rag.retrieve import POOL, RERANK_POOL, Hit, Mode

FIELDS = ["chunk_id", "nct_id", "title", "section", "text"]  # what a Hit needs; vectors are never returned
VECTOR_FIELD = {"local": "vec_bge", "azure_openai": "vec_te3"}
ANALYZER = "standard.lucene"  # Azure's default: lower-cases and splits words, no stemming (like our tokenizer)
BM25_K1, BM25_B = 1.2, 0.75  # the same as rag/retrieve.py, set explicitly so the comparison is like for like
_OPERATORS = re.compile(r"[^0-9A-Za-z]+")


def keyword_query(q: str) -> str:
    """Plain words only. Azure's simple query syntax reads `"..."` as a phrase, `-x` as NOT x, `*` as a prefix.

    The eval questions quote trial titles and contain hyphens ("Phase 2/3", "COVID-19"), so without this the
    keyword mode would silently run phrase and exclusion queries instead of the bag-of-words BM25 it is
    compared with.
    """
    return " ".join(_OPERATORS.sub(" ", q).split())


def index_definition(name: str, bge_dims: int, te3_dims: int | None):
    """The index schema. Built from SDK models, so a typo in a property name fails here and not on Azure."""
    from azure.search.documents.indexes.models import (
        BM25SimilarityAlgorithm,
        HnswAlgorithmConfiguration,
        HnswParameters,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        VectorSearch,
        VectorSearchProfile,
    )

    def vector_field(field: str, dims: int) -> SearchField:
        # retrievable/stored = False: the original vectors are kept only inside the HNSW graph, which roughly
        # halves the storage, and the Free tier has very little of it
        return SearchField(
            name=field,
            type=SearchFieldDataType.Collection(SearchFieldDataType.SINGLE),
            searchable=True,
            retrievable=False,
            stored=False,
            vector_search_dimensions=dims,
            vector_search_profile_name="hnsw-cosine",
        )

    def plain(name: str, **kw) -> SearchField:
        # string fields are *searchable by default* in the REST API: without searchable=False the keyword query
        # would also match title / nct_id / section, and Azure's BM25 would score a different text than ours
        return SearchField(name=name, type=SearchFieldDataType.STRING, searchable=False, **kw)

    fields = [
        plain("id", key=True, filterable=True),
        plain("chunk_id", filterable=True),
        plain("nct_id", filterable=True),
        plain("title"),
        plain("section", filterable=True),
        # the chunk text already starts with "[NCT id] title | section", exactly what the local BM25 indexes
        SearchField(name="text", type=SearchFieldDataType.STRING, searchable=True, analyzer_name=ANALYZER),
        vector_field("vec_bge", bge_dims),
    ]
    if te3_dims:
        fields.append(vector_field("vec_te3", te3_dims))
    return SearchIndex(
        name=name,
        fields=fields,
        similarity=BM25SimilarityAlgorithm(k1=BM25_K1, b=BM25_B),
        vector_search=VectorSearch(
            # Qdrant's defaults are m=16, ef_construct=100; Azure's are m=4, efConstruction=400, efSearch=500
            algorithms=[HnswAlgorithmConfiguration(name="hnsw", parameters=HnswParameters(metric="cosine"))],
            profiles=[VectorSearchProfile(name="hnsw-cosine", algorithm_configuration_name="hnsw")],
        ),
    )


def to_documents(chunks: list[dict], vec_bge: np.ndarray, vec_te3: np.ndarray | None) -> list[dict]:
    """Chunks → Azure documents. The key is the row number: chunk ids contain '#', which keys may not."""
    docs = []
    for i, c in enumerate(chunks):
        d = {"id": str(i), **{f: c[f] for f in FIELDS}, "vec_bge": vec_bge[i].tolist()}
        if vec_te3 is not None:
            d["vec_te3"] = vec_te3[i].tolist()
        docs.append(d)
    return docs


def search_client(s: Settings, credential=None):
    from azure.search.documents import SearchClient

    from rag.azure_auth import get_credential

    # azure-core retries 429/503 with the Retry-After header: the Free tier throttles bursts of queries
    return SearchClient(s.azure_search_endpoint, s.azure_search_index, credential or get_credential())


def index_client(s: Settings, credential=None):
    from azure.search.documents.indexes import SearchIndexClient

    from rag.azure_auth import get_credential

    return SearchIndexClient(s.azure_search_endpoint, credential or get_credential())


class AzureSearchRetriever:
    """Same public API as rag.retrieve.Retriever: search(), timed_search(), n_chunks, name, close()."""

    def __init__(self, s: Settings, client=None, embedder=None, reranker=None):
        from rag.embed import get_embedder, get_reranker

        self.s, self.chunking = s, s.chunking
        self.client = client or search_client(s)
        self.embedder = embedder or get_embedder(s)
        self.reranker = reranker or get_reranker(s)
        self.vector_field = VECTOR_FIELD[s.embed_provider]
        self.name = f"azure_search:{s.azure_search_index}({self.embedder.name})"
        self._n: int | None = None

    @property
    def n_chunks(self) -> int:
        if self._n is None:
            self._n = self.client.get_document_count()
        return self._n

    def close(self) -> None:
        self.client.close()

    def _vector_query(self, q: str, k: int):
        from azure.search.documents.models import VectorizedQuery

        qv = self.embedder.encode([self.embedder.query_prefix + q])[0]
        return VectorizedQuery(vector=qv.tolist(), k_nearest_neighbors=k, fields=self.vector_field)

    def _run(self, top: int, text: str | None = None, vector=None) -> list[tuple[dict, float]]:
        res = self.client.search(
            search_text=text,
            vector_queries=[vector] if vector else None,
            select=FIELDS,
            top=top,
            search_mode="any",  # OR over the words, scored by BM25: the same as the local keyword retriever
        )
        return [({f: r[f] for f in FIELDS}, float(r["@search.score"])) for r in res]

    def search(self, q: str, mode: Mode = "rerank", k: int = 5) -> list[Hit]:
        if mode == "bm25":
            return [Hit(c, sc, {"bm25": sc}) for c, sc in self._run(k, text=keyword_query(q))]
        if mode == "dense":
            return [Hit(c, sc, {"dense": sc}) for c, sc in self._run(k, vector=self._vector_query(q, k))]

        # hybrid: Azure fuses the text and the vector ranking itself with RRF; the score it returns is the RRF sum
        top = k if mode == "hybrid" else RERANK_POOL
        fused = self._run(top, text=keyword_query(q), vector=self._vector_query(q, POOL))
        if mode == "hybrid":
            return [Hit(c, sc, {"rrf": sc}) for c, sc in fused]

        ce = self.reranker.score(q, [c["text"] for c, _ in fused])
        best = np.argsort(-ce)[:k]
        return [Hit(fused[j][0], float(ce[j]), {"rrf": fused[j][1], "rerank": float(ce[j])}) for j in best]

    def timed_search(self, q: str, mode: Mode, k: int = 5) -> tuple[list[Hit], float]:
        t0 = time.perf_counter()
        hits = self.search(q, mode, k)
        return hits, (time.perf_counter() - t0) * 1000
