"""Four retrieval modes, each one step more complex, so the eval can show what every step buys.

bm25    exact word overlap. Great for names, codes and numbers ("NCT01234567", "metformin"), blind to synonyms.
dense   embedding similarity. Understands "elderly" ≈ "aged 65 and over", but fuzzy on exact names.
hybrid  both lists merged with Reciprocal Rank Fusion: score = Σ 1 / (60 + rank). RRF uses only ranks, so the
        incomparable scales of BM25 scores and cosine similarities never have to be calibrated against each other.
rerank  hybrid gives 20 candidates, a cross-encoder reads (question, chunk) *together* and re-orders them.
        Slower per pair than a bi-encoder, so it only sees a short list: that's the classic two-stage design.

The same four modes exist on Azure AI Search (rag/azure_search.py). `get_retriever()` picks the stack from
SEARCH_PROVIDER, and both return `Hit`s, so the API, the UI and the eval do not know which one is running.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from rag.config import Settings, get_settings
from rag.index import collection_name, load_chunks, qdrant_client

Mode = Literal["bm25", "dense", "hybrid", "rerank"]
MODES: tuple[Mode, ...] = ("bm25", "dense", "hybrid", "rerank")
RRF_K = 60  # the constant from the original RRF paper; dampens the gap between rank 1 and rank 2
POOL = 50  # candidates each retriever contributes to the fusion
RERANK_POOL = 20

_STOP = set(
    "a an the of in on for to and or is are was were be by with at as from that this what which who how".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


class BM25:
    """Okapi BM25 with the Lucene idf, log(1 + (N - n + 0.5) / (n + 0.5)), which never goes negative.

    Written out instead of using rank_bm25: its classic idf turns negative for words in over half of the
    documents, and 40 lines make the formula visible: idf × tf·(k1+1) / (tf + k1·(1 − b + b·len/avglen)).
    k1 caps how much repeating a word helps; b punishes long chunks for containing everything.
    """

    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b, self.n = k1, b, len(docs)
        lens = np.array([len(d) for d in docs], dtype=np.float32)
        self.norm = k1 * (1 - b + b * lens / lens.mean())  # per-document length penalty, precomputed
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        tmp: dict[str, dict[int, int]] = {}
        for i, d in enumerate(docs):
            for t in d:
                tmp.setdefault(t, {}).setdefault(i, 0)
                tmp[t][i] += 1
        for t, m in tmp.items():
            self.postings[t] = (np.fromiter(m.keys(), dtype=np.int64), np.fromiter(m.values(), dtype=np.float32))

    def idf(self, term: str) -> float:
        n = len(self.postings[term][0]) if term in self.postings else 0
        return float(np.log(1 + (self.n - n + 0.5) / (n + 0.5)))

    def get_scores(self, query: list[str]) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        for t in query:
            if t in self.postings:
                idx, tf = self.postings[t]
                scores[idx] += self.idf(t) * tf * (self.k1 + 1) / (tf + self.norm[idx])
        return scores


@dataclass
class Hit:
    chunk: dict
    score: float
    scores: dict[str, float] = field(default_factory=dict)  # per-stage scores, shown in the UI

    def to_dict(self) -> dict:
        return {**self.chunk, "score": round(self.score, 4), "scores": {k: round(v, 4) for k, v in self.scores.items()}}


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion of several ranked lists of chunk indices."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


class Retriever:
    def __init__(self, chunking: str | None = None, s: Settings | None = None, dense: bool = True):
        self.s = s or get_settings()
        self.chunking = chunking or self.s.chunking
        self.chunks = load_chunks(self.s, self.chunking)
        self.bm25_index = BM25([tokenize(c["text"]) for c in self.chunks])
        self.embedder = self.reranker = self.qdrant = None
        self.name = "local:bm25"
        if dense:
            from rag.embed import get_local_embedder, get_reranker

            self.embedder = get_local_embedder(self.s)
            self.reranker = get_reranker(self.s)
            self.qdrant = qdrant_client(self.s)
            self.name = f"local:bm25+qdrant({self.embedder.name})"

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    def close(self) -> None:
        if self.qdrant:
            self.qdrant.close()

    # --- single retrievers: return (indices, scores) best first ---
    def _bm25(self, q: str, k: int) -> tuple[list[int], list[float]]:
        scores = self.bm25_index.get_scores(tokenize(q))
        top = np.argsort(-scores)[:k]
        return top.tolist(), scores[top].tolist()

    def _dense(self, q: str, k: int) -> tuple[list[int], list[float]]:
        # bge needs an instruction on the query side, OpenAI embeddings take the raw text (Embedder.query_prefix)
        qv = self.embedder.encode([self.embedder.query_prefix + q])[0]
        res = self.qdrant.query_points(collection_name(self.chunking), query=qv.tolist(), limit=k).points
        return [p.id for p in res], [p.score for p in res]

    # --- public API ---
    def search(self, q: str, mode: Mode = "rerank", k: int = 5) -> list[Hit]:
        if mode == "bm25":
            idx, sc = self._bm25(q, k)
            return [Hit(self.chunks[i], s, {"bm25": s}) for i, s in zip(idx, sc, strict=True)]
        if mode == "dense":
            idx, sc = self._dense(q, k)
            return [Hit(self.chunks[i], s, {"dense": s}) for i, s in zip(idx, sc, strict=True)]

        b_idx, b_sc = self._bm25(q, POOL)
        d_idx, d_sc = self._dense(q, POOL)
        fused = rrf([b_idx, d_idx])
        per = {i: {"bm25": s} for i, s in zip(b_idx, b_sc, strict=True)}
        for i, s in zip(d_idx, d_sc, strict=True):
            per.setdefault(i, {})["dense"] = s
        order = sorted(fused, key=fused.get, reverse=True)
        if mode == "hybrid":
            return [Hit(self.chunks[i], fused[i], {**per[i], "rrf": fused[i]}) for i in order[:k]]

        cand = order[:RERANK_POOL]
        ce = self.reranker.score(q, [self.chunks[i]["text"] for i in cand])
        best = np.argsort(-ce)[:k]
        return [
            Hit(self.chunks[cand[j]], float(ce[j]), {**per[cand[j]], "rrf": fused[cand[j]], "rerank": float(ce[j])})
            for j in best
        ]

    def timed_search(self, q: str, mode: Mode, k: int = 5) -> tuple[list[Hit], float]:
        t0 = time.perf_counter()
        hits = self.search(q, mode, k)
        return hits, (time.perf_counter() - t0) * 1000


def get_retriever(chunking: str | None = None, s: Settings | None = None, dense: bool = True):
    """The retrieval stack chosen by SEARCH_PROVIDER: own BM25 + Qdrant, or Azure AI Search."""
    s = s or get_settings()
    if s.search_provider == "azure_search":
        from rag.azure_search import AzureSearchRetriever

        return AzureSearchRetriever(s)
    return Retriever(chunking, s, dense=dense)
