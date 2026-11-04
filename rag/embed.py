"""Embedding and reranking behind one interface, with two interchangeable backends.

torch  sentence-transformers on PyTorch: what steps 14–15 used. Great for research and the Mac (MPS),
       but PyTorch alone makes a serving image several GB.
onnx   the same models exported to ONNX and run by onnxruntime through fastembed: no PyTorch at all, so the API
       image is a fraction of the size. Step 17 checks that retrieval quality survives the switch (it does:
       identical evidence@k on all 141 questions).
azure  an Azure OpenAI embedding deployment (text-embedding-3-small), called over HTTPS with Entra ID. It costs
       money per token, so it counts the tokens it used; the eval turns them into a cost.

Every embedder says which prefix its *queries* need: bge was trained with an instruction on the query side,
OpenAI embedding models take the raw text.
"""

from typing import Protocol

import numpy as np

from rag.config import Settings

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge: instruction on queries only
ONNX_RERANK = {"cross-encoder/ms-marco-MiniLM-L-6-v2": "Xenova/ms-marco-MiniLM-L-6-v2"}


class Embedder(Protocol):
    name: str
    query_prefix: str

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray: ...


class Reranker(Protocol):
    def score(self, query: str, docs: list[str]) -> np.ndarray: ...


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class TorchEmbedder:
    def __init__(self, s: Settings):
        from sentence_transformers import SentenceTransformer

        self.name, self.query_prefix = s.embed_model, QUERY_PREFIX
        self.m = SentenceTransformer(s.embed_model, device=s.device)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.m.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)


class TorchReranker:
    def __init__(self, s: Settings):
        from sentence_transformers import CrossEncoder

        self.m = CrossEncoder(s.rerank_model, device=s.device)

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        return np.asarray(self.m.predict([(query, d) for d in docs]))


class OnnxEmbedder:
    def __init__(self, s: Settings):
        from fastembed import TextEmbedding

        self.name, self.query_prefix = s.embed_model, QUERY_PREFIX
        self.m = TextEmbedding(s.embed_model)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return _normalize(np.array(list(self.m.embed(texts, batch_size=batch_size)), dtype=np.float32))


class OnnxReranker:
    def __init__(self, s: Settings):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.m = TextCrossEncoder(ONNX_RERANK.get(s.rerank_model, s.rerank_model))

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        return np.array(list(self.m.rerank(query, docs)), dtype=np.float32)


class AzureOpenAIEmbedder:
    """Azure OpenAI embeddings. Vectors come back unit-length already; normalized again so cosine = dot product."""

    def __init__(self, s: Settings, client=None):
        from rag.azure_auth import openai_client

        self.deployment, self.dimensions = s.azure_openai_embed_deployment, s.azure_openai_embed_dimensions
        self.name, self.query_prefix = f"azure_openai:{self.deployment}", ""
        self.client = client or openai_client(s.azure_openai_endpoint, max_retries=8, timeout=60)
        self.tokens, self.model = 0, ""  # running total, for the cost in results.json

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        # small batches: one request must fit into the deployment's tokens-per-minute quota on its own
        out: list[list[float]] = []
        extra = {"dimensions": self.dimensions} if self.dimensions else {}
        for i in range(0, len(texts), batch_size):
            resp = self.client.embeddings.create(model=self.deployment, input=texts[i : i + batch_size], **extra)
            out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
            self.tokens += resp.usage.prompt_tokens
            self.model = resp.model or self.model
        return _normalize(np.array(out, dtype=np.float32))


def get_embedder(s: Settings) -> Embedder:
    if s.embed_provider == "azure_openai":
        return AzureOpenAIEmbedder(s)
    return OnnxEmbedder(s) if s.embed_backend == "onnx" else TorchEmbedder(s)


def get_local_embedder(s: Settings) -> Embedder:
    """The bge model regardless of EMBED_PROVIDER: the Qdrant index and the Azure `vec_bge` field both use it."""
    return OnnxEmbedder(s) if s.embed_backend == "onnx" else TorchEmbedder(s)


def get_reranker(s: Settings) -> Reranker:
    return OnnxReranker(s) if s.embed_backend == "onnx" else TorchReranker(s)
