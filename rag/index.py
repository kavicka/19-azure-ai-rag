"""Build the search indexes for one chunking strategy.

Sparse (BM25) is rebuilt in memory at start-up (a few seconds), so only chunks are stored for it.
Dense vectors go to Qdrant: embedded local mode by default (a folder in data/, no server needed),
or a real Qdrant server when QDRANT_URL is set (docker compose up -d qdrant).
The bge vectors are also saved to vectors_bge.npy, so rag/azure_index.py uploads *exactly* the same vectors to
Azure AI Search: any difference between the two stacks then comes from the search engine, not the embeddings.

Run:  uv run python -m rag.index --chunking section
      uv run python -m rag.index --chunking fixed
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from rag.chunking import CHUNKERS, Chunk
from rag.config import Settings, get_settings

BATCH = 256


def chunks_path(s: Settings, chunking: str) -> Path:
    return s.data_dir / "index" / chunking / "chunks.jsonl"


def collection_name(chunking: str) -> str:
    return f"chunks_{chunking}"


def qdrant_client(s: Settings) -> QdrantClient:
    if s.qdrant_url:
        key = s.qdrant_api_key.get_secret_value() if s.qdrant_api_key else None
        return QdrantClient(url=s.qdrant_url, api_key=key)
    return QdrantClient(path=str(s.data_dir / "qdrant"))


def load_docs(s: Settings) -> list[dict]:
    with (s.data_dir / "studies.jsonl").open() as f:
        return [json.loads(line) for line in f]


def load_chunks(s: Settings, chunking: str) -> list[dict]:
    with chunks_path(s, chunking).open() as f:
        return [json.loads(line) for line in f]


def build(chunking: str, s: Settings | None = None, sparse_only: bool = False) -> dict:
    from rag.embed import get_local_embedder  # heavy import, only needed here

    s = s or get_settings()
    docs = load_docs(s)
    chunks: list[Chunk] = [c for d in docs for c in CHUNKERS[chunking](d)]
    p = chunks_path(s, chunking)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    stats = {
        "chunking": chunking,
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "avg_words": round(sum(len(c.text.split()) for c in chunks) / len(chunks), 1),
    }
    if sparse_only:  # BM25 needs only the chunk file; useful where embedding models can't be downloaded
        (p.parent / "stats.json").write_text(json.dumps(stats, indent=1))
        print(json.dumps(stats, indent=1))
        return stats

    model = get_local_embedder(s)
    t0 = time.perf_counter()
    # both backends return unit vectors, so cosine = dot product (step 10)
    vecs = model.encode([c.text for c in chunks], batch_size=32)
    embed_s = time.perf_counter() - t0
    np.save(p.parent / "vectors_bge.npy", vecs.astype(np.float32))

    client = qdrant_client(s)
    name = collection_name(chunking)
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name, vectors_config=models.VectorParams(size=vecs.shape[1], distance=models.Distance.COSINE)
    )
    for i in range(0, len(chunks), BATCH):
        client.upsert(
            name,
            points=[
                models.PointStruct(id=j, vector=vecs[j].tolist(), payload={"chunk_id": chunks[j].chunk_id})
                for j in range(i, min(i + BATCH, len(chunks)))
            ],
        )
    client.close()
    stats |= {"embed_model": s.embed_model, "embed_backend": s.embed_backend, "embed_seconds": round(embed_s, 1)}
    (p.parent / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunking", choices=list(CHUNKERS), default=None, help="default: both")
    ap.add_argument("--sparse-only", action="store_true", help="write chunks for BM25 only, skip embeddings")
    args = ap.parse_args()
    for c in [args.chunking] if args.chunking else list(CHUNKERS):
        build(c, sparse_only=args.sparse_only)


if __name__ == "__main__":
    main()
