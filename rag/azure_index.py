"""Upload the chunks to Azure AI Search with two embeddings each: the local bge vectors and text-embedding-3-small.

Needs `python -m rag.index --chunking section` first: it writes the chunks and vectors_bge.npy, so Azure gets
exactly the vectors the local Qdrant has. text-embedding-3-small vectors cost money, so they are cached in
data/index/<chunking>/vectors_te3_<dims>.npy together with the token count, and a re-run does not pay twice.

Run:  uv run python -m rag.azure_index            (index + both embeddings)
      uv run python -m rag.azure_index --no-te3   (bge only, e.g. while there is no embedding quota yet)
"""

import argparse
import json
import time
from datetime import UTC, datetime

import numpy as np

from rag.azure_search import index_client, index_definition, search_client, to_documents
from rag.config import Settings, get_settings
from rag.index import chunks_path, load_chunks
from rag.pricing import cost_usd, load_prices
from rag.results import merge

BATCH = 100  # documents per upload request; with two 1536-dim vectors that is ~3 MB of JSON, under the 16 MB cap


def te3_vectors(s: Settings, chunks: list[dict]) -> tuple[np.ndarray, dict]:
    from rag.embed import AzureOpenAIEmbedder

    if not s.azure_openai_endpoint:
        raise SystemExit("AZURE_OPENAI_ENDPOINT is not set: run `make env-from-tf`, or pass --no-te3")
    folder = chunks_path(s, s.chunking).parent
    dims = s.azure_openai_embed_dimensions or "native"
    cache, meta_path = folder / f"vectors_te3_{dims}.npy", folder / f"vectors_te3_{dims}.json"
    if cache.exists() and meta_path.exists():
        print(f"using cached {cache.name}")
        return np.load(cache), json.loads(meta_path.read_text())

    emb = AzureOpenAIEmbedder(s)
    t0 = time.perf_counter()
    vecs = emb.encode([c["text"] for c in chunks])
    meta = {
        "deployment": s.azure_openai_embed_deployment,
        "model": emb.model,
        "dimensions": int(vecs.shape[1]),
        "tokens": emb.tokens,
        "cost_usd": cost_usd(emb.model, emb.tokens),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    np.save(cache, vecs)
    meta_path.write_text(json.dumps(meta, indent=1))
    return vecs, meta


def wait_for_count(client, n: int, timeout_s: float = 120) -> int:
    """Uploads are indexed asynchronously (near real time): wait until the service reports every document."""
    t0 = time.perf_counter()
    while (count := client.get_document_count()) < n and time.perf_counter() - t0 < timeout_s:
        time.sleep(3)
    return count


def build(s: Settings, with_te3: bool = True) -> dict:
    if not s.azure_search_endpoint:
        raise SystemExit("AZURE_SEARCH_ENDPOINT is not set: run `make env-from-tf` after `make infra-apply`")
    chunks = load_chunks(s, s.chunking)
    bge_path = chunks_path(s, s.chunking).parent / "vectors_bge.npy"
    if not bge_path.exists():
        raise SystemExit(f"{bge_path} is missing: run `uv run python -m rag.index --chunking {s.chunking}` first")
    bge = np.load(bge_path)
    assert len(bge) == len(chunks), "vectors_bge.npy and chunks.jsonl are out of sync: rebuild the local index"

    te3, te3_meta = te3_vectors(s, chunks) if with_te3 else (None, None)
    idx = index_client(s)
    name = s.azure_search_index
    if name in set(idx.list_index_names()):
        idx.delete_index(name)  # schema changes (e.g. other dimensions) need a fresh index
    idx.create_index(index_definition(name, bge.shape[1], te3.shape[1] if te3 is not None else None))

    client = search_client(s)
    docs = to_documents(chunks, bge, te3)
    t0 = time.perf_counter()
    for i in range(0, len(docs), BATCH):
        failed = [r.key for r in client.upload_documents(docs[i : i + BATCH]) if not r.succeeded]
        if failed:
            raise RuntimeError(f"{len(failed)} documents were rejected, e.g. {failed[:3]}")
    upload_s = time.perf_counter() - t0
    count = wait_for_count(client, len(docs))
    st = idx.get_index_statistics(name)

    out = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "region": s.azure_region,
        "index": name,
        "chunking": s.chunking,
        "n_chunks": len(chunks),
        "document_count": count,
        "storage_mb": round(st.storage_size / 2**20, 2),
        "vector_index_mb": round(st.vector_index_size / 2**20, 2),
        "upload_seconds": round(upload_s, 1),
        "bge": {"model": s.embed_model, "dimensions": int(bge.shape[1])},
        "te3": te3_meta,
        "prices": {k: load_prices()[k] for k in ("source", "checked", "verified")},
    }
    merge(["azure_index"], out)
    print(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-te3", action="store_true", help="skip text-embedding-3-small (bge vectors only)")
    args = ap.parse_args()
    build(get_settings(), with_te3=not args.no_te3)


if __name__ == "__main__":
    main()
