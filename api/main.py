"""FastAPI backend: search, streamed answers (Server-Sent Events) and the source records.

Why SSE and not WebSockets: the stream only goes one way (server → browser), SSE is plain HTTP, works through
proxies and load balancers (step 17/18), and needs no extra protocol.

Run:  uv run uvicorn api.main:app --reload --port 8000
"""

import json
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag.answer import finalize, stream_answer
from rag.config import get_settings
from rag.index import load_docs
from rag.llm import LLM, get_llm
from rag.retrieve import MODES, Retriever, get_retriever

Variant = Literal["no_rag", "bm25", "dense", "hybrid", "rerank"]


class SearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: Literal["bm25", "dense", "hybrid", "rerank"] = "rerank"
    k: int = Field(default=5, ge=1, le=10)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: Variant = "rerank"
    k: int = Field(default=5, ge=1, le=10)


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(retriever: Retriever | None = None, llm: LLM | None = None, docs: list[dict] | None = None) -> FastAPI:
    """Factory, so tests can inject a small retriever and a fake LLM instead of loading real models."""
    s = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.retriever = retriever or get_retriever(s.chunking, s)  # local or Azure AI Search
        app.state.llm = llm or get_llm(s)
        app.state.docs = {d["nct_id"]: d for d in (docs if docs is not None else load_docs(s))}
        yield

    app = FastAPI(title="Clinical trials RAG", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=s.allowed_origins, allow_methods=["GET", "POST"], allow_headers=["*"]
    )

    @app.get("/api/health")
    def health(request: Request) -> dict:
        st = request.app.state
        return {
            "status": "ok",
            "llm": st.llm.name,
            "search": st.retriever.name,
            "chunking": st.retriever.chunking,
            "n_chunks": st.retriever.n_chunks,
            "n_studies": len(st.docs),
            "modes": ["no_rag", *MODES],
        }

    @app.post("/api/search")
    def search(req: SearchRequest, request: Request) -> dict:
        hits, ms = request.app.state.retriever.timed_search(req.question, req.mode, req.k)
        return {"hits": [h.to_dict() for h in hits], "retrieval_ms": round(ms, 1)}

    @app.post("/api/ask")
    def ask(req: AskRequest, request: Request) -> StreamingResponse:
        st = request.app.state

        def events() -> Iterator[str]:
            t0 = time.perf_counter()
            hits = None
            if req.mode != "no_rag":
                hits, ms = st.retriever.timed_search(req.question, req.mode, req.k)
                yield sse("sources", {"hits": [h.to_dict() for h in hits], "retrieval_ms": round(ms, 1)})
            parts: list[str] = []
            first_ms = None
            try:
                for tok in stream_answer(st.llm, req.question, hits):
                    first_ms = first_ms or (time.perf_counter() - t0) * 1000
                    parts.append(tok)
                    yield sse("token", {"text": tok})
            except Exception as e:  # e.g. Ollama not running: tell the UI instead of cutting the stream silently
                yield sse("error", {"message": f"{type(e).__name__}: {e}"})
                return
            a = finalize("".join(parts), hits)
            yield sse(
                "done",
                {
                    "citations": a.citations,
                    "abstained": a.abstained,
                    "unsupported_citations": a.unsupported_citations,
                    "first_token_ms": round(first_ms or 0),
                    "total_ms": round((time.perf_counter() - t0) * 1000),
                },
            )

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/studies/{nct_id}")
    def study(nct_id: str, request: Request) -> dict:
        if (d := request.app.state.docs.get(nct_id)) is None:
            raise HTTPException(404, f"{nct_id} is not in the corpus")
        return {
            "nct_id": d["nct_id"],
            "title": d["title"],
            "sections": d["sections"],
            "url": f"https://clinicaltrials.gov/study/{d['nct_id']}",
        }

    return app


app = create_app()
