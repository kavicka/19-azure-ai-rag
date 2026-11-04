"""Measure retrieval and generation on eval/questions.jsonl and merge the numbers into results.json.

The stack under test comes from the environment (SEARCH_PROVIDER, EMBED_PROVIDER, LLM_PROVIDER), and --tag
names the run in results.json, so local and Azure runs sit side by side:
  results["retrieval"][tag]   = {meta, <chunking>: {<mode>: metrics}}
  results["generation"][tag]  = {meta, <variant>: metrics}

Retrieval (no LLM): where does the chunk that holds the answer rank?
  evidence@k   a chunk with the gold evidence text is in the top k (the main metric)
  trial@k      any chunk of the gold trial is in the top k
Generation: no retrieval vs a retrieval mode, top 5 chunks.
  accuracy        answerable questions answered correctly (rule check vs the gold value)
  grounded        correct AND cites the gold trial: the answer is right for the right reason
  false_abstain   answerable questions where the model said "I don't know"
  abstain_unans   unanswerable questions where it correctly said "I don't know"
  bad_citations   answers citing a trial that was not among the sources (made-up citation)
  ttft / latency  time to the first token / to the full answer, retrieval included
  cost            real token counts from the provider × pricing.json; local models count as 0 USD

Run:  uv run python eval/run_eval.py --retrieval --tag local
      uv run python eval/run_eval.py --generation --tag A_local --variants no_rag rerank [--limit 20]
"""

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag import results  # noqa: E402
from rag.answer import CITE_RE, finalize, stream_answer  # noqa: E402
from rag.config import ROOT, Settings, get_settings  # noqa: E402
from rag.llm import get_llm  # noqa: E402
from rag.pricing import cost_usd, load_prices  # noqa: E402
from rag.results import hardware, load, merge  # noqa: E402
from rag.retrieve import MODES, get_retriever  # noqa: E402
from rag.scoring import is_correct  # noqa: E402

RUNS = ROOT / "eval" / "runs"
K = 5
LOCAL_LLMS = {"ollama", "transformers", "fake"}  # run on this machine: no per-token bill


def load_questions() -> list[dict]:
    return [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").open()]


def _ws(text: str) -> str:
    return " ".join(text.split())


def has_evidence(chunk: dict, q: dict) -> bool:
    return chunk["nct_id"] == q["nct_id"] and _ws(q["evidence"]) in _ws(chunk["text"])


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1)


def stack_meta(s: Settings) -> dict:
    """Everything that defines the stack, stored next to its numbers."""
    return {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "hardware": hardware(),
        "search_provider": s.search_provider,
        "embed_provider": s.embed_provider,
        "embed_model": s.azure_openai_embed_deployment if s.embed_provider == "azure_openai" else s.embed_model,
        "rerank_model": s.rerank_model,
        "chunking": s.chunking,
        "azure_region": s.azure_region,
    }


def eval_retrieval(qs: list[dict], modes: list[str], tag: str, s: Settings | None = None) -> dict:
    s = s or get_settings()
    qs = [q for q in qs if q["nct_id"]]  # made-up trials have no gold document
    r = get_retriever(s.chunking, s, dense=any(m != "bm25" for m in modes))
    out: dict = {"meta": {**stack_meta(s), "retriever": r.name, "n_questions": len(qs), "modes": modes}}
    res: dict = {"n_chunks": r.n_chunks}
    for mode in modes:
        ranks, ev_ranks, lat = [], [], []
        for q in qs:
            hits, ms = r.timed_search(q["question"], mode, k=10)
            ids = [h.chunk["nct_id"] for h in hits]
            ranks.append(ids.index(q["nct_id"]) + 1 if q["nct_id"] in ids else None)
            if q["evidence"]:  # budget questions have no evidence chunk
                ev = [i for i, h in enumerate(hits, start=1) if has_evidence(h.chunk, q)]
                ev_ranks.append(ev[0] if ev else None)
            lat.append(ms)
        n, ne = len(ranks), len(ev_ranks)
        res[mode] = {
            **{f"evidence@{k}": round(sum(1 for x in ev_ranks if x and x <= k) / ne, 3) for k in (1, 5, 10)},
            "evidence_mrr@10": round(sum(1 / x for x in ev_ranks if x) / ne, 3),
            **{f"trial@{k}": round(sum(1 for x in ranks if x and x <= k) / n, 3) for k in (1, 5)},
            "latency_ms_p50": pct(lat, 0.5),
            "latency_ms_p95": pct(lat, 0.95),
        }
        print(tag, mode, res[mode], flush=True)
    out[s.chunking] = res
    if hasattr(r.embedder, "tokens"):  # Azure embeddings: every query was a paid API call
        tokens = r.embedder.tokens
        out["meta"] |= {"query_embed_tokens": tokens, "query_embed_cost_usd": cost_usd(r.embedder.model, tokens)}
    r.close()
    merge(["retrieval", tag], out)
    return out


def summarize(records: list[dict]) -> dict:
    a_rec = [x for x in records if x["kind"] != "unanswerable"]
    u_rec = [x for x in records if x["kind"] == "unanswerable"]
    costs = [x.get("cost_usd") for x in records]
    priced = all(c is not None for c in costs)
    tin = [x["input_tokens"] for x in records if x.get("input_tokens") is not None]
    tout = [x["output_tokens"] for x in records if x.get("output_tokens") is not None]
    return {
        "n": len(records),
        "accuracy": round(sum(x["correct"] for x in a_rec) / len(a_rec), 3),
        "grounded": round(sum(x["grounded"] for x in a_rec) / len(a_rec), 3),
        "false_abstain": round(sum(x["abstained"] for x in a_rec) / len(a_rec), 3),
        "abstain_unans": round(sum(x["abstained"] for x in u_rec) / len(u_rec), 3) if u_rec else None,
        "bad_citations": round(sum(bool(x["bad_citations"]) for x in records) / len(records), 3),
        "errors": sum(bool(x.get("error")) for x in records),
        "latency_ms_p50": pct([x["ms"] for x in records], 0.5),
        "latency_ms_p95": pct([x["ms"] for x in records], 0.95),
        "ttft_ms_p50": pct([x["ttft_ms"] for x in records if x.get("ttft_ms") is not None], 0.5),
        "ttft_ms_p95": pct([x["ttft_ms"] for x in records if x.get("ttft_ms") is not None], 0.95),
        "input_tokens_mean": round(statistics.mean(tin), 1) if tin else None,
        "output_tokens_mean": round(statistics.mean(tout), 1) if tout else None,
        "cost_usd_per_answer": round(sum(costs) / len(costs), 7) if priced else None,
        "cost_usd_total": round(sum(costs), 5) if priced else None,
        "by_kind": {
            k: round(statistics.mean(x["correct"] for x in a_rec if x["kind"] == k), 3)
            for k in sorted({x["kind"] for x in a_rec})
        },
    }


def answer_one(llm, retriever, q: dict, variant: str, local_llm: bool) -> dict:
    t0 = time.perf_counter()
    hits = None if variant == "no_rag" else retriever.search(q["question"], variant, K)
    retrieval_ms = (time.perf_counter() - t0) * 1000
    parts, ttft, error = [], None, None
    try:
        for tok in stream_answer(llm, q["question"], hits):
            ttft = ttft if ttft is not None else (time.perf_counter() - t0) * 1000
            parts.append(tok)
    except Exception as e:  # e.g. a content-filter refusal on Azure: keep the run going, count it as wrong
        error = f"{type(e).__name__}: {e}"[:300]
    ms = (time.perf_counter() - t0) * 1000
    a = finalize("".join(parts), hits)
    ans = q["kind"] != "unanswerable"
    correct = ans and not error and not a.abstained and is_correct(q["kind"], q["gold"], a.text)
    u = llm.last_usage
    if local_llm:
        cost = 0.0
    else:
        cost = cost_usd(u.model, u.input_tokens, u.output_tokens) if u else None
    return {
        "id": q["id"],
        "kind": q["kind"],
        "answer": a.text,
        "correct": correct,
        "grounded": correct and q["nct_id"] in a.citations,
        "abstained": a.abstained,
        "bad_citations": a.unsupported_citations,
        "retrieved": [h.chunk["nct_id"] for h in hits or []],
        "gold_retrieved": bool(hits) and q["nct_id"] in [h.chunk["nct_id"] for h in hits],
        "ms": round(ms),
        "retrieval_ms": round(retrieval_ms),
        "ttft_ms": round(ttft) if ttft is not None else None,
        "input_tokens": u.input_tokens if u else None,
        "output_tokens": u.output_tokens if u else None,
        "model": u.model if u else None,
        "cost_usd": cost,
        "error": error,
    }


def eval_generation(qs: list[dict], variants: list[str], tag: str, s: Settings | None = None, llm=None) -> dict:
    s = s or get_settings()
    llm = llm or get_llm(s)
    needs_search = any(v != "no_rag" for v in variants)
    dense = any(v not in ("no_rag", "bm25") for v in variants)  # bm25 alone needs no embedding model
    r = get_retriever(s.chunking, s, dense=dense) if needs_search else None
    local_llm = s.llm_provider in LOCAL_LLMS
    RUNS.mkdir(parents=True, exist_ok=True)
    out: dict = {
        "meta": {
            **stack_meta(s),
            "llm": llm.name,
            "retriever": r.name if r else None,
            "k": K,
            "n_questions": len(qs),
            "temperature": s.llm_temperature,
            "max_tokens": s.llm_max_tokens,
            "prices": None if local_llm else {k: load_prices()[k] for k in ("source", "checked", "verified")},
        }
    }
    for v in variants:
        records = [answer_one(llm, r, q, v, local_llm) for q in qs]
        models = sorted({x["model"] for x in records if x["model"]})
        out["meta"]["llm_model_versions"] = models  # what Azure/Ollama says actually answered
        path = RUNS / f"{tag}__{v}.jsonl"
        path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records))
        out[v] = summarize(records)
        print(tag, v, {k: val for k, val in out[v].items() if k != "by_kind"}, flush=True)
        merge(["generation", tag], out)  # after every variant, so a run that dies keeps what it measured
    if r:
        r.close()
    return out


def rescore(qs: list[dict]) -> None:
    """Re-apply the answer rules and the current prices to the stored answers, without calling any LLM again."""
    by_id = {q["id"]: q for q in qs}
    res = load()
    for path in sorted(RUNS.glob("*__*.jsonl")):
        tag, v = path.stem.split("__", 1)
        records = [json.loads(line) for line in path.open()]
        for x in records:
            q = by_id[x["id"]]
            x["correct"] = (
                q["kind"] != "unanswerable"
                and not x.get("error")
                and not x["abstained"]
                and is_correct(q["kind"], q["gold"], x["answer"])
            )
            x["grounded"] = x["correct"] and q["nct_id"] in CITE_RE.findall(x["answer"])
            # priced answers get today's pricing.json (e.g. after the prices were verified); local ones stay 0
            if x.get("cost_usd") != 0.0 and x.get("model") and x.get("input_tokens") is not None:
                x["cost_usd"] = cost_usd(x["model"], x["input_tokens"], x["output_tokens"] or 0)
        path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records))
        gen = res.setdefault("generation", {}).setdefault(tag, {"meta": {}})
        gen[v] = summarize(records)
        gen["meta"]["rescored"] = datetime.now(UTC).isoformat(timespec="seconds")
        print(tag, v, {k: val for k, val in gen[v].items() if k != "by_kind"})
    results.RESULTS.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--generation", action="store_true")
    ap.add_argument("--tag", default=None, help="name of this run in results.json, e.g. local, azure_bge, A_local")
    ap.add_argument("--modes", nargs="+", default=list(MODES), help="retrieval modes to evaluate")
    ap.add_argument("--variants", nargs="+", default=["no_rag", "rerank"], help="no_rag and/or retrieval modes")
    ap.add_argument("--rescore", action="store_true", help="re-apply the scoring rules to stored answers")
    ap.add_argument("--limit", type=int, default=None, help="every n-th question only, for a quick smoke run")
    args = ap.parse_args()
    if (args.retrieval or args.generation) and not args.tag:
        ap.error("--tag is required: it names the run in results.json")
    qs = load_questions()
    if args.limit:
        qs = qs[:: max(1, len(qs) // args.limit)]
    if args.retrieval:
        eval_retrieval(qs, args.modes, args.tag)
    if args.generation:
        eval_generation(qs, args.variants, args.tag)
    if args.rescore:
        rescore(qs)


if __name__ == "__main__":
    main()
