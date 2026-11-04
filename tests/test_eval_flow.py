"""The eval and the Azure indexer end to end on a two-trial corpus: no models, no network, temp results.json."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from rag import results
from rag.chunking import chunk_section
from rag.config import ROOT, Settings
from rag.ingest import render_study
from rag.llm import Usage
from rag.pricing import price_for
from tests.test_rag import OTHER, STUDY


def load_run_eval():
    spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


QUESTIONS = [
    {
        "id": "q1",
        "kind": "phase",
        "question": "What clinical phase is the metformin prediabetes trial?",
        "gold": ["Phase 2", "Phase 3"],
        "nct_id": "NCT00000001",
        "evidence": "Phase: Phase 2, Phase 3",
    },
    {
        "id": "q2",
        "kind": "enrollment",
        "question": "How many participants did the insulin prediabetes trial enrol?",
        "gold": 240,
        "nct_id": "NCT00000002",
        "evidence": "Enrollment: 240 participants",
    },
    {
        "id": "q3",
        "kind": "unanswerable",
        "question": "What was the budget?",
        "gold": None,
        "nct_id": None,
        "evidence": None,
    },
]


class PricedLLM:
    """Behaves like AzureOpenAILLM: streams, then reports usage. Fails on one question to test the error path."""

    name = "azure_openai:gpt-4o-mini"

    def __init__(self, fail_on: str | None = None):
        self.last_usage, self.fail_on = None, fail_on

    def stream(self, messages):
        user = messages[-1]["content"]
        if self.fail_on and self.fail_on in user:
            raise RuntimeError("content_filter")
        if "budget" in user:
            yield "I don't know."
        elif "insulin" in user:
            yield from ["240 participants ", "[NCT00000002]."]
        else:
            yield from ["Phase 2/3 ", "[NCT00000001]."]
        self.last_usage = Usage(1000, 10, "gpt-4o-mini-2024-07-18")


@pytest.fixture
def env(tmp_path, monkeypatch):
    docs = [render_study(STUDY), render_study(OTHER)]
    d = tmp_path / "index" / "section"
    d.mkdir(parents=True)
    chunks = [c.to_dict() for doc in docs for c in chunk_section(doc)]
    (d / "chunks.jsonl").write_text("".join(json.dumps(c) + "\n" for c in chunks))
    (d / "stats.json").write_text(json.dumps({"n_chunks": len(chunks), "avg_words": 10}))
    monkeypatch.setattr(results, "RESULTS", tmp_path / "results.json")
    ev = load_run_eval()
    monkeypatch.setattr(ev, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(ev, "load_questions", lambda: QUESTIONS)
    s = Settings(_env_file=None, data_dir=tmp_path, llm_provider="fake")
    return ev, s, tmp_path, chunks


def test_retrieval_eval_writes_a_tagged_section(env):
    ev, s, tmp, chunks = env
    ev.eval_retrieval(QUESTIONS, ["bm25"], "local", s)
    res = json.loads((tmp / "results.json").read_text())["retrieval"]["local"]
    assert res["meta"]["retriever"] == "local:bm25" and res["meta"]["n_questions"] == 2  # q3 has no gold trial
    m = res["section"]["bm25"]
    assert res["section"]["n_chunks"] == len(chunks)
    assert m["trial@5"] == 1.0 and m["evidence@10"] == 1.0
    assert m["latency_ms_p50"] is not None


def test_generation_eval_counts_tokens_and_cost(env):
    ev, s, tmp, _ = env
    s = s.model_copy(update={"llm_provider": "azure_openai"})  # priced: not a local model
    out = ev.eval_generation(QUESTIONS, ["no_rag", "bm25"], "B_test", s, llm=PricedLLM())
    g = json.loads((tmp / "results.json").read_text())["generation"]["B_test"]
    assert g == json.loads(json.dumps(out))
    bm = g["bm25"]
    assert bm["accuracy"] == 1.0 and bm["grounded"] == 1.0 and bm["abstain_unans"] == 1.0 and bm["errors"] == 0
    # gpt-4o-mini: 1000 × 0.15 + 10 × 0.60 per 1M tokens = 0.000156 USD per answer
    assert bm["cost_usd_per_answer"] == pytest.approx(0.000156)
    assert bm["cost_usd_total"] == pytest.approx(3 * 0.000156, abs=1e-5)
    assert bm["input_tokens_mean"] == 1000 and bm["ttft_ms_p50"] is not None
    assert g["meta"]["llm_model_versions"] == ["gpt-4o-mini-2024-07-18"]
    assert g["meta"]["prices"]["verified"] in (True, False)
    rec = [json.loads(x) for x in (tmp / "runs" / "B_test__bm25.jsonl").read_text().splitlines()]
    assert {r["id"] for r in rec} == {"q1", "q2", "q3"} and rec[0]["model"] == "gpt-4o-mini-2024-07-18"


def test_local_models_cost_zero_and_unknown_prices_stay_unknown(env):
    ev, s, tmp, _ = env
    g = ev.eval_generation(QUESTIONS, ["bm25"], "A_test", s, llm=PricedLLM())  # llm_provider=fake → local
    assert g["bm25"]["cost_usd_per_answer"] == 0.0

    class Unpriced(PricedLLM):
        def stream(self, messages):
            yield from super().stream(messages)
            self.last_usage = Usage(5, 5, "some-new-model")

    s2 = s.model_copy(update={"llm_provider": "azure_openai"})
    g2 = ev.eval_generation(QUESTIONS, ["bm25"], "X_test", s2, llm=Unpriced())
    assert g2["bm25"]["cost_usd_per_answer"] is None  # not in pricing.json → no made-up number


def test_a_failing_answer_is_recorded_not_fatal(env):
    ev, s, tmp, _ = env
    g = ev.eval_generation(QUESTIONS, ["bm25"], "E_test", s, llm=PricedLLM(fail_on="insulin"))
    assert g["bm25"]["errors"] == 1 and g["bm25"]["accuracy"] == 0.5
    rec = {r["id"]: r for r in map(json.loads, (tmp / "runs" / "E_test__bm25.jsonl").read_text().splitlines())}
    assert rec["q2"]["error"].startswith("RuntimeError") and rec["q2"]["correct"] is False


def test_rescore_reproduces_the_stored_summary(env):
    ev, s, tmp, _ = env
    first = ev.eval_generation(QUESTIONS, ["bm25"], "R_test", s, llm=PricedLLM())["bm25"]
    ev.rescore(QUESTIONS)
    again = json.loads((tmp / "results.json").read_text())["generation"]["R_test"]["bm25"]
    assert {k: v for k, v in again.items()} == json.loads(json.dumps(first))


# --- Azure indexer with fake clients -----------------------------------------------------------------------------


class FakeIndexClient:
    def __init__(self):
        self.created, self.deleted = None, []

    def list_index_names(self):
        return ["chunks-section"]  # already there: must be replaced, not appended to

    def delete_index(self, name):
        self.deleted.append(name)

    def create_index(self, index):
        self.created = index.as_dict()

    def get_index_statistics(self, name):
        class St:
            storage_size, vector_index_size = 3 * 2**20, 2**20

        return St()


class FakeUploadClient:
    def __init__(self):
        self.batches = []

    def upload_documents(self, docs):
        self.batches.append(docs)

        class R:
            def __init__(self, key):
                self.key, self.succeeded = key, True

        return [R(d["id"]) for d in docs]

    def get_document_count(self):
        return sum(len(b) for b in self.batches)


def test_azure_index_uploads_the_local_bge_vectors_and_caches_te3(env, monkeypatch):
    from rag import azure_index, embed

    ev, s, tmp, chunks = env
    s = s.model_copy(
        update={
            "search_provider": "azure_search",
            "azure_search_endpoint": "https://x.search.windows.net",
            "azure_openai_endpoint": "https://x.openai.azure.com",
            "azure_openai_embed_dimensions": 8,
        }
    )
    bge = np.random.default_rng(0).normal(size=(len(chunks), 4)).astype(np.float32)
    np.save(tmp / "index" / "section" / "vectors_bge.npy", bge)
    idx, up = FakeIndexClient(), FakeUploadClient()
    monkeypatch.setattr(azure_index, "index_client", lambda s: idx)
    monkeypatch.setattr(azure_index, "search_client", lambda s: up)
    monkeypatch.setattr(azure_index, "BATCH", 2)

    calls = []

    class FakeTe3:
        def __init__(self, s):
            self.tokens, self.model = 0, ""

        def encode(self, texts, batch_size=16):
            calls.append(len(texts))
            self.tokens, self.model = 100 * len(texts), "text-embedding-3-small"
            return np.ones((len(texts), 8), dtype=np.float32)

    monkeypatch.setattr(embed, "AzureOpenAIEmbedder", FakeTe3)
    out = azure_index.build(s)

    assert idx.deleted == ["chunks-section"]
    dims = {f["name"]: f.get("dimensions") for f in idx.created["fields"]}
    assert dims["vec_bge"] == 4 and dims["vec_te3"] == 8
    sent = [d for b in up.batches for d in b]
    assert len(sent) == len(chunks) and max(len(b) for b in up.batches) == 2
    assert np.allclose(sent[3]["vec_bge"], bge[3])  # exactly the local vectors
    assert out["document_count"] == len(chunks) and out["storage_mb"] == 3.0
    assert out["te3"]["tokens"] == 100 * len(chunks)
    assert out["te3"]["cost_usd"] == pytest.approx(100 * len(chunks) * 0.02 / 1e6)
    assert json.loads((tmp / "results.json").read_text())["azure_index"]["n_chunks"] == len(chunks)

    azure_index.build(s)  # second run: te3 vectors come from the cache, nothing is paid twice
    assert calls == [len(chunks)]


def test_azure_index_refuses_without_the_local_index(env):
    from rag import azure_index

    ev, s, tmp, _ = env
    s = s.model_copy(update={"azure_search_endpoint": "https://x.search.windows.net"})
    with pytest.raises(SystemExit, match="rag.index"):
        azure_index.build(s, with_te3=False)


def test_results_merge_keeps_other_sections(tmp_path):
    p = Path(tmp_path / "r.json")
    results.merge(["retrieval", "local"], {"a": 1}, p)
    results.merge(["retrieval", "azure_bge"], {"b": 2}, p)
    results.merge(["azure_index"], {"c": 3}, p)
    assert json.loads(p.read_text()) == {
        "retrieval": {"local": {"a": 1}, "azure_bge": {"b": 2}},
        "azure_index": {"c": 3},
    }


def test_rescore_applies_updated_prices(env, monkeypatch):
    ev, s, tmp, _ = env
    s = s.model_copy(update={"llm_provider": "azure_openai"})
    ev.eval_generation(QUESTIONS, ["bm25"], "P_test", s, llm=PricedLLM())
    cheaper = {"usd_per_1m_tokens": {"gpt-4o-mini": {"input": 0.1, "output": 0.5}}}
    monkeypatch.setattr(ev, "cost_usd", lambda m, i, o=0: (i * 0.1 + o * 0.5) / 1e6 if price_for(m, cheaper) else None)
    ev.rescore(QUESTIONS)
    again = json.loads((tmp / "results.json").read_text())["generation"]["P_test"]["bm25"]
    assert again["cost_usd_per_answer"] == pytest.approx((1000 * 0.1 + 10 * 0.5) / 1e6)
