"""Fast tests: no model downloads. Dense retrieval and real LLMs are covered by the eval, not here."""

import json

import pytest
from fastapi.testclient import TestClient

from rag.answer import build_messages, citations, finalize, is_abstain
from rag.chunking import chunk_fixed, chunk_section
from rag.config import Settings
from rag.ingest import render_study
from rag.llm import FakeLLM
from rag.retrieve import Hit, Retriever, rrf
from rag.scoring import is_correct, name_match

STUDY = {
    "protocolSection": {
        "identificationModule": {"nctId": "NCT00000001", "briefTitle": "Metformin in Prediabetes"},
        "statusModule": {"overallStatus": "COMPLETED", "startDateStruct": {"date": "2015-03"}},
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Acme Pharma (ACME)"}},
        "descriptionModule": {"briefSummary": "word " * 500},
        "conditionsModule": {"conditions": ["Prediabetes"]},
        "designModule": {
            "studyType": "INTERVENTIONAL",
            "phases": ["PHASE2", "PHASE3"],
            "designInfo": {"allocation": "RANDOMIZED", "maskingInfo": {"masking": "DOUBLE"}},
            "enrollmentInfo": {"count": 240, "type": "ACTUAL"},
        },
        "armsInterventionsModule": {"interventions": [{"type": "DRUG", "name": "Metformin"}]},
        "eligibilityModule": {"minimumAge": "18 Years", "sex": "ALL"},
    }
}
OTHER = json.loads(json.dumps(STUDY).replace("NCT00000001", "NCT00000002").replace("Metformin", "Insulin"))


def test_render_study_makes_readable_sections():
    d = render_study(STUDY)
    design = next(s["text"] for s in d["sections"] if s["name"] == "Design")
    assert "Phase: Phase 2, Phase 3" in design and "Enrollment: 240 participants" in design
    assert "Masking: Double" in design
    assert d["meta"]["phases"] == ["Phase 2", "Phase 3"]


def test_section_chunks_carry_a_header_and_fixed_chunks_do_not():
    d = render_study(STUDY)
    sec = chunk_section(d)
    assert all(c.text.startswith("[NCT00000001] Metformin in Prediabetes |") for c in sec)
    assert len([c for c in sec if c.section == "Summary"]) == 3  # 500 words, windows of 220 with overlap 40
    assert not chunk_fixed(d)[0].text.startswith("[NCT")


def test_rrf_rewards_documents_both_retrievers_like():
    fused = rrf([[1, 2, 3], [3, 1, 4]])
    assert max(fused, key=fused.get) == 1  # rank 1 + rank 2 beats rank 3 + rank 1
    assert fused[4] < fused[2] or fused[4] == pytest.approx(fused[2])


@pytest.mark.parametrize(
    ("kind", "gold", "answer", "ok"),
    [
        ("phase", ["Phase 2", "Phase 3"], "It is a Phase 2/3 trial [NCT1].", True),
        ("phase", ["Phase 2"], "This is a phase II study.", True),
        ("phase", ["Phase 3"], "Phase 2.", False),
        ("enrollment", 1240, "1,240 participants were enrolled.", True),
        ("enrollment", 240, "1,240 participants were enrolled.", False),
        ("sponsor", "National Institute on Aging (NIA)", "Sponsored by the NIA.", True),
        ("sponsor", "Eli Lilly and Company", "Eli Lilly", True),
        ("masking", "NONE", "It was an open-label study.", True),
        ("masking", "NONE", "The trial did not use any masking [NCT1].", True),
        ("masking", "NONE", "The trial did not specify the type of masking used.", False),
        ("masking", "QUADRUPLE", "Quadruple-blind.", True),
        ("intervention", ["Placebo", "Metformin"], "Metformin was tested.", True),
    ],
)
def test_is_correct(kind, gold, answer, ok):
    assert is_correct(kind, gold, answer) is ok


def test_name_match_needs_most_words():
    assert not name_match("Novo Nordisk A/S", "Sanofi")


def test_citations_abstain_and_made_up_ids():
    hit = Hit({"nct_id": "NCT00000001", "text": "x"}, 1.0)
    a = finalize("Phase 2 [NCT00000001], also [NCT99999999] and [NCT00000001].", [hit])
    assert a.citations == ["NCT00000001", "NCT99999999"]
    assert a.unsupported_citations == ["NCT99999999"]
    assert is_abstain("I don’t know.") and not is_abstain("Phase 2")
    assert citations("no ids") == []


def test_prompt_labels_every_source_with_its_trial_id():
    hit = Hit({"nct_id": "NCT00000001", "text": "Enrollment: 240"}, 1.0)
    msgs = build_messages("How many?", [hit])
    assert "Source 1 [NCT00000001]: Enrollment: 240" in msgs[-1]["content"]
    assert "ONLY" in msgs[0]["content"]


@pytest.fixture
def app_client(tmp_path):
    from api.main import create_app

    docs = [render_study(STUDY), render_study(OTHER)]
    s = Settings(data_dir=tmp_path)
    p = tmp_path / "index" / "section"
    p.mkdir(parents=True)
    (p / "chunks.jsonl").write_text("".join(json.dumps(c.to_dict()) + "\n" for d in docs for c in chunk_section(d)))
    retr = Retriever("section", s, dense=False)
    with TestClient(create_app(retr, FakeLLM(), docs)) as c:
        yield c


def test_bm25_search_finds_the_right_trial(app_client):
    r = app_client.post("/api/search", json={"question": "insulin enrollment", "mode": "bm25", "k": 3})
    assert r.status_code == 200
    assert r.json()["hits"][0]["nct_id"] == "NCT00000002"


def test_ask_streams_sources_tokens_and_done(app_client):
    r = app_client.post("/api/ask", json={"question": "Which sponsor runs the insulin trial?", "mode": "bm25"})
    events = [line.split(": ", 1)[1] for line in r.text.splitlines() if line.startswith("event: ")]
    assert events[0] == "sources" and "token" in events and events[-1] == "done"
    done = json.loads(r.text.strip().splitlines()[-1].removeprefix("data: "))
    assert done["citations"] and not done["unsupported_citations"]


def test_validation_and_404(app_client):
    assert app_client.post("/api/ask", json={"question": "x", "mode": "bm25"}).status_code == 422
    assert app_client.post("/api/ask", json={"question": "hello", "mode": "magic"}).status_code == 422
    assert app_client.get("/api/studies/NCT12345678").status_code == 404
    assert app_client.get("/api/studies/NCT00000001").json()["title"] == "Metformin in Prediabetes"
