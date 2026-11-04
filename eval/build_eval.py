"""Generate the eval set from structured fields of the records (gold answers are exact, not guessed).

Questions name the trial by its title, never by NCT id: a user asks "the DPP outcomes study", and an id in the
question would let BM25 win by string matching alone. Two kinds of unanswerable questions test abstention:
a fact no record contains (budget) and a trial that does not exist.

Run:  uv run python eval/build_eval.py   → eval/questions.jsonl (committed, so every run uses the same set)
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.config import ROOT, get_settings  # noqa: E402
from rag.index import load_docs  # noqa: E402
from rag.ingest import _pretty  # noqa: E402

SEED = 0
PER_KIND = 18  # 7 kinds × 18 = 126 answerable
N_BUDGET, N_FAKE = 15, 15

TEMPLATES = {
    "phase": 'What clinical phase is the trial "{t}"?',
    "enrollment": 'How many participants were enrolled in the trial "{t}"?',
    "sponsor": 'Who is the lead sponsor of the trial "{t}"?',
    "start_year": 'In which year did the trial "{t}" start?',
    "masking": 'What type of masking (blinding) was used in the trial "{t}"?',
    "intervention": 'Which intervention was tested in the trial "{t}"?',
    "min_age": 'What is the minimum age for participants in the trial "{t}"?',
}


def gold_for(kind: str, m: dict):
    match kind:
        case "phase":
            return m["phases"] or None
        case "enrollment":
            return m["enrollment"]
        case "sponsor":
            return m["lead_sponsor"]
        case "start_year":
            return int(m["start_date"][:4]) if m.get("start_date") else None
        case "masking":
            return m["masking"]
        case "intervention":
            return [n for n in m["interventions"] if n] or None
        case "min_age":
            return int(m["minimum_age"].split()[0]) if (m.get("minimum_age") or "").endswith("Years") else None


def evidence_for(kind: str, m: dict) -> str:
    """The exact text that proves the answer. Retrieval only counts as a hit if a chunk contains it:
    finding *some* chunk of the right trial is not enough when the fact sits in another chunk."""
    match kind:
        case "phase":
            return "Phase: " + ", ".join(m["phases"])
        case "enrollment":
            return f"Enrollment: {m['enrollment']} participants"
        case "sponsor":
            return f"Lead sponsor: {m['lead_sponsor']}"
        case "start_year":
            return f"Start date: {m['start_date']}"
        case "masking":
            return f"Masking: {_pretty(m['masking'])}"
        case "intervention":
            return next(n for n in m["interventions"] if n)
        case "min_age":
            return f"Minimum age: {m['minimum_age']}"
    raise ValueError(kind)


def main() -> None:
    rng = random.Random(SEED)
    docs = load_docs(get_settings())
    titles = Counter(d["title"].lower() for d in docs)
    docs = [d for d in docs if titles[d["title"].lower()] == 1]  # ambiguous titles have no single gold trial

    qs: list[dict] = []
    for kind, tpl in TEMPLATES.items():
        pool = [d for d in docs if gold_for(kind, d["meta"]) is not None]
        for d in rng.sample(pool, PER_KIND):
            qs.append(
                {
                    "kind": kind,
                    "question": tpl.format(t=d["title"]),
                    "gold": gold_for(kind, d["meta"]),
                    "nct_id": d["nct_id"],
                    "evidence": evidence_for(kind, d["meta"]),
                }
            )
    for d in rng.sample(docs, N_BUDGET):
        q = f'What was the total budget in US dollars of the trial "{d["title"]}"?'
        qs.append({"kind": "unanswerable", "question": q, "gold": None, "nct_id": d["nct_id"], "evidence": None})
    conditions = sorted({c for d in docs for c in d["meta"]["conditions"]})
    for _ in range(N_FAKE):
        code = f"{rng.choice('BKQXZ')}{rng.choice('BKQXZ')}-{rng.randint(1000, 9999)}"
        q = f'How many participants were enrolled in the trial "A Phase 2 Study of {code} in {rng.choice(conditions)}"?'
        qs.append({"kind": "unanswerable", "question": q, "gold": None, "nct_id": None, "evidence": None})

    for i, q in enumerate(qs):
        q["id"] = f"q{i:03d}"
    out = ROOT / "eval" / "questions.jsonl"
    out.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs))
    print(f"{len(qs)} questions → {out}", Counter(q["kind"] for q in qs))


if __name__ == "__main__":
    main()
