"""Download public clinical-trial records from the ClinicalTrials.gov API v2 and turn them into text documents.

Why this source: it is free, needs no key, is the reference registry for clinical trials (the "CTM" world)
and every record has structured fields next to free text, so eval questions get exact gold answers.

Run:  uv run python -m rag.ingest              # first time: search by conditions, save the NCT id list
      uv run python -m rag.ingest --from-ids   # later: re-download exactly the ids in eval/study_ids.txt
"""

import argparse
import json
import time
from typing import Any

import httpx

from rag.config import ROOT, get_settings

API = "https://clinicaltrials.gov/api/v2/studies"
IDS_FILE = ROOT / "eval" / "study_ids.txt"
# Several therapeutic areas, so retrieval has to tell similar trials apart (that is what makes it hard).
CONDITIONS = [
    "type 2 diabetes",
    "breast cancer",
    "asthma",
    "alzheimer disease",
    "rheumatoid arthritis",
    "hypertension",
    "major depressive disorder",
    "psoriasis",
]
PER_CONDITION = 50


def _get(client: httpx.Client, params: dict[str, Any]) -> dict[str, Any]:
    for attempt in range(4):
        try:
            r = client.get(API, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def fetch_by_conditions(client: httpx.Client) -> list[dict]:
    studies: dict[str, dict] = {}
    for cond in CONDITIONS:
        page = _get(
            client,
            {
                "query.cond": cond,
                "filter.overallStatus": "COMPLETED",
                "filter.advanced": "AREA[StudyType]INTERVENTIONAL AND AREA[HasResults]true",
                "pageSize": PER_CONDITION,
                "sort": "LastUpdatePostDate:desc",
            },
        )
        for s in page["studies"]:
            studies.setdefault(s["protocolSection"]["identificationModule"]["nctId"], s)
        print(f"{cond:28s} → {len(page['studies'])} studies (total unique {len(studies)})")
    return list(studies.values())


def fetch_by_ids(client: httpx.Client, ids: list[str]) -> list[dict]:
    out = []
    for i in range(0, len(ids), 100):
        batch = ids[i : i + 100]
        page = _get(client, {"filter.ids": ",".join(batch), "pageSize": 100})
        out.extend(page["studies"])
    return out


# ---------- record → text document ----------


def _join(items: list[str] | None) -> str:
    return ", ".join(i for i in (items or []) if i)


def _pretty(code: str | None) -> str:
    """API enums like PHASE2 / DOUBLE / RANDOMIZED → readable words, because the LLM and BM25 read text."""
    if not code:
        return ""
    return code.replace("PHASE", "Phase ").replace("_", " ").replace("EARLY Phase ", "Early Phase ").strip().title()


def render_study(study: dict) -> dict:
    """One study → {nct_id, title, sections: [{name, text}], meta}. Sections mirror how a reader scans a record."""
    p = study["protocolSection"]
    ident, status = p["identificationModule"], p.get("statusModule", {})
    design, desc = p.get("designModule", {}), p.get("descriptionModule", {})
    arms, outcomes = p.get("armsInterventionsModule", {}), p.get("outcomesModule", {})
    elig, sponsor = p.get("eligibilityModule", {}), p.get("sponsorCollaboratorsModule", {})
    conds = p.get("conditionsModule", {})
    info = design.get("designInfo", {})

    nct, title = ident["nctId"], ident.get("briefTitle", "")
    phases = [_pretty(x) for x in design.get("phases", []) if x != "NA"]
    enroll = design.get("enrollmentInfo", {})
    interventions = arms.get("interventions", [])

    overview = [
        f"Official title: {ident.get('officialTitle', title)}",
        f"Conditions: {_join(conds.get('conditions'))}",
        f"Keywords: {_join(conds.get('keywords'))}" if conds.get("keywords") else "",
        f"Lead sponsor: {sponsor.get('leadSponsor', {}).get('name', '')}",
        f"Collaborators: {_join([c['name'] for c in sponsor.get('collaborators', [])])}"
        if sponsor.get("collaborators")
        else "",
        f"Overall status: {_pretty(status.get('overallStatus'))}",
        f"Start date: {status.get('startDateStruct', {}).get('date', '')}",
        f"Completion date: {status.get('completionDateStruct', {}).get('date', '')}",
    ]
    design_lines = [
        f"Study type: {_pretty(design.get('studyType'))}",
        f"Phase: {', '.join(phases) if phases else 'Not applicable'}",
        f"Allocation: {_pretty(info.get('allocation'))}",
        f"Intervention model: {_pretty(info.get('interventionModel'))}",
        f"Primary purpose: {_pretty(info.get('primaryPurpose'))}",
        f"Masking: {_pretty(info.get('maskingInfo', {}).get('masking'))}",
        f"Enrollment: {enroll.get('count', '')} participants ({_pretty(enroll.get('type'))})",
    ]
    arm_lines = [
        f"Arm '{a.get('label', '')}' ({_pretty(a.get('type'))}): {a.get('description', '')}"
        for a in arms.get("armGroups", [])
    ] + [
        f"Intervention ({_pretty(i.get('type'))}): {i.get('name', '')}. {i.get('description', '')}"
        for i in interventions
    ]
    outcome_lines = [
        f"{kind} outcome: {o.get('measure', '')}. {o.get('description', '')} Time frame: {o.get('timeFrame', '')}"
        for kind, key in (("Primary", "primaryOutcomes"), ("Secondary", "secondaryOutcomes"))
        for o in outcomes.get(key, [])
    ]
    elig_lines = [
        f"Sex: {_pretty(elig.get('sex'))}",
        f"Minimum age: {elig.get('minimumAge', 'not stated')}",
        f"Maximum age: {elig.get('maximumAge', 'not stated')}",
        f"Healthy volunteers: {'yes' if elig.get('healthyVolunteers') else 'no'}",
        elig.get("eligibilityCriteria", ""),
    ]
    sections = [
        ("Overview", "\n".join(x for x in overview if x)),
        ("Summary", desc.get("briefSummary", "")),
        ("Detailed description", desc.get("detailedDescription", "")),
        ("Design", "\n".join(design_lines)),
        ("Arms and interventions", "\n".join(arm_lines)),
        ("Outcomes", "\n".join(outcome_lines)),
        ("Eligibility", "\n".join(x for x in elig_lines if x)),
    ]
    meta = {
        "phases": phases,
        "enrollment": enroll.get("count"),
        "lead_sponsor": sponsor.get("leadSponsor", {}).get("name"),
        "start_date": status.get("startDateStruct", {}).get("date"),
        "masking": info.get("maskingInfo", {}).get("masking"),
        "allocation": info.get("allocation"),
        "interventions": [i.get("name") for i in interventions],
        "minimum_age": elig.get("minimumAge"),
        "conditions": conds.get("conditions", []),
    }
    return {
        "nct_id": nct,
        "title": title,
        "sections": [{"name": n, "text": t.strip()} for n, t in sections if t and t.strip()],
        "meta": meta,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-ids", action="store_true", help="re-download the frozen id list instead of searching")
    args = ap.parse_args()
    s = get_settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(headers={"User-Agent": "rag-fullstack-clinical (learning project)"}) as client:
        if args.from_ids:
            raw = fetch_by_ids(client, IDS_FILE.read_text().split())
        else:
            raw = fetch_by_conditions(client)
            # Freeze the corpus: the search results change every day, the eval set must not.
            IDS_FILE.write_text(
                "\n".join(sorted(st["protocolSection"]["identificationModule"]["nctId"] for st in raw)) + "\n"
            )

    docs = sorted((render_study(st) for st in raw), key=lambda d: d["nct_id"])
    out = s.data_dir / "studies.jsonl"
    with out.open("w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    n_words = sum(len(sec["text"].split()) for d in docs for sec in d["sections"])
    print(f"wrote {len(docs)} studies, {n_words:,} words → {out}")


if __name__ == "__main__":
    main()
