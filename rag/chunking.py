"""Two ways to cut documents into chunks, compared in the eval.

- "fixed":   the naive baseline. Glue the whole record together and cut every N words. A chunk can start in the
             middle of the eligibility criteria and has no idea which trial it belongs to.
- "section": cut along the record's own sections and prefix every chunk with a short header
             (NCT id, title, section). Why: a chunk like "Enrollment: 240 participants" is useless on its own;
             with the header, both BM25 and the embedding model can match it to a question about that trial.
"""

from dataclasses import asdict, dataclass

MAX_WORDS = 220  # ~300 tokens: small enough for precise retrieval, big enough to hold one idea
OVERLAP = 40  # words repeated between neighbouring windows so a sentence on the border is not lost


@dataclass
class Chunk:
    chunk_id: str
    nct_id: str
    title: str
    section: str
    text: str  # what gets embedded / BM25-indexed / shown to the LLM

    def to_dict(self) -> dict:
        return asdict(self)


def _windows(words: list[str], size: int = MAX_WORDS, overlap: int = OVERLAP) -> list[list[str]]:
    if len(words) <= size:
        return [words]
    step = size - overlap
    return [words[i : i + size] for i in range(0, len(words) - overlap, step)]


def chunk_fixed(doc: dict) -> list[Chunk]:
    words = " ".join(s["text"] for s in doc["sections"]).split()
    return [
        Chunk(f"{doc['nct_id']}#f{i}", doc["nct_id"], doc["title"], "fixed", " ".join(w))
        for i, w in enumerate(_windows(words))
    ]


def chunk_section(doc: dict) -> list[Chunk]:
    out: list[Chunk] = []
    for sec in doc["sections"]:
        header = f"[{doc['nct_id']}] {doc['title']} | {sec['name']}"
        for i, w in enumerate(_windows(sec["text"].split())):
            out.append(
                Chunk(
                    f"{doc['nct_id']}#{sec['name'].lower().replace(' ', '-')}-{i}",
                    doc["nct_id"],
                    doc["title"],
                    sec["name"],
                    f"{header}\n{' '.join(w)}",
                )
            )
    return out


CHUNKERS = {"fixed": chunk_fixed, "section": chunk_section}
