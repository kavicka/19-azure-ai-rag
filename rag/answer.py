"""Retrieve → build a grounded prompt → stream the answer → check its citations.

The prompt does three jobs, each one measured in the eval:
1. answer only from the sources (otherwise RAG is just a slower chatbot that still hallucinates),
2. cite the trial id after every claim, so a human can verify it in one click,
3. say exactly "I don't know" when the sources do not contain the answer (abstention).
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from rag.llm import LLM, Message
from rag.retrieve import Hit

ABSTAIN = "I don't know"
CITE_RE = re.compile(r"\[(NCT\d{8})\]")

SYSTEM_RAG = f"""You answer questions about clinical trials using ONLY the numbered sources below.
Rules:
- Use only facts stated in the sources. Do not use outside knowledge.
- After each fact, cite the trial id in square brackets, e.g. [NCT01234567].
- If the sources do not contain the answer, reply exactly: "{ABSTAIN}."
- Be short: one or two sentences."""

SYSTEM_NO_RAG = f"""You answer questions about clinical trials.
If you do not know the answer, reply exactly: "{ABSTAIN}."
Be short: one or two sentences."""


def build_messages(question: str, hits: list[Hit] | None) -> list[Message]:
    """hits=None → the no-retrieval baseline (the LLM alone)."""
    if hits is None:
        return [{"role": "system", "content": SYSTEM_NO_RAG}, {"role": "user", "content": question}]
    # The trial id is always in the source label (from metadata), so even header-less "fixed" chunks can be cited.
    sources = "\n\n".join(f"Source {i} [{h.chunk['nct_id']}]: {h.chunk['text']}" for i, h in enumerate(hits, start=1))
    user = f"Sources:\n{sources}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_RAG}, {"role": "user", "content": user}]


def citations(answer: str) -> list[str]:
    return list(dict.fromkeys(CITE_RE.findall(answer)))  # unique, in order of appearance


def is_abstain(answer: str) -> bool:
    a = answer.lower().replace("’", "'")
    return "i don't know" in a or "i do not know" in a


@dataclass
class Answer:
    text: str = ""
    citations: list[str] = field(default_factory=list)
    abstained: bool = False
    # cited ids that were not among the retrieved sources = the model made up a citation
    unsupported_citations: list[str] = field(default_factory=list)


def finalize(text: str, hits: list[Hit] | None) -> Answer:
    cites = citations(text)
    retrieved = {h.chunk["nct_id"] for h in hits or []}
    return Answer(text.strip(), cites, is_abstain(text), [c for c in cites if c not in retrieved])


def stream_answer(llm: LLM, question: str, hits: list[Hit] | None) -> Iterator[str]:
    yield from llm.stream(build_messages(question, hits))
