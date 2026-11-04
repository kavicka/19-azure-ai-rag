"""Rule-based answer checking against gold values from the structured record.

Why rules and not an LLM judge: the gold answers are short facts (a phase, a number, a sponsor name), so a
transparent string check is cheap, deterministic and easy to audit. The cost: it can't judge long free-text
answers, which is why the eval only asks short-fact questions (see Limitations in the README).
"""

import re

_ROMAN = {"iv": "4", "iii": "3", "ii": "2", "i": "1"}
_STOP = {"the", "of", "and", "for", "inc", "ltd", "llc", "co", "company", "corporation", "a", "an", "&"}
# "did not use any masking" is a correct answer for NONE; "did not specify the masking" is not (the record does).
NO_MASKING = re.compile(
    r"\bnone\b|open[- ]label|\bno masking|not masked|unmasked|not blinded|unblinded"
    r"|(not|n't) use any (masking|blinding)"
)


def _norm(s: str) -> str:
    s = s.lower().replace("’", "'")
    s = re.sub(r"phase\s*(\d)\s*/\s*(?:phase\s*)?(\d)", r"phase \1 / phase \2", s)  # "Phase 1/2" → both phases
    return re.sub(r"phase\s+(iv|iii|ii|i)\b", lambda m: "phase " + _ROMAN[m.group(1)], s)


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _STOP}


def _numbers(s: str) -> set[str]:
    return set(re.findall(r"\d+", s.replace(",", "")))


def name_match(gold: str, answer: str, min_recall: float = 0.6) -> bool:
    """A name counts as found if most of its words are in the answer, or its acronym in brackets is."""
    if m := re.search(r"\(([A-Za-z0-9-]{2,})\)", gold):
        if m.group(1).lower() in answer.lower():
            return True
    g = _tokens(re.sub(r"\([^)]*\)", "", gold))
    return bool(g) and len(g & _tokens(answer)) / len(g) >= min_recall


def is_correct(kind: str, gold, answer: str) -> bool:
    a = _norm(answer)
    match kind:
        case "phase":
            return all(re.sub(r"early ", "", p.lower()) in a for p in gold)
        case "enrollment" | "min_age":
            return str(gold) in _numbers(a)
        case "start_year":
            return str(gold) in _numbers(a)
        case "sponsor":
            return name_match(gold, answer)
        case "intervention":
            return any(name_match(g, answer) for g in gold)
        case "masking":
            if gold == "NONE":
                return bool(NO_MASKING.search(a))
            return gold.lower() in a
        case "unanswerable":
            return False  # scored as abstention instead
    raise ValueError(kind)
