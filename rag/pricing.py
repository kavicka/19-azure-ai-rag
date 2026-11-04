"""Token counts → USD, from the price list in pricing.json (prices copied from the Azure pricing page, with the date).

Why a file and not constants: prices change, and every number in the README has to be traceable. pricing.json
says where each price came from and when it was read. A model that is not in the file gets no price (None),
never a guess.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from rag.config import ROOT

PRICING = ROOT / "pricing.json"


@lru_cache
def load_prices(path: Path = PRICING) -> dict:
    return json.loads(path.read_text())


def price_for(model: str, prices: dict | None = None) -> dict | None:
    """Prices for a model as Azure reports it ("gpt-4o-mini-2024-07-18" → the "gpt-4o-mini" entry).

    Only the exact name or the name + a version date matches: "gpt-4.1-mini-2025-04-14" must not get the
    "gpt-4.1" price, and "gpt-4o-mini-tts" (another model) must not get the "gpt-4o-mini" one.
    """
    table = (prices or load_prices())["usd_per_1m_tokens"]
    m = model.lower()
    for name in table:
        if m == name or re.fullmatch(re.escape(name) + r"-\d{4}-\d{2}-\d{2}", m):
            return table[name]
    return None


def cost_usd(model: str, input_tokens: int, output_tokens: int = 0, prices: dict | None = None) -> float | None:
    p = price_for(model, prices)
    if p is None:
        return None
    return (input_tokens * p["input"] + output_tokens * p.get("output", 0.0)) / 1_000_000
