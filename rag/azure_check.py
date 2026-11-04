"""Preflight after `make infra-apply`: is the keyless access working, and what does the Free tier really allow?

1. Entra ID: can the logged-in identity get a token for Azure OpenAI?
2. Azure OpenAI: one tiny embedding and one 5-token chat call (fractions of a cent) prove the RBAC role and the
   deployment names; a 401/403 here means the role assignment has not propagated yet (wait a few minutes).
3. Azure AI Search: the service's own counters show the real storage and vector quota of the Free tier, so the
   README quotes measured limits instead of a docs page.

Run:  uv run python -m rag.azure_check
"""

import json
from datetime import UTC, datetime

from rag.config import Settings, get_settings
from rag.results import merge

MB = 2**20


def counter(c, scale: float = 1) -> dict:
    """ResourceCounter → {"usage", "quota"}; quota None = no limit on this tier."""
    return {k: (round(v / scale, 2) if v is not None else None) for k, v in (("usage", c.usage), ("quota", c.quota))}


def check(s: Settings) -> dict:
    from rag.azure_auth import COGNITIVE_SCOPE, get_credential, openai_client
    from rag.azure_search import index_client

    if not (s.azure_openai_endpoint and s.azure_search_endpoint):
        raise SystemExit("AZURE_OPENAI_ENDPOINT / AZURE_SEARCH_ENDPOINT missing: run `make env-from-tf` first")
    out: dict = {"date": datetime.now(UTC).isoformat(timespec="seconds"), "region": s.azure_region}
    cred = get_credential()
    cred.get_token(COGNITIVE_SCOPE)  # raises with a clear message if `az login` is missing
    out["entra_token"] = "ok"

    client = openai_client(s.azure_openai_endpoint, max_retries=3, timeout=30)
    emb = client.embeddings.create(model=s.azure_openai_embed_deployment, input=["ping"])
    out["embedding"] = {"model": emb.model, "dimensions": len(emb.data[0].embedding)}
    chat = client.chat.completions.create(
        model=s.azure_chat_deployment, messages=[{"role": "user", "content": "Say OK."}], max_completion_tokens=5
    )
    out["chat"] = {"model": chat.model, "reply": chat.choices[0].message.content}

    c = index_client(s).get_service_statistics().counters
    out["search_free_tier"] = {
        "indexes": counter(c.index_counter),
        "documents": counter(c.document_counter),
        "storage_mb": counter(c.storage_size_counter, MB),
        "vector_index_mb": counter(c.vector_index_size_counter, MB),
    }
    merge(["azure_service"], out)
    return out


def main() -> None:
    print(json.dumps(check(get_settings()), indent=1, default=str))


if __name__ == "__main__":
    main()
