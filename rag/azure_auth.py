"""Keyless access to Azure: one Entra ID credential for Azure OpenAI and Azure AI Search.

Why no API keys: a key is a long-lived secret that ends up in `.env` files, shell history and CI logs. With
Entra ID the Mac uses the `az login` session, a cloud workload uses its managed identity, and what each one may
do is an RBAC role assignment in Terraform (infra/main.tf). The resources have local (key) auth switched off,
so a leaked key could not exist in the first place.
"""

from collections.abc import Callable
from functools import lru_cache

COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"  # Azure OpenAI data plane


@lru_cache
def get_credential():
    from azure.identity import DefaultAzureCredential  # uv sync --extra azure

    # Tries environment → workload/managed identity → Azure CLI (`az login`) → … in that order.
    return DefaultAzureCredential(exclude_interactive_browser_credential=True)


def openai_token_provider(credential=None) -> Callable[[], str]:
    """A callable returning a fresh bearer token; the openai SDK calls it before every request."""
    from azure.identity import get_bearer_token_provider

    return get_bearer_token_provider(credential or get_credential(), COGNITIVE_SCOPE)


def openai_client(endpoint: str, credential=None, **kwargs):
    """OpenAI SDK client for the Azure OpenAI v1 API (no api-version juggling), authenticated with Entra ID."""
    from openai import OpenAI

    base_url = endpoint.rstrip("/") + "/openai/v1/"
    return OpenAI(base_url=base_url, api_key=openai_token_provider(credential), **kwargs)
