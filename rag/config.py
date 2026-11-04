"""All settings come from environment variables (or a local, git-ignored `.env`).

Why: the same code runs fully local (Ollama + Qdrant) or on Azure (Azure OpenAI + Azure AI Search) with nothing
but a different environment. There are no secrets in this file or in `.env`: Azure is reached with Microsoft
Entra ID (`az login` on the Mac, a managed identity in the cloud) and API keys are disabled on the resources.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # .env = your switches (git-ignored); .env.azure = endpoints written by `make env-from-tf` (git-ignored).
        # Real environment variables win over both, so `SEARCH_PROVIDER=azure_search make …` works.
        env_file=(ROOT / ".env", ROOT / ".env.azure"),
        env_file_encoding="utf-8",
        extra="ignore",
        env_parse_none_str="none",
    )

    # data / index
    data_dir: Path = ROOT / "data"
    chunking: Literal["section", "fixed"] = "section"

    # where retrieval runs: own BM25 + Qdrant on this machine, or Azure AI Search
    search_provider: Literal["local", "azure_search"] = "local"
    qdrant_url: str | None = None  # e.g. http://localhost:6333 from docker compose; empty = embedded local mode
    qdrant_api_key: SecretStr | None = None

    # embeddings: a local bge model (onnx or torch), or an Azure OpenAI embedding deployment
    embed_provider: Literal["local", "azure_openai"] = "local"
    embed_backend: Literal["torch", "onnx"] = "onnx"  # onnx = no PyTorch (step 17), identical retrieval quality
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # the reranker always runs locally
    device: str | None = None  # torch backend only: None = mps on the Mac, cpu in the cloud

    # generation
    llm_provider: Literal["ollama", "transformers", "bedrock", "azure_openai", "fake"] = "ollama"
    llm_model: str = "qwen3:4b"  # ollama tag / HF repo id / Bedrock model id (Azure: AZURE_CHAT_DEPLOYMENT)
    llm_temperature: float | None = 0.0  # None = provider default (reasoning models accept nothing else)
    llm_max_tokens: int = 256
    ollama_url: str = "http://localhost:11434"
    aws_region: str = "eu-central-1"  # AWS credentials come from the standard AWS chain, never from here

    # Azure: endpoints and names only (terraform output → `make env-from-tf`); identity comes from Entra ID
    azure_region: str | None = None  # recorded in results.json: latency depends on the distance to the region
    azure_openai_endpoint: str | None = None  # https://<resource>.openai.azure.com
    azure_chat_deployment: str = "gpt-4o-mini"  # Azure routes by *deployment* name (infra/main.tf)
    azure_openai_embed_deployment: str = "text-embedding-3-small"
    azure_openai_embed_dimensions: int | None = None  # None = the model's native size (1536 for -3-small)
    azure_search_endpoint: str | None = None  # https://<service>.search.windows.net
    azure_search_index: str = "chunks-section"

    # API
    allowed_origins: list[str] = ["http://localhost:5173"]

    @model_validator(mode="after")
    def _check_azure(self) -> Self:
        """Fail at start-up with a clear message instead of deep inside an SDK call."""
        needs_aoai = self.llm_provider == "azure_openai" or self.embed_provider == "azure_openai"
        if needs_aoai and not self.azure_openai_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required for the azure_openai provider (make env-from-tf)")
        if self.search_provider == "azure_search" and not self.azure_search_endpoint:
            raise ValueError("AZURE_SEARCH_ENDPOINT is required for SEARCH_PROVIDER=azure_search (make env-from-tf)")
        if self.search_provider == "local" and self.embed_provider == "azure_openai":
            # the Qdrant collection holds bge vectors; Azure embeddings are compared inside Azure AI Search only
            raise ValueError("EMBED_PROVIDER=azure_openai needs SEARCH_PROVIDER=azure_search")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
