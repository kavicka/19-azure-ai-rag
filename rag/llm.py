"""One small interface in front of every LLM provider, chosen with LLM_PROVIDER in the environment.

Why: the RAG logic (retrieve → prompt → stream → check citations) must not care whether the tokens come
from Ollama on the Mac, AWS Bedrock (step 18) or Azure OpenAI (this step). Switching is a config change,
not a code change. Credentials are never parameters here: AWS uses its standard credential chain,
Azure uses Entra ID (rag/azure_auth.py).

Every provider that can report token counts puts them in `last_usage` after a stream ends, so the eval can
compute a real cost per answer instead of guessing from character counts.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import httpx

from rag.config import Settings, get_settings

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    model: str  # the model that actually answered, e.g. "gpt-4o-mini-2024-07-18" (for pricing + the README)


class LLM(Protocol):
    name: str
    last_usage: Usage | None

    def stream(self, messages: list[Message]) -> Iterator[str]: ...


def complete(llm: LLM, messages: list[Message]) -> str:
    return "".join(llm.stream(messages))


class OllamaLLM:
    """Local models through Ollama's HTTP API (the default: 0 €, data never leaves the machine)."""

    def __init__(self, s: Settings):
        self.s, self.name, self.last_usage = s, f"ollama:{s.llm_model}", None

    def stream(self, messages: list[Message]) -> Iterator[str]:
        import json

        self.last_usage = None
        body = {
            "model": self.s.llm_model,
            "messages": messages,
            "stream": True,
            "think": False,  # Qwen3 would otherwise spend hundreds of tokens "thinking" before answering
            "options": {"temperature": self.s.llm_temperature or 0.0, "num_predict": self.s.llm_max_tokens, "seed": 0},
        }
        with httpx.stream("POST", f"{self.s.ollama_url}/api/chat", json=body, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                part = json.loads(line)
                if tok := part.get("message", {}).get("content"):
                    yield tok
                if part.get("done"):  # the last line carries the token counts
                    self.last_usage = Usage(
                        part.get("prompt_eval_count", 0), part.get("eval_count", 0), part.get("model", "")
                    )


class TransformersLLM:
    """A Hugging Face model run in-process. Used where Ollama is not available (cloud CPU, Colab)."""

    def __init__(self, s: Settings):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.s, self.name, self.last_usage = s, f"transformers:{s.llm_model}", None
        torch.manual_seed(0)
        self.tok = AutoTokenizer.from_pretrained(s.llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(s.llm_model, torch_dtype="auto")
        self.model.eval()

    def stream(self, messages: list[Message]) -> Iterator[str]:
        from threading import Thread

        from transformers import TextIteratorStreamer

        kwargs = {"enable_thinking": False} if "qwen3" in self.s.llm_model.lower() else {}
        ids = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True, **kwargs
        )
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        gen = {**ids, "max_new_tokens": self.s.llm_max_tokens, "do_sample": False, "streamer": streamer}
        Thread(target=self.model.generate, kwargs=gen, daemon=True).start()
        yield from streamer


class BedrockLLM:
    """AWS Bedrock Converse API. Prepared for step 18; not run in this step."""

    def __init__(self, s: Settings):
        import boto3  # uv sync --extra bedrock

        self.s, self.name, self.last_usage = s, f"bedrock:{s.llm_model}", None
        self.client = boto3.client("bedrock-runtime", region_name=s.aws_region)  # credentials: AWS default chain

    def stream(self, messages: list[Message]) -> Iterator[str]:
        system = [{"text": m["content"]} for m in messages if m["role"] == "system"]
        msgs = [{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages if m["role"] != "system"]
        resp = self.client.converse_stream(
            modelId=self.s.llm_model,
            system=system,
            messages=msgs,
            inferenceConfig={"temperature": self.s.llm_temperature, "maxTokens": self.s.llm_max_tokens},
        )
        for ev in resp["stream"]:
            if delta := ev.get("contentBlockDelta", {}).get("delta", {}).get("text"):
                yield delta


class AzureOpenAILLM:
    """Azure OpenAI chat completions over the v1 API, authenticated with Entra ID (no API key).

    AZURE_CHAT_DEPLOYMENT is the *deployment* name (infra/main.tf), not the model name: Azure routes by deployment.
    The model that really answered (name + version) comes back in every response and is stored with the usage.
    """

    def __init__(self, s: Settings, client=None):
        from rag.azure_auth import openai_client

        self.s, self.name, self.last_usage = s, f"azure_openai:{s.azure_chat_deployment}", None
        # max_retries: Azure answers 429 when the deployment's tokens-per-minute quota is used up; the SDK waits
        # for the Retry-After header and tries again instead of failing the eval run.
        self.client = client or openai_client(s.azure_openai_endpoint, max_retries=8, timeout=60)

    def stream(self, messages: list[Message]) -> Iterator[str]:
        self.last_usage = None
        # reasoning models (gpt-5, o-series) reject any temperature but the default: LLM_TEMPERATURE=none omits it
        extra = {} if self.s.llm_temperature is None else {"temperature": self.s.llm_temperature}
        resp = self.client.chat.completions.create(
            model=self.s.azure_chat_deployment,
            messages=messages,
            # max_completion_tokens replaced max_tokens; reasoning models reject max_tokens with a 400
            max_completion_tokens=self.s.llm_max_tokens,
            stream=True,
            stream_options={"include_usage": True},  # one extra final chunk with the token counts
            **extra,
        )
        for ev in resp:
            if ev.choices and (tok := ev.choices[0].delta.content):
                yield tok
            if ev.usage:  # the final chunk has no choices, only usage
                self.last_usage = Usage(ev.usage.prompt_tokens, ev.usage.completion_tokens, ev.model or "")


class FakeLLM:
    """Deterministic stand-in for tests: answers with the first line of the first source and cites it."""

    name = "fake"
    last_usage = None

    def stream(self, messages: list[Message]) -> Iterator[str]:
        import re

        user = messages[-1]["content"]
        m = re.search(r"\[(NCT\d{8})\][^\n]*\n([^\n]+)", user)
        answer = f"{m.group(2)} [{m.group(1)}]" if m else "I don't know."
        for i, word in enumerate(answer.split(" ")):
            yield word if i == 0 else " " + word


PROVIDERS = {
    "ollama": OllamaLLM,
    "transformers": TransformersLLM,
    "bedrock": BedrockLLM,
    "azure_openai": AzureOpenAILLM,
}


def get_llm(s: Settings | None = None) -> LLM:
    s = s or get_settings()
    return FakeLLM() if s.llm_provider == "fake" else PROVIDERS[s.llm_provider](s)
