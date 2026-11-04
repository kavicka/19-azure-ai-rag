# Where this code comes from

`rag/`, `api/`, `eval/`, `tests/test_rag.py` and `frontend/` are the RAG app from
[step 17](https://github.com/kavicka/17-containers-k8s-cicd) (`app/` at commit `d0124ee`), which is the
[step 15](https://github.com/kavicka/15-rag-fullstack-clinical) app plus an onnx embedding backend. They are copied,
not a submodule, so this repo runs on its own. Changes made here for Azure:

| File | Change | Why |
|---|---|---|
| `rag/config.py` | `SEARCH_PROVIDER`, `EMBED_PROVIDER`, Azure endpoints/names, `.env.azure`; **no API key field**; onnx is the default backend | switch stacks by config; keyless access |
| `rag/azure_auth.py` | new: one `DefaultAzureCredential`, OpenAI v1 client with an Entra ID token provider | no keys anywhere |
| `rag/llm.py` | `AzureOpenAILLM` on the v1 API, `max_completion_tokens`, optional temperature; every provider reports `last_usage` (tokens + model version) | real cost per answer |
| `rag/embed.py` | `AzureOpenAIEmbedder` (token counting, optional shortened `dimensions`); every embedder declares its `query_prefix` | bge needs a query instruction, OpenAI embeddings do not |
| `rag/azure_search.py` | new: index schema + the four retrieval modes on Azure AI Search | same `Hit`s as the local retriever |
| `rag/azure_index.py` | new: upload chunks with the local bge vectors and cached text-embedding-3-small vectors | identical vectors on both stacks |
| `rag/azure_check.py` | new: preflight (token, one tiny call per deployment, Free-tier quota counters) | fail fast, measure the real limits |
| `rag/index.py` | also saves `vectors_bge.npy` | reused by `rag/azure_index.py` |
| `rag/retrieve.py` | `get_retriever()` factory, `name`, `n_chunks`, `close()` | API / eval don't care which stack runs |
| `rag/pricing.py`, `pricing.json`, `rag/results.py` | new | traceable prices, tagged sections in results.json |
| `eval/run_eval.py` | `--tag`, time to first token, tokens, cost, error capture; runs saved as `eval/runs/<tag>__<variant>.jsonl` | local and Azure runs side by side |
| `api/main.py`, `frontend/src/*` | health shows the search stack | the UI says which stack answered |

Unchanged: chunking, ingest, prompts (`rag/answer.py`), scoring rules (`rag/scoring.py`), the question set.
