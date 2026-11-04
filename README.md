# 19 · Azure AI: the same RAG on Azure

🇸🇰 [Slovenská verzia](README.sk.md)

> **Status: code, infrastructure and tests are done; the Azure measurements have not been run yet.**
> The results tables below fill in from `results.json` after the runs in [Run it](#run-it). Until then this README
> contains no Azure numbers on purpose: every number here has to come from an actual run.

## TL;DR
- The clinical-trials RAG from [step 15](https://github.com/kavicka/15-rag-fullstack-clinical) runs on **two
  interchangeable stacks**, picked in `.env`, with no code change:
  - **local:** own BM25 + Qdrant, bge-small embeddings, `qwen3:4b` in Ollama;
  - **Azure:** Azure AI Search (Free tier), Azure OpenAI (`gpt-4o-mini` + `text-embedding-3-small`).
- **Keyless:** Microsoft Entra ID (`az login`) + RBAC roles from Terraform. API keys are **disabled** on both Azure
  resources, so there is no secret in `.env`, in CI or in this repo.
- **Measured side by side** on the same 141 retrieval / 156 generation questions as step 15: evidence@k,
  accuracy, grounding, abstention, time to first token, latency and **real cost per answer** (token counts from the
  provider × `pricing.json`).
- **0 €:** Azure for Students credit with a hard spending limit, Search Free tier, a budget alert in Terraform, and
  `make infra-destroy` after the runs.
- **Tested without Azure:** 48 Python tests run the real `openai` and `azure-search-documents` SDKs against fake
  HTTP transports; 7 `terraform test` runs check the plan with a mocked provider (keys off, Free SKU,
  least-privilege roles, pinned models, budget).

## Problem
Step 15 showed that retrieval turns a 1.7B model from 15.9 % to 96.8 % correct. The open question for a real team
is the platform: **is a managed cloud stack better, faster or cheaper than the one that runs on a laptop, and by
how much?** Three concrete questions:
1. **Search engine:** does Azure AI Search find the passage with the answer as well as own BM25 + Qdrant + RRF,
   when it gets exactly the same chunks and the same vectors?
2. **Embeddings:** is `text-embedding-3-small` better than the local `bge-small-en-v1.5` on these questions?
3. **LLM:** what does a cheap hosted "mini" model buy over local `qwen3:4b` in accuracy, latency and cost?

## Setup
- **Data:** the same 396 ClinicalTrials.gov trials as step 15 (`eval/study_ids.txt`), section-aware chunking,
  5,218 chunks. Questions: `eval/questions.jsonl` from step 15, unchanged.
- **Local stack:** `BAAI/bge-small-en-v1.5` through onnxruntime (step 17), Qdrant in embedded mode, own BM25
  (k1 = 1.2, b = 0.75), RRF with k = 60, cross-encoder `ms-marco-MiniLM-L-6-v2`, `qwen3:4b` in Ollama (thinking off).
- **Azure stack:** Azure AI Search **Free** (BM25 with the same k1/b, `standard.lucene` analyzer, HNSW cosine),
  Azure OpenAI `gpt-4o-mini` (2024-07-18, pinned) and `text-embedding-3-small` (shortened to 512 dimensions),
  GlobalStandard deployments, Entra ID only.
- **Generation:** temperature 0, at most 256 output tokens, top 5 chunks, the step 15 prompt.
- **Hardware:** the measurements are meant to run on the Mac (Apple Silicon), local and Azure from the same machine
  and network, so latencies are comparable. Region and machine are stored in each run's `meta`.

## How it works
```
                    ┌─ SEARCH_PROVIDER=local ──── own BM25 + Qdrant (bge) ─────────┐
question ─► get_retriever()                                                       ├─► Hits ─► prompt ─► get_llm() ─► answer
                    └─ SEARCH_PROVIDER=azure_search ─ Azure AI Search ─────────────┘          LLM_PROVIDER=ollama | azure_openai
                                                       EMBED_PROVIDER=local (vec_bge) | azure_openai (vec_te3)
```
- **One interface, two stacks** (`rag/retrieve.py`, `rag/azure_search.py`). Both retrievers return the same `Hit`
  objects for the same four modes, so the API, the React UI and the eval never know which stack runs:

  | mode | local | Azure AI Search |
  |---|---|---|
  | `bm25` | own BM25 | full-text search, BM25 with the same k1/b |
  | `dense` | bge vectors in Qdrant | the same bge vectors, or text-embedding-3-small, in HNSW |
  | `hybrid` | own RRF over both lists | Azure's built-in RRF over the text + vector query |
  | `rerank` | hybrid top 20 → cross-encoder | Azure hybrid top 20 → **the same local cross-encoder** |

  Azure's own reranker (semantic ranker) is not available on the Free tier. Using the same cross-encoder on both
  stacks also isolates what Azure's first stage finds.
- **A fair engine comparison needs care** (`rag/azure_search.py`):
  - Azure gets **the very same bge vectors** as Qdrant: `rag/index.py` saves them, `rag/azure_index.py` uploads them.
  - Only the chunk `text` is searchable. In the REST API string fields are searchable **by default**, so `title`,
    `nct_id` and `section` are set to `searchable=False` explicitly; otherwise BM25 would score other text.
  - Questions are reduced to plain words before the keyword query. Azure's simple syntax reads `"…"` as a phrase,
    `-x` as NOT and `*` as a prefix, and the questions quote trial titles and contain hyphens.
- **Keyless access** (`rag/azure_auth.py`, `infra/main.tf`):
  - Terraform sets `local_auth_enabled = false` (OpenAI) and `local_authentication_enabled = false` (Search).
  - It gives the logged-in user three roles and nothing more: *Cognitive Services OpenAI User*, *Search Service
    Contributor* (the index schema) and *Search Index Data Contributor* (documents and queries).
  - The code uses `DefaultAzureCredential`. The openai SDK gets a token provider instead of a key and talks to the
    **v1 API** (`…/openai/v1/`, no `api-version`).
- **Real cost per answer** (`rag/llm.py`, `rag/pricing.py`):
  - Azure streams with `stream_options={"include_usage": true}`; the last chunk carries the token counts and the
    exact model version.
  - `pricing.json` turns them into USD and records where each price came from. A model that is not in the file
    gets no price, never a guess.
  - Local models count as 0 USD; electricity is not included.
- **Terraform** (`infra/`): azurerm 5.x, one resource group, Azure OpenAI with two pinned deployments
  (`NoAutoUpgrade`, so the model cannot change between runs), Search Free, three role assignments and a budget
  alert. Resource names get a hash of the subscription, so they are globally unique and stable without a random
  provider.

## Results
**Pending.** Run the steps in [Run it](#run-it), then `make plot`. The tables come from `results.json`:

| Retrieval (141 questions) | evidence@1 | evidence@5 | MRR@10 | p50 latency |
|---|---|---|---|---|
| local · bm25 / dense / hybrid / rerank | – | – | – | – |
| Azure AI Search + bge · bm25 / dense / hybrid / rerank | – | – | – | – |
| Azure AI Search + text-embedding-3-small · dense / hybrid / rerank | – | – | – | – |

| Generation (156 questions) | accuracy | grounded | abstain (unanswerable) | TTFT p50 | latency p50 | USD / answer |
|---|---|---|---|---|---|---|
| A · qwen3:4b alone / + local search | – | – | – | – | – | 0 |
| B · Azure mini alone | – | – | – | – | – | – |
| C · Azure mini + local search | – | – | – | – | – | – |
| D · Azure mini + Azure hybrid (+ reranker) | – | – | – | – | – | – |

The plots (`assets/retrieval.png`, `assets/generation.png`, `assets/cost.png`) are generated by `make plot` from the
same file and will be embedded here with the numbers.

### Verification run of the local stack (this repo's code, cloud CPU)
Before any Azure run, the local stack was rebuilt from scratch with this repo's code and measured on the cloud
machine it was built on (Linux, 4 CPU cores, onnxruntime; `measurements/verify_local_cloud_cpu.json`). The
numbers are **identical to step 15/17** on all 141 questions, so the refactoring for Azure did not change local
retrieval. Latencies are cloud CPU latencies, not Mac ones.

| mode | evidence@1 | evidence@5 | MRR@10 | p50 latency |
|---|---|---|---|---|
| bm25 | 0.421 | 0.905 | 0.613 | 0.2 ms |
| dense | 0.667 | 0.968 | 0.788 | 58.4 ms |
| hybrid | 0.492 | 0.976 | 0.705 | 67.7 ms |
| rerank | 0.857 | 0.992 | 0.917 | 1141.4 ms |

## Error analysis
To be written from the runs in `eval/runs/*.jsonl`. Planned checks:
- `bm25` local vs Azure: the same BM25 formula, different tokenization (Lucene standard tokenizer vs own regex,
  no stop words on Azure). Which questions flip?
- `dense` bge in Qdrant vs bge in Azure: identical vectors, so any difference comes from the two HNSW
  implementations and their settings (Qdrant m = 16, Azure m = 4, efSearch = 500).
- Per question type (`by_kind`): where does the mini model differ from qwen3:4b?

## What I learned
- **Keyless is not harder than keys**, it only moves the work into Terraform: three role assignments instead of
  copying a key into `.env`. A leaked `.env` now contains nothing that grants access.
- **"Same search, different engine" is full of hidden defaults.** String fields searchable by default, query
  operators inside natural-language questions, a different analyzer and different HNSW parameters would each have
  skewed the comparison without an error message. Pinning them down explicitly is most of the work.
- **The cloud SDKs can be tested without the cloud.** A fake HTTP transport under the real SDK checks the URL, the
  auth header and the exact JSON body. That caught a real bug before any Azure call: model-name matching would
  have priced `gpt-4o-mini-tts` as `gpt-4o-mini`.
- **Azure for Students is not "free Azure OpenAI".** Model quota may be zero and regions can be locked by policy,
  so the run starts with a quota and region check (`make azure-preflight`).

## Run it
```bash
make sync                      # Python (onnx + azure extras) + frontend
cp .env.example .env
make data index-local          # 396 trials, local index (~10 min on 4 CPU cores)
make test                      # 48 Python tests + terraform test, no Azure needed
```
Azure, in order (on the Mac, logged in with `az login`):
```bash
make azure-preflight LOCATION=swedencentral   # account, region policy, models, quota
cp infra/terraform.tfvars.example infra/terraform.tfvars   # region, model, capacity, budget e-mail
make infra-apply               # resource group, Azure OpenAI + 2 deployments, Search Free, roles, budget
make env-from-tf               # endpoints → .env.azure (nothing secret)
make azure-check               # token, one tiny call per deployment, real Free-tier quota → results.json
make index-azure               # upload chunks + bge + text-embedding-3-small vectors
make eval-local                # retrieval + run A (needs `ollama serve`, `ollama pull qwen3:4b`)
make eval-azure-retrieval      # Azure + bge, Azure + text-embedding-3-small
make eval-azure-generation     # runs B, C, D  (D_EMBED=azure_openai if te3 retrieves better)
make plot
make infra-destroy             # nothing keeps running or billing
```
Run the app on either stack: set `SEARCH_PROVIDER` / `LLM_PROVIDER` in `.env`, then `make api` and `make web`.

## Limitations
- **Prices are not verified yet.** The official Azure price page was not reachable from the machine this was built
  on. `pricing.json` has `"verified": false` and prices from third-party trackers. Confirm them on the official
  page, then run `make rescore`.
- **Free tier, one run.** Azure AI Search Free runs on shared infrastructure without an SLA, so its latency can
  vary from day to day. Latency also includes the network round trip from the Mac to the region.
- **No semantic ranker:** not available on the Free tier, so Azure's own reranker is not measured.
- **GlobalStandard** deployments may process prompts in any Azure region. That is acceptable for public trial
  records, not for patient data (use DataZoneStandard: `deployment_sku` in `infra/terraform.tfvars`).
- The app itself is not hosted on Azure. That would be Container Apps with the images from step 17, out of scope
  here.
- **Why no Racing Lab chapter:** swapping a cloud provider behind an interface is not a concept that maps to cars.
  The visual example is the React UI, which shows which stack answered, plus the plots above.
