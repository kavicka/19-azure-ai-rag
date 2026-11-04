# Step 19 · the step 15 RAG, local vs Azure. The order of the targets below is the order of a full run (README → Run it).
SHELL := /bin/bash
.PHONY: sync lint test data index-local azure-preflight infra-init infra-plan infra-apply infra-destroy env-from-tf \
        azure-check index-azure eval-local eval-azure-retrieval eval-azure-generation rescore plot api web

LOCATION ?= swedencentral    # region to check in azure-preflight (must match infra/terraform.tfvars)
D_EMBED  ?= local            # embeddings for the all-Azure run D: local (bge) or azure_openai (text-embedding-3-small)
TF := terraform -chdir=infra
# the subscription for Terraform comes from the Azure CLI login, never from a file in the repo
ARM := ARM_SUBSCRIPTION_ID=$$(az account show --query id -o tsv)

sync:
	uv sync --extra onnx --extra azure
	cd frontend && npm ci

lint:
	uv run ruff check . && uv run ruff format --check .
	$(TF) fmt -check -recursive

test:
	uv run pytest -q
	$(TF) init -backend=false -input=false >/dev/null && $(TF) test

# --- local stack -------------------------------------------------------------------------------------------------
data:             ## download the frozen list of 396 trials (eval/questions.jsonl is committed)
	uv run python -m rag.ingest --from-ids

index-local:      ## chunks + bge vectors → embedded Qdrant + data/index/section/vectors_bge.npy
	uv run python -m rag.index --chunking section

# --- phase 0: what does this subscription allow? ------------------------------------------------------------------
azure-preflight:
	az account show --query "{subscription:name, offer:id, user:user.name}" -o table
	@echo "--- policies (look for 'Allowed resource deployment regions'):"
	az policy assignment list --query "[].displayName" -o tsv
	@echo "--- chat/embedding models in $(LOCATION):"
	az cognitiveservices model list -l $(LOCATION) --query "[?model.name=='gpt-4o-mini' || model.name=='gpt-4.1-mini' || model.name=='text-embedding-3-small'].{model:model.name, version:model.version, skus:join(',', model.skus[].name)}" -o table
	@echo "--- quota (tokens per minute, in thousands) in $(LOCATION):"
	az cognitiveservices usage list -l $(LOCATION) --query "[?contains(name.value, 'gpt-4o-mini') || contains(name.value, 'text-embedding-3-small')].{name:name.value, used:currentValue, limit:limit}" -o table

# --- Azure resources (Terraform) ---------------------------------------------------------------------------------
infra-init:
	$(TF) init -input=false

infra-plan: infra-init
	$(ARM) $(TF) plan

infra-apply: infra-init
	$(ARM) $(TF) apply

infra-destroy:
	$(ARM) $(TF) destroy

env-from-tf:      ## endpoints + deployment names (nothing secret) → .env.azure
	$(TF) output -raw dotenv > .env.azure && cat .env.azure

azure-check:      ## keyless access works? + the Free tier's real storage / vector quota → results.json
	uv run python -m rag.azure_check

index-azure:      ## upload chunks + bge vectors + text-embedding-3-small vectors to Azure AI Search
	uv run python -m rag.azure_index

# --- measurements --------------------------------------------------------------------------------------------------
eval-local:       ## needs `ollama serve` with qwen3:4b pulled
	uv run python eval/run_eval.py --retrieval --tag local
	LLM_PROVIDER=ollama LLM_MODEL=qwen3:4b uv run python eval/run_eval.py --generation --tag A_local --variants no_rag rerank

eval-azure-retrieval:
	SEARCH_PROVIDER=azure_search EMBED_PROVIDER=local uv run python eval/run_eval.py --retrieval --tag azure_bge
	SEARCH_PROVIDER=azure_search EMBED_PROVIDER=azure_openai uv run python eval/run_eval.py --retrieval --tag azure_te3 --modes dense hybrid rerank

eval-azure-generation:
	LLM_PROVIDER=azure_openai uv run python eval/run_eval.py --generation --tag B_azure_llm --variants no_rag
	LLM_PROVIDER=azure_openai SEARCH_PROVIDER=local uv run python eval/run_eval.py --generation --tag C_azure_llm_local_search --variants rerank
	LLM_PROVIDER=azure_openai SEARCH_PROVIDER=azure_search EMBED_PROVIDER=$(D_EMBED) uv run python eval/run_eval.py --generation --tag D_azure_full --variants hybrid rerank

rescore:          ## re-apply scoring rules and prices to stored answers (no LLM calls)
	uv run python eval/run_eval.py --rescore

plot:
	uv run python plot.py

# --- the app -------------------------------------------------------------------------------------------------------
api:              ## SEARCH_PROVIDER / LLM_PROVIDER from .env decide which stack answers
	uv run uvicorn api.main:app --reload --port 8000

web:
	cd frontend && npm run dev
