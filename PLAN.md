# Plán: krok 19 · Azure AI: ten istý RAG na Azure

Pracovný plán implementácie. Po dokončení kroku sa zmaže alebo presunie do histórie, výsledok bude v `README.md` / `README.sk.md`.

## Cieľ
Vziať RAG nad klinickými štúdiami z [kroku 15](https://github.com/kavicka/15-rag-fullstack-clinical) (v serving podobe z
[kroku 17](https://github.com/kavicka/17-containers-k8s-cicd)) a vymeniť jeho časti za Azure služby **za tým istým
rozhraním**, nastavením v `.env`, nie zmenou kódu. Potom číslami odpovedať:

1. **Vyhľadávanie:** nájde Azure AI Search (keyword / vector / hybrid) pasáž s odpoveďou rovnako dobre ako lokálny
   BM25 + Qdrant + RRF?
2. **Embeddingy:** je Azure OpenAI `text-embedding-3-small` lepší ako lokálny `bge-small-en-v1.5` na tých istých otázkach?
3. **Generovanie:** o koľko je lacný Azure „mini“ model lepší alebo horší ako lokálny `qwen3:4b` v presnosti,
   groundingu, abstain, latencii (p50/p95, čas do prvého tokenu) a **cene za odpoveď**?

Baseline = lokálny stack na tom istom Macu. Metriky sú rovnaké ako v kroku 15, takže čísla sú priamo porovnateľné.

## Rozhodnutia (odsúhlasené)
| Téma | Rozhodnutie |
|---|---|
| Podpiska | **Azure for Students** (kredit bez karty, tvrdý spending limit → reálne 0 €) |
| Čo ide do Azure | Azure OpenAI (chat + embeddingy) + Azure AI Search (Free tier). Appka beží lokálne, hosting na Azure nie je v rozsahu |
| Eval | **všetkých 156 otázok** (generovanie) a 141 otázok (vyhľadávanie), rovnaké ako krok 15 |
| Prihlásenie | **Entra ID bez kľúčov**: `az login` + `DefaultAzureCredential`, RBAC role z Terraformu, lokálne kľúče vypnuté |
| Lokálny baseline | nový beh na Macu: `qwen3:4b` v Ollame + Qdrant, aby latencie boli z toho istého stroja a siete |
| Modely | jeden lacný chat „mini“ model (napr. `gpt-4o-mini` / `gpt-4.1-mini`, podľa kvóty a regiónu) + `text-embedding-3-small` |
| Kto čo robí | Claude píše kód, Terraform, testy a README. Majiteľ spúšťa `terraform apply`, indexovanie a evaly na Macu a commitne `results.json` |
| Základ kódu | kópia `app/` z kroku 17 (má `onnx` backend pre embeddingy, bez PyTorchu) + `SOURCE.md` s commitom, ako v kroku 17 |
| Chunking | iba `section` (víťaz z kroku 15), aby sa zmestil do Free tieru a eval bol kratší |

## Riziká, ktoré treba overiť ako prvé (fáza 0)
- **Kvóta na OpenAI modely na Students podpiske** býva malá alebo nulová, niekedy treba žiadosť o navýšenie
  ([Q&A](https://learn.microsoft.com/en-ie/answers/questions/2224916/azure-ai-foundry-quota-insufficient-issues-student),
  [quotas](https://learn.microsoft.com/azure/foundry/openai/quotas-limits)).
- **Povolené regióny:** Students podpisky môžu mať politiku, ktorá zamyká regióny.
- **Azure AI Search Free:** jedna Free služba na podpisku, malé úložisko a vektorová kvóta, **bez semantic rankera**
  ([tier guide](https://learn.microsoft.com/azure/search/search-sku-tier)). Presné limity sa overia na stránke Service limits.
  Odhad: 5 218 chunkov × 384 dimenzií × 4 B ≈ 8 MB na jedno vektorové pole.
- **RBAC na Search Free tieri** (data-plane role namiesto admin kľúča): treba overiť, či funguje. Ak nie, záložné
  riešenie je admin kľúč iba v git-ignored `.env`, uvedené v Limitations.
- **Throttling Free tieru** pri 141 + 156 dotazoch: retry s backoffom v klientovi.
- Ceny a `api_version` sa zapíšu až pri implementácii, s odkazom na zdroj a dátumom. Nič sa nevymýšľa z pamäte.

## Fázy

### Fáza 0 · Prerekvizity a kontrola kvóty (majiteľ, Mac, ~30 min)
```bash
az login
az account show --query "{name:name, offer:id}"             # potvrdiť Azure for Students
az policy assignment list --query "[].displayName"          # je tam obmedzenie regiónov?
az cognitiveservices model list -l swedencentral -o table   # aké modely sú v regióne
az cognitiveservices usage list -l swedencentral -o table   # kvóta (TPM) na chat aj embedding
```
- Vybrať región (preferencia EÚ: Sweden Central / West Europe / France Central, podľa toho, čo politika a kvóta dovolia).
- **Brána:** ak je kvóta na chat model 0 → požiadať o navýšenie (Quota → Azure OpenAI). Kým nepríde, dajú sa robiť
  fázy 1–4 (vyhľadávanie a embeddingy), generovanie počká.
- Výstup: región + názvy modelov a verzií zapísané do `infra/terraform.tfvars` (necitlivé, môže byť v gite).

### Fáza 1 · Kostra repa (Claude)
- `pyproject.toml` (meno `azure-ai-rag`, Python 3.12, extras `onnx`, `azure` = `openai`, `azure-identity`,
  `azure-search-documents`), `uv.lock`, ruff.
- `rag/`, `api/`, `eval/`, `frontend/`, `tests/` skopírované z `17-containers-k8s-cicd/app` + `SOURCE.md` (commit, čo
  sa zmenilo a prečo). `.env.example` bez akýchkoľvek kľúčov.
- `Makefile`: `sync`, `lint`, `test`, `data`, `index-local`, `index-azure`, `eval-local`, `eval-azure`, `plot`,
  `infra-plan`, `infra-apply`, `infra-destroy`, `env-from-tf`.
- Testy prejdú s `LLM_PROVIDER=fake` bez siete. Commit: „Scaffold from step 17“.

### Fáza 2 · Terraform pre Azure (Claude píše, majiteľ aplikuje)
`infra/` (provider `azurerm` ~> 4.x, lokálny state, `*.tfstate` v `.gitignore`):
- `azurerm_resource_group`
- `azurerm_cognitive_account` (OpenAI / AI Services) s `local_auth_enabled = false` (bez kľúčov)
- 2 × `azurerm_cognitive_deployment`: chat mini model + `text-embedding-3-small`, s malou kapacitou (TPM) v rámci kvóty
- `azurerm_search_service` so `sku = "free"`, RBAC autentifikácia, ak ju Free tier podporuje
- `azurerm_role_assignment` pre prihláseného používateľa (`data.azurerm_client_config`):
  `Cognitive Services OpenAI User`, `Search Index Data Contributor`, `Search Service Contributor`
- rozpočtový alarm (`azurerm_consumption_budget_subscription`, napr. 5 €), ak ho Students offer podporuje.
  Inak stačí spending limit podpisky, uvedie sa to v README.
- `outputs.tf`: endpointy a názvy deploymentov → `make env-from-tf` ich zapíše do `.env` (nie sú tajné).
- CI: `terraform fmt -check` + `terraform validate` (bez prihlásenia do Azure).

### Fáza 3 · Azure adaptéry za existujúcimi rozhraniami (Claude)
- `rag/config.py`: nové nastavenia `search_provider: qdrant | azure_search`, `embed_provider: local | azure_openai`,
  `azure_search_endpoint`, `azure_search_index`, `azure_openai_chat_deployment`, `azure_openai_embed_deployment`,
  `azure_openai_api_version`. Pole `azure_openai_api_key` sa odstráni (keyless).
- `rag/llm.py` · `AzureOpenAILLM`: `DefaultAzureCredential` + `get_bearer_token_provider`, streamovanie s
  `stream_options={"include_usage": True}`, aby sa zapísali skutočné tokeny vstup / výstup. Ošetriť modely, ktoré
  chcú `max_completion_tokens` namiesto `max_tokens`.
- `rag/embed.py` · `AzureOpenAIEmbedder`: dávkovanie, `dimensions` podľa toho, čo sa zmestí do Free tieru.
- `rag/azure_search.py`: schéma indexu (`id`, `nct_id`, `section`, `text`, `vec_bge`, `vec_te3`), vytvorenie a nahratie
  po dávkach, dotazy:
  - `keyword`: full-text (Azure BM25, analyzer `en.lucene`)
  - `vector`: HNSW nad `vec_bge` alebo `vec_te3`
  - `hybrid`: keyword + vector, RRF robí sám Azure
  - `rerank`: Azure hybrid top 20 → lokálny cross-encoder (ako v kroku 15, keďže semantic ranker na Free nie je)
- `rag/retrieve.py`: `get_retriever()` vráti lokálny alebo Azure retriever s tým istým `search(q, mode, k) -> list[Hit]`,
  takže `answer.py`, API a UI sa nemenia.
- `rag/pricing.py` + `pricing.json`: cena za 1M tokenov vstup / výstup pre použité modely, s URL zdroja a dátumom.
  Lokálne modely 0 € (bez elektriny, uvedie sa to).
- Testy (bez siete): mock `SearchClient` (správne parametre dotazu pre každý režim), výpočet ceny, výber providera
  z nastavení, Azure LLM s fake streamom vrátane usage.

### Fáza 4 · Dáta a indexy (majiteľ spúšťa, ~15 min)
```bash
make data          # ten istý korpus (eval/study_ids.txt, 396 štúdií)
make index-local   # chunks + bge vektory + lokálny Qdrant
make index-azure   # vytvorí Azure index, nahrá text + bge vektory + text-embedding-3-small vektory
```
`index-azure` zapíše do `results.json` počet dokumentov, veľkosť indexu, počet tokenov a cenu embeddingov.

### Fáza 5 · Merania (majiteľ, Mac)
Všetko s rovnakými otázkami, `k = 5`, teplota 0, thinking vypnutý pri Qwen3.

**Vyhľadávanie (141 otázok, bez LLM, Search Free nestojí nič):**
| Stack | Režimy |
|---|---|
| lokálne (Qdrant + vlastný BM25, bge) | bm25, dense, hybrid, rerank |
| Azure AI Search + bge vektory | keyword, vector, hybrid, rerank |
| Azure AI Search + text-embedding-3-small | vector, hybrid, rerank |

Metriky: evidence@1/5/10, MRR@10, trial@1/5, latencia p50/p95 (pri Azure vrátane siete).

**Generovanie (156 otázok):**
| Beh | LLM | Vyhľadávanie | Prečo |
|---|---|---|---|
| A | qwen3:4b (Ollama) | lokálny rerank | lokálny baseline |
| B | Azure mini | žiadne (no RAG) | koľko vie model sám |
| C | Azure mini | lokálny rerank | izoluje vplyv LLM (rovnaké pasáže ako A) |
| D | Azure mini | najlepší Azure režim | celý stack na Azure |

Metriky: accuracy, grounded, false_abstain, abstain_unans, bad_citations, latencia p50/p95, čas do prvého tokenu,
tokeny vstup / výstup, **cena za odpoveď** a za celý beh. Odhad: rádovo centy z kreditu.

```bash
make eval-local    # retrieval + beh A
make eval-azure    # retrieval na Azure + behy B, C, D
```
Každý beh zapíše `meta` (dátum, hardvér, región, model + verzia, api_version) a uloží surové odpovede do `eval/runs/`.

### Fáza 6 · Grafy, README a vizuálny príklad (Claude, z `results.json`)
- `plot.py` → `assets/retrieval.png` (evidence@1 a @5 podľa stacku a režimu), `assets/generation.png` (presnosť vs
  latencia, veľkosť bodu = cena za odpoveď), `assets/cost.png`.
- **Vizuálny príklad:** autíčka sa nepoužijú, výmena cloudového providera nie je téma pre autá. Namiesto toho
  screenshot React UI, ktoré odpovedá cez Azure (odznak providera + skóre zdrojov z Azure AI Search), plus graf
  kvalita / latencia / cena.
- `README.md` + `README.sk.md` podľa štandardu (TL;DR · Problem · Setup · How it works · Results · Error analysis ·
  What I learned · Run it · Limitations). Analýza chýb: kde sa Azure keyword líši od vlastného BM25 (analyzer,
  stop slová), ktoré typy otázok mini model zvláda inak ako Qwen.

### Fáza 7 · Upratanie a uzavretie
- `make infra-destroy` (Free Search služba je jedna na podpisku, OpenAI deploymenty netreba nechávať bežať).
- Skutočnú spotrebu z Cost Management zapísať do `results.json` (`actual_cost_eur`) a porovnať s vypočítanou cenou.
- Index `kavicka/ai-ml-learning`: riadok 19 presunúť z „Plánované“ do hlavnej tabuľky v `README.md` aj `README.sk.md`
  a upraviť popis (celý eval namiesto „pár otázok“, appka beží lokálne).
- Skontrolovať viditeľnosť repa: `19-azure-ai-rag` je teraz verejné, AGENTS.md hovorí, že kroky sú privátne.

## Commity (jeden na hotový kus)
1. Scaffold from step 17 · 2. Terraform for Azure OpenAI + AI Search · 3. Azure adapters + tests ·
4. Azure indexing · 5. Eval: local vs Azure (results.json) · 6. Plots + README EN/SK · 7. Index row in ai-ml-learning

## Hotovo, keď
- `uv run pytest` a `ruff` prejdú, CI je zelené (lint, testy, `terraform validate`), v gite nie je žiaden kľúč.
- `results.json` obsahuje všetky behy z tabuliek vyššie, s hardvérom a regiónom v `meta`.
- README v oboch jazykoch, každé číslo pochádza z `results.json`.
- Azure zdroje sú zmazané a skutočná cena je zapísaná.

## Otvorené body (rozhodnú sa vo fáze 0)
- Konkrétny región a chat model (podľa kvóty a politiky Students).
- Či `text-embedding-3-small` pôjde na 1 536 alebo zmenšených dimenziách (podľa vektorovej kvóty Free tieru).
- Či Search Free tier dovolí RBAC bez admin kľúča.
