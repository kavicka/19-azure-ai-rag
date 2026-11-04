# 19 · Azure AI: ten istý RAG na Azure

🇬🇧 [English version](README.md)

> **Stav: kód, infraštruktúra a testy sú hotové; merania na Azure ešte neprebehli.**
> Tabuľky výsledkov nižšie sa vyplnia z `results.json` po behoch zo sekcie [Spustenie](#spustenie). Dovtedy toto
> README zámerne neobsahuje žiadne Azure čísla: každé číslo tu musí pochádzať zo skutočného behu.

## TL;DR
- RAG nad klinickými štúdiami z [kroku 15](https://github.com/kavicka/15-rag-fullstack-clinical) beží na **dvoch
  zameniteľných stackoch**, ktoré sa volia v `.env`, bez zmeny kódu:
  - **lokálne:** vlastný BM25 + Qdrant, embeddingy bge-small, `qwen3:4b` v Ollame;
  - **Azure:** Azure AI Search (Free tier), Azure OpenAI (`gpt-4o-mini` + `text-embedding-3-small`).
- **Bez kľúčov:** Microsoft Entra ID (`az login`) + RBAC role z Terraformu. API kľúče sú na oboch Azure zdrojoch
  **vypnuté**, takže v `.env`, v CI ani v tomto repe nie je žiadne tajomstvo.
- **Merané vedľa seba** na tých istých 141 otázkach pre vyhľadávanie a 156 pre generovanie ako v kroku 15:
  evidence@k, presnosť, grounding, abstain, čas do prvého tokenu, latencia a **skutočná cena za odpoveď** (počty
  tokenov od poskytovateľa × `pricing.json`).
- **0 €:** kredit Azure for Students s tvrdým limitom míňania, Search Free tier, rozpočtový alarm v Terraforme a
  `make infra-destroy` po behoch.
- **Otestované bez Azure:** 48 Python testov púšťa skutočné SDK `openai` a `azure-search-documents` proti falošnému
  HTTP transportu; 7 behov `terraform test` kontroluje plán s mockovaným providerom (vypnuté kľúče, Free SKU,
  minimálne role, pripnuté modely, rozpočet).

## Problém
Krok 15 ukázal, že vyhľadávanie posunie 1,7B model z 15,9 % na 96,8 % správnych odpovedí. Otvorená otázka pre
skutočný tím je platforma: **je spravovaný cloudový stack lepší, rýchlejší alebo lacnejší ako ten, ktorý beží na
notebooku, a o koľko?** Tri konkrétne otázky:
1. **Vyhľadávač:** nájde Azure AI Search pasáž s odpoveďou rovnako dobre ako vlastný BM25 + Qdrant + RRF, keď
   dostane presne tie isté chunky a tie isté vektory?
2. **Embeddingy:** je `text-embedding-3-small` na týchto otázkach lepší ako lokálny `bge-small-en-v1.5`?
3. **LLM:** čo prinesie lacný hosťovaný „mini“ model oproti lokálnemu `qwen3:4b` v presnosti, latencii a cene?

## Setup
- **Dáta:** tých istých 396 štúdií z ClinicalTrials.gov ako v kroku 15 (`eval/study_ids.txt`), chunking podľa
  sekcií, 5 218 chunkov. Otázky: `eval/questions.jsonl` z kroku 15, bez zmeny.
- **Lokálny stack:** `BAAI/bge-small-en-v1.5` cez onnxruntime (krok 17), Qdrant v embedded režime, vlastný BM25
  (k1 = 1,2, b = 0,75), RRF s k = 60, cross-encoder `ms-marco-MiniLM-L-6-v2`, `qwen3:4b` v Ollame (bez thinkingu).
- **Azure stack:** Azure AI Search **Free** (BM25 s rovnakými k1/b, analyzer `standard.lucene`, HNSW cosine),
  Azure OpenAI `gpt-4o-mini` (2024-07-18, pripnutý) a `text-embedding-3-small` (skrátený na 512 dimenzií),
  deploymenty GlobalStandard, iba Entra ID.
- **Generovanie:** teplota 0, najviac 256 výstupných tokenov, top 5 chunkov, prompt z kroku 15.
- **Hardvér:** merania majú bežať na Macu (Apple Silicon), lokálne aj Azure z toho istého stroja a siete, aby boli
  latencie porovnateľné. Región a stroj sa ukladajú do `meta` každého behu.

## Ako to funguje
```
                    ┌─ SEARCH_PROVIDER=local ──── vlastný BM25 + Qdrant (bge) ─────┐
otázka ─► get_retriever()                                                         ├─► Hits ─► prompt ─► get_llm() ─► odpoveď
                    └─ SEARCH_PROVIDER=azure_search ─ Azure AI Search ─────────────┘          LLM_PROVIDER=ollama | azure_openai
                                                       EMBED_PROVIDER=local (vec_bge) | azure_openai (vec_te3)
```
- **Jedno rozhranie, dva stacky** (`rag/retrieve.py`, `rag/azure_search.py`). Oba retrievery vracajú tie isté
  objekty `Hit` pre tie isté štyri režimy, takže API, React UI ani eval nevedia, ktorý stack beží:

  | režim | lokálne | Azure AI Search |
  |---|---|---|
  | `bm25` | vlastný BM25 | full-text vyhľadávanie, BM25 s rovnakými k1/b |
  | `dense` | bge vektory v Qdrante | tie isté bge vektory alebo text-embedding-3-small v HNSW |
  | `hybrid` | vlastné RRF nad oboma zoznamami | vstavané RRF Azure nad textovým + vektorovým dotazom |
  | `rerank` | hybrid top 20 → cross-encoder | Azure hybrid top 20 → **ten istý lokálny cross-encoder** |

  Vlastný reranker Azure (semantic ranker) vo Free tieri nie je. Rovnaký cross-encoder na oboch stackoch navyše
  izoluje, čo nájde prvá fáza Azure.
- **Férové porovnanie vyhľadávačov chce pozornosť** (`rag/azure_search.py`):
  - Azure dostane **presne tie isté bge vektory** ako Qdrant: `rag/index.py` ich uloží, `rag/azure_index.py` nahrá.
  - Prehľadávať sa dá iba `text` chunku. V REST API sú textové polia **predvolene** prehľadávateľné, preto majú
    `title`, `nct_id` a `section` explicitne `searchable=False`; inak by BM25 skóroval iný text.
  - Otázky sa pred keyword dotazom zredukujú na obyčajné slová. Jednoduchá syntax Azure číta `"…"` ako frázu,
    `-x` ako NOT a `*` ako prefix, a otázky citujú názvy štúdií a obsahujú pomlčky.
- **Prístup bez kľúčov** (`rag/azure_auth.py`, `infra/main.tf`):
  - Terraform nastaví `local_auth_enabled = false` (OpenAI) a `local_authentication_enabled = false` (Search).
  - Prihlásenému používateľovi dá tri role a nič viac: *Cognitive Services OpenAI User*, *Search Service
    Contributor* (schéma indexu) a *Search Index Data Contributor* (dokumenty a dotazy).
  - Kód používa `DefaultAzureCredential`. SDK openai dostane namiesto kľúča poskytovateľa tokenov a komunikuje cez
    **v1 API** (`…/openai/v1/`, bez `api-version`).
- **Skutočná cena za odpoveď** (`rag/llm.py`, `rag/pricing.py`):
  - Azure streamuje so `stream_options={"include_usage": true}`; posledný chunk nesie počty tokenov a presnú
    verziu modelu.
  - `pricing.json` ich prevedie na USD a zaznamená, odkiaľ ktorá cena pochádza. Model, ktorý v súbore nie je,
    nedostane žiadnu cenu, nikdy odhad.
  - Lokálne modely sa rátajú ako 0 USD; elektrina nie je započítaná.
- **Terraform** (`infra/`): azurerm 5.x, jedna resource group, Azure OpenAI s dvoma pripnutými deploymentmi
  (`NoAutoUpgrade`, aby sa model medzi behmi nezmenil), Search Free, tri priradenia rolí a rozpočtový alarm. Mená
  zdrojov dostanú hash podpisky, takže sú globálne unikátne a stabilné aj bez random providera.

## Výsledky
**Čaká na beh.** Spusti kroky zo sekcie [Spustenie](#spustenie), potom `make plot`. Tabuľky pochádzajú z
`results.json`:

| Vyhľadávanie (141 otázok) | evidence@1 | evidence@5 | MRR@10 | p50 latencia |
|---|---|---|---|---|
| lokálne · bm25 / dense / hybrid / rerank | – | – | – | – |
| Azure AI Search + bge · bm25 / dense / hybrid / rerank | – | – | – | – |
| Azure AI Search + text-embedding-3-small · dense / hybrid / rerank | – | – | – | – |

| Generovanie (156 otázok) | presnosť | grounded | abstain (nezodpovedateľné) | TTFT p50 | latencia p50 | USD / odpoveď |
|---|---|---|---|---|---|---|
| A · qwen3:4b sám / + lokálne vyhľadávanie | – | – | – | – | – | 0 |
| B · Azure mini sám | – | – | – | – | – | – |
| C · Azure mini + lokálne vyhľadávanie | – | – | – | – | – | – |
| D · Azure mini + Azure hybrid (+ reranker) | – | – | – | – | – | – |

Grafy (`assets/retrieval.png`, `assets/generation.png`, `assets/cost.png`) vygeneruje `make plot` z toho istého
súboru a vložia sa sem spolu s číslami.

### Overovací beh lokálneho stacku (kód tohto repa, cloudové CPU)
Pred akýmkoľvek behom na Azure bol lokálny stack postavený odznova kódom tohto repa a zmeraný na cloudovom stroji,
na ktorom vznikal (Linux, 4 jadrá CPU, onnxruntime; `measurements/verify_local_cloud_cpu.json`). Čísla sú
**identické s krokom 15/17** na všetkých 141 otázkach, takže úpravy pre Azure lokálne vyhľadávanie nezmenili.
Latencie sú z cloudového CPU, nie z Macu.

| režim | evidence@1 | evidence@5 | MRR@10 | p50 latencia |
|---|---|---|---|---|
| bm25 | 0.421 | 0.905 | 0.613 | 0.2 ms |
| dense | 0.667 | 0.968 | 0.788 | 58.4 ms |
| hybrid | 0.492 | 0.976 | 0.705 | 67.7 ms |
| rerank | 0.857 | 0.992 | 0.917 | 1141.4 ms |

## Analýza chýb
Napíše sa z behov v `eval/runs/*.jsonl`. Plánované kontroly:
- `bm25` lokálne vs Azure: ten istý vzorec BM25, iná tokenizácia (štandardný Lucene tokenizer vs vlastný regex,
  bez stop slov na Azure). Ktoré otázky sa preklopia?
- `dense` bge v Qdrante vs bge v Azure: identické vektory, takže každý rozdiel pochádza z dvoch implementácií HNSW
  a ich nastavení (Qdrant m = 16, Azure m = 4, efSearch = 500).
- Podľa typu otázky (`by_kind`): kde sa mini model líši od qwen3:4b?

## Čo som sa naučil
- **Bez kľúčov to nie je ťažšie ako s kľúčmi**, len sa práca presunie do Terraformu: tri priradenia rolí namiesto
  kopírovania kľúča do `.env`. Uniknutý `.env` teraz neobsahuje nič, čo dáva prístup.
- **„Rovnaké vyhľadávanie, iný engine“ je plné skrytých predvolieb.** Textové polia prehľadávateľné predvolene,
  operátory dotazov v otázkach prirodzeným jazykom, iný analyzer a iné parametre HNSW by každé samo skreslilo
  porovnanie bez chybovej hlášky. Explicitne ich zafixovať je väčšina práce.
- **Cloudové SDK sa dajú testovať bez cloudu.** Falošný HTTP transport pod skutočným SDK skontroluje URL,
  autorizačnú hlavičku aj presné JSON telo. Ešte pred prvým volaním Azure to chytilo skutočnú chybu: párovanie mien
  modelov by ocenilo `gpt-4o-mini-tts` ako `gpt-4o-mini`.
- **Azure for Students nie je „Azure OpenAI zadarmo“.** Kvóta na modely môže byť nulová a regióny môže zamykať
  politika, preto beh začína kontrolou kvóty a regiónu (`make azure-preflight`).

## Spustenie
```bash
make sync                      # Python (extras onnx + azure) + frontend
cp .env.example .env
make data index-local          # 396 štúdií, lokálny index (~10 min na 4 jadrách CPU)
make test                      # 48 Python testov + terraform test, Azure netreba
```
Azure, v tomto poradí (na Macu, prihlásený cez `az login`):
```bash
make azure-preflight LOCATION=swedencentral   # účet, politika regiónov, modely, kvóta
cp infra/terraform.tfvars.example infra/terraform.tfvars   # región, model, kapacita, e-mail pre rozpočet
make infra-apply               # resource group, Azure OpenAI + 2 deploymenty, Search Free, role, rozpočet
make env-from-tf               # endpointy → .env.azure (nič tajné)
make azure-check               # token, jedno malé volanie na deployment, skutočná kvóta Free tieru → results.json
make index-azure               # nahrá chunky + bge + text-embedding-3-small vektory
make eval-local                # vyhľadávanie + beh A (treba `ollama serve`, `ollama pull qwen3:4b`)
make eval-azure-retrieval      # Azure + bge, Azure + text-embedding-3-small
make eval-azure-generation     # behy B, C, D  (D_EMBED=azure_openai, ak te3 vyhľadáva lepšie)
make plot
make infra-destroy             # nič nezostane bežať ani účtovať
```
Appku na ľubovoľnom stacku: nastav `SEARCH_PROVIDER` / `LLM_PROVIDER` v `.env`, potom `make api` a `make web`.

## Obmedzenia
- **Ceny zatiaľ nie sú overené.** Oficiálna stránka s cenami Azure nebola zo stroja, na ktorom to vznikalo,
  dostupná. `pricing.json` má `"verified": false` a ceny z cudzích cenníkov. Potvrď ich na oficiálnej stránke a
  potom spusti `make rescore`.
- **Free tier, jeden beh.** Azure AI Search Free beží na zdieľanej infraštruktúre bez SLA, takže jeho latencia sa
  môže zo dňa na deň meniť. Latencia obsahuje aj cestu po sieti z Macu do regiónu.
- **Bez semantic rankera:** vo Free tieri nie je, takže vlastný reranker Azure sa nemeria.
- **GlobalStandard** deploymenty môžu spracovať prompt v ľubovoľnom regióne Azure. Pre verejné záznamy štúdií je to
  v poriadku, pre dáta pacientov nie (použi DataZoneStandard: `deployment_sku` v `infra/terraform.tfvars`).
- Samotná appka nie je hosťovaná na Azure. To by boli Container Apps s obrazmi z kroku 17, mimo rozsahu.
- **Prečo nie kapitola v Racing Labe:** výmena cloudového poskytovateľa za rozhraním nie je koncept, ktorý sa dá
  ukázať na autách. Vizuálny príklad je React UI, ktoré ukazuje, ktorý stack odpovedal, plus grafy vyššie.
