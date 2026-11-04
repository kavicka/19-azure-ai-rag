# AGENTS.md

Guidance for coding agents (Claude Code, Codex, Cursor, …) working in this repository.

## What this repo is
A ~28-week AI/ML engineering roadmap (setup & math → ML fundamentals → transformers → AI engineering → serving & MLOps → capstone). **Every step is its own GitHub repository** under `kavicka` (public or private, see *Public step repos*), named like its folder (`kavicka/01-setup-benchmark`, `kavicka/02-calculus-chain-rule`, …). The repo `kavicka/ai-ml-learning` is only the index: root READMEs with links to the step repos, plus this file. Each step doubles as a portfolio piece.

The owner is learning. Explain concepts when asked. In projects that build things **from scratch to learn them** (P1 autograd/makemore, P4 GPT from scratch) the owner writes the code: explain, review and give hints, but do not write the solution unless explicitly asked.

## Layout
On the Mac all step repos are cloned side by side in `~/programming/ai-ml-learning/`:
```
ai-ml-learning/              # kavicka/ai-ml-learning: index only
├── README.md / README.sk.md # table of all steps with links to their repos
├── AGENTS.md, CLAUDE.md
└── NN-short-name/           # each one a separate repo kavicka/NN-short-name
    ├── README.md            # English write-up (main)
    ├── README.sk.md         # Slovak write-up
    ├── AGENTS.md, CLAUDE.md # copy of these rules
    ├── .gitignore
    ├── pyproject.toml       # own uv project
    ├── uv.lock
    ├── .python-version      # 3.12
    ├── *.py                 # scripts, run from the folder
    ├── results.json         # raw numbers the plots and README are built from
    └── assets/              # plots, screenshots, GIFs referenced by the READMEs
```
Links between steps (e.g. to a Racing Lab chapter) are absolute GitHub URLs, never `../`. Racing Lab chapter 04 imports `micrograd` from `04-autograd-makemore` cloned next to it.

## Environment
- Hardware: Mac with Apple Silicon. Use PyTorch with `device="mps"` when `torch.backends.mps.is_available()`, fall back to CPU. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for missing ops. CUDA-only tools (vLLM, bitsandbytes, flash-attention) run on Colab/Kaggle, not locally.
- Local LLMs: Ollama and MLX / mlx-lm.
- Budget is 0 €: prefer local or free-tier tools; never add paid APIs without asking.
- Every project is its own `uv` project. Inside a project folder:
  ```bash
  uv sync                    # create .venv from uv.lock
  uv add <package>           # add a dependency (never pip install)
  uv run python <script>.py  # run anything
  ```
- Lint/format with `ruff` (`uv run ruff check . && uv run ruff format .`).

## Adding a new step
1. Create `NN-short-name/` with the next two-digit number, from the template repo `kavicka/ml-project-template` (README EN/SK skeleton, uv + ruff, Makefile, CI) or with `uv init --python 3.12`. It becomes its own repo `kavicka/NN-short-name` (one commit per finished piece of work, pushed when done). It may be public; check *Public step repos* before switching it.
2. Keep measurement and presentation separate: scripts write raw numbers to `results.json`, a plotting script turns them into PNGs in `assets/`.
3. Write `README.md` (English) and `README.sk.md` (Slovak) with the same content, each linking to the other on the first line under the title (`🇸🇰 [Slovenská verzia](README.sk.md)` / `🇬🇧 [English version](README.md)`).
4. Add a row to the project table in both index READMEs of `kavicka/ai-ml-learning` (`README.md` and `README.sk.md`).
5. **Visual example for every step (only where it makes sense):** every step gets one hands-on visual example of its main concept. If the concept maps naturally to the cars, add a Racing Lab chapter in the `05-racing-lab` repo (`chapters/`) (live pygame window + GIF in `assets/`) and a "🏎️ On the cars" / "🏎️ Na autíčkach" section in both step READMEs. If it does not (e.g. RAG, LLM fine-tuning, agents), do NOT force the cars: build a different visual example that fits the topic inside the step itself, and say in one line why the cars were not used.

## Public step repos
Step repos may be public: they are portfolio pieces. Before a repo is made public, its **whole git history** must be free of the following (and every later push to a public repo must not add any of it):
- secrets: API keys, tokens, passwords, connection strings, private keys, cloud credentials, real subscription/tenant/account ids;
- local config and state: `.env` (only `.env.example` is committed), `*.tfvars` (only `*.tfvars.example`), `*.tfstate`, cloud CLI config;
- personal data: real e-mail addresses (use `you@example.com` in examples; commits use the GitHub noreply address), and any non-public or patient data (only public datasets);
- datasets, model weights, `.venv/`, `mlruns/` (see `.gitignore`).

Prefer designs where nothing in the repo grants access (e.g. keyless Entra ID / IAM roles instead of API keys). CI in a public repo uses no secrets, runs with `permissions: contents: read`, and never uses `pull_request_target`. If something sensitive was ever committed, making the repo private is not enough: rotate the secret and rewrite the history before going public.

## README standard (both languages)
Sections, in this order, skipping ones that don't apply:
`TL;DR` (key numbers) · `Problem` · `Setup` · `How it works` (code walkthrough explaining *why*, not just *what*) · `Results` (table + plots from `assets/`) · `Error analysis` · `What I learned` · `Run it` · `Limitations`.

Rules:
- Every project has a baseline, a metric and a results table. "Without a number it's not done."
- Numbers in the README must come from `results.json` / an actual run. Never invent or round up results. If a run was on different hardware (e.g. CPU instead of MPS), say so.
- English is the main README; the Slovak one is a full translation, not a summary.

## Code style
- Python 3.12, type hints where they help, small functions, short comments explaining *why*.
- Seed randomness and log hyperparameters so runs are reproducible.
- Don't commit `.venv/`, datasets, model weights, or `mlruns/` (see `.gitignore`).
