# AGENTS.md

Guidance for coding agents (Claude Code, Codex, Cursor, …) working in this repository.

## What this repo is
A learning monorepo that follows a ~28-week AI/ML engineering roadmap (setup & math → ML fundamentals → transformers → AI engineering → serving & MLOps → capstone). Each numbered folder is one self-contained step, and each step doubles as a portfolio piece.

The owner is learning. Explain concepts when asked. In projects that build things **from scratch to learn them** (P1 autograd/makemore, P4 GPT from scratch) the owner writes the code: explain, review and give hints, but do not write the solution unless explicitly asked.

## Layout
```
ai-ml-learning/
├── README.md            # English index of all projects (main)
├── README.sk.md         # Slovak index
├── AGENTS.md            # this file
├── CLAUDE.md            # points to AGENTS.md
└── NN-short-name/       # one folder per step, two-digit prefix: 01-, 02-, …
    ├── README.md        # English write-up (main)
    ├── README.sk.md     # Slovak write-up
    ├── pyproject.toml   # own uv project
    ├── uv.lock
    ├── .python-version  # 3.12
    ├── *.py             # scripts, run from the folder
    ├── results.json     # raw numbers the plots and README are built from
    └── assets/          # plots, screenshots, GIFs referenced by the READMEs
```

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
1. Create `NN-short-name/` with the next two-digit number and `uv init --python 3.12` inside it.
2. Keep measurement and presentation separate: scripts write raw numbers to `results.json`, a plotting script turns them into PNGs in `assets/`.
3. Write `README.md` (English) and `README.sk.md` (Slovak) with the same content, each linking to the other on the first line under the title (`🇸🇰 [Slovenská verzia](README.sk.md)` / `🇬🇧 [English version](README.md)`).
4. Add a row to the project table in both root READMEs (`README.md` and `README.sk.md`).
5. **Visual example for every step (only where it makes sense):** every step gets one hands-on visual example of its main concept. If the concept maps naturally to the cars, add a Racing Lab chapter in `05-racing-lab/chapters/` (live pygame window + GIF in `assets/`) and a "🏎️ On the cars" / "🏎️ Na autíčkach" section in both step READMEs. If it does not (e.g. RAG, LLM fine-tuning, agents), do NOT force the cars: build a different visual example that fits the topic inside the step itself, and say in one line why the cars were not used.

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
