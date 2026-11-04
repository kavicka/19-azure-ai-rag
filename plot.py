"""Turn results.json into the plots used in the README (measurement and presentation stay separate).

assets/retrieval.png   evidence@1 / @5 per mode: local vs Azure AI Search (bge) vs Azure (text-embedding-3-small)
assets/generation.png  accuracy and median latency for every generation run (A–D), side by side
assets/cost.png        cost per 1,000 answers for every generation run

Sections that are not in results.json yet are skipped with a message, so the script works after any phase.
Run:  uv run python plot.py
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("RESULTS_PATH", ROOT / "results.json"))
ASSETS = Path(os.environ.get("ASSETS_DIR", ROOT / "assets"))

# validated categorical slots (light surface): blue, orange, aqua; fixed order = fixed meaning
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e4e3de"
STACKS = [  # results["retrieval"] tag → label, color
    ("local", "Local: own BM25 + Qdrant (bge)", "#2a78d6"),
    ("azure_bge", "Azure AI Search (bge vectors)", "#eb6834"),
    ("azure_te3", "Azure AI Search (text-embedding-3-small)", "#1baf7a"),
]
MODES = ["bm25", "dense", "hybrid", "rerank"]
MODE_LABEL = {"bm25": "keyword\n(BM25)", "dense": "vector", "hybrid": "hybrid\n(RRF)", "rerank": "hybrid +\nreranker"}
RUNS = [  # (results["generation"] tag, variant, label, LLM color, marker)
    ("A_local", "no_rag", "A · qwen3:4b alone", "#2a78d6", "o"),
    ("A_local", "rerank", "A · qwen3:4b + local search", "#2a78d6", "s"),
    ("B_azure_llm", "no_rag", "B · Azure mini alone", "#eb6834", "o"),
    ("C_azure_llm_local_search", "rerank", "C · Azure mini + local search", "#eb6834", "s"),
    ("D_azure_full", "hybrid", "D · Azure mini + Azure hybrid", "#eb6834", "^"),
    ("D_azure_full", "rerank", "D · Azure mini + Azure hybrid + reranker", "#eb6834", "D"),
]


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / name, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", ASSETS / name)


def plot_retrieval(res: dict) -> None:
    stacks = [(t, lab, c) for t, lab, c in STACKS if t in res.get("retrieval", {})]
    if not stacks:
        print("skip retrieval.png: no results['retrieval'] yet")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor=SURFACE)
    width = 0.8 / len(stacks)
    for ax, metric in zip(axes, ["evidence@1", "evidence@5"], strict=True):
        style(ax)
        for i, (tag, label, color) in enumerate(stacks):
            run = res["retrieval"][tag]
            data = run[run["meta"]["chunking"]]
            xs = [m for m in range(len(MODES)) if MODES[m] in data]
            ys = [data[MODES[m]][metric] for m in xs]
            pos = [x - 0.4 + width * (i + 0.5) for x in xs]
            ax.bar(pos, ys, width=width - 0.03, color=color, label=label, edgecolor=SURFACE, linewidth=1)
            for x, y in zip(pos, ys, strict=True):
                ax.text(x, y + 0.015, f"{y:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(range(len(MODES)), [MODE_LABEL[m] for m in MODES])
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{metric}: passage with the answer in the top {metric[-1]}", fontsize=10, color=INK, loc="left")
    axes[0].set_ylabel("share of 141 questions", color=MUTED, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=len(stacks), labelcolor=INK)
    fig.subplots_adjust(bottom=0.24)
    save(fig, "retrieval.png")


def gen_points(res: dict) -> list[tuple[str, str, str, dict]]:
    gen = res.get("generation", {})
    return [(label, color, marker, gen[tag][v]) for tag, v, label, color, marker in RUNS if v in gen.get(tag, {})]


def plot_generation(res: dict) -> None:
    """Two aligned panels with the same rows, not a scatter: the runs cluster near the top and labels would collide."""
    pts = gen_points(res)
    if not pts:
        print("skip generation.png: no results['generation'] yet")
        return
    fig, (acc, lat) = plt.subplots(1, 2, figsize=(10, 0.5 * len(pts) + 1.4), sharey=True, facecolor=SURFACE)
    rows = range(len(pts))
    colors = [c for _, c, _, _ in pts]
    for ax, key, scale, fmt, title in [
        (acc, "accuracy", 1, "{:.3f}", "Accuracy (126 answerable questions)"),
        (lat, "latency_ms_p50", 1000, "{:.1f} s", "Median time to the full answer (retrieval included)"),
    ]:
        style(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        vals = [m[key] / scale for _, _, _, m in pts]
        ax.barh(rows, vals, color=colors, height=0.6, edgecolor=SURFACE)
        for i, v in enumerate(vals):
            ax.text(v, i, "  " + fmt.format(v), va="center", fontsize=8, color=INK)
        ax.set_xlim(0, max(vals) * 1.3 if key != "accuracy" else 1.18)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    acc.set_yticks(list(rows), [lab for lab, _, _, _ in pts])
    acc.invert_yaxis()
    fig.suptitle("blue = local LLM (qwen3:4b), orange = Azure OpenAI", fontsize=9, color=MUTED, x=0.02, ha="left")
    save(fig, "generation.png")


def plot_cost(res: dict) -> None:
    pts = [(lab, c, m) for lab, c, _, m in gen_points(res)]
    if not pts:
        print("skip cost.png: no results['generation'] yet")
        return
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(pts) + 1.2), facecolor=SURFACE)
    style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    labels = [lab for lab, _, _ in pts]
    vals = [(m["cost_usd_per_answer"] or 0) * 1000 for _, _, m in pts]
    ax.barh(range(len(pts)), vals, color=[c for _, c, _ in pts], height=0.6, edgecolor=SURFACE)
    for i, (_, _, m) in enumerate(pts):
        c = m["cost_usd_per_answer"]
        text = "price unknown" if c is None else ("0 (runs locally)" if c == 0 else f"${c * 1000:.3f}")
        ax.text(vals[i], i, "  " + text, va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(pts)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals + [0.01]) * 1.35)
    ax.set_xlabel("USD per 1,000 answers (token counts from the provider × pricing.json)", color=MUTED, fontsize=9)
    ax.set_title("What an answer costs", fontsize=10, color=INK, loc="left")
    save(fig, "cost.png")


def main() -> None:
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    plot_retrieval(res)
    plot_generation(res)
    plot_cost(res)


if __name__ == "__main__":
    main()
