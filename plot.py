"""Turn results.json into the plots used in the README (measurement and presentation stay separate)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

R = json.loads(Path("results.json").read_text())
Path("assets").mkdir(exist_ok=True)
# the colours used across all steps
BLUE, RED, GREEN, GRAY, ORANGE, DARK = "#2563eb", "#dc2626", "#16a34a", "#94a3b8", "#ea580c", "#1e293b"


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(f"assets/{name}", dpi=150)
    plt.close(fig)


def main() -> None:
    fig, a = plt.subplots(figsize=(6, 4))
    a.bar(["baseline"], [R["baseline"]], color=GRAY)
    a.set(ylabel="metric", title="Results")
    a.grid(alpha=0.3, axis="y")
    save(fig, "results.png")


if __name__ == "__main__":
    main()
