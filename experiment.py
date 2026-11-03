"""NN · Step title.

What is measured and why, in 2–4 lines. Every number in the README comes from results.json written here.

Run:  uv run python experiment.py   → results.json
      uv run python plot.py         → assets/*.png
"""

import json
from pathlib import Path

import numpy as np

SEED = 0
rng = np.random.default_rng(SEED)


def baseline(x: np.ndarray) -> float:
    """The simplest thing that could work. Every step needs one: without a number it's not done."""
    return float(x.mean())


def main() -> None:
    x = rng.normal(size=1000)
    res = {"seed": SEED, "baseline": baseline(x)}
    print(json.dumps(res, indent=1))
    Path("results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
