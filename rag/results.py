"""results.json: the one file every number in the README comes from. Each script merges its own section."""

import json
import os
import platform
from pathlib import Path

from rag.config import ROOT

RESULTS = Path(os.environ.get("RESULTS_PATH", ROOT / "results.json"))


def load(path: Path | None = None) -> dict:
    path = path or RESULTS
    return json.loads(path.read_text()) if path.exists() else {}


def merge(keys: list[str], data: dict, path: Path | None = None) -> None:
    """results[keys[0]][keys[1]]… = data, keeping every other section as it is."""
    path = path or RESULTS  # looked up at call time, so tests can point it at a temp file
    res = load(path)
    node = res
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = data
    path.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")


def hardware() -> str:
    """Where the numbers come from: latencies on the Mac and in the cloud are not comparable."""
    chip = platform.processor() or platform.machine()
    try:
        import torch
    except ImportError:  # the onnx stack has no PyTorch
        return f"{platform.system()} {chip}, onnxruntime cpu, {os.cpu_count()} cores"
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    return f"{platform.system()} {chip}, torch device {dev}, {os.cpu_count()} cores"
