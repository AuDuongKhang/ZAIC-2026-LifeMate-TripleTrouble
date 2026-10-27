from __future__ import annotations

import json
import os
import random
import subprocess
import sys

from .data import REPO_ROOT


def seed_everything(seed: int = 0) -> None:
    """Rules require fixed seeds + deterministic decoding. Call after the models are built: numpy/torch are only
    seeded when already imported, so the stdlib-only baselines never pay for importing torch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if "numpy" in sys.modules:
        sys.modules["numpy"].random.seed(seed)
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_kw(items: list[str] | None) -> dict[str, object]:
    """["key=value", ...] -> {key: value} for backend constructors. Values are JSON when they parse (6, 12.5, true),
    otherwise plain strings, so model ids need no quoting."""
    out: dict[str, object] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"expected key=value, got {item!r}")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {"commit": run("rev-parse", "--short", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}
