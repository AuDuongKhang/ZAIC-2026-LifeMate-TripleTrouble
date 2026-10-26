"""Read-only access to a split: samples, tool catalog, audio paths.

Never touches <split>_environment/ (competition rule): app data is only reachable through ToolEnv.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SYS_TOOL = "sys.ask_missing_info"


def default_data_root() -> Path:
    """Folder holding the split folders (train/, public_test/, ...); override with LIFEMATE_DATA_ROOT."""
    return Path(os.environ.get("LIFEMATE_DATA_ROOT", REPO_ROOT / "data"))


def load_samples(data_root: Path, split: str) -> list[dict]:
    path = data_root / split / f"{split}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_catalog(data_root: Path, split: str) -> dict[str, dict]:
    """{tool name: tool spec} for every tool of the split, sys.ask_missing_info included."""
    with open(data_root / split / f"{split}_tools_catalog.json", encoding="utf-8") as f:
        catalog = json.load(f)
    return {t["name"]: t for app in catalog["apps"] for t in app["tools"]}


def installed_tool_names(sample: dict) -> list[str]:
    names = [t for app in sample["context"]["installed_apps"].values() for t in app["tools"]]
    return names + [SYS_TOOL]


def fuzzy_params(catalog: dict[str, dict]) -> dict[str, set[str]]:
    """{tool name: names of its x-match: fuzzy parameters} (scored by token-F1 instead of equality)."""
    return {
        name: {p for p, s in spec["parameters"].get("properties", {}).items() if s.get("x-match") == "fuzzy"}
        for name, spec in catalog.items()
    }


def audio_path(data_root: Path, split: str, rel: str) -> Path:
    return data_root / split / rel


def import_zaic_tools(data_root: Path, split: str):
    """Import the zaic_tools shipped with `split`. Every split has its own version under the same module name,
    so one process can only work on one split."""
    loaded = sys.modules.get("zaic_tools")
    wanted = (data_root / split / "zaic_tools").resolve()
    if loaded is not None:
        if Path(loaded.__file__).resolve().parent != wanted:
            raise RuntimeError(f"zaic_tools already imported from {loaded.__file__}; use one process per split")
        return loaded
    sys.path.insert(0, str(data_root / split))
    import zaic_tools

    return zaic_tools
