from __future__ import annotations

import json
import os
from pathlib import Path

from ..data import REPO_ROOT
from .base import ASR


def cache_root() -> Path:
    """Folder of transcript caches, asr_cache/<split>/<name>.jsonl; override with LIFEMATE_ASR_CACHE."""
    return Path(os.environ.get("LIFEMATE_ASR_CACHE", REPO_ROOT / "asr_cache"))


def cache_path(split: str, name: str) -> Path:
    return cache_root() / split / f"{name}.jsonl"


def read_cache(path: Path) -> dict[str, dict]:
    """{audio path as written in the sample: cache line}; later lines win."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {row["audio"]: row for row in map(json.loads, f) if row}


def sample_clips(sample: dict) -> list[str]:
    """Audio paths (as in the sample) of the final request and of every history user turn."""
    return [sample["audio"]] + [t["audio"] for t in sample["history"] if t["role"] == "user" and t.get("audio")]


class CachedASR(ASR):
    """Replays transcripts written by `python -m lifemate.transcribe`, so LLM experiments share identical ASR output
    without rerunning ASR. A clip missing from the cache is an error, never a silent ""."""

    name = "cached"

    def __init__(self, name: str, split: str | None = None):
        self.cache_name, self.split = name, split
        self._cache: dict[str, dict[str, dict]] = {}

    def transcribe(self, audio: Path, sample: dict) -> str:
        rel = next((c for c in sample_clips(sample) if Path(c).name == audio.name), None)
        if rel is None:
            raise KeyError(f"{sample['id']}: {audio.name} is not one of the sample's clips")
        split = self.split or audio.parts[-len(Path(rel).parts) - 1]
        if split not in self._cache:
            self._cache[split] = read_cache(cache_path(split, self.cache_name))
        row = self._cache[split].get(rel)
        if row is None:
            raise KeyError(f"{self.cache_name}: no cached transcript for {rel}")
        return row["text"]
