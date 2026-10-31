"""ASR fine-tuning manifests (B2): one line per clip, {"audio", "text", "source", "sample"}.

Train clips come from the pool only; dev and holdout clips are never trained on and serve as the eval set.
Public datasets (VietSpeech, ...) are added later through configs/asr_mix.json.
"""
from __future__ import annotations

import random

from .splits import load_splits


def sample_rows(sample: dict) -> list[dict]:
    """The final request and every history user turn that has audio, with its gold transcript."""
    turns = [sample] + [t for t in sample["history"] if t["role"] == "user" and t.get("audio")]
    return [{"audio": t["audio"], "text": t["audio_text"], "source": "train", "sample": sample["id"]} for t in turns]


def build_manifest(samples: list[dict], ids: set[str]) -> list[dict]:
    return [row for s in samples if s["id"] in ids for row in sample_rows(s)]


def build_train_manifest(samples: list[dict], pool_ids: set[str]) -> list[dict]:
    return build_manifest(samples, pool_ids)


def build_eval_manifest(samples: list[dict], limit: int | None = None, seed: int = 0) -> list[dict]:
    """Dev + holdout clips (the clips asr_report scores), optionally a fixed random subset for in-training eval."""
    s = load_splits()
    rows = build_manifest(samples, set(s["dev"]) | set(s["holdout"]))
    if limit and limit < len(rows):
        rows = sorted(random.Random(seed).sample(rows, limit), key=lambda r: r["audio"])
    return rows
