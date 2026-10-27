"""Word error rate of ASR transcripts against train audio_text (NFC, lowercase, punctuation stripped)."""
from __future__ import annotations

import unicodedata


def normalize(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    return text.split()


def wer(ref: str, hyp: str) -> float:
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))  # word-level Levenshtein, one row at a time
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)


def mean_wer(traces: list[dict], samples: dict[str, dict]) -> float | None:
    """Mean final-turn WER over trace rows that have a transcript; None when there are none."""
    scores = [wer(samples[t["id"]]["audio_text"], t["request"]) for t in traces if t.get("request") is not None]
    return round(sum(scores) / len(scores), 4) if scores else None
