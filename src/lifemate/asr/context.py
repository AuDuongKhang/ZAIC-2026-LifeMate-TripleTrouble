"""Context prompt for Whisper: words the speaker is likely to say, taken only from what is known before the request.

Sources are the user profile and, from the history, the assistant's text and the tool results. audio_text, label and
user turns are never read: on train they hold the transcripts being predicted, so using them would fake a lower WER.
"""
from __future__ import annotations

import re

_OPAQUE = re.compile(r"^\d{4}-\d{2}-\d{2}|@")  # dates and emails: nothing to bias the decoder towards


def _useful(value: str) -> bool:
    """Spoken words, not ids or enum values: alm_01, confirmed and direct stay out, Minh and họp sớm go in."""
    text = value.strip()
    if not text or _OPAQUE.search(text) or not any(c.isalpha() for c in text):
        return False
    return " " in text or text[0].isupper() or not text.isascii()


def _strings(value) -> list[str]:
    """String values of a tool result, in order; dict keys are schema names, not speech."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        return [s for item in value for s in _strings(item)]
    return []


def context_prompt(sample: dict) -> str:
    """Comma-joined context, oldest first. A repeated phrase keeps its latest place, because a long prompt is cut from
    the start and the latest turns matter most."""
    parts = _strings((sample.get("context") or {}).get("user_profile"))
    for turn in sample.get("history") or []:
        if turn.get("role") == "assistant":
            parts += _strings(turn.get("tool_results"))
            parts += _strings(turn.get("text"))
    kept = [p.strip() for p in parts if _useful(p)]
    return ", ".join(reversed(list(dict.fromkeys(reversed(kept)))))
