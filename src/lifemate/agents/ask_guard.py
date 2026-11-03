"""Ask guard: send `sys.ask_missing_info` instead of a write whose id was picked from several candidates that the
request does not tell apart. Shared by the simulator (`lifemate.guard_sim`) and, once it passes its gate, the agent.

A candidate list is a list of at least two dicts with an `id` inside a tool result. An item's match score is the
largest share of the words of one of its string fields (except `id`) that also occur in the request, after folding
case and Vietnamese diacritics. The pick stands when it scores at least `min_score` and beats every other candidate
by at least `margin`; otherwise the guard asks for that parameter.
"""
from __future__ import annotations

import re
import unicodedata

SYS_TOOL = "sys.ask_missing_info"
_WORD = re.compile(r"[a-z0-9]+")


def fold(text: str) -> list[str]:
    """Lowercase words without diacritics: "Đường Lò Đúc" -> ["duong", "lo", "duc"]."""
    text = unicodedata.normalize("NFD", text.lower().replace("đ", "d").replace("Đ", "d"))
    return _WORD.findall("".join(ch for ch in text if not unicodedata.combining(ch)))


def match_score(request: str, item: dict) -> float:
    words = set(fold(request))
    best = 0.0
    for key, value in item.items():
        if key == "id" or not isinstance(value, str):
            continue
        tokens = fold(value)
        if tokens:
            best = max(best, sum(t in words for t in tokens) / len(tokens))
    return best


def candidate_lists(result, out: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Add {id: its list} for every list of >= 2 dicts with an `id` anywhere inside a tool result."""
    if isinstance(result, dict):
        for v in result.values():
            candidate_lists(v, out)
    elif isinstance(result, list):
        if len(result) >= 2 and all(isinstance(x, dict) and "id" in x for x in result):
            for x in result:
                out[str(x["id"])] = result
        for v in result:
            candidate_lists(v, out)
    return out


def ask_instead(request: str, call: dict, spec: dict | None, candidates: dict[str, list[dict]],
                min_score: float = 0.5, margin: float = 0.2) -> dict | None:
    """The ask to send instead of `call`, or None when the call may go ahead."""
    if not spec or spec.get("kind") != "write":
        return None
    props = (spec.get("parameters") or {}).get("properties", {})
    for param, value in call.get("arguments", {}).items():
        if not (props.get(param) or {}).get("x-ref") or not isinstance(value, str) or value not in candidates:
            continue
        items = candidates[value]
        chosen = next(x for x in items if str(x["id"]) == value)
        score = match_score(request, chosen)
        others = max((match_score(request, x) for x in items if x is not chosen), default=0.0)
        if score < min_score or score - others < margin:
            return {"name": SYS_TOOL, "arguments": {"tool_name": call["name"], "params": [param]}}
    return None
