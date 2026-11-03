"""Complete a free-text argument from the tool results the agent has seen.

The model often copies the start of a title from a result and stops ("Đọc bài logistics xanh" for the article
"Logistics xanh: kho hàng Bắc Ninh ..."). Free-text keys are scored by token-F1 > 0.6 against the gold text, which
is usually the full title, so a truncated copy fails. The only rule: if the last n >= `min_tokens` folded tokens of the
value equal the first n folded tokens of a result string that has more than n tokens, the value becomes the text before
that overlap plus the result string (its own casing and diacritics). The longest n wins, then the first string in
result order. Anything else is left alone. Shared by the simulator (`lifemate.sim`, rewrite "freetext") and the agent.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from .ask_guard import fold

_WORD = re.compile(r"[^\W_]+")


def _strings(obj, out: list[str]) -> list[str]:
    """Every string scalar inside a tool result, in order."""
    if isinstance(obj, dict):
        for v in obj.values():
            _strings(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _strings(v, out)
    elif isinstance(obj, str):
        out.append(obj)
    return out


def complete_free_text(value: str, texts: Iterable[str], min_tokens: int = 2) -> str:
    """`value` with its trailing partial copy of one of `texts` replaced by that text, or `value` unchanged."""
    text = unicodedata.normalize("NFC", value)
    words = [(tok, m.start()) for m in _WORD.finditer(text) for tok in fold(m.group())]
    folded = [w for w, _ in words]
    best: tuple[int, str] | None = None
    for t in texts:
        theirs = fold(t)
        lowest = max(min_tokens, 1, best[0] + 1 if best else 0)  # strictly longer than the best so far: first text wins ties
        for n in range(min(len(folded), len(theirs) - 1), lowest - 1, -1):
            if folded[-n:] == theirs[:n]:
                best = (n, text[:words[-n][1]] + t)
                break
    return best[1] if best else value


def complete_call(call: dict, spec: dict | None, texts: Iterable[str], min_tokens: int = 2) -> dict:
    """A copy of `call` with `complete_free_text` applied to every `x-match: fuzzy` string argument."""
    texts = list(texts)
    props = ((spec or {}).get("parameters") or {}).get("properties", {})
    args = {k: complete_free_text(v, texts, min_tokens)
            if isinstance(v, str) and (props.get(k) or {}).get("x-match") == "fuzzy" else v
            for k, v in call.get("arguments", {}).items()}
    return dict(call, arguments=args)


def freetext_rewrite(min_tokens: int = 2):
    """Simulator rewrite (see lifemate.sim): complete each call from the results of the calls before it.
    The trace does not record the history turns' results, so only this turn's results are used."""
    def rewrite(trace: dict, catalog: dict[str, dict]) -> list[dict] | None:
        seen: list[str] = []
        calls, changed = [], False
        for c in trace["calls"]:
            new = complete_call(c, catalog.get(c["name"]), seen, min_tokens)
            changed |= new["arguments"] != c["arguments"]
            calls.append({"name": new["name"], "arguments": new["arguments"]})
            _strings(c.get("result"), seen)
        return calls if changed else None
    return rewrite
