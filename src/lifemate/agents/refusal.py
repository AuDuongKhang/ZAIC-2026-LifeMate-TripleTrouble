"""Second look before a refusal: which installed tools could the request plausibly be about?

The score of a tool is the number of distinct word bigrams of the request that also occur as consecutive word pairs
in the tool's own text. Words are folded (lowercase, no Vietnamese diacritics, `ask_guard.fold`), so "Kết thúc phiên"
and "ket thuc phien" are the same. The request's bigrams are the distinct pairs of consecutive folded words
("ket thuc phien gui xe" -> ket thuc, thuc phien, phien gui, gui xe). The tool's text is two word sequences, its name
(split on `.` and `_`: `parking.end_session` -> parking end session) and its description; their bigrams are collected
separately, so no pair straddles the boundary between the two. A one-word request has no bigram and matches nothing.
`sys.*` tools are never candidates.
"""
from __future__ import annotations

from .ask_guard import fold


def _bigrams(words: list[str]) -> set[tuple[str, str]]:
    return set(zip(words, words[1:]))


def matching_tools(request: str, tools: list[dict], min_hits: int = 1, k: int = 2) -> list[str]:
    """Names of up to `k` tools sharing at least `min_hits` word bigrams with `request`, best first, ties by name."""
    wanted = _bigrams(fold(request))
    scored = []
    for spec in tools:
        name = spec["name"]
        if name.startswith("sys."):
            continue
        have = _bigrams(fold(name)) | _bigrams(fold(spec.get("description") or ""))
        hits = len(wanted & have)
        if hits >= min_hits:
            scored.append((-hits, name))
    return [name for _, name in sorted(scored)[:k]]
