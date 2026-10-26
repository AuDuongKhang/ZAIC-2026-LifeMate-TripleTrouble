"""Local re-implementation of the official metric (context/EVALUATION.md).

Per sample: gold [] -> 1 iff submitted []. Otherwise 0 if state_hash differs or nothing matches, else F1 over calls
(P = M / (M + U), R = M / |G|). A call matches a gold call when the name is equal, every non-fuzzy argument is
identical and every fuzzy (x-match) argument has token-F1 > 0.6 after lowercasing + NFC; one-to-one, order ignored.

Our assumptions where the spec is silent: argument key sets must be equal, "identical" is plain == (no case folding),
tokens are whitespace-split, and the one-to-one matching is a maximum bipartite matching.
"""
from __future__ import annotations

import unicodedata
from collections import Counter

FUZZY_THRESHOLD = 0.6


def _tokens(s) -> list[str]:
    return unicodedata.normalize("NFC", str(s).lower()).split()


def token_f1(pred, gold) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p and not g:
        return 1.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def call_matches(pred: dict, gold: dict, fuzzy: set[str]) -> bool:
    if pred.get("name") != gold.get("name"):
        return False
    pa, ga = pred.get("arguments") or {}, gold.get("arguments") or {}
    if set(pa) != set(ga):
        return False
    for key, gv in ga.items():
        if key in fuzzy:
            if token_f1(pa[key], gv) <= FUZZY_THRESHOLD:
                return False
        elif pa[key] != gv:
            return False
    return True


def count_matches(pred_calls: list[dict], gold_calls: list[dict], fuzzy_by_tool: dict[str, set[str]]) -> int:
    """Size of a maximum one-to-one matching between submitted and gold calls."""
    edges = [[j for j, g in enumerate(gold_calls) if call_matches(p, g, fuzzy_by_tool.get(g.get("name"), set()))]
             for p in pred_calls]
    owner: dict[int, int] = {}  # gold index -> pred index

    def augment(i: int, seen: set[int]) -> bool:
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    return sum(augment(i, set()) for i in range(len(pred_calls)))


def score_sample(pred_calls: list[dict], pred_hash: str | None, gold_calls: list[dict], gold_hash: str,
                 fuzzy_by_tool: dict[str, set[str]]) -> dict:
    """{"score": official per-sample score, "f1": call F1 ignoring the hash (tie-break metric), "hash_ok", "matched"}."""
    if not gold_calls:
        ok = float(not pred_calls)
        return {"score": ok, "f1": ok, "hash_ok": pred_hash == gold_hash, "matched": 0}
    m = count_matches(pred_calls, gold_calls, fuzzy_by_tool)
    if m == 0:
        f1 = 0.0
    else:
        precision, recall = m / len(pred_calls), m / len(gold_calls)
        f1 = 2 * precision * recall / (precision + recall)
    hash_ok = pred_hash == gold_hash
    return {"score": f1 if hash_ok else 0.0, "f1": f1, "hash_ok": hash_ok, "matched": m}
