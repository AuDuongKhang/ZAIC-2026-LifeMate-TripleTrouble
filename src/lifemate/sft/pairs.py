"""Preference pairs from the model's own wrong decisions (design docs/superpowers/specs/2026-11-12-onpolicy-preference-design.md).

Development only: never imported by agents/, infer.py or docker_entry.py.

A step's decision is what the agent does next: write to a tool, read a tool, ask (`sys.ask_missing_info`), answer
without a call, or other (unparsable output, a tool the record does not show). A sampled step makes a pair with the gold
step when the two decisions differ in a way gold is the single right answer for (pair_type), or when both write to the
same tool and a non-fuzzy argument has another value (args_differ, type `write:args`). Reads against other reads,
read arguments and fuzzy free text never make a pair: gold is not consistent there.
"""
from __future__ import annotations

from ..data import SYS_TOOL

Decision = tuple[str, "str | None"]


def tool_kinds(catalog: dict[str, dict]) -> dict[str, str]:
    """{tool name: "read" | "write"} for every catalog tool except sys.ask_missing_info."""
    return {name: spec["kind"] for name, spec in catalog.items() if name != SYS_TOOL}


def _first_call(message: dict) -> dict | None:
    calls = message.get("tool_calls") or []
    return calls[0]["function"] if calls else None


def decision(message: dict, name_map: dict[str, str], kinds: dict[str, str]) -> Decision:
    """The decision of an assistant message (its first tool call), with masked names mapped back by name_map."""
    call = _first_call(message)
    if call is None:
        if "<tool_call>" in (message.get("content") or ""):  # a call that did not parse
            return ("other", None)
        return ("answer", None)
    name = name_map.get(call["name"], call["name"])
    if name == SYS_TOOL:
        target = (call.get("arguments") or {}).get("tool_name")
        return ("ask", name_map.get(target, target) if isinstance(target, str) else None)
    kind = kinds.get(name)
    if kind not in ("read", "write"):
        return ("other", None)
    return (kind, name)


def pair_type(gold: Decision, model: Decision) -> str | None:
    """The pair type for a model decision against the gold one, or None when they do not make a pair."""
    g, m = gold[0], model[0]
    if m == "other" or g == "other":
        return None
    if g == "write":
        if m == "write":
            return "write:other_write" if model[1] != gold[1] else None
        return f"write:{m}"
    if g == "ask":
        return None if m == "ask" else f"ask:{m}"
    if g == "answer":
        return f"answer:{m}" if m in ("write", "ask") else None
    if g == "read":
        return None if m == "read" else f"read:{m}"
    return None


def args_differ(gold: dict, model: dict, name_map: dict[str, str], fuzzy: dict[str, set[str]]) -> bool:
    """True when both messages call the same tool and a non-fuzzy argument present on both sides has another value
    (compared with != as scoring.call_matches does). The caller checks that gold is a write."""
    g, m = _first_call(gold), _first_call(model)
    if g is None or m is None:
        return False
    tool = name_map.get(g["name"], g["name"])
    if name_map.get(m["name"], m["name"]) != tool:
        return False
    model_args = {name_map.get(k, k): v for k, v in (m.get("arguments") or {}).items()}
    loose = fuzzy.get(tool, set())
    return any(k in model_args and k not in loose and model_args[k] != v
               for k, v in (g.get("arguments") or {}).items())
