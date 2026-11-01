"""Which ids a conversation has actually shown: scalars in tool results, and x-ref arguments that never appeared."""
from __future__ import annotations


def scalar_strings(obj, out: set[str]) -> set[str]:
    """Every scalar inside obj, as a string: the ids and values the agent has actually seen."""
    if isinstance(obj, dict):
        for v in obj.values():
            scalar_strings(v, out)
    elif isinstance(obj, list):
        for v in obj:
            scalar_strings(v, out)
    elif obj is not None:
        out.add(str(obj))
    return out


def unseen_refs(spec: dict | None, arguments: dict, seen: set[str]) -> list[str]:
    """'param=value' for each x-ref argument (an id into another app's data) whose value never appeared in a tool
    result. On train, 1277 of 1278 gold x-ref values appear in an earlier result, so these are invented ids."""
    props = ((spec or {}).get("parameters") or {}).get("properties", {})
    bad = []
    for name, value in arguments.items():
        prop = props.get(name, {})
        if not (prop.get("x-ref") or (prop.get("items") or {}).get("x-ref")):
            continue
        for v in value if isinstance(value, list) else [value]:
            if str(v) not in seen:
                bad.append(f"{name}={v!r}")
    return bad
