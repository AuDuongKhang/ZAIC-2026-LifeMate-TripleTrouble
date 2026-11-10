"""Pass an entity's id where a write call names it.

The model sometimes copies an entity's name from a tool result into a write call ("program": "Cao Nguyên Rewards") where
the gold call passes the entity's id ("prog_815"). On train, gold string arguments that point at an entity seen earlier use
its id 1136 times and its name 6 times, all six a search `query` of a read. So, for a write call only: a string argument
that is not free text (`x-match: fuzzy`) and equals the name, title or label of exactly one entity seen so far (a dict
with a string `id` in a tool result, compared after strip + casefold) becomes that entity's id. It changes an argument of a call the model sent and never sends a new call; no app or tool name decides anything.
Shared by the simulator (`lifemate.sim`, rewrite "id_fix") and the agent (`id_fix=True`).
"""
from __future__ import annotations

_NAME_KEYS = ("name", "title", "label")


def entity_names(obj, out: dict[str, set[str]]) -> dict[str, set[str]]:
    """Add {folded name/title/label: ids} for every dict with a string `id` inside obj; returns out."""
    if isinstance(obj, dict):
        if isinstance(obj.get("id"), str):
            for key in _NAME_KEYS:
                if isinstance(obj.get(key), str) and obj[key].strip():
                    out.setdefault(obj[key].strip().casefold(), set()).add(obj["id"])
        for v in obj.values():
            entity_names(v, out)
    elif isinstance(obj, list):
        for v in obj:
            entity_names(v, out)
    return out


def fix_call_ids(call: dict, spec: dict | None, names: dict[str, set[str]]) -> dict:
    """`call` itself when nothing changes, else a copy whose entity names are replaced by their ids (module docstring)."""
    if not spec or spec.get("kind") != "write":
        return call
    props = (spec.get("parameters") or {}).get("properties", {})
    args = dict(call.get("arguments") or {})
    changed = False
    for key, value in args.items():
        if not isinstance(value, str) or (props.get(key) or {}).get("x-match") == "fuzzy":
            continue
        ids = names.get(value.strip().casefold(), set())
        if len(ids) == 1 and value not in ids:
            args[key] = next(iter(ids))
            changed = True
    return dict(call, arguments=args) if changed else call


def fix_calls_ids(calls: list[dict], specs: dict[str, dict], names: dict[str, set[str]]) -> list[dict]:
    """fix_call_ids over a batch; `calls` itself when no call changes."""
    fixed = [fix_call_ids(c, specs.get(c["name"]), names) for c in calls]
    return calls if all(a is b for a, b in zip(fixed, calls)) else fixed
