"""Pass an entity's id where a write call names it.

The model sometimes copies an entity's name from a tool result into a write call ("program": "Cao Nguyên Rewards") where
the gold call passes the entity's id ("prog_815"). On train, gold string arguments that point at an entity seen earlier use
its id 1136 times and its name 6 times, all six a search `query` of a read. So, for a write call only: a string argument
that is not free text (`x-match: fuzzy`) and equals the name, title or label of exactly one entity seen so far (a dict
with a string `id` in a tool result, compared after strip + casefold) becomes that entity's id, unless the argument
already is the id of an entity seen so far. Plain string parameters count, not only `x-ref` ones: on train, 42 gold write
arguments carrying a seen entity's id are plain strings (`loyalty.redeem.program`, `.reward_id`), against 1 non-fuzzy
write argument whose gold value is an entity name. It changes an argument of a call the model sent and never sends a new call; no app or tool name decides anything.
Shared by the simulator (`lifemate.sim`, rewrite "id_fix") and the agent (`id_fix=True`).
"""
from __future__ import annotations

_NAME_KEYS = ("name", "title", "label")
IDS = ""  # key of every seen entity id in the entity_names map; never a name, since names are stripped and non-empty


def entity_names(obj, out: dict[str, set[str]]) -> dict[str, set[str]]:
    """Add {folded name/title/label: ids} for every dict with a string `id` inside obj, and each such id under IDS;
    returns out."""
    if isinstance(obj, dict):
        if isinstance(obj.get("id"), str):
            out.setdefault(IDS, set()).add(obj["id"])
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
        folded = value.strip().casefold()
        if not folded or value in names.get(IDS, ()):
            continue
        ids = names.get(folded, set())
        if len(ids) == 1:
            args[key] = next(iter(ids))
            changed = True
    return dict(call, arguments=args) if changed else call


def fix_calls_ids(calls: list[dict], specs: dict[str, dict], names: dict[str, set[str]]) -> list[dict]:
    """fix_call_ids over a batch; `calls` itself when no call changes."""
    fixed = [fix_call_ids(c, specs.get(c["name"]), names) for c in calls]
    return calls if all(a is b for a, b in zip(fixed, calls)) else fixed
