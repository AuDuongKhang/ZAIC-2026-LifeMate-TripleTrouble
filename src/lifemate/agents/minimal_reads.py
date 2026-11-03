"""Drop optional filters on read calls that the user never said.

A read with fewer filters returns a superset, and the gold calls usually carry only what the request names. The model
often adds a guessed filter (an enum code such as "dishwasher" or a keyword that is not in the request). Rule: for a call
to a tool whose spec `kind` is "read", every optional string argument (enum values included, they are often English codes
the user never says) whose folded value tokens are not all among the folded tokens of what the user said is dropped.
Required arguments, non-string values and every argument of non-read tools are kept. Shared by the simulator
(`lifemate.sim`, rewrite "minimal_reads") and, once it passes its gate, the agent.
"""
from __future__ import annotations

from .ask_guard import fold


def drop_unstated_optionals(call: dict, spec: dict | None, said: str) -> dict:
    """A copy of `call` without its unstated optional string arguments (only for `kind == "read"` tools)."""
    if not spec or spec.get("kind") != "read":
        return call
    params = spec.get("parameters") or {}
    required, words = set(params.get("required") or []), set(fold(said))
    args = {k: v for k, v in call.get("arguments", {}).items()
            if k in required or not isinstance(v, str) or all(t in words for t in fold(v))}
    return dict(call, arguments=args)


def minimal_reads_rewrite():
    """Simulator rewrite (see lifemate.sim). The trace does not record the history turns, so `said` is the request."""
    def rewrite(trace: dict, catalog: dict[str, dict]) -> list[dict] | None:
        calls, changed = [], False
        for c in trace["calls"]:
            new = drop_unstated_optionals(c, catalog.get(c["name"]), trace.get("request", ""))
            changed |= new["arguments"] != c["arguments"]
            calls.append({"name": new["name"], "arguments": new["arguments"]})
        return calls if changed else None
    return rewrite
