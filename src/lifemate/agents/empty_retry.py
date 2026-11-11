"""Re-send a read that found nothing without its optional filters.

The model sometimes narrows a search with a guessed filter (`contacts.search {"query": "Sếp Nguyễn"}` when the contact is
"Sếp Dũng", `loyalty.get_points {"program": "Cine Việt"}`) and then asks the user or stops. A gold read comes back empty in
4 of 1,545 train samples, but the agent gets an empty read in 40 of 634 public samples, and those samples average about
0.1. relaxed_call builds the retry: for a read tool only (`kind == "read"` in the spec), every argument that is not in
`parameters.required` and is not a date- or time-valued string is dropped. A date filter is kept, so a read that is empty
for a real reason (no event on that day) is not retried. Nothing to drop gives None. Only the schema decides; no app or
tool name does. The agent (`empty_retry=True`) sends the retry; this is the only rule that sends a call the model did not
write.
"""
from __future__ import annotations

import re

_DATED = re.compile(r"^(?:\d{4}-\d{2}-\d{2}|\d{2}:\d{2})")


def relaxed_call(call: dict, spec: dict | None) -> dict | None:
    """A copy of `call` without its optional, non-date arguments, or None (not a read, no spec, nothing to drop)."""
    if not spec or spec.get("kind") != "read":
        return None
    required = set((spec.get("parameters") or {}).get("required") or [])
    args = call.get("arguments") or {}
    kept = {k: v for k, v in args.items() if k in required or (isinstance(v, str) and _DATED.match(v))}
    if len(kept) == len(args):
        return None
    return dict(call, arguments=kept)
