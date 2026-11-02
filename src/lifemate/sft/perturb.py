"""Schema perturbation for SFT records: shuffle each tool's property order and drop optional properties the record
never uses, so the model learns to read the schema in front of it rather than a memorised one.

Applied to function-masked copies only (`records --mask K --perturb`); gold and ASR copies keep the exact inference
text. A property counts as used when any call in the messages (history and gold), the `params` of an ask about that
tool, or a few-shot `<tool_call>` line in a system message passes it, so the training text stays schema-valid.
"""
from __future__ import annotations

import copy
import json
import random
from collections import defaultdict

from ..agents.parsing import BLOCK

SYS_TOOL = "sys.ask_missing_info"


def _used_params(record: dict) -> dict[str, set[str]]:
    calls = [c["function"] for m in record["messages"] for c in m.get("tool_calls", [])]
    for m in record["messages"]:
        if m["role"] != "system":
            continue
        for line in m["content"].split("\n"):
            if line.startswith("<tool_call>"):
                try:
                    calls.append(json.loads(BLOCK.match(line).group(1)))
                except (json.JSONDecodeError, AttributeError):
                    continue
    used: dict[str, set[str]] = defaultdict(set)
    for call in calls:
        args = call.get("arguments") or {}
        used[call.get("name")].update(args)
        if call.get("name") == SYS_TOOL and isinstance(args.get("params"), list):
            used[args.get("tool_name")].update(args["params"])
    return used


def perturb_record(record: dict, rng: random.Random, drop_p: float = 0.5) -> dict:
    """A copy of record whose tools have shuffled properties and some unused optional properties dropped."""
    used = _used_params(record)
    tools = copy.deepcopy(record["tools"])
    for tool in tools:
        fn = tool["function"]
        if fn["name"] == SYS_TOOL:
            continue
        params = fn.get("parameters") or {}
        props = params.get("properties") or {}
        required = set(params.get("required", []))
        keep = [k for k in props if k in required or k in used[fn["name"]] or rng.random() >= drop_p]
        rng.shuffle(keep)
        params["properties"] = {k: props[k] for k in keep}
    return dict(record, tools=tools, perturbed=True)
