"""Read native tool-call blocks (<tool_call>{"name": ..., "arguments": {...}}</tool_call>, Qwen format) from model output."""
from __future__ import annotations

import json
import re

# A block may be cut off by the end of generation (stop token or token limit), hence `|$`.
BLOCK = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL)


def parse_tool_calls(text: str) -> tuple[list[dict], list[str]]:
    """Returns (calls, errors): one error message per block that is not a valid {"name", "arguments"} object."""
    calls, errors = [], []
    for raw in BLOCK.findall(text):
        raw = raw.strip()
        try:
            obj = json.loads(raw)
            args = obj.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(obj.get("name"), str) or not isinstance(args, dict):
                raise ValueError('expected {"name": str, "arguments": object}')
        except (ValueError, AttributeError) as e:  # json.JSONDecodeError is a ValueError
            errors.append(f"{raw[:200]!r}: {e}")
            continue
        calls.append({"name": obj["name"], "arguments": args})
    return calls, errors
