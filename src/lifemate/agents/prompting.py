"""Turn a Task into chat messages + tool schemas in the format chat templates expect (OpenAI-style functions)."""
from __future__ import annotations

import json
from typing import Callable

from ..task import Task
from .prompts import DEFAULT_PROMPT, PROMPTS


def _clean_property(prop: dict) -> dict:
    """Drop catalog x-* keys; keep what they tell the model as description text."""
    out = {k: (_clean_property(v) if isinstance(v, dict) else v) for k, v in prop.items() if not k.startswith("x-")}
    notes = []
    if prop.get("x-ref"):
        notes.append(f"id from {prop['x-ref']}")
    if prop.get("x-match") == "fuzzy":
        notes.append("free text")
    if notes:
        out["description"] = (out.get("description", "") + f" ({', '.join(notes)})").strip()
    return out


def tool_schema(spec: dict) -> dict:
    params = spec.get("parameters") or {"type": "object", "properties": {}}
    clean = {k: v for k, v in params.items() if not k.startswith("x-") and k != "properties"}
    clean["properties"] = {name: _clean_property(p) for name, p in params.get("properties", {}).items()}
    description = spec.get("description", "")
    if params.get("x-one-of"):
        groups = " | ".join(" + ".join(group) for group in params["x-one-of"])
        description = f"{description} Provide one of: {groups}."
    return {"type": "function", "function": {"name": spec["name"], "description": description, "parameters": clean}}


def system_prompt(context: dict, version: str = DEFAULT_PROMPT) -> str:
    ctx = {k: context[k] for k in ("current_time", "timezone", "location", "user_profile") if k in context}
    return f"{PROMPTS[version]}\n\nContext: {json.dumps(ctx, ensure_ascii=False)}"


def history_messages(history: list[dict]) -> list[dict]:
    msgs = []
    for turn in history:
        if turn["role"] == "user":
            msgs.append({"role": "user", "content": turn.get("text", "")})
            continue
        calls = turn.get("tool_calls") or []
        if not calls:
            msgs.append({"role": "assistant", "content": turn.get("text", "")})
            continue
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})
        for result in turn.get("tool_results") or []:
            msgs.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
        if turn.get("text"):  # said after seeing the results
            msgs.append({"role": "assistant", "content": turn["text"]})
    return msgs


def build_messages(task: Task, count_tokens: Callable[[list[dict]], int], budget: int,
                   prompt_version: str = DEFAULT_PROMPT, examples_text: str = "") -> list[dict]:
    """[system, *history, user request]; drops the oldest history turns (a user turn and the replies after it)
    until the prompt fits the token budget. The system prompt (with examples_text, when given) and the request are
    never dropped."""
    turns: list[list[dict]] = []
    for msg in history_messages(task.history):
        if msg["role"] == "user" or not turns:
            turns.append([])
        turns[-1].append(msg)
    system = system_prompt(task.context, prompt_version) + (f"\n\n{examples_text}" if examples_text else "")
    head, tail = [{"role": "system", "content": system}], [{"role": "user", "content": task.request}]
    while True:
        msgs = head + [m for turn in turns for m in turn] + tail
        if not turns or count_tokens(msgs) <= budget:
            return msgs
        turns.pop(0)
