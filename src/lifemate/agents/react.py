"""Training-free tool-use agent: the LLM proposes calls in its native format, env.validate gates every call
before env.call (rejected calls are fed back, never sent), results go back into the conversation."""
from __future__ import annotations

import json
import time
from typing import Callable

from ..data import SYS_TOOL
from ..task import Task
from .base import Agent
from .llm import ChatLLM, HFChatLLM
from .parsing import BLOCK, parse_tool_calls
from .prompting import build_messages, tool_schema
from .prompts import DEFAULT_PROMPT


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


class ReActAgent(Agent):
    name = "react"

    def __init__(self, llm: ChatLLM | None = None, model: str = "Qwen/Qwen3-4B-Instruct-2507", max_steps: int = 8,
                 max_invalid: int = 2, time_budget_s: float = 20.0, min_step_s: float = 3.0,
                 prompt_version: str = DEFAULT_PROMPT, check_refs: bool = False,
                 clock: Callable[[], float] = time.perf_counter):
        """time_budget_s counts from task.started_at (sample start, ASR included) when the runner sets it.
        A step only starts with at least min_step_s left, and its generation is capped at the time remaining.
        check_refs holds back calls whose x-ref ids were never seen in a tool result (off by default: with it the
        model dodged the check, e.g. calling by phone_number instead of asking which contact)."""
        self.llm = llm if llm is not None else HFChatLLM(model)
        self.max_steps, self.max_invalid, self.clock = max_steps, max_invalid, clock
        self.time_budget_s, self.min_step_s = time_budget_s, min_step_s
        self.prompt_version, self.check_refs = prompt_version, check_refs
        # Recorded in meta.json, so every RESULTS.md row traces back to its prompt and settings.
        self.info = {"model": model if llm is None else type(llm).__name__, "prompt": prompt_version,
                     "check_refs": check_refs, "max_steps": max_steps, "time_budget_s": time_budget_s}

    def run(self, task: Task, env) -> None:
        start = task.started_at if task.started_at is not None else self.clock()
        tools = [tool_schema(spec) for spec in task.tools]
        specs = {spec["name"]: spec for spec in task.tools}
        seen = scalar_strings([turn.get("tool_results") for turn in task.history], set())
        budget = self.llm.context_limit - self.llm.max_new_tokens
        messages = build_messages(task, lambda m: self.llm.count_tokens(m, tools), budget, self.prompt_version)
        done: set[str] = set()  # canonical (name, arguments) of calls already sent
        invalid = 0
        for _ in range(self.max_steps):
            remaining = self.time_budget_s - (self.clock() - start)
            if remaining < self.min_step_s or self.llm.count_tokens(messages, tools) > budget:
                return
            text = self.llm.generate(messages, tools, max_time=remaining)
            calls, errors = parse_tool_calls(text)
            if not calls and not errors:
                return  # a plain answer: finished, or nothing to do (out of scope)
            messages.append({"role": "assistant", "content": BLOCK.sub("", text).strip(), "tool_calls": [
                {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})
            for call in calls:
                key = json.dumps([call["name"], call["arguments"]], sort_keys=True, ensure_ascii=False)
                if key in done:
                    invalid += 1
                    messages.append({"role": "tool", "content": "Already called; reuse the earlier result."})
                    continue
                check = env.validate(call["name"], call["arguments"])
                if not check.get("ok"):
                    invalid += 1
                    messages.append({"role": "tool", "content": f"Rejected: {check.get('error')}. Fix the call."})
                    continue
                unseen = unseen_refs(specs.get(call["name"]), call["arguments"], seen) if self.check_refs else []
                if unseen:
                    invalid += 1
                    messages.append({"role": "tool", "content": f"Not sent: unknown id {', '.join(unseen)}. Ids must "
                                     "come from tool results; call a read or search tool to look it up first."})
                    continue
                result = env.call(call["name"], call["arguments"])
                scalar_strings(result, seen)
                done.add(key)
                if call["name"] == SYS_TOOL:
                    return  # asking the user ends the turn; later calls in the batch are dropped
                messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
            if errors:
                invalid += 1
                messages.append({"role": "user", "content": "Your tool call was not valid JSON: " + "; ".join(errors)})
            if invalid > self.max_invalid:
                return
