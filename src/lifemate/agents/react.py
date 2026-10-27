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


class ReActAgent(Agent):
    name = "react"

    def __init__(self, llm: ChatLLM | None = None, model: str = "Qwen/Qwen3-4B-Instruct-2507", max_steps: int = 8,
                 max_invalid: int = 2, time_budget_s: float = 20.0, min_step_s: float = 3.0,
                 clock: Callable[[], float] = time.perf_counter):
        """time_budget_s counts from task.started_at (sample start, ASR included) when the runner sets it.
        A step only starts with at least min_step_s left, and its generation is capped at the time remaining."""
        self.llm = llm if llm is not None else HFChatLLM(model)
        self.max_steps, self.max_invalid, self.clock = max_steps, max_invalid, clock
        self.time_budget_s, self.min_step_s = time_budget_s, min_step_s

    def run(self, task: Task, env) -> None:
        start = task.started_at if task.started_at is not None else self.clock()
        tools = [tool_schema(spec) for spec in task.tools]
        budget = self.llm.context_limit - self.llm.max_new_tokens
        messages = build_messages(task, lambda m: self.llm.count_tokens(m, tools), budget)
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
                result = env.call(call["name"], call["arguments"])
                done.add(key)
                if call["name"] == SYS_TOOL:
                    return  # asking the user ends the turn; later calls in the batch are dropped
                messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
            if errors:
                invalid += 1
                messages.append({"role": "user", "content": "Your tool call was not valid JSON: " + "; ".join(errors)})
            if invalid > self.max_invalid:
                return
