"""Training-free tool-use agent: the LLM proposes calls in its native format, env.validate gates every call
before env.call (rejected calls are fed back, never sent), results go back into the conversation."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from ..data import SYS_TOOL, default_data_root
from ..task import Task
from .base import Agent
from .fewshot import ExampleBank, render_examples
from .llm import ChatLLM, HFChatLLM
from .parsing import BLOCK, parse_tool_calls
from .prompting import build_messages, tool_schema
from .prompts import DEFAULT_PROMPT
from .refs import scalar_strings, unseen_refs
from .refusal import matching_tools

EMPTY_HINT = ("\nNo results. Try once more with other values: another date (up to 7 days away), fewer optional filters "
              "or a shorter query. If that also finds nothing, reply in one short sentence without calling any tool.")
EMPTY_REPEAT = "Already called and it returned nothing. Change the arguments as suggested, or stop."
REFUSAL_CHECK = ("Check again: the installed tool(s) {names} may handle this request. If one does, call it. If the "
                 "request is out of scope or the user cancelled it, reply in one short sentence without calling any tool.")


def is_empty_result(result) -> bool:
    """A search that found nothing: `[]`, or a dict with at least one list value where every list value is empty."""
    if isinstance(result, list):
        return not result
    lists = [v for v in result.values() if isinstance(v, list)] if isinstance(result, dict) else []
    return bool(lists) and not any(lists)


class ReActAgent(Agent):
    name = "react"

    def __init__(self, llm: ChatLLM | None = None, model: str = "Qwen/Qwen3-4B-Instruct-2507", max_steps: int = 8,
                 max_invalid: int = 2, time_budget_s: float = 20.0, min_step_s: float = 3.0,
                 prompt_version: str = DEFAULT_PROMPT, check_refs: bool = False,
                 examples: int = 0, split: str = "train", data_root: Path | None = None, bank: ExampleBank | None = None,
                 clock: Callable[[], float] = time.perf_counter, revision: str | None = None,
                 date_hints: bool = False, empty_hint: bool = False,
                 refusal_check: bool = False, refusal_min_hits: int = 1):
        """time_budget_s counts from task.started_at (sample start, ASR included) when the runner sets it.
        A step only starts with at least min_step_s left, and its generation is capped at the time remaining.
        check_refs holds back calls whose x-ref ids were never seen in a tool result (off by default: with it the
        model dodged the check, e.g. calling by phone_number instead of asking which contact).
        examples=k puts the k most similar solved training requests into the system prompt. split is the split being
        run (infer sets it): on train the bank is the experiment pool, on any other split every train sample.
        revision pins the HF model repo commit (shipped presets set it, so a later push cannot change the system).
        date_hints appends a weekday table (lifemate.agents.dates) to the system prompt.
        empty_hint adds a retry hint to the tool message after an empty read; only the message changes, never the calls.
        refusal_check: when the model's first output is a plain answer and an installed tool shares at least
        refusal_min_hits word bigrams with the request (agents/refusal.py), ask it once more before accepting the
        refusal. Never used when SFT records are rendered (ReplayLLM has no second answer, so the default is off there)."""
        self.llm = llm if llm is not None else HFChatLLM(model, revision=revision)
        self.max_steps, self.max_invalid, self.clock = max_steps, max_invalid, clock
        self.time_budget_s, self.min_step_s = time_budget_s, min_step_s
        self.prompt_version, self.check_refs, self.examples = prompt_version, check_refs, examples
        self.bank, self.date_hints, self.empty_hint = bank, date_hints, empty_hint
        self.refusal_check, self.refusal_min_hits = refusal_check, refusal_min_hits
        if self.bank is None and examples:
            self.bank = ExampleBank.from_data(data_root or default_data_root(), split)
        # Recorded in meta.json, so every RESULTS.md row traces back to its prompt and settings.
        self.info = {"model": model if llm is None else type(llm).__name__, "revision": revision,
                     "prompt": prompt_version, "check_refs": check_refs, "max_steps": max_steps, "time_budget_s": time_budget_s,
                     "examples": examples, "bank_size": len(self.bank) if self.bank is not None else 0,
                     "date_hints": date_hints, "empty_hint": empty_hint, "refusal_check": refusal_check, "refusal_min_hits": refusal_min_hits}

    def run(self, task: Task, env) -> None:
        trace = self.last_trace = {"steps": [], "stop": None}  # None: a step raised
        start = task.started_at if task.started_at is not None else self.clock()
        tools = [tool_schema(spec) for spec in task.tools]
        specs = {spec["name"]: spec for spec in task.tools}
        seen = scalar_strings([turn.get("tool_results") for turn in task.history], set())
        budget = self.llm.context_limit - self.llm.max_new_tokens
        examples_text = ""
        if self.bank is not None:
            picked = self.bank.select(task.request, set(task.context["installed_apps"]), self.examples, exclude_id=task.id)
            examples_text = render_examples(picked)
        messages = build_messages(task, lambda m: self.llm.count_tokens(m, tools), budget, self.prompt_version,
                                  examples_text, self.date_hints)
        done: set[str] = set()  # canonical (name, arguments) of calls already sent
        empty: set[str] = set()  # the subset of done whose result was empty (only tracked with empty_hint)
        invalid = 0
        checked = False  # the refusal check runs at most once, and only after the first output
        for _ in range(self.max_steps):
            remaining = self.time_budget_s - (self.clock() - start)
            if remaining < self.min_step_s:
                trace["stop"] = "time"
                return
            if self.llm.count_tokens(messages, tools) > budget:
                trace["stop"] = "tokens"
                return
            text = self.llm.generate(messages, tools, max_time=remaining)
            calls, errors = parse_tool_calls(text)
            rejected: list[str] = []
            trace["steps"].append({"output": text[:2000], "rejected": rejected})
            if not calls and not errors and self.refusal_check and not checked and len(trace["steps"]) == 1:
                checked = True
                names = matching_tools(task.request, task.tools, self.refusal_min_hits)
                if names:
                    trace["refusal_check"] = names
                    messages.append({"role": "assistant", "content": BLOCK.sub("", text).strip(), "tool_calls": []})
                    messages.append({"role": "user", "content": REFUSAL_CHECK.format(names=", ".join(names))})
                    continue
            if not calls and not errors:
                trace["stop"] = "no_call"  # a plain answer: finished, or nothing to do (out of scope)
                return
            messages.append({"role": "assistant", "content": BLOCK.sub("", text).strip(), "tool_calls": [
                {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})
            for call in calls:
                key = json.dumps([call["name"], call["arguments"]], sort_keys=True, ensure_ascii=False)
                if key in done:
                    invalid += 1
                    rejected.append(f"{call['name']}: already called")
                    messages.append({"role": "tool", "content": EMPTY_REPEAT if key in empty
                                     else "Already called; reuse the earlier result."})
                    continue
                check = env.validate(call["name"], call["arguments"])
                if not check.get("ok"):
                    invalid += 1
                    rejected.append(f"{call['name']}: {check.get('error')}")
                    messages.append({"role": "tool", "content": f"Rejected: {check.get('error')}. Fix the call."})
                    continue
                unseen = unseen_refs(specs.get(call["name"]), call["arguments"], seen) if self.check_refs else []
                if unseen:
                    invalid += 1
                    rejected.append(f"{call['name']}: unknown id {', '.join(unseen)}")
                    messages.append({"role": "tool", "content": f"Not sent: unknown id {', '.join(unseen)}. Ids must "
                                     "come from tool results; call a read or search tool to look it up first."})
                    continue
                result = env.call(call["name"], call["arguments"])
                scalar_strings(result, seen)
                done.add(key)
                if call["name"] == SYS_TOOL:
                    trace["stop"] = "asked"
                    return  # asking the user ends the turn; later calls in the batch are dropped
                content = json.dumps(result, ensure_ascii=False)
                if self.empty_hint and is_empty_result(result):
                    empty.add(key)
                    content += EMPTY_HINT
                messages.append({"role": "tool", "content": content})
            if errors:
                invalid += 1
                rejected.extend(f"invalid JSON: {e}" for e in errors)
                messages.append({"role": "user", "content": "Your tool call was not valid JSON: " + "; ".join(errors)})
            if invalid > self.max_invalid:
                trace["stop"] = "invalid"
                return
        trace["stop"] = "max_steps"
