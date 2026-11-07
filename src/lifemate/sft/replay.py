"""Render one train sample as an SFT conversation by running the real ReActAgent with an LLM that replays the gold
calls. The agent builds the prompt, validates, executes in the ToolEnv and feeds the results back exactly as at
inference, so the training text is the inference text by construction."""
from __future__ import annotations

import copy
import json
from typing import Callable

from ..agents.fewshot import ExampleBank
from ..agents.parsing import BLOCK, parse_tool_calls
from ..agents.prompts import DEFAULT_PROMPT, FINAL_TEXT, REFUSAL_TEXT
from ..agents.react import ReActAgent
from ..evaluate import label_kind
from ..task import Task


class ReplayLLM:
    """ChatLLM that answers with the gold calls, one per step, then final_text (None: stop after the last call).
    With `thoughts` (one sentence per step), each call is preceded by its sentence on its own line; for a sample with
    no gold calls the single sentence goes before final_text (the refusal). The closing text never gets one.
    Records (messages, tools, output) of every generate call."""

    context_limit = 16384
    max_new_tokens = 0

    def __init__(self, gold_calls: list[dict], final_text: str | None, thoughts: list[str] | None = None):
        self.outputs = ["<tool_call>" + json.dumps({"name": c["name"], "arguments": c["arguments"]}, ensure_ascii=False)
                        + "</tool_call>" for c in gold_calls]
        if final_text is not None:
            self.outputs.append(final_text)
        if thoughts is not None:
            for i, thought in enumerate(thoughts):
                self.outputs[i] = f"{thought}\n{self.outputs[i]}"
        self.recorded: list[tuple[list[dict], list[dict], str]] = []

    def generate(self, messages: list[dict], tools: list[dict], max_time: float | None = None) -> str:
        out = self.outputs.pop(0) if self.outputs else ""
        self.recorded.append((copy.deepcopy(messages), tools, out))
        return out

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return 1  # no history trimming: records that are too long are skipped at tokenization, never cut


def final_text(gold_calls: list[dict]) -> str | None:
    kind = label_kind(gold_calls)
    return {"act": FINAL_TEXT, "empty": REFUSAL_TEXT}.get(kind)  # ask samples end on the ask call


def assistant_message(output: str) -> dict:
    """The assistant turn for one model output, in the shape ReActAgent appends to the conversation."""
    calls, _ = parse_tool_calls(output)
    if not calls:
        return {"role": "assistant", "content": output}
    return {"role": "assistant", "content": BLOCK.sub("", output).strip(), "tool_calls": [
        {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]}


def replay(sample: dict, task: Task, env, prompt_version: str = DEFAULT_PROMPT, examples: int = 0,
           bank: ExampleBank | None = None, date_hints: bool = False, thoughts: list[dict] | None = None,
           leave_gold_apps_out: bool = False, fresh_env: Callable | None = None) -> dict | None:
    """{"id", "messages", "tools"}: the agent's last prompt plus the gold answer to it, or None when the agent did not
    send exactly the gold calls (a rejected or repeated call, calls after an ask). Never a record with partial calls.
    `thoughts` is a thoughts row's steps: one {"call", "thought"} per gold call ([None] for no gold calls). The calls
    must match the gold names, else ValueError naming the sample.
    leave_gold_apps_out leaves the apps of the gold calls out of the few-shot examples (the task needs its label)."""
    gold = sample["label"]["tool_calls"]
    if thoughts is not None and [s["call"] for s in thoughts] != ([c["name"] for c in gold] or [None]):
        raise ValueError(f"{sample['id']}: thought steps {[s['call'] for s in thoughts]} do not match the gold calls "
                         f"{[c['name'] for c in gold]}")
    expected = gold
    prefix = sample.get("prefix_calls") or []
    if prefix:  # recovery: the model's own earlier calls come first, then the gold calls they did not already make
        if fresh_env is None:
            raise ValueError("a sample with prefix_calls needs fresh_env, a way to build the gold replay's environment")
        made = {(c["name"], json.dumps(c["arguments"], sort_keys=True)) for c in prefix}
        expected = prefix + [c for c in gold if (c["name"], json.dumps(c["arguments"], sort_keys=True)) not in made]
    llm = ReplayLLM(expected, final_text(gold), None if thoughts is None else [s["thought"] for s in thoughts])
    agent = ReActAgent(llm=llm, prompt_version=prompt_version, examples=examples, bank=bank, clock=lambda: 0.0,
                       date_hints=date_hints, leave_gold_apps_out=leave_gold_apps_out)
    agent.run(task, env)
    if [(c["name"], c["arguments"]) for c in env.get_tool_calls()] != [(c["name"], c["arguments"]) for c in expected]:
        return None
    if prefix:  # the prefix must leave the app exactly where the gold calls leave it
        reference = fresh_env()
        for c in gold:
            reference.call(c["name"], c["arguments"])
        if env.get_state_hash() != reference.get_state_hash():
            return None
    if not llm.recorded or llm.outputs:
        return None
    messages, tools, output = llm.recorded[-1]
    record = {"id": sample["id"], "messages": messages + [assistant_message(output)], "tools": tools}
    if prefix:
        record["context_turns"] = len(prefix)  # these agent turns are context, never labelled (sft.tokenize)
    return record
