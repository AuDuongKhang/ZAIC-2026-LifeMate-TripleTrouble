"""Render one train sample as an SFT conversation by running the real ReActAgent with an LLM that replays the gold
calls. The agent builds the prompt, validates, executes in the ToolEnv and feeds the results back exactly as at
inference, so the training text is the inference text by construction."""
from __future__ import annotations

import copy
import json

from ..agents.fewshot import ExampleBank
from ..agents.parsing import BLOCK, parse_tool_calls
from ..agents.prompts import DEFAULT_PROMPT
from ..agents.react import ReActAgent
from ..evaluate import label_kind
from ..task import Task

FINAL_TEXT = "Đã xong."  # act samples end on this after their last call
REFUSAL_TEXT = "Xin lỗi, tôi không làm được việc này."  # out-of-scope samples answer this without any call


class ReplayLLM:
    """ChatLLM that answers with the gold calls, one per step, then final_text (None: stop after the last call).
    Records (messages, tools, output) of every generate call."""

    context_limit = 16384
    max_new_tokens = 0

    def __init__(self, gold_calls: list[dict], final_text: str | None):
        self.outputs = ["<tool_call>" + json.dumps({"name": c["name"], "arguments": c["arguments"]}, ensure_ascii=False)
                        + "</tool_call>" for c in gold_calls]
        if final_text is not None:
            self.outputs.append(final_text)
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
           bank: ExampleBank | None = None, date_hints: bool = False) -> dict | None:
    """{"id", "messages", "tools"}: the agent's last prompt plus the gold answer to it, or None when the agent did not
    send exactly the gold calls (a rejected or repeated call, calls after an ask). Never a record with partial calls."""
    gold = sample["label"]["tool_calls"]
    llm = ReplayLLM(gold, final_text(gold))
    agent = ReActAgent(llm=llm, prompt_version=prompt_version, examples=examples, bank=bank, clock=lambda: 0.0,
                       date_hints=date_hints)
    agent.run(task, env)
    if [(c["name"], c["arguments"]) for c in env.get_tool_calls()] != [(c["name"], c["arguments"]) for c in gold]:
        return None
    if not llm.recorded or llm.outputs:
        return None
    messages, tools, output = llm.recorded[-1]
    return {"id": sample["id"], "messages": messages + [assistant_message(output)], "tools": tools}
