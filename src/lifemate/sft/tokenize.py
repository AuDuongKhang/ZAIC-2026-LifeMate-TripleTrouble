"""Token ids + labels of an SFT record, with the loss on the agent's own assistant turns only.

Each labelled turn is found by rendering the conversation up to it (with the generation prompt, as at inference) and
up to its end: the extra tokens are that turn. This needs the chat template to render earlier turns identically when a
turn is appended, which is checked on every record instead of assumed (a shifted mask would train on the wrong tokens).
"""
from __future__ import annotations

import re

MAX_LEN = 12288


def render(tokenizer, messages: list[dict], tools: list[dict], add_generation_prompt: bool = False) -> list[int]:
    ids = tokenizer.apply_chat_template(messages, tools=tools or None, tokenize=True,
                                        add_generation_prompt=add_generation_prompt)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)  # transformers 5 returns a BatchEncoding


TOOL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
NAME_VALUE = re.compile(r'"name"\s*:\s*"((?:[^"\\]|\\.)*)"')
ARGUMENTS = re.compile(r'"arguments"\s*:\s*(?=\{)')


def _top_level_keys(text: str, start: int):
    """(begin, end) of the content of every key of the JSON object that opens at text[start], at depth 1 only."""
    depth, i, expect_key = 0, start, False
    while i < len(text):
        c = text[i]
        if c == '"':
            j = i + 1
            while j < len(text) and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            if depth == 1 and expect_key:
                yield i + 1, j
            expect_key = False
            i = j + 1
            continue
        if c in "{[":
            depth += 1
            expect_key = c == "{" and depth == 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                return
        elif c == ",":
            expect_key = depth == 1
        i += 1


def decision_positions(turn_ids: list[int], tokenizer) -> set[int]:
    """Indexes in `turn_ids` (the token ids of one assistant turn) of the tokens that decide what the turn does: its first
    token (a call or an answer), the tokens of every tool name, and the tokens of every top-level argument key. Values
    and nested keys are not decisions: they are long, free-form and already dominate a mean over tokens."""
    if not turn_ids:
        return set()
    positions = {0}
    text = tokenizer.decode(turn_ids)
    spans: list[tuple[int, int]] = []
    for block in TOOL_BLOCK.finditer(text):
        base = block.start(1)
        name = NAME_VALUE.search(block.group(1))
        if name:
            spans.append((base + name.start(1), base + name.end(1)))
        arguments = ARGUMENTS.search(block.group(1))
        if arguments:
            spans += [(base + a, base + b) for a, b in _top_level_keys(block.group(1), arguments.end())]
    if not spans:
        return positions
    ends, previous = [], 0  # char offset after each token; a token that ends inside a character adds none
    for k in range(1, len(turn_ids) + 1):
        previous = max(previous, len(tokenizer.decode(turn_ids[:k]).rstrip("�")))
        ends.append(previous)
    begin = 0
    for k, end in enumerate(ends):
        if end > begin and any(begin < b and a < end for a, b in spans):
            positions.add(k)
        begin = end
    return positions


def encode_record(record: dict, tokenizer, max_len: int = MAX_LEN, decision_weight: float = 1.0) -> dict | None:
    """{"input_ids", "labels"}, labels -100 outside the assistant turns after the last user message (the turns the
    agent generates; history turns are context). None when longer than max_len: skipped by the caller, never cut.
    With decision_weight != 1 the result also has "weights", one per token: decision_weight at the decision_positions
    of each labelled turn and 1.0 elsewhere. Without it the result is as it always was."""
    messages, tools = record["messages"], record["tools"]
    full = render(tokenizer, messages, tools)
    if len(full) > max_len:
        return None
    labels = [-100] * len(full)
    turns: list[tuple[int, int]] = []
    last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
    for i in range(last_user + 1, len(messages)):
        if messages[i]["role"] != "assistant":
            continue
        before = render(tokenizer, messages[:i], tools, add_generation_prompt=True)
        upto = render(tokenizer, messages[: i + 1], tools)
        if upto[: len(before)] != before or full[: len(upto)] != upto:
            raise ValueError(f"{record['id']}: chat template does not render message {i} as a prefix of the next turn")
        labels[len(before): len(upto)] = upto[len(before):]
        turns.append((len(before), len(upto)))
    enc = {"input_ids": full, "labels": labels}
    if decision_weight != 1.0:
        weights = [1.0] * len(full)
        for start, end in turns:
            for p in decision_positions(full[start:end], tokenizer):
                weights[start + p] = float(decision_weight)
        enc["weights"] = weights
    return enc
