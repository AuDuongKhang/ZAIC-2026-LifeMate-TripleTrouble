"""Token ids + labels of an SFT record, with the loss on the agent's own assistant turns only.

Each labelled turn is found by rendering the conversation up to it (with the generation prompt, as at inference) and
up to its end: the extra tokens are that turn. This needs the chat template to render earlier turns identically when a
turn is appended, which is checked on every record instead of assumed (a shifted mask would train on the wrong tokens).
"""
from __future__ import annotations

MAX_LEN = 12288


def render(tokenizer, messages: list[dict], tools: list[dict], add_generation_prompt: bool = False) -> list[int]:
    ids = tokenizer.apply_chat_template(messages, tools=tools or None, tokenize=True,
                                        add_generation_prompt=add_generation_prompt)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)  # transformers 5 returns a BatchEncoding


def encode_record(record: dict, tokenizer, max_len: int = MAX_LEN) -> dict | None:
    """{"input_ids", "labels"}, labels -100 outside the assistant turns after the last user message (the turns the
    agent generates; history turns are context). None when longer than max_len: skipped by the caller, never cut."""
    messages, tools = record["messages"], record["tools"]
    full = render(tokenizer, messages, tools)
    if len(full) > max_len:
        return None
    labels = [-100] * len(full)
    last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
    for i in range(last_user + 1, len(messages)):
        if messages[i]["role"] != "assistant":
            continue
        before = render(tokenizer, messages[:i], tools, add_generation_prompt=True)
        upto = render(tokenizer, messages[: i + 1], tools)
        if upto[: len(before)] != before or full[: len(upto)] != upto:
            raise ValueError(f"{record['id']}: chat template does not render message {i} as a prefix of the next turn")
        labels[len(before): len(upto)] = upto[len(before):]
    return {"input_ids": full, "labels": labels}
