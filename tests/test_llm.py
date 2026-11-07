from pathlib import Path

import pytest

from conftest import replay_sample
from lifemate.agents.llm import finish_text
from lifemate.evaluate import label_kind
from lifemate.sft.tokenize import encode_record, render

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
TEMPLATE = Path(__file__).resolve().parents[1] / "configs" / "chat_templates" / "qwen_json_nothink.jinja"
EMPTY_THINK = "<think>\n\n</think>\n\n"


def test_finish_text_drops_a_call_written_inside_thinking():
    call = '<tool_call>{"name":"x","arguments":{}}</tool_call>'
    assert finish_text(f"draft {call}</think>\n\n{call}<|im_end|>", {"<|im_end|>"}, True) == call


def test_finish_text_with_cut_off_thinking_is_empty():
    assert finish_text('reasoning <tool_call>{"name":"x"', set(), True) == ""


def test_finish_text_without_thinking_is_unchanged():
    assert finish_text("Đã xong.", set(), False) == "Đã xong."


def test_finish_text_keeps_only_what_follows_the_last_close():
    assert finish_text("a</think>b</think>\n\nc<|im_end|>", {"<|im_end|>"}, False) == "c"


def test_finish_text_without_open_think_and_without_close_keeps_the_text():
    assert finish_text("  plain <|im_end|>", {"<|im_end|>"}, False) == "plain"


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except OSError:
        pytest.skip(f"{MODEL} tokenizer not in the local HF cache")
    tok.chat_template = TEMPLATE.read_text(encoding="utf-8")
    return tok


def test_json_template_prompt_has_empty_think(tok):
    prompt = tok.apply_chat_template([{"role": "user", "content": "chào"}], add_generation_prompt=True, tokenize=False)
    assert prompt.endswith("<|im_start|>assistant\n" + EMPTY_THINK)


def test_json_template_think_only_after_the_last_user_message(tok):
    call = [{"type": "function", "function": {"name": "a.b", "arguments": {"k": 1}}}]
    messages = [{"role": "user", "content": "một"}, {"role": "assistant", "content": "cũ", "tool_calls": []},
                {"role": "user", "content": "hai"}, {"role": "assistant", "content": "", "tool_calls": call},
                {"role": "tool", "content": "{}"}, {"role": "assistant", "content": "Xong.", "tool_calls": []}]
    text = tok.apply_chat_template(messages, tokenize=False)
    assert text.count("<think>") == 2  # the two turns after the last user message, not the history turn
    assert "<|im_start|>assistant\ncũ" in text
    assert "<|im_start|>assistant\n" + EMPTY_THINK + '<tool_call>\n{"name": "a.b"' in text
    assert "<|im_start|>assistant\n" + EMPTY_THINK + "Xong." in text


def test_json_template_renders_prefixes(tok, train_samples):
    s = next(x for x in train_samples
             if label_kind(x["label"]["tool_calls"]) == "act" and len(x["label"]["tool_calls"]) == 2 and not x["history"])
    record = replay_sample(s)
    enc = encode_record(record, tok)  # raises when a turn is not a prefix of the next one
    labelled = tok.decode([t for t, l in zip(enc["input_ids"], enc["labels"]) if l != -100])
    assert labelled.count("<tool_call>") == 2 and "<think>" not in labelled  # the empty think belongs to the prompt
    assert render(tok, record["messages"][:1], record["tools"], True)  # renders with tools


def test_finish_text_with_a_self_opened_think_that_never_closes_is_empty():
    """A hybrid model that writes <think> itself and is cut off: its drafted call must not reach the agent."""
    draft = '<think>\nmaybe <tool_call>{"name":"phone.call","arguments":{}}</tool_call> but'
    assert finish_text(draft, set(), False) == ""
    assert finish_text("  " + draft, {"<|im_end|>"}, False) == ""


def test_finish_text_keeps_a_closed_self_opened_think():
    call = '<tool_call>{"name":"x","arguments":{}}</tool_call>'
    assert finish_text(f"<think>\nhmm</think>\n\n{call}<|im_end|>", {"<|im_end|>"}, False) == call
