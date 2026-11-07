import pytest

from conftest import replay_sample
from lifemate.evaluate import label_kind
from lifemate.sft.replay import FINAL_TEXT
from lifemate.sft.tokenize import encode_record

MODEL = "Qwen/Qwen3-4B-Instruct-2507"


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except OSError:
        pytest.skip(f"{MODEL} tokenizer not in the local HF cache")


@pytest.fixture(scope="module")
def two_call_record(train_samples):
    s = next(x for x in train_samples
             if label_kind(x["label"]["tool_calls"]) == "act" and len(x["label"]["tool_calls"]) == 2 and not x["history"])
    return replay_sample(s)


def _labelled(enc):
    return [t for t, l in zip(enc["input_ids"], enc["labels"]) if l != -100]


def test_only_assistant_tokens_are_labelled(tok, two_call_record):
    enc = encode_record(two_call_record, tok)
    assert len(enc["input_ids"]) == len(enc["labels"])
    labelled = tok.decode(_labelled(enc))
    assert labelled.count("<tool_call>") == 2 and FINAL_TEXT in labelled
    assert "<tool_response>" not in labelled and "You are a voice assistant" not in labelled
    assert "<|im_start|>" not in labelled  # the generation prompt belongs to the input


def test_labels_end_with_end_of_turn(tok, two_call_record):
    last = _labelled(encode_record(two_call_record, tok))[-3:]
    assert tok.convert_tokens_to_ids("<|im_end|>") in last


def test_labels_equal_the_input_tokens_they_cover(tok, two_call_record):
    enc = encode_record(two_call_record, tok)
    assert all(l in (-100, t) for t, l in zip(enc["input_ids"], enc["labels"]))


def test_history_assistant_turns_are_not_labelled(tok, train_samples):
    s = next(x for x in train_samples if label_kind(x["label"]["tool_calls"]) == "act" and len(x["label"]["tool_calls"]) == 1
             and any(t["role"] == "assistant" and t.get("tool_calls") for t in x["history"]))
    labelled = tok.decode(_labelled(encode_record(replay_sample(s), tok)))
    assert labelled.count("<tool_call>") == 1


def test_too_long_record_is_skipped(tok, two_call_record):
    assert encode_record(two_call_record, tok, max_len=100) is None


class ShiftingTokenizer:
    """Renders the system prompt differently once an assistant turn exists: not prefix-stable."""

    def apply_chat_template(self, messages, tools=None, tokenize=True, add_generation_prompt=False):
        ids = [1 if any(m["role"] == "assistant" for m in messages) else 2] + [3] * len(messages)
        return ids + [4] if add_generation_prompt else ids


def test_render_that_is_not_a_prefix_raises():
    rec = {"id": "r1", "tools": [], "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                                                  {"role": "assistant", "content": "a"}]}
    with pytest.raises(ValueError, match="r1"):
        encode_record(rec, ShiftingTokenizer())


def test_the_sentence_is_labelled_before_its_call(tok, train_samples):
    s = next(x for x in train_samples if label_kind(x["label"]["tool_calls"]) == "act" and not x["history"])
    steps = [{"call": c["name"], "thought": "Cần tìm thông tin trước."} for c in s["label"]["tool_calls"]]
    labelled = tok.decode(_labelled(encode_record(replay_sample(s, thoughts=steps), tok)))
    assert labelled.count("Cần tìm thông tin trước.") == len(steps)
    assert labelled.index("Cần tìm thông tin trước.") < labelled.index("<tool_call>")
