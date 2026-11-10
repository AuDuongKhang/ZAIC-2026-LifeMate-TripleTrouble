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


# --- end of turn ------------------------------------------------------------------------------------------------------

def test_finish_text_cuts_at_the_end_of_turn():
    """A model whose generate() did not stop at <|im_end|> writes made-up turns after it: they must never reach the agent."""
    a = '<tool_call>\n{"name": "x.get", "arguments": {}}\n</tool_call>'
    b = '<tool_call>\n{"name": "bank.transfer", "arguments": {"to": "1"}}\n</tool_call>'
    run_on = f"{a}<|im_end|>\n<|im_start|>user\n{b}<|im_end|>\n<|im_start|>user\nĐã xong."
    assert finish_text(run_on, {"<|im_end|>", "<|endoftext|>"}, False) == a
    assert finish_text("Đã xong.<|endoftext|>rác", {"<|im_end|>", "<|endoftext|>"}, False) == "Đã xong."


def test_finish_text_of_a_turn_that_ends_once_is_unchanged():
    call = '<tool_call>{"name":"x","arguments":{}}</tool_call>'
    assert finish_text(f"{call}<|im_end|>", {"<|im_end|>", "<|endoftext|>"}, False) == call


class _Tok:
    def __init__(self, vocab, eos_token):
        self.vocab, self.eos_token = vocab, eos_token

    def get_vocab(self):
        return dict(self.vocab)


class _GenConfig:
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id


def test_stop_token_ids_add_the_end_of_turn_token():
    from lifemate.agents.llm import stop_token_ids
    tok = _Tok({"<|im_end|>": 7, "<|endoftext|>": 5, "a": 1}, eos_token="<|endoftext|>")
    assert stop_token_ids(tok, _GenConfig(5)) == [5, 7]  # a converted checkpoint that only stops at <|endoftext|>
    assert stop_token_ids(tok, _GenConfig(None)) == [5, 7]


def test_stop_token_ids_of_a_model_that_already_stops_are_unchanged():
    from lifemate.agents.llm import stop_token_ids
    tok = _Tok({"<|im_end|>": 151645, "<|endoftext|>": 151643}, eos_token="<|im_end|>")
    assert stop_token_ids(tok, _GenConfig([151645, 151643])) == [151643, 151645]  # Qwen3-4B-Instruct-2507


def test_stop_token_ids_skip_tokens_missing_from_the_vocab():
    from lifemate.agents.llm import stop_token_ids
    assert stop_token_ids(_Tok({"</s>": 2}, eos_token="</s>"), _GenConfig(2)) == [2]


@pytest.mark.parametrize("model", ["Qwen/Qwen3.5-4B", MODEL])
def test_stop_token_ids_of_real_tokenizers_contain_im_end(model):
    from lifemate.agents.llm import stop_token_ids
    transformers = pytest.importorskip("transformers")
    try:
        real = transformers.AutoTokenizer.from_pretrained(model, local_files_only=True)
    except OSError:
        pytest.skip(f"{model} tokenizer not in the local HF cache")
    assert real.convert_tokens_to_ids("<|im_end|>") in stop_token_ids(real, None)
