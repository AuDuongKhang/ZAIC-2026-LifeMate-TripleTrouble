import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from lifemate.sft.pref import encode_pair, rpo_loss, step_logps  # noqa: E402
from lifemate.sft.tokenize import render  # noqa: E402


def t(x):
    return torch.tensor(float(x))


def test_rpo_loss_at_the_reference_is_log_two():
    loss, margin = rpo_loss(t(-3), t(-5), t(-3), t(-5), t(0), beta=0.1, sft_weight=1.0)
    assert loss.item() == pytest.approx(math.log(2)) and margin.item() == pytest.approx(0)


def test_rpo_loss_adds_the_weighted_nll():
    loss, _ = rpo_loss(t(-3), t(-5), t(-3), t(-5), t(2), beta=0.1, sft_weight=0.5)
    assert loss.item() == pytest.approx(math.log(2) + 1)


def test_margin_sign_follows_the_preference():
    loss, margin = rpo_loss(t(-2), t(-6), t(-3), t(-5), t(0), beta=0.1, sft_weight=1.0)
    assert margin.item() == pytest.approx(0.2) and loss.item() < math.log(2)


@pytest.fixture(scope="module")
def tok():
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", local_files_only=True)
    except OSError:
        pytest.skip("tokenizer not in the local HF cache")


PAIR = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "đặt báo thức tám giờ"}],
        "tools": [],
        "chosen": {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "alarm.create", "arguments": {"time": "08:00"}}}]},
        "rejected": {"role": "assistant", "content": "Đã xong."}}


def test_encode_pair_shares_the_prefix(tok):
    enc = encode_pair(PAIR, tok)
    assert enc["chosen"] and enc["rejected"]
    assert render(tok, PAIR["messages"] + [PAIR["chosen"]], []) == enc["prefix"] + enc["chosen"]
    assert render(tok, PAIR["messages"] + [PAIR["rejected"]], []) == enc["prefix"] + enc["rejected"]


def test_encode_pair_skips_a_pair_over_max_len(tok):
    assert encode_pair(PAIR, tok, max_len=10) is None


def test_step_logps_matches_a_full_forward(tok):
    cfg = transformers.Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=1, head_dim=16)
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(cfg).eval()
    prefix, step = [1, 2, 3, 4], [5, 6, 7]
    with torch.no_grad():
        logp, nll = step_logps(model, prefix, step, "cpu")
        logits = model(input_ids=torch.tensor([prefix + step])).logits[0].float().log_softmax(-1)
    expected = sum(logits[len(prefix) - 1 + j, s].item() for j, s in enumerate(step))
    assert logp.item() == pytest.approx(expected, rel=1e-4)
    assert nll.item() == pytest.approx(-expected / len(step), rel=1e-4)
