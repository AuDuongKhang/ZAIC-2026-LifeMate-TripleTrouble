import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from lifemate.sft.interpolate import interpolate_state  # noqa: E402


def test_alpha_one_keeps_a_and_half_is_the_mean():
    a = {"w": torch.tensor([1.0, 3.0]), "steps": torch.tensor([7])}
    b = {"w": torch.tensor([3.0, 5.0]), "steps": torch.tensor([9])}
    assert torch.equal(interpolate_state(a, b, 1.0)["w"], a["w"])
    half = interpolate_state(a, b, 0.5)
    assert torch.equal(half["w"], torch.tensor([2.0, 4.0]))
    assert torch.equal(half["steps"], a["steps"])  # integer tensors are taken from a


def test_mismatched_keys_and_shapes_are_refused():
    with pytest.raises(ValueError, match="x"):
        interpolate_state({"w": torch.zeros(2), "x": torch.zeros(1)}, {"w": torch.zeros(2)}, 0.5)
    with pytest.raises(ValueError, match="w"):
        interpolate_state({"w": torch.zeros(2)}, {"w": torch.zeros(3)}, 0.5)


def test_cli_writes_the_mean_of_two_tiny_models(tmp_path):
    from lifemate.agents.llm import load_causal_lm
    from lifemate.sft import interpolate

    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", local_files_only=True)
    except OSError:
        pytest.skip("tokenizer not in the local HF cache")
    cfg = transformers.Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=1, head_dim=16)
    for name, seed in (("a", 0), ("b", 1)):
        torch.manual_seed(seed)
        transformers.AutoModelForCausalLM.from_config(cfg).save_pretrained(tmp_path / name)
        tok.save_pretrained(tmp_path / name)
    out = interpolate.main(["--a", str(tmp_path / "a"), "--b", str(tmp_path / "b"), "--alpha", "0.5",
                            "--out", str(tmp_path / "mix")])
    a, b, mix = (load_causal_lm(str(p), torch.float32).state_dict() for p in (tmp_path / "a", tmp_path / "b", out))
    key = next(k for k in mix if mix[k].is_floating_point())
    assert torch.allclose(mix[key], (a[key] + b[key]) / 2, atol=1e-2)  # saved in bf16
    assert (out / "tokenizer_config.json").exists()
