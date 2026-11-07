import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
for name in ("Qwen3_5Config", "Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM"):
    if not hasattr(transformers, name):
        pytest.skip(f"transformers has no {name}", allow_module_level=True)

from transformers import AutoModelForCausalLM, Qwen3_5Config, Qwen3_5ForConditionalGeneration  # noqa: E402

from lifemate.agents.llm import load_causal_lm  # noqa: E402
from lifemate.sft.text_only import convert_model  # noqa: E402

TEXT = dict(vocab_size=120, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=2,
            num_key_value_heads=1, head_dim=16, max_position_embeddings=128, tie_word_embeddings=False,
            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
            linear_conv_kernel_dim=4)
VISION = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=32, patch_size=4,
              spatial_merge_size=2, temporal_patch_size=1, num_position_embeddings=16)


@pytest.fixture(scope="module")
def multimodal():
    torch.manual_seed(0)
    return Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=TEXT, vision_config=VISION)).eval()


def test_text_only_keeps_logits(multimodal, tmp_path):
    ids = torch.randint(0, 120, (1, 9))
    with torch.no_grad():
        before = multimodal(input_ids=ids).logits
    text = convert_model(multimodal).eval()
    with torch.no_grad():
        after = text(input_ids=ids).logits
    assert torch.allclose(before, after, atol=1e-4), (before - after).abs().max()
    text.save_pretrained(tmp_path)
    _, info = AutoModelForCausalLM.from_pretrained(tmp_path, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]


def test_text_only_has_no_vision_params(multimodal):
    text = convert_model(multimodal)
    assert not [n for n, _ in text.named_parameters() if "visual" in n or "mtp" in n]
    assert sum(p.numel() for p in text.parameters()) < sum(p.numel() for p in multimodal.parameters())


def test_a_checkpoint_with_unmatched_weights_is_refused_by_the_loader(multimodal, tmp_path):
    """The loader must raise when weights find no match: they would stay random and the run would look fine."""
    from safetensors.torch import load_file, save_file

    convert_model(multimodal).save_pretrained(tmp_path / "bad")
    for f in (tmp_path / "bad").glob("*.safetensors"):
        save_file({k.replace("model.layers", "model.blocks"): v.contiguous() for k, v in load_file(f).items()}, f,
                  metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="missing"):
        load_causal_lm(str(tmp_path / "bad"), torch.float32)


def test_the_multimodal_checkpoint_is_converted_or_refused_never_random(multimodal, tmp_path):
    """Whatever this transformers version does with the multimodal folder, a loaded model must equal the original."""
    multimodal.save_pretrained(tmp_path / "mm")
    ids = torch.randint(0, 120, (1, 7))
    try:
        loaded = load_causal_lm(str(tmp_path / "mm"), torch.float32).eval()
    except RuntimeError:
        return  # refused: fine
    with torch.no_grad():
        assert torch.allclose(multimodal(input_ids=ids).logits, loaded(input_ids=ids).logits, atol=1e-4)


def test_tied_embeddings_are_not_reported_missing(tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    Qwen3ForCausalLM(Qwen3Config(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                 num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                                 tie_word_embeddings=True)).save_pretrained(tmp_path)
    assert load_causal_lm(str(tmp_path), torch.float32) is not None


def test_the_converted_checkpoint_loads(multimodal, tmp_path):
    convert_model(multimodal).save_pretrained(tmp_path / "text")
    assert load_causal_lm(str(tmp_path / "text"), torch.float32) is not None


def test_tied_embeddings_stay_tied_and_the_parameter_count_is_the_text_part_only():
    torch.manual_seed(0)
    cfg = Qwen3_5Config(text_config=TEXT, vision_config=VISION, tie_word_embeddings=True)
    tied = Qwen3_5ForConditionalGeneration(cfg).eval()
    assert tied.config.tie_word_embeddings is True
    text = convert_model(tied)
    non_vision = sum(p.numel() for n, p in tied.named_parameters() if "visual" not in n)  # shared weights counted once
    assert sum(p.numel() for p in text.parameters()) == non_vision
    assert text.lm_head.weight.data_ptr() == text.model.embed_tokens.weight.data_ptr()
    ids = torch.randint(0, 120, (1, 6))
    with torch.no_grad():
        assert torch.allclose(tied(input_ids=ids).logits, text.eval()(input_ids=ids).logits, atol=1e-4)
