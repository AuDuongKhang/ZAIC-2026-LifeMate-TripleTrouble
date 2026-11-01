"""End-to-end trainer check on a tiny random Qwen3 (CPU, a few seconds): a run stopped and resumed gives the same
log as an uninterrupted one, and the adapter merges. Skipped without torch + peft (they are not in requirements.txt)."""
import json

import pytest

from conftest import replay_sample
from lifemate.evaluate import label_kind

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")
MODEL = "Qwen/Qwen3-4B-Instruct-2507"


@pytest.fixture(scope="module")
def tiny(tmp_path_factory, train_samples):
    try:
        tok = transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except OSError:
        pytest.skip(f"{MODEL} tokenizer not in the local HF cache")
    root = tmp_path_factory.mktemp("tiny")
    cfg = transformers.Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=16384)
    torch.manual_seed(0)
    transformers.AutoModelForCausalLM.from_config(cfg).save_pretrained(root / "model")
    tok.save_pretrained(root / "model")
    picked = [s for s in train_samples if not s["history"]]
    records = [replay_sample(s) for s in [next(s for s in picked if label_kind(s["label"]["tool_calls"]) == k)
                                          for k in ("act", "ask", "empty")] + picked[:3]]
    (root / "records").mkdir()
    (root / "records" / "r.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                                              encoding="utf-8")
    return root


def _train(tiny, out, *extra):
    from lifemate.sft import train

    return train.main(["--records", "r", "--eval-records", "r", "--records-dir", str(tiny / "records"), "--out", str(out),
                       "--model", str(tiny / "model"), "--epochs", "1", "--grad-accum", "2", "--eval-every", "2",
                       "--lr", "1e-2", *extra])


def _log(out):
    return [json.loads(line) for line in (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]


def test_resumed_run_matches_an_uninterrupted_one(tiny, tmp_path):
    full = tmp_path / "full"
    _train(tiny, full)
    part = tmp_path / "part"
    _train(tiny, part, "--max-steps", "2")
    assert [r["step"] for r in _log(part)] == [1, 2]
    _train(tiny, part, "--resume")
    a, b = _log(full), _log(part)
    assert [r["step"] for r in a] == [r["step"] for r in b] == [1, 2, 3]
    assert [r["loss"] for r in a] == pytest.approx([r["loss"] for r in b], rel=1e-4)
    assert a[0]["loss"] > a[-1]["loss"]
    assert "eval_loss" in a[1] and "eval_loss" in a[2]
    assert all((full / f).exists() for f in ("loss.png", "last/adapter_model.safetensors", "last/state.json"))


def test_fresh_run_refuses_an_existing_folder(tiny, tmp_path):
    _train(tiny, tmp_path / "x", "--max-steps", "1")
    with pytest.raises(SystemExit):
        _train(tiny, tmp_path / "x")


def test_merge_writes_a_plain_model(tiny, tmp_path):
    from lifemate.sft import merge

    ckpt = _train(tiny, tmp_path / "run", "--max-steps", "1")
    out = merge.main(["--adapter", str(ckpt), "--out", str(tmp_path / "merged")])
    model = transformers.AutoModelForCausalLM.from_pretrained(out)
    assert not any("lora" in name for name, _ in model.named_parameters())
    assert (out / "tokenizer_config.json").exists()
