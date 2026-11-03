"""End-to-end trainer check on a tiny random Qwen3 (CPU, a few seconds): a run stopped and resumed gives the same
log as an uninterrupted one, and the adapter merges. Skipped without torch + peft (they are not in requirements.txt)."""
import json
import shutil
from pathlib import Path

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


def test_resume_refuses_changed_training_args(tiny, tmp_path):
    _train(tiny, tmp_path / "r", "--max-steps", "1")
    with pytest.raises(SystemExit, match="epochs"):
        _train(tiny, tmp_path / "r", "--resume", "--epochs", "2")


def test_stop_file_saves_and_resume_finishes_the_same_run(tiny, tmp_path):
    import threading
    import time

    full = tmp_path / "full"
    _train(tiny, full, "--grad-accum", "1")
    part = tmp_path / "part"

    def request_stop():  # what the notebook's Stop button does, once the first step is logged
        while not (part / "train_log.jsonl").exists():
            time.sleep(0.05)
        (part / "STOP").write_text("", encoding="utf-8")

    threading.Thread(target=request_stop, daemon=True).start()
    _train(tiny, part, "--grad-accum", "1")
    stopped = _log(part)
    assert len(stopped) < len(_log(full)) and not (part / "STOP").exists()
    assert json.loads((part / "last" / "state.json").read_text(encoding="utf-8"))["step"] == stopped[-1]["step"]
    _train(tiny, part, "--grad-accum", "1", "--resume")
    assert [r["loss"] for r in _log(part)] == pytest.approx([r["loss"] for r in _log(full)], rel=1e-4)


def test_orphan_log_without_checkpoint_does_not_block_a_fresh_run(tiny, tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "train_log.jsonl").write_text('{"step": 1, "loss": 9.0}\n', encoding="utf-8")
    _train(tiny, out, "--max-steps", "1")
    assert [r["loss"] for r in _log(out)] != [9.0]


class FakeHfApi:
    files: list[str] = []

    def create_repo(self, *a, **k):
        pass

    def list_repo_files(self, *a, **k):
        return self.files

    def upload_folder(self, *a, **k):
        pass


def test_fresh_run_refuses_a_run_already_on_the_hub(tiny, tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    monkeypatch.setattr(FakeHfApi, "files", ["checkpoints/h/last/state.json"])
    with pytest.raises(SystemExit, match="--resume"):
        _train(tiny, tmp_path / "h", "--hf-repo", "me/x")


def test_resume_takes_the_newer_hub_checkpoint_over_a_stale_local_one(tiny, tmp_path, monkeypatch, capsys):
    import shutil

    import huggingface_hub

    full = tmp_path / "full"
    _train(tiny, full)
    part, hub = tmp_path / "part", tmp_path / "hub"
    _train(tiny, part, "--max-steps", "1")
    shutil.copytree(part / "last", tmp_path / "stale")
    _train(tiny, part, "--max-steps", "2", "--resume")
    shutil.copytree(part / "last", hub / "checkpoints" / "part" / "last")  # what the upload at step 2 left on HF
    shutil.rmtree(part / "last")
    shutil.copytree(tmp_path / "stale", part / "last")  # the molab copy is older

    def fake_download(repo_id, allow_patterns, local_dir, **kw):
        shutil.copytree(hub / "checkpoints", Path(local_dir) / "checkpoints")

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    capsys.readouterr()
    _train(tiny, part, "--resume", "--hf-repo", "me/x")
    assert "resuming part at step 2" in capsys.readouterr().out
    assert [r["loss"] for r in _log(part)] == pytest.approx([r["loss"] for r in _log(full)], rel=1e-4)


def _fake_hub(hub):
    def fake_download(repo_id, allow_patterns, local_dir, ignore_patterns=(), **kw):
        shutil.copytree(hub / "checkpoints", Path(local_dir) / "checkpoints",
                        ignore=shutil.ignore_patterns(*ignore_patterns))
    return fake_download


def test_merge_takes_the_hub_adapter_when_the_local_folder_is_incomplete(tiny, tmp_path, monkeypatch, capsys):
    import huggingface_hub

    from lifemate.sft import merge

    hub = tmp_path / "hub"
    shutil.copytree(_train(tiny, tmp_path / "run", "--max-steps", "1"), hub / "checkpoints" / "run" / "last")
    local = tmp_path / "session" / "run" / "last"
    local.mkdir(parents=True)
    (local / "state.json").write_text((hub / "checkpoints" / "run" / "last" / "state.json").read_text())  # cut short
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_hub(hub))
    out = merge.main(["--adapter", str(local), "--adapter-repo", "me/x", "--out", str(tmp_path / "merged")])
    assert "adapter source: me/x checkpoints/run/last (step 1)" in capsys.readouterr().out
    assert not (local / "optimizer.pt").exists()  # merging needs the adapter only
    model = transformers.AutoModelForCausalLM.from_pretrained(out)
    assert not any("lora" in name for name, _ in model.named_parameters())


def test_merge_without_an_adapter_anywhere_says_so(tiny, tmp_path, monkeypatch):
    import huggingface_hub

    from lifemate.sft import merge

    (tmp_path / "hub" / "checkpoints").mkdir(parents=True)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_hub(tmp_path / "hub"))
    with pytest.raises(SystemExit, match="no complete adapter"):
        merge.main(["--adapter", str(tmp_path / "run" / "last"), "--adapter-repo", "me/x", "--out", str(tmp_path / "m")])


def test_resume_replaces_an_incomplete_local_checkpoint_with_the_hub_one(tiny, tmp_path, monkeypatch, capsys):
    import huggingface_hub

    full = tmp_path / "full"
    _train(tiny, full)
    part, hub = tmp_path / "part", tmp_path / "hub"
    _train(tiny, part, "--max-steps", "1")
    shutil.copytree(part / "last", hub / "checkpoints" / "part" / "last")
    _train(tiny, part, "--max-steps", "2", "--resume")
    (part / "last" / "adapter_model.safetensors").unlink()  # newer than the hub, but the session died during a save
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_hub(hub))
    capsys.readouterr()
    _train(tiny, part, "--resume", "--hf-repo", "me/x")
    assert "resuming part at step 1" in capsys.readouterr().out
    assert [r["loss"] for r in _log(part)] == pytest.approx([r["loss"] for r in _log(full)], rel=1e-4)
