"""End-to-end trainer check on a tiny random Qwen3 (CPU, a few seconds): a run stopped and resumed gives the same
log as an uninterrupted one, and the adapter merges. Skipped without torch + peft (they are not in requirements.txt)."""
import json
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


def test_weight_one_keeps_the_loss(tiny, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _train(tiny, a, "--max-steps", "2")
    _train(tiny, b, "--max-steps", "2", "--decision-weight", "1.0")
    assert [r["loss"] for r in _log(a)] == [r["loss"] for r in _log(b)]
    assert [r["grad_norm"] for r in _log(a)] == [r["grad_norm"] for r in _log(b)]


def test_decision_weight_changes_the_updates_but_not_what_is_logged(tiny, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _train(tiny, a, "--max-steps", "2")
    _train(tiny, b, "--max-steps", "2", "--decision-weight", "3")
    la, lb = _log(a), _log(b)
    assert la[0]["loss"] == pytest.approx(lb[0]["loss"], rel=1e-6)  # the first step starts from the same model
    assert la[0]["grad_norm"] != pytest.approx(lb[0]["grad_norm"], rel=1e-3)  # but weights the tokens differently
    from safetensors.torch import load_file

    wa, wb = (load_file(r / "last" / "adapter_model.safetensors") for r in (a, b))
    assert any(not torch.equal(wa[k], wb[k]) for k in wa)  # so the adapters end up different
    assert "eval_loss" in lb[1]


def test_resume_refuses_a_changed_decision_weight(tiny, tmp_path):
    run = tmp_path / "run"
    _train(tiny, run, "--max-steps", "1")
    with pytest.raises(SystemExit):
        _train(tiny, run, "--resume", "--decision-weight", "3")
    _train(tiny, run, "--resume", "--max-steps", "2")  # the same weight resumes


def test_eval_loss_ignores_the_weights(tiny, tmp_path):
    from lifemate.sft import train

    a = tmp_path / "a"
    train.main(["--records", "r", "--eval-records", "r", "--records-dir", str(tiny / "records"), "--out", str(a),
                "--model", str(tiny / "model"), "--epochs", "1", "--grad-accum", "2", "--eval-every", "1",
                "--lr", "0", "--max-steps", "1", "--decision-weight", "5"])
    b = tmp_path / "b"
    train.main(["--records", "r", "--eval-records", "r", "--records-dir", str(tiny / "records"), "--out", str(b),
                "--model", str(tiny / "model"), "--epochs", "1", "--grad-accum", "2", "--eval-every", "1",
                "--lr", "0", "--max-steps", "1"])
    assert _log(a)[0]["eval_loss"] == _log(b)[0]["eval_loss"]  # lr 0: same model, so the same unweighted eval loss


def _teacher(tiny):
    return str(tiny / "model")


def test_kd_run_logs_both_terms_and_keeps_the_eval_loss_plain(tiny, tmp_path):
    out = tmp_path / "kd"
    _train(tiny, out, "--kd-teachers", _teacher(tiny))
    rows = _log(out)
    assert all("kd" in r for r in rows)
    assert rows[-1]["eval_loss"] > 0
    assert (out / "kd_report.json").exists()
    assert json.loads((out / "kd_report.json").read_text(encoding="utf-8"))["positions"] > 0


def test_no_teachers_keeps_the_log_free_of_kd(tiny, tmp_path):
    out = tmp_path / "plain"
    _train(tiny, out, "--max-steps", "1")
    assert "kd" not in _log(out)[0]
    assert not (out / "kd_report.json").exists()


def test_alpha_zero_gives_the_loss_of_a_plain_run(tiny, tmp_path):
    plain, zero = tmp_path / "plain", tmp_path / "zero"
    _train(tiny, plain)
    _train(tiny, zero, "--kd-teachers", _teacher(tiny), "--kd-alpha", "0")
    assert [r["loss"] for r in _log(plain)] == pytest.approx([r["loss"] for r in _log(zero)], rel=1e-4)


def test_kd_resume_matches_an_uninterrupted_run(tiny, tmp_path):
    full, part = tmp_path / "full", tmp_path / "part"
    kd = ("--kd-teachers", _teacher(tiny))
    _train(tiny, full, *kd)
    _train(tiny, part, *kd, "--max-steps", "2")
    _train(tiny, part, *kd, "--resume")
    assert [r["loss"] for r in _log(full)] == pytest.approx([r["loss"] for r in _log(part)], rel=1e-4)
    assert [r["kd"] for r in _log(full)] == pytest.approx([r["kd"] for r in _log(part)], rel=1e-3, abs=1e-5)


def test_report_only_stops_before_training(tiny, tmp_path):
    out = tmp_path / "report"
    _train(tiny, out, "--kd-teachers", _teacher(tiny), "--kd-report-only")
    assert (out / "kd_report.json").exists()
    assert not (out / "last").exists() and not (out / "train_log.jsonl").exists()


def test_resume_refuses_changed_kd_args(tiny, tmp_path):
    out = tmp_path / "kd"
    _train(tiny, out, "--kd-teachers", _teacher(tiny), "--max-steps", "1")
    with pytest.raises(SystemExit, match="kd-alpha"):
        _train(tiny, out, "--kd-teachers", _teacher(tiny), "--kd-alpha", "0.2", "--resume")


def test_teacher_with_another_vocabulary_is_refused(tiny, tmp_path):
    other = tmp_path / "other"
    tok = transformers.AutoTokenizer.from_pretrained(tiny / "model")
    tok.add_tokens(["<not_in_the_student>"])
    tok.save_pretrained(other)
    with pytest.raises(SystemExit, match="vocabulary"):
        _train(tiny, tmp_path / "kd", "--kd-teachers", str(other))


def test_lora_targets_option_limits_the_adapter(tiny, tmp_path):
    out = tmp_path / "attn"
    _train(tiny, out, "--max-steps", "1", "--lora-targets", "q_proj,v_proj")
    config = json.loads((out / "last" / "adapter_config.json").read_text(encoding="utf-8"))
    assert set(config["target_modules"]) == {"q_proj", "v_proj"}


def test_distillation_and_decision_weight_work_together(tiny, tmp_path):
    """Both options change the training loss; the logged loss and the eval loss stay plain, and kd is logged."""
    both, kd_only = tmp_path / "both", tmp_path / "kd"
    _train(tiny, both, "--max-steps", "2", "--kd-teachers", _teacher(tiny), "--decision-weight", "3")
    _train(tiny, kd_only, "--max-steps", "2", "--kd-teachers", _teacher(tiny))
    a, b = _log(both), _log(kd_only)
    assert all("kd" in r for r in a) and "eval_loss" in a[1]
    assert a[0]["loss"] == pytest.approx(b[0]["loss"], rel=1e-6)  # same start, the logged loss is the plain one
    assert a[0]["grad_norm"] != pytest.approx(b[0]["grad_norm"], rel=1e-3)  # but the gradient weights the tokens


# --- preference mode (--pairs) ---------------------------------------------------------------------------------------

import math  # noqa: E402


def _pair_rows(records):
    rows = []
    for r in records:
        msgs = r["messages"]
        last = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
        chosen = msgs[last]
        rejected = ({"role": "assistant", "content": "Đã xong."} if chosen.get("tool_calls") else
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"type": "function", "function": {"name": "alarm.list", "arguments": {}}}]})
        rows.append({"id": r["id"], "record": 0, "turn": last, "type": "t", "greedy_wrong": True,
                     "messages": msgs[:last], "tools": r["tools"], "chosen": chosen, "rejected": rejected})
    return rows


@pytest.fixture(scope="module")
def tiny_pairs(tiny):
    records = [json.loads(line) for line in (tiny / "records" / "r.jsonl").read_text(encoding="utf-8").splitlines()]
    (tiny / "records" / "pairs").mkdir(exist_ok=True)
    (tiny / "records" / "pairs" / "p.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in _pair_rows(records)), encoding="utf-8")
    return tiny


def test_pairs_run_starts_at_log_two(tiny_pairs, tmp_path):
    out = tmp_path / "pref"
    _train(tiny_pairs, out, "--pairs", "p", "--sft-share", "0", "--sft-weight", "0", "--max-steps", "1")
    first = _log(out)[0]
    assert first["dpo"] == pytest.approx(math.log(2), abs=1e-4)
    assert first["margin"] == pytest.approx(0, abs=1e-4)
    assert first["pref_acc"] == 0.0  # a margin of exactly zero is not a win


def test_pairs_run_logs_preference_terms_and_eval_accuracy(tiny_pairs, tmp_path):
    out = tmp_path / "pref"
    _train(tiny_pairs, out, "--pairs", "p", "--eval-pairs", "p")
    rows = _log(out)
    assert all({"dpo", "nll", "margin", "pref_acc"} <= set(r) for r in rows if r.get("pairs"))
    evals = [r for r in rows if "eval_loss" in r]
    assert evals and all("eval_pref_acc" in r and 0 <= r["eval_pref_acc"] <= 1 for r in evals)


def test_pairs_resume_matches_an_uninterrupted_run(tiny_pairs, tmp_path):
    full, part = tmp_path / "full", tmp_path / "part"
    _train(tiny_pairs, full, "--pairs", "p")
    _train(tiny_pairs, part, "--pairs", "p", "--max-steps", "1")
    _train(tiny_pairs, part, "--pairs", "p", "--resume")
    a, b = _log(full), _log(part)
    assert [r["step"] for r in a] == [r["step"] for r in b]
    assert [r["loss"] for r in a] == pytest.approx([r["loss"] for r in b], rel=1e-4)


def test_reference_cache_is_rebuilt_when_the_pairs_change(tiny_pairs, tmp_path):
    out = tmp_path / "pref"
    _train(tiny_pairs, out, "--pairs", "p", "--max-steps", "1")
    first = torch.load(out / "ref_logps.pt", weights_only=False)
    path = tiny_pairs / "records" / "pairs" / "p2.jsonl"
    rows = (tiny_pairs / "records" / "pairs" / "p.jsonl").read_text(encoding="utf-8").splitlines()
    path.write_text("".join(line + "\n" for line in rows[:-1]), encoding="utf-8")
    out2 = tmp_path / "pref2"
    out2.mkdir()
    (out2 / "ref_logps.pt").write_bytes((out / "ref_logps.pt").read_bytes())
    _train(tiny_pairs, out2, "--pairs", "p2", "--max-steps", "1")
    second = torch.load(out2 / "ref_logps.pt", weights_only=False)
    assert second["key"]["pairs_sha"] != first["key"]["pairs_sha"]
    assert len(second["train"]) == len(first["train"]) - 1


def test_pairs_refuse_kd_and_decision_weight(tiny_pairs, tmp_path):
    with pytest.raises(SystemExit):
        _train(tiny_pairs, tmp_path / "a", "--pairs", "p", "--decision-weight", "2")
    with pytest.raises(SystemExit):
        _train(tiny_pairs, tmp_path / "b", "--pairs", "p", "--kd-teachers", "x")


def test_merge_accepts_a_revision(tmp_path, monkeypatch):
    import lifemate.agents.llm as llm
    from lifemate.sft import merge

    seen = {}

    def fake_load(model, dtype, revision=None):
        seen["revision"] = revision
        raise RuntimeError("stop after loading")

    monkeypatch.setattr(llm, "load_causal_lm", fake_load)
    (tmp_path / "adapter").mkdir()
    with pytest.raises(RuntimeError, match="stop after loading"):
        merge.main(["--adapter", str(tmp_path / "adapter"), "--out", str(tmp_path / "out"), "--model", "m",
                    "--revision", "abc123"])
    assert seen["revision"] == "abc123"
