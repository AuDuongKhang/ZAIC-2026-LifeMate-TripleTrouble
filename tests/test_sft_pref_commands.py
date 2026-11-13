import sys

import pytest

from lifemate.sft.pref_commands import BASE_REVISION, LORA_TARGETS, commands

F = {"name": "pilot", "model": "TripleTrouble/lifemate-sft-q1c", "revision": "8cb4336e529f0a4b2d6f10cdcfc63d15c45b651d",
     "k": 8, "temperature": "0.7,1.0", "top_p": 0.95, "limit": 100, "lr": 1e-5, "beta": 0.1, "sft_weight": 1.0,
     "owner": "TripleTrouble", "data_repo": "TripleTrouble/zaic2026-lifemate-data", "run": ""}
PY = [sys.executable, "-m"]


def test_records_transcribes_then_writes_the_four_record_files():
    cmds = commands("records", F)
    assert cmds[0] == PY + ["lifemate.transcribe", "--split", "train", "--asr", "whisper", "--asr-kw",
                            "model=vinai/PhoWhisper-large", "--name", "pho-large-fp16", "--subset", "pool"]
    assert cmds[1] == PY + ["lifemate.sft.records", "--name", "pref-pilot-gold", "--subset", "pool", "--text", "gold",
                            "--mask", "1", "--leave-app-out", "1", "--seed", "7", "--examples", "4", "--tokenizer", "none",
                            "--limit", "100"]
    assert cmds[2] == PY + ["lifemate.sft.records", "--name", "pref-pilot-asr", "--subset", "pool", "--text",
                            "cache:pho-large-fp16", "--mask", "1", "--leave-app-out", "1", "--seed", "7", "--examples",
                            "4", "--tokenizer", "none", "--limit", "100"]
    assert cmds[3] == PY + ["lifemate.sft.records", "--name", "gold-pool-m1-asr-pho-large-fp16-lao50", "--subset", "pool",
                            "--text", "gold", "--text", "cache:pho-large-fp16", "--mask", "1", "--leave-app-out", "0.5",
                            "--examples", "4", "--tokenizer", "none"]
    assert cmds[4] == PY + ["lifemate.sft.records", "--name", "gold-dev", "--subset", "dev", "--examples", "4",
                            "--tokenizer", "none"]
    assert cmds[5] == PY + ["lifemate.sft.hf_files", "push", "--repo", F["data_repo"], "--folder", "records", "--dir",
                            "train_records", "pref-pilot-gold.jsonl", "pref-pilot-asr.jsonl",
                            "gold-pool-m1-asr-pho-large-fp16-lao50.jsonl", "gold-dev.jsonl"]
    assert len(cmds) == 6


def pull(folder, *names):
    return PY + ["lifemate.sft.hf_files", "pull", "--repo", F["data_repo"], "--folder", folder, "--dir",
                 "train_records" if folder == "records" else f"train_records/{folder}", *names]


def only(cmds, module):
    return [c for c in cmds if module in c]


def test_sample_runs_once_per_temperature():
    cmds = commands("sample", F)
    assert cmds[0] == pull("records", "pref-pilot-gold.jsonl", "pref-pilot-asr.jsonl")
    assert [c[c.index("--name") + 1] for c in only(cmds, "lifemate.sft.sample_steps")] == ["pilot-t0.7", "pilot-t1.0"]
    assert cmds[1] == PY + ["lifemate.sft.sample_steps", "--records", "pref-pilot-gold", "--records", "pref-pilot-asr",
                            "--name", "pilot-t0.7", "--model", F["model"], "--revision", F["revision"], "--k", "8",
                            "--temperature", "0.7", "--top-p", "0.95", "--seed", "0", "--hf-repo", F["data_repo"],
                            "--limit", "100"]


def test_pairs_runs_once_per_temperature_and_shares_its_files():
    cmds = commands("pairs", F)
    assert cmds[0] == pull("records", "pref-pilot-gold.jsonl", "pref-pilot-asr.jsonl")
    assert cmds[1:4] == [pull("steps", "pilot-t0.7.jsonl"),
                         PY + ["lifemate.sft.pairs", "--records", "pref-pilot-gold", "--records", "pref-pilot-asr",
                               "--steps", "pilot-t0.7", "--out", "pilot-t0.7"],
                         PY + ["lifemate.sft.hf_files", "push", "--repo", F["data_repo"], "--folder", "pairs", "--dir",
                               "train_records/pairs", "pilot-t0.7.jsonl", "pilot-t0.7-heldout.jsonl",
                               "pilot-t0.7-review.md"]]
    assert len(only(cmds, "lifemate.sft.pairs")) == 2


def test_train_and_merge_need_one_temperature():
    with pytest.raises(ValueError, match="one temperature"):
        commands("train", F)
    one = dict(F, temperature="0.8", run="a")
    assert commands("train", one) == [
        pull("records", "gold-pool-m1-asr-pho-large-fp16-lao50.jsonl", "gold-dev.jsonl"),
        pull("pairs", "pilot-t0.8.jsonl", "pilot-t0.8-heldout.jsonl"), PY + [
        "lifemate.sft.train", "--records", "gold-pool-m1-asr-pho-large-fp16-lao50", "--pairs", "pilot-t0.8",
        "--eval-pairs", "pilot-t0.8-heldout", "--eval-records", "gold-dev", "--model", F["model"], "--revision",
        F["revision"], "--out", "checkpoints/sft-q1d-pilot-a", "--hf-repo", "TripleTrouble/lifemate-sft-q1d", "--epochs",
        "1", "--lr", "1e-05", "--beta", "0.1", "--sft-weight", "1.0", "--lora-targets", LORA_TARGETS]]
    assert commands("merge", one) == [PY + [
        "lifemate.sft.merge", "--adapter", "checkpoints/sft-q1d-pilot-a/last", "--out", "merged/sft-q1d-pilot-a",
        "--hf-repo", "TripleTrouble/lifemate-sft-q1d", "--model", F["model"], "--revision", F["revision"]]]


def test_interpolate_makes_both_alphas_from_the_pinned_base():
    cmds = commands("interpolate", F)
    assert cmds[1] == PY + ["lifemate.sft.interpolate", "--a", F["model"], "--a-revision", F["revision"], "--b",
                            "TripleTrouble/qwen35-4b-text", "--b-revision", BASE_REVISION, "--alpha", "0.85", "--out",
                            "merged/sft-q1c-wise85", "--hf-repo", "TripleTrouble/lifemate-sft-q1c-wise85"]
    assert [c[c.index("--alpha") + 1] for c in cmds] == ["0.7", "0.85"]
    assert BASE_REVISION == "300ccc142143c6479998714ca6cbc1c5b61ae411"


def test_no_limit_means_the_whole_pool():
    assert "--limit" not in only(commands("sample", dict(F, limit=0)), "lifemate.sft.sample_steps")[0]


def test_unknown_step_is_refused():
    with pytest.raises(ValueError):
        commands("deploy", F)


def test_progress_reads_the_last_progress_line():
    from lifemate.sft.pref_commands import progress
    log = "loading\nprogress 3/40 points (8%), 2.0 s/point, eta 1 min\nprogress 12/40 points (30%), 2.0 s/point, eta 1 min\n"
    assert progress(log) == (12, 40)
    assert progress("loading model\n") is None
