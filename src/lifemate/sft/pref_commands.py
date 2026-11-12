"""The commands notebooks/molab_pref.py runs for each step of the preference training (plan
docs/superpowers/plans/2026-11-12-onpolicy-preference.md). Pure, so the exact commands are unit-tested.

Steps: records (ASR cache + masked leave-app-out pool records + the sft-q1c records and gold-dev), sample (one steps file
per temperature), pairs, train, merge, interpolate (WiSE-FT, alpha 0.7 and 0.85). `f` holds the notebook's fields.
"""
from __future__ import annotations

import sys

ASR_CACHE = "pho-large-fp16"
ASR_MODEL = "vinai/PhoWhisper-large"
SFT_RECORDS = "gold-pool-m1-asr-pho-large-fp16-lao50"  # the sft-q1c train records, for the SFT share of the mix
EVAL_RECORDS = "gold-dev"
MASK_SEED = "7"  # sft-q1c's records used seed 0: other names than the model trained on
BASE_MODEL = "qwen35-4b-text"
BASE_REVISION = "300ccc142143c6479998714ca6cbc1c5b61ae411"  # TripleTrouble/qwen35-4b-text, base of sft-q1c
LORA_TARGETS = "q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,gate_proj,up_proj,down_proj"
ALPHAS = ("0.7", "0.85")
STEPS = ("records", "sample", "pairs", "train", "merge", "interpolate")


def _py(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def _temperatures(f: dict) -> list[str]:
    return [t.strip() for t in str(f["temperature"]).split(",") if t.strip()]


def _limit(f: dict) -> list[str]:
    return ["--limit", str(int(f["limit"]))] if f.get("limit") else []


def _one(f: dict) -> str:
    temps = _temperatures(f)
    if len(temps) != 1:
        raise ValueError(f"train and merge need one temperature, got {temps}")
    return f"{f['name']}-t{temps[0]}"


def _run(f: dict) -> str:
    return f"sft-q1d-{f['name']}" + (f"-{f['run']}" if f.get("run") else "")


def commands(step: str, f: dict) -> list[list[str]]:
    name, sources = f["name"], ["--records", f"pref-{f['name']}-gold", "--records", f"pref-{f['name']}-asr"]
    if step == "records":
        masked = ["--mask", "1", "--leave-app-out", "1", "--seed", MASK_SEED, "--examples", "4", "--tokenizer", "none"]
        return [
            _py("lifemate.transcribe", "--split", "train", "--asr", "whisper", "--asr-kw", f"model={ASR_MODEL}",
                "--name", ASR_CACHE, "--subset", "pool"),
            _py("lifemate.sft.records", "--name", f"pref-{name}-gold", "--subset", "pool", "--text", "gold", *masked,
                *_limit(f)),
            _py("lifemate.sft.records", "--name", f"pref-{name}-asr", "--subset", "pool", "--text", f"cache:{ASR_CACHE}",
                *masked, *_limit(f)),
            _py("lifemate.sft.records", "--name", SFT_RECORDS, "--subset", "pool", "--text", "gold", "--text",
                f"cache:{ASR_CACHE}", "--mask", "1", "--leave-app-out", "0.5", "--examples", "4", "--tokenizer", "none"),
            _py("lifemate.sft.records", "--name", EVAL_RECORDS, "--subset", "dev", "--examples", "4", "--tokenizer", "none"),
        ]
    if step == "sample":
        return [_py("lifemate.sft.sample_steps", *sources, "--name", f"{name}-t{t}", "--model", f["model"], "--revision",
                    f["revision"], "--k", str(int(f["k"])), "--temperature", t, "--top-p", str(f["top_p"]), "--seed", "0",
                    "--hf-repo", f["data_repo"], *_limit(f)) for t in _temperatures(f)]
    if step == "pairs":
        return [_py("lifemate.sft.pairs", *sources, "--steps", f"{name}-t{t}", "--out", f"{name}-t{t}")
                for t in _temperatures(f)]
    if step == "train":
        pairs = _one(f)
        return [_py("lifemate.sft.train", "--records", SFT_RECORDS, "--pairs", pairs, "--eval-pairs", f"{pairs}-heldout",
                    "--eval-records", EVAL_RECORDS, "--model", f["model"], "--revision", f["revision"], "--out",
                    f"checkpoints/{_run(f)}", "--hf-repo", f"{f['owner']}/lifemate-sft-q1d", "--epochs", "1", "--lr",
                    str(f["lr"]), "--beta", str(f["beta"]), "--sft-weight", str(f["sft_weight"]), "--lora-targets",
                    LORA_TARGETS)]
    if step == "merge":
        _one(f)
        return [_py("lifemate.sft.merge", "--adapter", f"checkpoints/{_run(f)}/last", "--out", f"merged/{_run(f)}",
                    "--hf-repo", f"{f['owner']}/lifemate-sft-q1d", "--model", f["model"], "--revision", f["revision"])]
    if step == "interpolate":
        return [_py("lifemate.sft.interpolate", "--a", f["model"], "--a-revision", f["revision"], "--b",
                    f"{f['owner']}/{BASE_MODEL}", "--b-revision", BASE_REVISION, "--alpha", a, "--out",
                    f"merged/sft-q1c-wise{round(float(a) * 100)}", "--hf-repo",
                    f"{f['owner']}/lifemate-sft-q1c-wise{round(float(a) * 100)}") for a in ALPHAS]
    raise ValueError(f"unknown step {step!r}; one of {STEPS}")
