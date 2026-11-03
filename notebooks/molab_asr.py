# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo"]
# ///
"""LifeMate ASR trainer for molab: pull the repo's branch, fetch the data, fine-tune a Whisper-family model on the pool
clips in the background with a live log, transcribe clips into an ASR cache, report WER and the error analysis, and
upload the model (weights + processor only) to a private HF model repo. Only this file is synced from GitHub, so it
clones the repo itself; data, tokens and checkpoints never go into the repo."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    import shlex
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    import marimo as mo
    return Path, json, mo, os, shlex, shutil, subprocess, sys


@app.cell
def _(mo):
    mo.md("""
    # LifeMate ASR trainer

    1. Turn on the GPU with the *notebook specs* button in the header.
    2. Fill in the GitHub token, the HF read token (data, private models) and, for **Upload** only, an HF **write** token.
    3. **Pull repo**: pulls the branch, installs `requirements.txt` + `requirements-train.txt` (`--no-deps`, keeps
       molab's torch). Press it again whenever the branch has new commits (not while training runs).
       **Fetch data** once per session, after the first Pull repo.
    4. **Start training** runs `lifemate.train_asr` in the background on the pool clips (never dev or holdout). It
       evaluates and saves a checkpoint every *eval steps* and keeps the best by WER; the log refreshes every minute.
       **Stop** kills it: the saved checkpoints stay, and *resume* + **Start training** with the same run name continues.
       Checkpoints live in this session only: upload the model before the session ends.
    5. **Transcribe**: any model (a checkpoint of a run, or an HF id such as `vinai/PhoWhisper-large`) into an ASR cache.
       Subset `eval` (dev + holdout) also prints the WER report against the other caches and the error analysis.
       The pool cache with the *untuned* PhoWhisper-large (`pho-large-fp16`, subset `pool`) is the noisy text for SFT.
    6. **Upload model** pushes weights + processor (no optimizer state) to the HF model repo, then add an ASR preset with
       `"model": "${HF_MODEL_OWNER}/<repo name>"` on a branch. This notebook never deletes anything on the hub.
    """)
    return


@app.cell
def _(mo, os):
    repo_url = mo.ui.text(value=os.environ.get("LIFEMATE_REPO", "https://github.com/AuDuongKhang/ZAIC-2026-LifeMate-TripleTrouble.git"),
                          label="Repo", full_width=True)
    branch = mo.ui.text(value="main", label="Branch")
    token = mo.ui.text(kind="password", value=os.environ.get("GITHUB_TOKEN", ""), label="GitHub token (repo private)")
    hf_repo = mo.ui.text(value=os.environ.get("LIFEMATE_HF_REPO", "TripleTrouble/zaic2026-lifemate-data"),
                         label="HF dataset (user/name)")
    hf_token = mo.ui.text(kind="password", value=os.environ.get("HF_TOKEN", ""), label="HF token (read)")
    hf_write = mo.ui.text(kind="password", value="", label="HF token (write, Upload only)")
    pull = mo.ui.run_button(label="Pull repo")
    fetch = mo.ui.run_button(label="Fetch data")
    mo.vstack([repo_url, mo.hstack([branch, token]), mo.hstack([hf_repo, hf_token]), hf_write,
               mo.hstack([pull, fetch], justify="start")])
    return branch, fetch, hf_repo, hf_token, hf_write, pull, repo_url, token


@app.cell
def _(Path):
    work = Path.cwd() / "lifemate_work"
    work.mkdir(exist_ok=True)
    repo_dir, data_root = work / "repo", work / "data"
    return data_root, repo_dir, work


@app.cell
def _():
    procs = {}  # the background training process, kept across cell reruns
    return (procs,)


@app.cell
def _(branch, mo, procs, pull, repo_dir, repo_url, shutil, subprocess, sys, token):
    mo.stop(not pull.value, mo.md("_Press **Pull repo** to start._"))
    if procs.get("train") and procs["train"].poll() is None:
        raise RuntimeError("training is running: Stop it (or wait for it to finish) before pulling the repo again")

    def _git(*args, cwd=None):
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
        if r.returncode:
            msg = r.stderr.replace(_token, "***") if _token else r.stderr
            raise RuntimeError(f"git {args[0]} failed: {msg}")
        return r.stdout.strip()

    _token = token.value.strip()
    _url = repo_url.value.strip()
    if _token and _url.startswith("https://"):
        _url = _url.replace("https://", f"https://x-access-token:{_token}@", 1)
    if (repo_dir / ".git").is_dir():
        _git("remote", "set-url", "origin", _url, cwd=repo_dir)
        _git("fetch", "--depth", "1", "origin", branch.value.strip(), cwd=repo_dir)
        _git("reset", "--hard", "FETCH_HEAD", cwd=repo_dir)
    else:
        shutil.rmtree(repo_dir, ignore_errors=True)  # leftover of a failed clone (e.g. a bad token)
        _git("clone", "--depth", "1", "--branch", branch.value.strip(), _url, str(repo_dir))
    commit = _git("log", "-1", "--format=%h", cwd=repo_dir)
    commit_msg = _git("log", "-1", "--format=%s", cwd=repo_dir)

    def _install(*args):
        cmd = [sys.executable, "-m", "pip", "install", "-q", *args]
        if shutil.which("uv"):
            cmd = ["uv", "pip", "install", "--python", sys.executable, "-q", *args]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"pip install {' '.join(args)} failed:\n{r.stderr[-3000:]}")

    _install("-r", str(repo_dir / "requirements.txt"))
    _install("--no-deps", "-r", str(repo_dir / "requirements-train.txt"))  # accelerate (Trainer): keep molab's torch
    _torch = subprocess.run([sys.executable, "-c", "import torch, accelerate; print(torch.__version__, 'accelerate', "
                             "accelerate.__version__)"], capture_output=True, text=True)
    if _torch.returncode:
        raise RuntimeError(f"torch/accelerate import failed:\n{_torch.stderr[-2000:]}")

    _gpu = "no GPU"
    if shutil.which("nvidia-smi"):
        _gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    mo.md(f"**commit** `{commit}` {commit_msg}  \n**torch** `{_torch.stdout.strip()}`  \n**GPU** {_gpu}")
    return (commit,)


@app.cell
def _(commit, data_root, fetch, hf_repo, hf_token, mo, os, repo_dir, subprocess, sys):
    commit  # lifemate.fetch_data comes from the pulled repo; pulling again only re-checks the data, never refetches it
    if fetch.value:
        _fetch = [sys.executable, "-m", "lifemate.fetch_data", "--dest", str(data_root), "--hf-repo", hf_repo.value.strip()]
        _fetch_env = {**os.environ, "PYTHONPATH": str(repo_dir / "src")}
        if hf_token.value:
            _fetch_env["HF_TOKEN"] = hf_token.value.strip()
        _r = subprocess.run(_fetch, cwd=repo_dir, env=_fetch_env, capture_output=True, text=True)
        if _r.returncode:
            _err = _r.stderr[-2000:].replace(hf_token.value, "***") if hf_token.value else _r.stderr[-2000:]
            raise RuntimeError(f"fetching data failed (HF dataset + token filled in?):\n{_err}")
    _splits = sorted(p.name for p in data_root.iterdir() if p.is_dir()) if data_root.is_dir() else []
    mo.md(f"**data** `{data_root}`: {', '.join(_splits)}" if _splits
          else "**data**: _not fetched yet; press **Fetch data** (once per session)_")
    return


@app.cell
def _(commit, data_root, hf_token, os, repo_dir, work):
    commit  # everything below runs the pulled repo's code
    def step_env():
        env = {**os.environ, "PYTHONPATH": str(repo_dir / "src"), "LIFEMATE_DATA_ROOT": str(data_root), "PYTHONUTF8": "1",
               "LIFEMATE_ASR_CACHE": str(work / "asr_cache")}
        if hf_token.value:
            env["HF_TOKEN"] = hf_token.value.strip()  # private base models; uploads use the write token
        return env

    def run_dir(name):
        return work / "asr_runs" / name

    def hide(text):
        return text.replace(hf_token.value.strip(), "***") if hf_token.value.strip() else text
    return hide, run_dir, step_env


@app.cell
def _(mo):
    base_model = mo.ui.dropdown(["vinai/PhoWhisper-large", "vinai/PhoWhisper-medium",
                                 "Qualcomm-AI-Research/PhoASR-whisper-small"], value="vinai/PhoWhisper-large",
                                label="Base model")
    run_name = mo.ui.text(value="pho-large-ft-b", label="Run name")
    epochs = mo.ui.number(start=0.5, stop=20, step=0.5, value=3, label="epochs")
    lr = mo.ui.text(value="1e-5", label="lr")
    batch = mo.ui.number(start=1, stop=64, value=8, label="batch")
    grad_accum = mo.ui.number(start=1, stop=16, value=2, label="grad accum")
    eval_steps = mo.ui.number(start=25, stop=5000, step=25, value=100, label="eval steps")
    resume = mo.ui.checkbox(value=False, label="resume from the newest checkpoint (local or HF)")
    train_extra = mo.ui.text(value="", label="extra train_asr args (e.g. --grad-checkpointing --limit 64)", full_width=True)
    start = mo.ui.run_button(label="Start training")
    stop = mo.ui.run_button(label="Stop")
    mo.vstack([mo.md("## Train"), mo.hstack([base_model, run_name], justify="start"),
               mo.hstack([epochs, lr, batch, grad_accum, eval_steps], justify="start"), resume, train_extra,
               mo.hstack([start, stop], justify="start")])
    return base_model, batch, epochs, eval_steps, grad_accum, lr, resume, run_name, start, stop, train_extra


@app.cell
def _(base_model, batch, data_root, epochs, eval_steps, grad_accum, hide, lr, mo, procs, repo_dir, resume, run_dir,
      run_name, shlex, start, step_env, stop, subprocess, sys, train_extra):
    if stop.value and procs.get("train") and procs["train"].poll() is None:
        procs["train"].terminate()  # checkpoints saved so far stay; resume continues from the last one
    mo.stop(not start.value, mo.md("_Press **Start training**._"))
    if procs.get("train") and procs["train"].poll() is None:
        raise RuntimeError("training is already running; press Stop first")
    if not (data_root / "train").is_dir():
        raise RuntimeError(f"no train data in {data_root}: press **Fetch data** first")

    _out = run_dir(run_name.value.strip())
    _out.mkdir(parents=True, exist_ok=True)
    _cmd = [sys.executable, "-m", "lifemate.train_asr", "--model", base_model.value, "--out", str(_out),
            "--epochs", str(epochs.value), "--lr", lr.value.strip(), "--batch", str(int(batch.value)),
            "--grad-accum", str(int(grad_accum.value)), "--eval-steps", str(int(eval_steps.value)),
            *shlex.split(train_extra.value)]
    if resume.value:
        _cmd.append("--resume")
    (_out / "base_model.txt").write_text(base_model.value, encoding="utf-8")  # its processor, for checkpoint uploads
    _log = open(_out / "stdout.log", "a", encoding="utf-8")
    procs["train"] = subprocess.Popen(_cmd, cwd=repo_dir, env=step_env(), stdout=_log, stderr=subprocess.STDOUT,
                                      text=True)
    mo.md(f"Started `{hide(' '.join(_cmd))}`")
    return


@app.cell
def _(mo):
    refresh = mo.ui.refresh(options=["30s", "60s", "5m"], default_interval="60s")
    refresh
    return (refresh,)


@app.cell
def _(hide, json, mo, procs, refresh, run_dir, run_name):
    refresh
    _out = run_dir(run_name.value.strip())
    _p = procs.get("train")
    _status = "not started" if _p is None else ("running" if _p.poll() is None else f"exited with {_p.returncode}")
    _tail = ""
    if (_out / "stdout.log").exists():
        _lines = (_out / "stdout.log").read_text(encoding="utf-8", errors="replace").replace("\r", "\n").splitlines(True)
        _tail = hide("".join(_lines[-25:]))
    # eval WER per checkpoint, from the newest trainer_state.json (the Trainer writes it with every checkpoint)
    _ckpts = sorted(_out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])) if _out.is_dir() else []
    _evals, _best = [], ""
    if _ckpts and (_ckpts[-1] / "trainer_state.json").exists():
        _state = json.loads((_ckpts[-1] / "trainer_state.json").read_text(encoding="utf-8"))
        _evals = [{"step": h["step"], "epoch": round(h["epoch"], 2), "eval WER": round(h["eval_wer"], 4),
                   "eval CER": round(h["eval_cer"], 4)} for h in _state["log_history"] if "eval_wer" in h]
        if _state.get("best_model_checkpoint"):
            _best = f"best so far: `{_state['best_model_checkpoint'].split('/')[-1]}` WER {_state['best_metric']:.4f}"
    _items = [mo.md(f"**training** `{run_name.value.strip()}`: {_status}  \n{_best}"),
              mo.ui.code_editor(value=_tail, language="markdown", disabled=True)]
    if _evals:
        _items.append(mo.ui.table(_evals, selection=None))
    if (_out / "training.png").exists():
        _items.append(mo.image(src=(_out / "training.png").read_bytes(), width=900))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    rescan = mo.ui.run_button(label="Refresh model list")
    rescan
    return (rescan,)


@app.cell
def _(mo, rescan, run_dir, run_name):
    rescan  # not the minute timer: that would reset the pick below
    _out = run_dir(run_name.value.strip())
    _dirs = [p for p in [_out / "final", *sorted(_out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))]
             if (p / "config.json").exists()] if _out.is_dir() else []
    model_dir = mo.ui.dropdown({p.name: str(p) for p in _dirs}, value=_dirs[0].name if _dirs else None,
                               label=f"Model of run `{run_name.value.strip()}` (final = best by eval WER)")
    model_dir
    return (model_dir,)


@app.cell
def _(run_dir, run_name):
    def ready_model(path):
        """A checkpoint folder has the weights only: add the base model's processor so the whisper backend loads it."""
        from pathlib import Path as _Path

        p = _Path(path)
        if not (p / "preprocessor_config.json").exists():
            from transformers import WhisperProcessor

            _base = (run_dir(run_name.value.strip()) / "base_model.txt").read_text(encoding="utf-8").strip()
            WhisperProcessor.from_pretrained(_base).save_pretrained(str(p))
        return str(p)
    return (ready_model,)


@app.cell
def _(mo, model_dir):
    tr_model = mo.ui.text(value=model_dir.value or "vinai/PhoWhisper-large",
                          label="Model (folder above or HF id)", full_width=True)
    tr_name = mo.ui.text(value="", label="Cache name", placeholder="pho-large-ft-b")
    tr_subset = mo.ui.dropdown({"eval (dev + holdout): WER report": "eval", "pool (noisy text for SFT)": "pool",
                                "dev": "dev", "holdout": "holdout"}, value="eval (dev + holdout): WER report",
                               label="Subset")
    tr_extra = mo.ui.text(value="", label="extra --asr-kw (e.g. dtype=float32)")
    transcribe = mo.ui.run_button(label="Transcribe")
    mo.vstack([mo.md("## Transcribe and report"), tr_model, mo.hstack([tr_name, tr_subset, tr_extra], justify="start"),
               transcribe])
    return tr_extra, tr_model, tr_name, tr_subset, transcribe


@app.cell
def _(hide, mo, ready_model, repo_dir, step_env, subprocess, sys, tr_extra, tr_model, tr_name, tr_subset, transcribe,
      work):
    mo.stop(not transcribe.value, mo.md("_Press **Transcribe**._"))
    _name = tr_name.value.strip()
    if not _name:
        raise RuntimeError("fill in a cache name")
    _model = tr_model.value.strip()
    if _model.startswith("/"):
        _model = ready_model(_model)

    def _stream(cmd):
        out = []
        with subprocess.Popen(cmd, cwd=repo_dir, env=step_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8") as p:
            for line in p.stdout:
                line = hide(line)
                print(line, end="", flush=True)
                out.append(line)
        if p.returncode:
            raise RuntimeError(f"{cmd[2]} exited with {p.returncode}")
        return "".join(out)

    _kw = ["--asr-kw", f"model={_model}"] + [a for kv in tr_extra.value.split() for a in ("--asr-kw", kv)]
    _stream([sys.executable, "-m", "lifemate.transcribe", "--split", "train", "--asr", "whisper", *_kw, "--name", _name,
             "--subset", tr_subset.value])
    _report = ""
    if tr_subset.value == "eval":  # WER against every eval cache of this session, then where this one goes wrong
        _caches = sorted(p.stem for p in (work / "asr_cache" / "train").glob("*.jsonl"))
        _report = _stream([sys.executable, "-m", "lifemate.asr_report", "--split", "train", *_caches])
        _stream([sys.executable, "-m", "lifemate.asr_errors", _name, "--split", "train"])
    _cache = work / "asr_cache" / "train" / f"{_name}.jsonl"
    _files = [_cache, _cache.with_suffix(".errors.md"), _cache.parent / "report.md"]
    mo.vstack([mo.ui.code_editor(value=_report[-5000:], language="markdown", disabled=True),
               mo.md("Download to keep (the session's files go when it ends):"),
               mo.hstack([mo.download(data=p.read_bytes(), filename=p.name, label=p.name) for p in _files if p.exists()],
                         justify="start")])
    return


@app.cell
def _(mo, model_dir, run_name):
    up_repo = mo.ui.text(value=f"TripleTrouble/{run_name.value.strip()}", label="HF model repo (owner/name, private)")
    upload = mo.ui.run_button(label="Upload model")
    mo.vstack([mo.md(f"## Upload\nUploads **{model_dir.value or '(no model yet)'}**: weights, config and processor; "
                     "optimizer state and trainer files stay here. Needs the HF write token."), up_repo, upload])
    return up_repo, upload


@app.cell
def _(hf_write, mo, model_dir, ready_model, up_repo, upload):
    mo.stop(not upload.value, mo.md("_Press **Upload model**._"))
    if not hf_write.value.strip():
        raise RuntimeError("fill in the HF write token")
    if not model_dir.value:
        raise RuntimeError("no model picked: train first (or wait for the first checkpoint)")
    from huggingface_hub import HfApi

    _repo = up_repo.value.strip()
    _api = HfApi(token=hf_write.value.strip())
    _api.create_repo(_repo, private=True, exist_ok=True)
    _api.upload_folder(folder_path=ready_model(model_dir.value), repo_id=_repo,
                       commit_message=f"{model_dir.value.split('/')[-1]} of {model_dir.value.split('/')[-2]}",
                       ignore_patterns=["optimizer.pt", "scheduler.pt", "rng_state*", "trainer_state.json",
                                        "training_args.bin", "scaler.pt"])
    _name = _repo.split("/")[-1]
    mo.md(f"-> https://huggingface.co/{_repo} (private)  \nPreset for `configs/presets.json`: "
          f'`"{_name}": {{"asr": "whisper", "asr_kw": {{"model": "${{HF_MODEL_OWNER}}/{_name}"}}}}`')
    return


if __name__ == "__main__":
    app.run()
