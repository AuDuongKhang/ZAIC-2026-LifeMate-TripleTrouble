# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo"]
# ///
"""LifeMate SFT trainer for molab: pull the repo's branch, fetch the data, render the SFT records, train the LoRA
adapter in the background with live plots, then merge it and push it to a private HF model repo. Only this file is
synced from GitHub, so it clones the repo itself; data, tokens and checkpoints never go into the repo."""

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
    # LifeMate SFT trainer

    1. Turn on the GPU with the *notebook specs* button in the header.
    2. Fill in the GitHub token, the HF read token (data), the HF model repo and an HF **write** token for it.
    3. **Pull**: pulls the branch, installs `requirements.txt` + `requirements-train.txt` (`--no-deps`, keeps molab's
       torch), fetches the data.
    4. **Start training**: renders the records (`lifemate.sft.records`: pool for experiments, all train only for the final
       retrain; dev or holdout for the eval loss, never trained on), then
       runs `lifemate.sft.train` in the background. The log and plots below refresh every minute. Checkpoints go to
       `<model repo>/checkpoints/<run>/last` every 30 minutes and at the end.
    5. **Stop** saves a checkpoint at the next optimizer step, then exits. In a new session (or after Stop) tick
       *resume* and press **Start training** again with the same run name, records and extra args; training continues
       from the checkpoint in the HF model repo (the session's own copy only when it is newer, e.g. a failed upload). Do not press **Pull** or edit the fields above it while training runs.
    6. **Merge** once training has finished: merges the adapter into the base weights and uploads the model to the root of
       the model repo. Then add an agent preset with `"model": "${HF_MODEL_OWNER}/<repo name>"` on the branch.
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
    model_repo = mo.ui.text(value="", label="HF model repo (owner/name, private)", placeholder="TripleTrouble/lifemate-sft-v1")
    hf_write = mo.ui.text(kind="password", value="", label="HF token (write)")
    pull = mo.ui.run_button(label="Pull")
    mo.vstack([repo_url, mo.hstack([branch, token]), mo.hstack([hf_repo, hf_token]), mo.hstack([model_repo, hf_write]),
               pull])
    return branch, hf_repo, hf_token, hf_write, model_repo, pull, repo_url, token


@app.cell
def _(Path, branch, hf_repo, hf_token, mo, os, pull, repo_url, shutil, subprocess, sys, token):
    mo.stop(not pull.value, mo.md("_Press **Pull** to start._"))

    work = Path.cwd() / "lifemate_work"
    work.mkdir(exist_ok=True)
    repo_dir = work / "repo"

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
        _git("fetch", "--depth", "1", "origin", branch.value, cwd=repo_dir)
        _git("reset", "--hard", "FETCH_HEAD", cwd=repo_dir)
    else:
        shutil.rmtree(repo_dir, ignore_errors=True)  # leftover of a failed clone (e.g. a bad token)
        _git("clone", "--depth", "1", "--branch", branch.value, _url, str(repo_dir))
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
    _install("--no-deps", "-r", str(repo_dir / "requirements-train.txt"))  # peft, accelerate: keep molab's torch
    _torch = subprocess.run([sys.executable, "-c", "import torch, peft; print(torch.__file__, torch.__version__, "
                             "'peft', peft.__version__)"], capture_output=True, text=True)
    if _torch.returncode:
        raise RuntimeError(f"torch/peft import failed:\n{_torch.stderr[-2000:]}")

    data_root = work / "data"
    _fetch = [sys.executable, "-m", "lifemate.fetch_data", "--dest", str(data_root), "--hf-repo", hf_repo.value.strip()]
    _fetch_env = {**os.environ, "PYTHONPATH": str(repo_dir / "src")}
    if hf_token.value:
        _fetch_env["HF_TOKEN"] = hf_token.value
    _r = subprocess.run(_fetch, cwd=repo_dir, env=_fetch_env, capture_output=True, text=True)
    if _r.returncode:
        _err = _r.stderr[-2000:].replace(hf_token.value, "***") if hf_token.value else _r.stderr[-2000:]
        raise RuntimeError(f"fetching data failed (HF dataset + token filled in?):\n{_err}")

    _gpu = "no GPU"
    if shutil.which("nvidia-smi"):
        _gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    mo.md(f"**commit** `{commit}` {commit_msg}  \n**torch** `{_torch.stdout.strip()}`  \n**data** `{data_root}`  \n"
          f"**GPU** {_gpu}")
    return commit, data_root, repo_dir, work


@app.cell
def _(mo):
    run_name = mo.ui.text(value="sft-v1", label="Run name")
    # Record files are named gold-<subset> (gold-<subset>-k<k> when k != 4), so the same choices find the same file again
    records = mo.ui.dropdown({"pool (experiments)": "pool", "all train (final retrain only)": "all"},
                             value="pool (experiments)", label="Train records")
    eval_records = mo.ui.dropdown({"dev": "dev", "holdout": "holdout", "none": ""}, value="dev",
                                  label="Eval loss records")
    examples = mo.ui.number(start=0, stop=8, value=4, label="few-shot examples in the records")
    resume = mo.ui.checkbox(value=False, label="resume from the last checkpoint")
    max_steps = mo.ui.number(start=0, stop=100000, value=0, label="--max-steps (0 = full run)")
    extra = mo.ui.text(value="", label="extra train args (e.g. --epochs 1)", full_width=True)
    start = mo.ui.run_button(label="Start training")
    stop = mo.ui.run_button(label="Stop")
    mo.vstack([mo.hstack([run_name, records, eval_records], justify="start"),
               mo.hstack([examples, max_steps, resume], justify="start"), extra, mo.hstack([start, stop], justify="start")])
    return eval_records, examples, extra, max_steps, records, resume, run_name, start, stop


@app.cell
def _():
    procs = {}  # the background training process, kept across cell reruns
    return (procs,)


@app.cell
def _(data_root, hf_write, model_repo, os, repo_dir, work):
    def step_env():
        env = {**os.environ, "PYTHONPATH": str(repo_dir / "src"), "LIFEMATE_DATA_ROOT": str(data_root), "PYTHONUTF8": "1"}
        if hf_write.value:
            env["HF_TOKEN"] = hf_write.value  # writes checkpoints and the merged model
        return env

    def run_dir(name):
        return work / "checkpoints" / name

    def repo_name():
        return model_repo.value.strip()
    return repo_name, run_dir, step_env


@app.cell
def _(eval_records, examples, extra, hf_write, max_steps, mo, procs, records, repo_dir, repo_name, resume, run_dir,
      run_name, shlex, start, step_env, stop, subprocess, sys):
    if stop.value and procs.get("train") and procs["train"].poll() is None:
        (run_dir(run_name.value) / "STOP").write_text("", encoding="utf-8")  # saves at the next step, then exits
    mo.stop(not start.value, mo.md("_Press **Start training**._"))
    if procs.get("train") and procs["train"].poll() is None:
        raise RuntimeError("training is already running; press Stop first")

    def _run(cmd):
        r = subprocess.run(cmd, cwd=repo_dir, env=step_env(), capture_output=True, text=True, encoding="utf-8")
        print(r.stdout[-3000:], flush=True)
        if r.returncode:
            raise RuntimeError(f"{cmd[2]} failed:\n{r.stderr[-3000:]}")

    def _name(subset):
        return f"gold-{subset}" + ("" if int(examples.value) == 4 else f"-k{int(examples.value)}")

    _k = ["--examples", str(int(examples.value)), "--tokenizer", "none"]
    _run([sys.executable, "-m", "lifemate.sft.records", "--name", _name(records.value), "--subset", records.value, *_k])
    if eval_records.value:
        _run([sys.executable, "-m", "lifemate.sft.records", "--name", _name(eval_records.value), "--subset",
              eval_records.value, *_k])

    _out = run_dir(run_name.value)
    _out.mkdir(parents=True, exist_ok=True)
    _cmd = [sys.executable, "-m", "lifemate.sft.train", "--records", _name(records.value), "--out", str(_out),
            *shlex.split(extra.value)]
    if eval_records.value:
        _cmd += ["--eval-records", _name(eval_records.value)]
    if repo_name():
        _cmd += ["--hf-repo", repo_name()]
    if resume.value:
        _cmd.append("--resume")
    if max_steps.value:
        _cmd += ["--max-steps", str(int(max_steps.value))]
    _log = open(_out / "stdout.log", "a", encoding="utf-8")
    procs["train"] = subprocess.Popen(_cmd, cwd=repo_dir, env=step_env(), stdout=_log, stderr=subprocess.STDOUT,
                                      text=True)
    _shown = " ".join(_cmd).replace(hf_write.value, "***") if hf_write.value else " ".join(_cmd)
    mo.md(f"Started `{_shown}`")
    return


@app.cell
def _(mo):
    refresh = mo.ui.refresh(options=["30s", "60s", "5m"], default_interval="60s")
    refresh
    return (refresh,)


@app.cell
def _(hf_write, mo, procs, refresh, repo_dir, run_dir, run_name, sys):
    refresh
    _out = run_dir(run_name.value)
    _p = procs.get("train")
    _status = "not started" if _p is None else ("running" if _p.poll() is None else f"exited with {_p.returncode}")
    _tail = ""
    if (_out / "stdout.log").exists():
        _tail = "".join((_out / "stdout.log").read_text(encoding="utf-8", errors="replace").splitlines(True)[-25:])
        if hf_write.value:
            _tail = _tail.replace(hf_write.value, "***")
    if str(repo_dir / "src") not in sys.path:
        sys.path.insert(0, str(repo_dir / "src"))
    from lifemate.sft.plots import plot_log

    _pngs = plot_log(_out / "train_log.jsonl", _out / "live_plots")
    mo.vstack([mo.md(f"**training**: {_status}"),
               mo.ui.code_editor(value=_tail, language="markdown", disabled=True),
               mo.hstack([mo.image(src=p.read_bytes(), width=330) for p in _pngs], justify="start")])
    return


@app.cell
def _(mo):
    merge = mo.ui.run_button(label="Merge and upload")
    merge
    return (merge,)


@app.cell
def _(examples, merge, mo, procs, repo_dir, repo_name, run_dir, run_name, step_env, subprocess, sys, work):
    mo.stop(not merge.value, mo.md("_**Merge** once training has finished._"))
    if procs.get("train") and procs["train"].poll() is None:
        raise RuntimeError("training is still running")
    _cmd = [sys.executable, "-m", "lifemate.sft.merge", "--adapter", str(run_dir(run_name.value) / "last"),
            "--out", str(work / "merged" / run_name.value)]
    if repo_name():
        _cmd += ["--hf-repo", repo_name(), "--adapter-repo", repo_name()]
    _r = subprocess.run(_cmd, cwd=repo_dir, env=step_env(), capture_output=True, text=True, encoding="utf-8")
    if _r.returncode:
        raise RuntimeError(f"merge failed:\n{_r.stderr[-3000:]}")
    _name = repo_name().split("/")[-1] if repo_name() else "<repo name>"
    mo.md(f"```\n{_r.stdout[-2000:]}\n```\nPreset for `configs/presets.json`: "
          f'`"{run_name.value}": {{"agent": "react", "agent_kw": {{"model": "${{HF_MODEL_OWNER}}/{_name}", "examples": {int(examples.value)}}}}}`')
    return


if __name__ == "__main__":
    app.run()
