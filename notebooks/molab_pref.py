# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo"]
# ///
"""LifeMate preference training for molab (plan docs/superpowers/plans/2026-11-12-onpolicy-preference.md): pull the
repo's branch, fetch the data, then run one step in the background: records, sample, pairs, train, merge or interpolate.
The commands come from lifemate.sft.pref_commands (unit-tested). Only this file is synced from GitHub, so it clones the
repo itself; data, tokens, steps and pairs never go into the repo (steps files are uploaded to the private HF dataset)."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import shlex
    import shutil
    import signal
    import subprocess
    import sys
    import time
    from pathlib import Path

    import marimo as mo
    return Path, mo, os, shlex, shutil, signal, subprocess, sys, time


@app.cell
def _(mo):
    mo.md("""
    # LifeMate preference training (sft-q1d)

    1. Turn on the GPU (*notebook specs*). Fill in the GitHub token, the HF dataset, an HF **read** token and an HF
       **write** token (steps files, checkpoints and merged models go to private HF repos).
    2. **Pull repo** (branch `feat/onpolicy-pref` until it is merged), then **Fetch data** once per session.
    3. Pick a **step**, check the fields, press **Start**. Run the steps in this order: `records`, `sample`, `pairs`,
       then `train` (one temperature, one *run* suffix per configuration), `merge`; `interpolate` is the WiSE-FT side
       track and needs only *model* and *revision*. The fields of every run are in `docs/molab-runbook.md`, section C5.
    4. Files are shared through the HF dataset: `records` uploads its record files, `sample` its steps file and `pairs`
       its pair files; every later step downloads what this session lacks. So `records` runs **once per name**, even
       across sessions, and the HF **write** token is required (it also reads the private dataset).
    5. **Stop** ends the step. `sample` resumes where it stopped (same name and fields); `train` resumes with
       `--resume`, which this notebook does not add: start a new run name instead, or resume from a terminal.
    """)
    return


@app.cell
def _(mo, os):
    repo_url = mo.ui.text(value=os.environ.get("LIFEMATE_REPO", "https://github.com/AuDuongKhang/ZAIC-2026-LifeMate-TripleTrouble.git"),
                          label="Repo", full_width=True)
    branch = mo.ui.text(value="feat/onpolicy-pref", label="Branch")
    token = mo.ui.text(kind="password", value=os.environ.get("GITHUB_TOKEN", ""), label="GitHub token (repo private)")
    hf_repo = mo.ui.text(value=os.environ.get("LIFEMATE_HF_REPO", "TripleTrouble/zaic2026-lifemate-data"),
                         label="HF dataset (user/name)")
    hf_token = mo.ui.text(kind="password", value=os.environ.get("HF_TOKEN", ""), label="HF token (read)")
    hf_write = mo.ui.text(kind="password", value="", label="HF token (write)")
    pull = mo.ui.run_button(label="Pull repo")
    fetch = mo.ui.run_button(label="Fetch data")
    mo.vstack([repo_url, mo.hstack([branch, token]), mo.hstack([hf_repo, hf_token, hf_write]),
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
    procs = {}  # the background step, kept across cell reruns
    return (procs,)


@app.cell
def _(branch, mo, procs, pull, repo_dir, repo_url, shutil, subprocess, sys, token):
    mo.stop(not pull.value, mo.md("_Press **Pull repo** to start._"))
    if procs.get("step") and procs["step"].poll() is None:
        raise RuntimeError("a step is running: Stop it (or wait for it) before pulling the repo again")

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
        shutil.rmtree(repo_dir, ignore_errors=True)
        _git("clone", "--depth", "1", "--branch", branch.value, _url, str(repo_dir))
    commit = _git("log", "-1", "--format=%h", cwd=repo_dir)

    def _install(*args):
        cmd = [sys.executable, "-m", "pip", "install", "-q", *args]
        if shutil.which("uv"):
            cmd = ["uv", "pip", "install", "--python", sys.executable, "-q", *args]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"pip install {' '.join(args)} failed:\n{r.stderr[-3000:]}")

    _install("-r", str(repo_dir / "requirements.txt"))
    _install("--no-deps", "-r", str(repo_dir / "requirements-train.txt"))  # peft, accelerate: keep molab's torch
    _gpu = "no GPU"
    if shutil.which("nvidia-smi"):
        _gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    mo.md(f"**commit** `{commit}`  \n**GPU** {_gpu}")
    return (commit,)


@app.cell
def _(commit, data_root, fetch, hf_repo, hf_token, mo, os, repo_dir, subprocess, sys):
    commit
    if fetch.value:
        _fetch = [sys.executable, "-m", "lifemate.fetch_data", "--dest", str(data_root), "--hf-repo", hf_repo.value.strip()]
        _env = {**os.environ, "PYTHONPATH": str(repo_dir / "src")}
        if hf_token.value:
            _env["HF_TOKEN"] = hf_token.value
        _r = subprocess.run(_fetch, cwd=repo_dir, env=_env, capture_output=True, text=True)
        if _r.returncode:
            _err = _r.stderr[-2000:].replace(hf_token.value, "***") if hf_token.value else _r.stderr[-2000:]
            raise RuntimeError(f"fetching data failed:\n{_err}")
    _splits = sorted(p.name for p in data_root.iterdir() if p.is_dir()) if data_root.is_dir() else []
    mo.md(f"**data** `{data_root}`: {', '.join(_splits)}" if _splits else "**data**: _press **Fetch data**_")
    return


@app.cell
def _(mo):
    step = mo.ui.dropdown(["records", "sample", "pairs", "train", "merge", "interpolate"], value="records", label="Step")
    name = mo.ui.text(value="pilot", label="Name (records, steps and pairs files)")
    model = mo.ui.text(value="TripleTrouble/lifemate-sft-q1c", label="Model", full_width=True)
    revision = mo.ui.text(value="8cb4336e529f0a4b2d6f10cdcfc63d15c45b651d", label="Model revision (commit sha)",
                          full_width=True)
    k = mo.ui.number(start=1, stop=16, value=8, label="K samples per point")
    temperature = mo.ui.text(value="0.7,1.0", label="Temperature(s), comma-separated (train/merge: one)")
    top_p = mo.ui.number(start=0.1, stop=1.0, step=0.05, value=0.95, label="top-p")
    limit = mo.ui.number(start=0, stop=1000, value=100, label="limit: sample ids (0 = whole pool)")
    lr = mo.ui.text(value="1e-5", label="learning rate")
    beta = mo.ui.text(value="0.1", label="beta")
    sft_weight = mo.ui.text(value="1.0", label="SFT weight")
    owner = mo.ui.text(value="TripleTrouble", label="HF owner")
    run = mo.ui.text(value="", label="run suffix (train/merge, e.g. a)")
    start = mo.ui.run_button(label="Start")
    stop = mo.ui.run_button(label="Stop")
    mo.vstack([mo.hstack([step, name, run]), model, revision, mo.hstack([k, temperature, top_p, limit]),
               mo.hstack([lr, beta, sft_weight, owner]), mo.hstack([start, stop], justify="start")])
    return beta, k, limit, lr, model, name, owner, revision, run, sft_weight, start, step, stop, temperature, top_p


@app.cell
def _(commit, data_root, hf_write, os, repo_dir, work):
    commit  # everything below runs the pulled repo's code
    def step_env():
        env = {**os.environ, "PYTHONPATH": str(repo_dir / "src"), "LIFEMATE_DATA_ROOT": str(data_root), "PYTHONUTF8": "1",
               "LIFEMATE_ASR_CACHE": str(work / "asr_cache"), "PYTHONUNBUFFERED": "1"}  # the log fills as it runs
        if hf_write.value:
            env["HF_TOKEN"] = hf_write.value
        return env
    return (step_env,)


@app.cell
def _(beta, data_root, hf_repo, hf_write, k, limit, lr, mo, model, name, os, owner, procs, repo_dir, revision, run,
      sft_weight, shlex, signal, start, step, step_env, stop, subprocess, sys, temperature, time, top_p, work):
    if stop.value and procs.get("step") and procs["step"].poll() is None:
        os.killpg(procs["step"].pid, signal.SIGTERM)  # the whole chain: bash and the python it runs
    mo.stop(not start.value, mo.md("_Pick a step and press **Start**._"))
    if procs.get("step") and procs["step"].poll() is None:
        raise RuntimeError("a step is already running; press Stop first")
    if not (data_root / "train").is_dir():
        raise RuntimeError(f"no train data in {data_root}: press **Fetch data** first")
    sys.path.insert(0, str(repo_dir / "src"))
    from lifemate.sft.pref_commands import commands

    _f = {"name": name.value.strip(), "model": model.value.strip(), "revision": revision.value.strip(), "k": k.value,
          "temperature": temperature.value, "top_p": top_p.value, "limit": limit.value, "lr": lr.value.strip(),
          "beta": beta.value.strip(), "sft_weight": sft_weight.value.strip(), "owner": owner.value.strip(),
          "data_repo": hf_repo.value.strip(), "run": run.value.strip()}
    _cmds = commands(step.value, _f)
    _log_path = work / f"pref-{step.value}.log"
    procs["log_start"] = _log_path.stat().st_size if _log_path.exists() else 0  # bytes: this run's part of the log
    _log = open(_log_path, "a", encoding="utf-8")
    procs["step"] = subprocess.Popen(["bash", "-c", " && ".join(shlex.join(c) for c in _cmds)], cwd=repo_dir,
                                     env=step_env(), stdout=_log, stderr=subprocess.STDOUT, text=True,
                                     start_new_session=True)
    procs["log"] = work / f"pref-{step.value}.log"
    procs["name"], procs["started"] = step.value, time.time()
    from lifemate.sft.pref_commands import run_name

    procs["train_out"] = repo_dir / "checkpoints" / run_name(_f) if step.value == "train" else None
    _shown = "\n".join(shlex.join(c) for c in _cmds)
    mo.md(f"Started:\n```\n{_shown.replace(hf_write.value, '***') if hf_write.value else _shown}\n```")
    return


@app.cell
def _(mo):
    refresh = mo.ui.refresh(options=["10s", "30s", "60s", "5m"], default_interval="10s")
    refresh
    return (refresh,)


@app.cell
def _(hf_write, mo, procs, refresh, repo_dir, sys, time):
    refresh
    _p = procs.get("step")
    _status = "not started" if _p is None else ("running" if _p.poll() is None else f"exited with {_p.returncode}")
    _text = ""
    if procs.get("log") and procs["log"].exists():
        _text = procs["log"].read_bytes()[procs.get("log_start", 0):].decode("utf-8", errors="replace")
        if hf_write.value:
            _text = _text.replace(hf_write.value, "***")
    _bar = mo.md("")
    if (repo_dir / "src").is_dir():
        sys.path.insert(0, str(repo_dir / "src"))
        from lifemate.sft.pref_commands import progress

        _done = progress(_text)
        if _done:  # sample prints `progress d/t points ...`; other steps show their log only
            _bar = mo.Html(f'<progress value="{_done[0]}" max="{_done[1]}" style="width:100%"></progress>'
                           f"<div>{_done[0]} / {_done[1]} points ({100 * _done[0] / max(_done[1], 1):.0f}%)</div>")
    _elapsed = f", {(time.time() - procs['started']) / 60:.0f} min since Start" if procs.get("started") else ""
    _tail = "".join(_text.splitlines(True)[-30:])
    _plots = mo.md("")
    _out = procs.get("train_out")
    if _out is not None and (_out / "train_log.jsonl").exists():  # the training curves, as molab_train.py shows them
        from lifemate.sft.plots import plot_log

        _pngs = plot_log(_out / "train_log.jsonl", _out / "live_plots")
        _plots = mo.vstack([mo.hstack([mo.image(src=p.read_bytes(), width=330) for p in _pngs[i:i + 3]], justify="start")
                            for i in range(0, len(_pngs), 3)])
    mo.vstack([mo.md(f"**step** `{procs.get('name', '')}`: {_status}{_elapsed}"), _bar, _plots,
               mo.ui.code_editor(value=_tail, language="markdown", disabled=True)])
    return


if __name__ == "__main__":
    app.run()
