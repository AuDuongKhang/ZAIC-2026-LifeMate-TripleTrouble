# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo"]
# ///
"""LifeMate reasoning-sentence writer for molab: pull the repo's branch, fetch the data, then run
`lifemate.sft.thoughts` in the background. A local open teacher model writes one short sentence per gold step of the
train samples; the rows are uploaded to the private HF dataset as thoughts/<name>.jsonl. Only this file is synced from
GitHub, so it clones the repo itself; data, tokens and sentences never go into the repo."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import shlex
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    import marimo as mo
    return Path, mo, os, shlex, shutil, subprocess, sys


@app.cell
def _(mo):
    mo.md("""
    # LifeMate reasoning-sentence writer

    1. Turn on the GPU with the *notebook specs* button in the header. The teacher (`Qwen/Qwen3.5-27B` by default) needs
       about 56-61 GB at 16 bits, so pick the largest GPU molab offers; the Pull cell shows the GPU's memory.
    2. Fill in the GitHub token, the HF dataset, an HF **read** token (data) and an HF **write** token (the sentences
       are uploaded to the same private dataset).
    3. **Pull repo**: pulls the branch and installs `requirements.txt` plus the *extra pip packages* (this session
       only; a teacher whose architecture is new may need `transformers>=5`). Press it again whenever the branch has new
       commits (not while the writer runs). **Fetch data** once per session, after the first Pull repo.
    4. Read the teacher's model card yourself (licence, size) and paste its commit sha into *revision*: every row
       records the teacher and revision, and a file never mixes two of them.
    5. **Start**: runs `lifemate.sft.thoughts` in the background on the train split only. *limit* draws that many
       samples at random (0 = the whole subset). Files go to `<work>/thoughts/<name>.jsonl` and are uploaded to the HF
       dataset every 200 samples as `thoughts/<name>.jsonl`. The log refreshes every minute.
    6. **Stop** ends the run (every row is flushed, so nothing is lost). After Stop, a crash or in a new session press
       **Start** again with the same name, subset and revision: finished samples are skipped and the HF copy is pulled
       when the session has none. *holdout* is written only after the model trained on the pool has passed its gate.
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
    hf_write = mo.ui.text(kind="password", value="", label="HF token (write)")
    extra_pip = mo.ui.text(value="", label="extra pip packages (this session only)", placeholder="transformers>=5",
                           full_width=True)
    pull = mo.ui.run_button(label="Pull repo")
    fetch = mo.ui.run_button(label="Fetch data")
    mo.vstack([repo_url, mo.hstack([branch, token]), mo.hstack([hf_repo, hf_token, hf_write]), extra_pip,
               mo.hstack([pull, fetch], justify="start")])
    return branch, extra_pip, fetch, hf_repo, hf_token, hf_write, pull, repo_url, token


@app.cell
def _(Path):
    work = Path.cwd() / "lifemate_work"
    work.mkdir(exist_ok=True)
    repo_dir, data_root, thoughts_dir = work / "repo", work / "data", work / "thoughts"
    return data_root, repo_dir, thoughts_dir, work


@app.cell
def _():
    procs = {}  # the background writer process, kept across cell reruns
    return (procs,)


@app.cell
def _(branch, extra_pip, mo, procs, pull, repo_dir, repo_url, shlex, shutil, subprocess, sys, token):
    mo.stop(not pull.value, mo.md("_Press **Pull repo** to start._"))
    if procs.get("write") and procs["write"].poll() is None:
        raise RuntimeError("the writer is running: Stop it (or wait for it to finish) before pulling the repo again")

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
    if extra_pip.value.strip():
        _install(*shlex.split(extra_pip.value))
    _versions = subprocess.run([sys.executable, "-c", "import torch, transformers; print('torch', torch.__version__, "
                               "'transformers', transformers.__version__)"], capture_output=True, text=True)
    if _versions.returncode:
        raise RuntimeError(f"torch/transformers import failed:\n{_versions.stderr[-2000:]}")

    _gpu = "no GPU"
    if shutil.which("nvidia-smi"):
        _gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    mo.md(f"**commit** `{commit}` {commit_msg}  \n**libraries** `{_versions.stdout.strip()}`  \n**GPU** {_gpu}")
    return (commit,)


@app.cell
def _(commit, data_root, fetch, hf_repo, hf_token, mo, os, repo_dir, subprocess, sys):
    commit  # lifemate.fetch_data comes from the pulled repo; pulling again only re-checks the data, never refetches it
    if fetch.value:
        _fetch = [sys.executable, "-m", "lifemate.fetch_data", "--dest", str(data_root), "--hf-repo", hf_repo.value.strip()]
        _fetch_env = {**os.environ, "PYTHONPATH": str(repo_dir / "src")}
        if hf_token.value:
            _fetch_env["HF_TOKEN"] = hf_token.value
        _r = subprocess.run(_fetch, cwd=repo_dir, env=_fetch_env, capture_output=True, text=True)
        if _r.returncode:
            _err = _r.stderr[-2000:].replace(hf_token.value, "***") if hf_token.value else _r.stderr[-2000:]
            raise RuntimeError(f"fetching data failed (HF dataset + token filled in?):\n{_err}")
    _splits = sorted(p.name for p in data_root.iterdir() if p.is_dir()) if data_root.is_dir() else []
    mo.md(f"**data** `{data_root}`: {', '.join(_splits)}" if _splits
          else "**data**: _not fetched yet; press **Fetch data** (once per session)_")
    return


@app.cell
def _(mo):
    name = mo.ui.text(value="pilot1", label="Name (file thoughts/<name>.jsonl)")
    subset = mo.ui.dropdown({"pool": "pool", "dev": "dev", "holdout (only after the pool model passed its gate)": "holdout"},
                            value="pool", label="Subset")
    teacher = mo.ui.text(value="Qwen/Qwen3.5-27B", label="Teacher model", full_width=True)
    revision = mo.ui.text(value="", label="Teacher revision (commit sha, required)", full_width=True)
    limit = mo.ui.number(start=0, stop=5000, value=100, label="limit (0 = the whole subset)")
    start = mo.ui.run_button(label="Start")
    stop = mo.ui.run_button(label="Stop")
    mo.vstack([mo.hstack([name, subset, limit], justify="start"), teacher, revision, mo.hstack([start, stop], justify="start")])
    return limit, name, revision, start, stop, subset, teacher


@app.cell
def _(commit, data_root, hf_write, os, repo_dir):
    commit  # everything below runs the pulled repo's code
    def step_env():
        env = {**os.environ, "PYTHONPATH": str(repo_dir / "src"), "LIFEMATE_DATA_ROOT": str(data_root), "PYTHONUTF8": "1"}
        if hf_write.value:
            env["HF_TOKEN"] = hf_write.value  # pulls and uploads the sentence files
        return env
    return (step_env,)


@app.cell
def _(data_root, hf_repo, hf_write, limit, mo, name, procs, repo_dir, revision, start, step_env, stop, subprocess, subset,
      sys, teacher, thoughts_dir):
    if stop.value and procs.get("write") and procs["write"].poll() is None:
        procs["write"].terminate()  # every row is flushed after its sample: a stop loses at most the sample in progress
    mo.stop(not start.value, mo.md("_Press **Start**._"))
    if procs.get("write") and procs["write"].poll() is None:
        raise RuntimeError("the writer is already running; press Stop first")
    if not revision.value.strip():
        raise RuntimeError("the teacher revision is empty: paste its commit sha from the model page (it is pinned in every row)")
    if not name.value.strip():
        raise RuntimeError("the name is empty")
    if not (data_root / "train").is_dir():
        raise RuntimeError(f"no train data in {data_root}: press **Fetch data** first")

    _cmd = [sys.executable, "-m", "lifemate.sft.thoughts", "--name", name.value.strip(), "--subset", subset.value,
            "--revision", revision.value.strip(), "--teacher", teacher.value.strip(), "--hf-repo", hf_repo.value.strip(),
            "--data-root", str(data_root), "--out-dir", str(thoughts_dir)]
    if limit.value:
        _cmd += ["--limit", str(int(limit.value))]
    thoughts_dir.mkdir(parents=True, exist_ok=True)
    _log = open(thoughts_dir / f"{name.value.strip()}.log", "a", encoding="utf-8")
    procs["write"] = subprocess.Popen(_cmd, cwd=repo_dir, env=step_env(), stdout=_log, stderr=subprocess.STDOUT, text=True)
    _shown = " ".join(_cmd).replace(hf_write.value, "***") if hf_write.value else " ".join(_cmd)
    mo.md(f"Started `{_shown}`")
    return


@app.cell
def _(mo):
    refresh = mo.ui.refresh(options=["30s", "60s", "5m"], default_interval="60s")
    refresh
    return (refresh,)


@app.cell
def _(hf_write, mo, name, procs, refresh, thoughts_dir):
    refresh
    _p = procs.get("write")
    _status = "not started" if _p is None else ("running" if _p.poll() is None else f"exited with {_p.returncode}")
    _log, _rows = thoughts_dir / f"{name.value.strip()}.log", thoughts_dir / f"{name.value.strip()}.jsonl"
    _tail = ""
    if _log.exists():
        _tail = "".join(_log.read_text(encoding="utf-8", errors="replace").splitlines(True)[-25:])
        if hf_write.value:
            _tail = _tail.replace(hf_write.value, "***")
    _n = sum(1 for _ in open(_rows, encoding="utf-8")) if _rows.exists() else 0
    mo.vstack([mo.md(f"**writer**: {_status}; **{_n}** samples written to `{_rows.name}`"),
               mo.ui.code_editor(value=_tail, language="markdown", disabled=True)])
    return


if __name__ == "__main__":
    app.run()
