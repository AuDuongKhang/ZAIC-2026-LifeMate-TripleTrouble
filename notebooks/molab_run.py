# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo"]
# ///
"""LifeMate runner for molab: pull the repo's branch, fetch the data, run lifemate.infer (+ evaluate on train)
and hand back submission.jsonl plus a RESULTS.md row. Only this file is synced from GitHub, so it clones the repo
itself; the competition data is downloaded at run time and never committed."""

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
    from datetime import datetime
    from pathlib import Path

    import marimo as mo
    return Path, datetime, mo, os, shlex, shutil, subprocess, sys


@app.cell
def _(mo):
    mo.md("""
    # LifeMate runner

    1. Turn on the GPU with the *notebook specs* button in the header.
    2. Fill in the repo, a token (private repo) and the data zip link (only needed once per session).
    3. Pick split, agent and ASR, then press **Pull + run**. Every press pulls the latest commit of the branch.

    On `train` the run is scored locally; on `public_test` download `submission.jsonl` and submit it on the portal.
    """)
    return


@app.cell
def _(mo, os):
    repo_url = mo.ui.text(value=os.environ.get("LIFEMATE_REPO", "https://github.com/AuDuongKhang/ZAIC-2026-LifeMate-TripleTrouble.git"),
                          label="Repo", full_width=True)
    branch = mo.ui.text(value="main", label="Branch")
    token = mo.ui.text(kind="password", value=os.environ.get("GITHUB_TOKEN", ""), label="GitHub token (repo private)")
    data_url = mo.ui.text(value=os.environ.get("LIFEMATE_DATA_URL", ""), label="Data zip URL (train+public_test)",
                          full_width=True)
    split = mo.ui.dropdown(["train", "public_test"], value="train", label="Split")
    agent = mo.ui.text(value="oracle", label="--agent")
    asr = mo.ui.text(value="gold", label="--asr")
    limit = mo.ui.number(start=0, stop=100000, value=0, label="--limit (0 = all)")
    extra = mo.ui.text(value="", label="extra infer args", full_width=True)
    run = mo.ui.run_button(label="Pull + run")
    mo.vstack([repo_url, mo.hstack([branch, token]), data_url, mo.hstack([split, agent, asr, limit]), extra, run])
    return agent, asr, branch, data_url, extra, limit, repo_url, run, split, token


@app.cell
def _(Path, branch, data_url, mo, os, repo_url, run, shutil, subprocess, sys, token):
    mo.stop(not run.value, mo.md("_Press **Pull + run** to start._"))

    work = Path.cwd() / "lifemate_work"
    work.mkdir(exist_ok=True)
    repo_dir = work / "repo"

    def _git(*args, cwd=None):
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
        if r.returncode:
            msg = r.stderr.replace(token.value, "***") if token.value else r.stderr
            raise RuntimeError(f"git {args[0]} failed: {msg}")
        return r.stdout.strip()

    _url = repo_url.value.strip()
    if token.value and _url.startswith("https://"):
        _url = _url.replace("https://", f"https://x-access-token:{token.value}@", 1)
    if (repo_dir / ".git").is_dir():
        _git("remote", "set-url", "origin", _url, cwd=repo_dir)
        _git("fetch", "--depth", "1", "origin", branch.value, cwd=repo_dir)
        _git("reset", "--hard", "FETCH_HEAD", cwd=repo_dir)
    else:
        _git("clone", "--depth", "1", "--branch", branch.value, _url, str(repo_dir))
    commit = _git("log", "-1", "--format=%h", cwd=repo_dir)
    commit_msg = _git("log", "-1", "--format=%s", cwd=repo_dir)

    # same command as on laptops; a no-op when this session already has the data
    data_root = work / "data"
    _fetch = [sys.executable, "-m", "lifemate.fetch_data", "--dest", str(data_root)]
    if data_url.value.strip():
        _fetch += ["--url", data_url.value.strip()]
    _r = subprocess.run(_fetch, cwd=repo_dir, env={**os.environ, "PYTHONPATH": str(repo_dir / "src")},
                        capture_output=True, text=True)
    print(_r.stdout[-2000:], flush=True)
    if _r.returncode:
        raise RuntimeError(f"fetching data failed (is 'Data zip URL' filled in?):\n{_r.stderr[-2000:]}")

    _req = repo_dir / "requirements.txt"
    _pip = [sys.executable, "-m", "pip", "install", "-q", "-r", str(_req)]
    if shutil.which("uv"):
        _pip = ["uv", "pip", "install", "--python", sys.executable, "-q", "-r", str(_req)]
    _r = subprocess.run(_pip, capture_output=True, text=True)
    if _r.returncode:
        raise RuntimeError(f"installing requirements failed:\n{_r.stderr[-3000:]}")

    _gpu = "no GPU"
    if shutil.which("nvidia-smi"):
        _gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    mo.md(f"**commit** `{commit}` {commit_msg}  \n**data** `{data_root}`  \n**GPU** {_gpu}")
    return commit, data_root, repo_dir, work


@app.cell
def _(agent, asr, commit, data_root, datetime, extra, limit, os, repo_dir, shlex, split, subprocess, sys, work):
    run_dir = work / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}_{commit}_{split.value}_{agent.value}_{asr.value}"
    _env = {**os.environ, "PYTHONPATH": str(repo_dir / "src"), "LIFEMATE_DATA_ROOT": str(data_root), "PYTHONUTF8": "1"}
    _cmd = [sys.executable, "-m", "lifemate.infer", "--split", split.value, "--agent", agent.value,
            "--asr", asr.value, "--out", str(run_dir), *shlex.split(extra.value)]
    if limit.value:
        _cmd += ["--limit", str(int(limit.value))]

    def _stream(cmd):
        """Run a step in its own process (each split has its own zaic_tools; fresh code after every pull)."""
        out = []
        with subprocess.Popen(cmd, cwd=repo_dir, env=_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8") as p:
            for line in p.stdout:
                print(line, end="", flush=True)
                out.append(line)
        if p.returncode:
            raise RuntimeError(f"{cmd[2]} exited with {p.returncode}")
        return "".join(out)

    _stream(_cmd)
    if split.value == "train":
        _stream([sys.executable, "-m", "lifemate.evaluate", str(run_dir)])
    results_row = _stream([sys.executable, "-m", "lifemate.evaluate", "--row-only", str(run_dir)]).strip()
    return results_row, run_dir


@app.cell
def _(mo, results_row, run_dir):
    import json as _json

    _items = [mo.md("### Results"), mo.md("Row for `RESULTS.md`:"),
              mo.ui.code_editor(value=results_row, language="markdown", disabled=True)]
    _metrics_path = run_dir / "metrics.json"
    if _metrics_path.exists():
        _m = _json.loads(_metrics_path.read_text(encoding="utf-8"))
        _rows = [{"group": "overall", **_m["overall"]}] + [{"group": k, **v} for k, v in _m["by_group"].items()]
        _items.append(mo.ui.table(_rows, selection=None))
    _downloads = [mo.download(data=(run_dir / f).read_bytes(), filename=f, label=f)
                  for f in ("submission.jsonl", "meta.json", "metrics.json", "per_sample.jsonl", "trace.jsonl")
                  if (run_dir / f).exists()]
    _items.append(mo.hstack(_downloads, justify="start"))
    mo.vstack(_items)
    return


if __name__ == "__main__":
    app.run()
