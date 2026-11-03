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
    import json
    import os
    import shlex
    import shutil
    import subprocess
    import sys
    from datetime import datetime
    from pathlib import Path

    import marimo as mo
    return Path, datetime, json, mo, os, shlex, shutil, subprocess, sys


@app.cell
def _(mo):
    mo.md("""
    # LifeMate runner

    1. Turn on the GPU with the *notebook specs* button in the header.
    2. Fill in the GitHub token (private repo) and the HF read token.
    3. Press **Pull repo**: it pulls the latest commit of the branch, installs the requirements and lists the presets.
       Press it again whenever the branch has new commits.
    4. Press **Fetch data** once per session (after the first Pull repo); the data stays for the whole session.
    5. Pick an agent preset and an ASR preset (configs/presets.json on the pulled branch; `cached:<name>` entries are
       the ASR caches in this session) and press **Run**. Add a new setup as one line in presets.json on the branch.
       The HF model owner list holds the HF token's user and orgs; it fills `${HF_MODEL_OWNER}` in presets.

    On `train`, **Train subset** picks `dev`, `holdout` or `pool` (`all` = the whole split); it is ignored on `public_test`.
    On `train` the run is scored locally; on `public_test` download `submission.jsonl` and submit it on the portal.

    **Command** (optional) runs one other step instead of infer, as `python -m lifemate.<command>` from the work folder,
    for example `transcribe --split train --asr whisper --asr-kw model=vinai/PhoWhisper-large --name pho-large-fp16 --subset eval`,
    `asr_report --split train pho-large-fp16` or `compare runs/<a> runs/<b>`. ASR caches stay in `lifemate_work/asr_cache/`
    for the session, so later runs can use `--asr cached --asr-kw name=<cache>`. The HF token is also passed to every step,
    so private model repos load too.
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
    data_url = mo.ui.text(value=os.environ.get("LIFEMATE_DATA_URL", ""), label="or data zip URL", full_width=True)
    split = mo.ui.dropdown(["train", "public_test"], value="train", label="Split")
    subset = mo.ui.dropdown(["all", "dev", "holdout", "pool"], value="all", label="Train subset (--subset)")
    limit = mo.ui.number(start=0, stop=100000, value=0, label="--limit (0 = all)")
    extra = mo.ui.text(value="", label="extra infer args", full_width=True)
    command = mo.ui.text(value="", label="or command instead of infer (lifemate.<command> args)", full_width=True)
    pull = mo.ui.run_button(label="Pull repo")
    fetch = mo.ui.run_button(label="Fetch data")
    mo.vstack([repo_url, mo.hstack([branch, token]), mo.hstack([hf_repo, hf_token]), data_url,
               mo.hstack([split, subset, limit]), extra, command, mo.hstack([pull, fetch], justify="start")])
    return (branch, command, data_url, extra, fetch, hf_repo, hf_token, limit, pull, repo_url, split, subset, token)


@app.cell
def _(Path):
    work = Path.cwd() / "lifemate_work"
    work.mkdir(exist_ok=True)
    repo_dir, data_root = work / "repo", work / "data"
    return data_root, repo_dir, work


@app.cell
def _(branch, mo, pull, repo_dir, repo_url, shutil, subprocess, sys, token):
    mo.stop(not pull.value, mo.md("_Press **Pull repo** to start._"))

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
    mo.md(f"**commit** `{commit}` {commit_msg}  \n**GPU** {_gpu}")
    return (commit,)


@app.cell
def _(commit, data_root, data_url, fetch, hf_repo, hf_token, mo, os, repo_dir, subprocess, sys):
    commit  # lifemate.fetch_data comes from the pulled repo; pulling again only re-checks the data, never refetches it
    if fetch.value:
        # same command as on laptops; a no-op when this session already has the data
        _fetch = [sys.executable, "-m", "lifemate.fetch_data", "--dest", str(data_root)]
        if hf_repo.value.strip():
            _fetch += ["--hf-repo", hf_repo.value.strip()]
        elif data_url.value.strip():
            _fetch += ["--url", data_url.value.strip()]
        _fetch_env = {**os.environ, "PYTHONPATH": str(repo_dir / "src")}
        if hf_token.value:
            _fetch_env["HF_TOKEN"] = hf_token.value
        _r = subprocess.run(_fetch, cwd=repo_dir, env=_fetch_env, capture_output=True, text=True)
        print(_r.stdout[-2000:], flush=True)
        if _r.returncode:
            _err = _r.stderr[-2000:].replace(hf_token.value, "***") if hf_token.value else _r.stderr[-2000:]
            raise RuntimeError(f"fetching data failed (HF dataset + token filled in?):\n{_err}")
    _splits = sorted(p.name for p in data_root.iterdir() if p.is_dir()) if data_root.is_dir() else []
    mo.md(f"**data** `{data_root}`: {', '.join(_splits)}" if _splits
          else "**data**: _not fetched yet; press **Fetch data** (once per session)_")
    return


@app.cell
def _(commit, hf_token, json, mo, os, repo_dir, split, subprocess, sys, work):
    commit  # the presets of the pulled branch
    _r = subprocess.run([sys.executable, "-m", "lifemate.presets", "--list"], cwd=repo_dir, capture_output=True,
                        text=True, env={**os.environ, "PYTHONPATH": str(repo_dir / "src")})
    if _r.returncode:
        raise RuntimeError(f"listing presets failed:\n{_r.stderr[-2000:]}")
    _names = json.loads(_r.stdout)
    _cache = work / "asr_cache" / split.value
    _cached = [f"cached:{p.stem}" for p in sorted(_cache.glob("*.jsonl"))] if _cache.is_dir() else []
    agent_preset = mo.ui.dropdown(_names["agent"], value="oracle" if "oracle" in _names["agent"] else _names["agent"][0],
                                  label="Agent preset")
    asr_preset = mo.ui.dropdown(_names["asr"] + _cached, value="gold" if "gold" in _names["asr"] else _names["asr"][0],
                                label="ASR preset")
    # HF accounts the token can publish under (its user and orgs); huggingface_hub is installed by Pull
    _owners = []
    if hf_token.value:
        try:
            from huggingface_hub import HfApi

            _who = HfApi(token=hf_token.value).whoami()
            _owners = [_who["name"]] + [o["name"] for o in _who.get("orgs", [])]
        except Exception as e:  # a bad token only disables the dropdown; presets that need an owner then say so
            print(f"HF whoami failed: {type(e).__name__}", flush=True)
    _env_owner = os.environ.get("HF_MODEL_OWNER", "")
    if _env_owner and _env_owner not in _owners:
        _owners.insert(0, _env_owner)
    # team models live under an org, so the first org comes before the user's own name
    _default = _env_owner or (_owners[1] if len(_owners) > 1 else (_owners[0] if _owners else None))
    hf_model_owner = mo.ui.dropdown(_owners, value=_default, label="HF model owner (fills ${HF_MODEL_OWNER})")
    run = mo.ui.run_button(label="Run")
    mo.vstack([mo.hstack([agent_preset, asr_preset, hf_model_owner], justify="start"), run])
    return agent_preset, asr_preset, hf_model_owner, run


@app.cell
def _(agent_preset, asr_preset, command, commit, data_root, datetime, extra, hf_model_owner, hf_token, limit, mo, os,
      repo_dir, run, shlex, split, subprocess, subset, sys, work):
    mo.stop(not run.value, mo.md("_Press **Run**._"))
    _env = {**os.environ, "PYTHONPATH": str(repo_dir / "src"), "LIFEMATE_DATA_ROOT": str(data_root), "PYTHONUTF8": "1",
            "LIFEMATE_ASR_CACHE": str(work / "asr_cache")}
    if hf_model_owner.value:
        _env["HF_MODEL_OWNER"] = hf_model_owner.value
    if hf_token.value:
        _env["HF_TOKEN"] = hf_token.value  # private model repos (fine-tuned ASR / LLM)

    def _stream(cmd, cwd=repo_dir):
        """Run a step in its own process (each split has its own zaic_tools; fresh code after every pull)."""
        out = []
        with subprocess.Popen(cmd, cwd=cwd, env=_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8") as p:
            for line in p.stdout:
                line = line.replace(hf_token.value, "***") if hf_token.value else line
                print(line, end="", flush=True)
                out.append(line)
        if p.returncode:
            raise RuntimeError(f"{cmd[2]} exited with {p.returncode}")
        return "".join(out)

    run_dir, command_out = None, None
    if command.value.strip():
        _module, *_args = shlex.split(command.value)
        # from the work folder, so runs/<name> and asr_cache/ resolve as shown in the UI
        command_out = _stream([sys.executable, "-m", f"lifemate.{_module}", *_args], cwd=work)
        results_row = ""
    else:
        if not (data_root / split.value).is_dir():
            raise RuntimeError(f"no {split.value} data in {data_root}: press **Fetch data** first")
        _tag = f"{agent_preset.value}_{asr_preset.value}".replace(":", "-")
        _subset = subset.value if split.value == "train" and subset.value != "all" else None  # --subset is train only
        run_dir = work / "runs" / (f"{datetime.now():%Y%m%d-%H%M%S}_{commit}_{split.value}_{_tag}"
                                   + (f"_{_subset}" if _subset else ""))
        _cmd = [sys.executable, "-m", "lifemate.infer", "--split", split.value, "--agent-preset", agent_preset.value,
                "--asr-preset", asr_preset.value, "--out", str(run_dir),
                *(["--subset", _subset] if _subset else []), *shlex.split(extra.value)]
        if limit.value:
            _cmd += ["--limit", str(int(limit.value))]
        _stream(_cmd)
        if split.value == "train":
            _stream([sys.executable, "-m", "lifemate.evaluate", str(run_dir)])
        results_row = _stream([sys.executable, "-m", "lifemate.evaluate", "--row-only", str(run_dir)]).strip()
    return command_out, results_row, run_dir


@app.cell
def _(command_out, mo, results_row, run_dir, work):
    import json as _json

    if run_dir is None:
        _caches = sorted((work / "asr_cache").glob("*/*")) if (work / "asr_cache").is_dir() else []
        _out = mo.vstack([mo.md("### Command output"),
                          mo.ui.code_editor(value=command_out[-5000:], language="markdown", disabled=True),
                          mo.md("ASR caches in this session (download to keep them):"),
                          mo.hstack([mo.download(data=p.read_bytes(), filename=f"{p.parent.name}_{p.name}", label=f"{p.parent.name}/{p.name}")
                                     for p in _caches], justify="start")])
    mo.stop(run_dir is None, _out if run_dir is None else None)

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
