# Docker Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the best system (sft-final-h + pho-large-ft-all beam 5, public 57.5) as the Docker image the organiser
guide asks for, build it without a local GPU or disk, and pass the portal's *Test Solution* check before 15 Nov.

**Architecture:** `infer.py` gets a reusable per-sample step (`load_pipeline` / `run_sample`) that adds the `time`
field, so `predict.sh` (through `docker_entry`) and `predict_notebook.ipynb` run the same code. The image moves to a
CUDA 12.8 PyTorch base that runs on both the A30 and the RTX 5090, with the guide's `/code` layout. A GitHub Actions
workflow builds the image, smoke-tests it on CPU with `--network none`, and uploads `zac2026_TripleTrouble.tar.gz` to
Google Drive with its MD5. The GPU check is the portal's Test Solution (1 run per day, closes 15 Nov 23:59 GMT+7).

**Tech Stack:** Docker + BuildKit, `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`, jupyterlab + nbconvert,
GitHub Actions (free minutes of the private repo), rclone (Google Drive).

**Spec:** `context/Docker_Guide_ZAC2026_LifeMate.pdf` (organiser guide, untracked), `context/RULE.md`,
`context/SUBMISSION_GUIDANCE.md`, the organisers' 10 Nov email (Test Solution, quoted in the PR description).

## Global Constraints

- Base image CUDA ≤ 13.2 (driver 595.58.03). The guide names an RTX 5090 32 GB (sm_120); `RULE.md` and the guide's
  `nvidia-smi` screenshot show an A30 (sm_80). The image must run on both: torch ≥ 2.7 built for cu128.
- Input: the organisers mount a folder at `/data` that **contains** the split folder (`/data/private_test/...`). Read-only.
- Output: `/result/submission.jsonl` from `predict.sh`, `/result/jupyter_submission.jsonl` from the notebook. One line
  per sample: `id`, `tool_calls`, `state_hash`, `time` (int, milliseconds, from before the sample's ToolEnv and audio
  to the end of its inference). The code must create `/result`.
- `/code` holds `predict.sh`, `start_jupyter.sh` (`jupyter lab --port 9777 --ip 0.0.0.0` with password and token
  `zac2026`, `--allow-root --no-browser`), `predict_notebook.ipynb` (at least the cells: seed, load models, read test
  cases, predict), `README.md`, `requirements.txt`, `training_code/README.md`. jupyterlab is installed.
- Fully offline at run time (`--network none`); all weights in the image; no base model in the image (only the
  fine-tuned weights used at inference).
- Fixed seeds (Python, NumPy, torch, `cudnn.deterministic = True`, `cudnn.benchmark = False`), greedy/beam decoding
  only. Hyper-parameters are defaults in code, not run-time flags.
- ≤ 30 s per sample including ASR; whole system < 6B params (5.57B today).
- Submission file: `zac2026_TripleTrouble.tar.gz` (`docker save | gzip`), MD5 checksum, Google Drive link shared
  "Anyone with the link". Docker deadline 16 Nov 2026 23:59 GMT+7; Test Solution closes 15 Nov 23:59 GMT+7.
- No data, tokens or download links committed (CLAUDE.md). Repo text in English.

## Review Focus

1. `/data` mounted as the split folder itself (how molab and our old docs call it) must still work, not only
   `/data/<split>/` → `locate_split` accepts both (Task 2 test).
2. A sample whose agent or ASR raises must still produce a line with `tool_calls`, `state_hash` and `time`, never a
   short file → `run_sample` never raises (Task 1 test).
3. Running the notebook after `predict.sh` in the same container must not overwrite `submission.jsonl`, and both files
   must agree on `id`, `tool_calls`, `state_hash` → Task 3 test and the Task 5 smoke check.
4. A build that silently ships a torch without sm_120 kernels, or weights that are not in the HF cache → Dockerfile
   build-time assert on `torch.cuda.get_arch_list()` and `docker_entry --check-models` under `--network none` (Task 4).
5. Writing anywhere under the read-only `/data` (symlink target) → the smoke run mounts `/data` with `:ro` (Task 5).

---

### Task 1: Per-sample step with the `time` field

**Files:**
- Modify: `src/lifemate/infer.py` (extract the loop body of `main`)
- Modify: `src/lifemate/utils.py` (`seed_everything`)
- Test: `tests/test_infer_pipeline.py` (new), `tests/test_utils.py` if it exists, else the new file

**Interfaces:**
- Produces:
  - `@dataclass class Pipeline: split: str; data_root: Path; zaic_tools: ModuleType; catalog: dict[str, dict]; asr; agent`
  - `load_pipeline(split: str, data_root: Path, agent_name: str, agent_kw: dict, asr_name: str, asr_kw: dict) -> Pipeline`
    (imports zaic_tools, loads the catalog, builds ASR then agent with `with_split`, as `main` does now; no seeding)
  - `run_sample(p: Pipeline, sample: dict) -> tuple[dict, dict]`: `(line, trace)`; `line` =
    `{"id", "tool_calls", "state_hash", "time"}` with `time = int((perf_counter() - t0) * 1000)`, `t0` taken before
    `ToolEnv(...)`; `trace` = the record `main` writes to `trace.jsonl` today (plus `"seconds"`). Never raises: an
    exception goes into `trace["error"]`.
  - `main` keeps its CLI and outputs, now built on these two functions; `submission.jsonl` lines gain `time`.

- [x] **Step 1: Write failing tests** in `tests/test_infer_pipeline.py` with a fake `Pipeline` (a stub `zaic_tools`
  module whose `ToolEnv` returns fixed calls and hash, a stub ASR and agent):
  - `test_run_sample_line_has_time_in_ms`: `set(line) == {"id", "tool_calls", "state_hash", "time"}`,
    `isinstance(line["time"], int)`, and with an agent that sleeps 50 ms `line["time"] >= 50`.
  - `test_run_sample_keeps_going_when_the_agent_raises`: agent raises `RuntimeError("boom")` → line still has the env's
    calls and hash, `"boom" in trace["error"]`.
  - `test_seed_everything_sets_cudnn_flags` (skip without torch): after `seed_everything(0)`,
    `torch.backends.cudnn.deterministic is True` and `torch.backends.cudnn.benchmark is False`.
- [x] **Step 2: Run** `pytest tests/test_infer_pipeline.py -v` → FAIL (`load_pipeline`/`run_sample` not defined).
- [x] **Step 3: Implement** `Pipeline`, `load_pipeline`, `run_sample` in `infer.py` and rewrite the loop in `main` on
  top of them; add the two cudnn flags to `seed_everything` (inside the existing `if "torch" in sys.modules`).
- [x] **Step 4: Run** `pytest tests/test_infer_pipeline.py tests/test_pipeline.py -v` → PASS (the oracle end-to-end in
  `test_pipeline.py` still scores 100 on train with the extra key).
- [x] **Step 5: Commit** `git commit -m "Add a per-sample inference step and the time field"`

### Task 2: `docker_entry` for the guide's layout

**Files:**
- Modify: `src/lifemate/docker_entry.py`
- Create: `configs/smoke.json` (`{"agent_preset": "empty", "asr_preset": "none", "agent_kw": {}, "asr_kw": {}, "hf_model_owner": "TripleTrouble"}`)
- Test: `tests/test_docker_entry.py`

**Interfaces:**
- Consumes: `infer.load_pipeline`, `infer.main` (Task 1).
- Produces:
  - `locate_split(data: Path) -> tuple[Path, str]`: `(split_dir, split)`. `data` itself if it holds one
    `*_tools_catalog.json`, else the single child folder `<name>` holding `<name>_tools_catalog.json`. Otherwise
    `SystemExit` naming what it found. (`find_split` stays as the single-folder helper.)
  - `prepare(data: Path, work: Path, train_dir: Path) -> tuple[Path, str]`: the work data root (symlink to the split
    folder + copied train bank, as `main` does now) and the split name.
  - `load_shipped(config: dict, split: str, data_root: Path) -> infer.Pipeline`: sets the HF offline flags and
    `HF_MODEL_OWNER`, resolves presets + `agent_kw`/`asr_kw` like `main`, calls `load_pipeline`.
  - `check_models(config: dict) -> None` and CLI flag `--check-models`: `snapshot_download(..., local_files_only=True)`
    for both models; exits non-zero if a file is missing.
  - `fetch_models` passes `ignore_patterns=["checkpoints/*", "*.pt"]` (the sft-final-h repo carries 1.2 GB of other
    runs' checkpoints).
  - CLI defaults: `--data /data`, `--out /result/submission.jsonl`, `--config $LIFEMATE_SHIPPED` or
    `configs/shipped.json`. After the run it prints `load model: <s> s` and
    `predict: <n> samples, <s> s, mean <ms> ms/sample` (from `meta.json`).

- [x] **Step 1: Write failing tests**:
  - `test_locate_split_accepts_the_parent_folder`: `/tmp/x/private_test/private_test_tools_catalog.json` →
    `locate_split(tmp/"x") == (tmp/"x"/"private_test", "private_test")`.
  - `test_locate_split_accepts_the_split_folder_itself`: same answer when given the split folder.
  - `test_locate_split_refuses_two_splits`: two child splits → `SystemExit` whose message names both.
  - Update `_run` to mount the split one level down (`--data <parent>`) and keep the existing assertions.
  - `test_fetch_models_skips_checkpoints`: the fake `snapshot_download` records `ignore_patterns` containing `"checkpoints/*"`.
  - `test_check_models_fails_when_a_model_is_not_cached`: fake `snapshot_download` raising `LocalEntryNotFoundError`
    → `SystemExit` with non-zero code.
  - `test_smoke_config_names_known_presets`: like `test_committed_shipped_config_names_known_presets`, for `configs/smoke.json`.
- [x] **Step 2: Run** `pytest tests/test_docker_entry.py -v` → the new tests FAIL.
- [x] **Step 3: Implement** the functions above; `main` = `prepare` → `infer.main(...)` → copy + line-count check → print times.
- [x] **Step 4: Run** `pytest tests/test_docker_entry.py -v` → PASS (symlink tests may skip on Windows without
  developer mode; they run on the Actions runner in Task 5).
- [x] **Step 5: Commit** `git commit -m "Find the split inside /data and write /result like the organiser guide"`

### Task 3: `predict.sh`, `start_jupyter.sh` and the timing notebook

**Files:**
- Create: `docker/predict.sh`, `docker/start_jupyter.sh`, `docker/predict_notebook.ipynb`
- Test: `tests/test_docker_notebook.py`

**Interfaces:**
- Consumes: `docker_entry.prepare`, `docker_entry.load_shipped`, `docker_entry.load_config`, `infer.run_sample`,
  `utils.seed_everything`, `data.load_samples`.
- Produces: the three files copied to `/code` in Task 4. All read `LIFEMATE_DATA` (default `/data`), `LIFEMATE_RESULT`
  (default `/result`) and `LIFEMATE_SHIPPED` (default `/code/configs/shipped.json`).

- [x] **Step 1: Write the files.**
  - `predict.sh`: `set -euo pipefail; cd /code; python -m lifemate.docker_entry --data "${LIFEMATE_DATA:-/data}" --out "${LIFEMATE_RESULT:-/result}/submission.jsonl"`.
  - `start_jupyter.sh`: the guide's command verbatim, `cd /code` first.
  - Notebook cells (code only, short markdown headers): (1) `import torch` + `seed_everything(0)`;
    (2) `prepare(...)` into `/tmp/lifemate_nb` + `load_shipped(...)` + `seed_everything(0)` again (models built);
    (3) `samples = load_samples(data_root, split)`; (4) loop `run_sample`, collect lines, `mkdir /result`, write
    `jupyter_submission.jsonl`, print count and mean `time`. No outputs stored in the file.
- [x] **Step 2: Write the test** `test_notebook_matches_predict_on_public_test` (skip without `data/public_test` or
  symlinks): run `docker_entry.main` with `configs/smoke.json` on a temp parent of `data/public_test` into
  `tmp/result`, then exec the notebook's code cells in one namespace with the same env vars; assert both files have
  634 lines with equal `id`, `tool_calls`, `state_hash` per line, and both have an int `time`.
  `test_notebook_never_writes_submission_jsonl`: the notebook source contains `jupyter_submission.jsonl` and not
  `"/submission.jsonl"`.
- [x] **Step 3: Run** `pytest tests/test_docker_notebook.py -v` → PASS (or SKIP on Windows without symlinks).
- [x] **Step 4: Commit** `git commit -m "Add predict.sh, start_jupyter.sh and the timing notebook"`

### Task 4: Image for the A30 and the RTX 5090

**Files:**
- Modify: `Dockerfile`, `scripts/docker_build.sh`, `.dockerignore` (create if missing)
- Create: `docker/training_code/README.md`, `README.md` section "Docker submission"
- Test: `tests/test_dockerfile.py`

**Interfaces:**
- Consumes: Tasks 2–3 files and flags.
- Produces: image tag `zac2026-tripletrouble:<sha>`, `zac2026_TripleTrouble.tar.gz` + `.md5` from `docker_build.sh`.

- [x] **Step 1: Write failing tests** in `tests/test_dockerfile.py` (text checks only):
  - `test_base_image_is_cuda_12_8_with_torch_2_7_or_newer`: `ARG BASE=pytorch/pytorch:<v>-cuda<c>-...` with
    `<v> >= 2.7` and `12.8 <= <c> <= 13.2`.
  - `test_image_uses_the_code_layout`: `WORKDIR /code`; the Dockerfile copies `docker/predict.sh`,
    `docker/start_jupyter.sh`, `docker/predict_notebook.ipynb`, `docker/training_code`; installs `jupyterlab` and `nbconvert`.
  - `test_build_script_writes_md5`: `scripts/docker_build.sh` contains `md5sum` and `zac2026_TripleTrouble.tar.gz`.
- [x] **Step 2: Run** `pytest tests/test_dockerfile.py -v` → FAIL.
- [x] **Step 3: Implement.**
  - `ARG BASE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`; `WORKDIR /code`; apt `libsndfile1`; `pip install -r
    requirements.txt jupyterlab nbconvert`; copy `src`, `configs`, `notebooks`, `scripts`, `tests`, docs, the train
    bank files and the `docker/*` files to `/code`; `fetch_models` with the HF token as a build secret (unchanged).
  - Build-time asserts: `python -c "import torch; a = torch.cuda.get_arch_list(); assert 'sm_80' in a and 'sm_120' in a, a"`.
  - `ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, `CMD ["bash", "/code/predict.sh"]` (no ENTRYPOINT, so the guide's
    `docker run ... image /bin/bash /code/predict.sh` and `start_jupyter.sh` both work).
  - `.dockerignore`: `data/*` except the two train bank files, `runs/`, `train_records/`, `checkpoints/`, `merged/`,
    `.venv/`, `.git/`, `context/`.
  - `docker_build.sh`: tag `zac2026-tripletrouble:<sha>`, `docker save | gzip > zac2026_TripleTrouble.tar.gz`,
    `md5sum ... | tee zac2026_TripleTrouble.tar.gz.md5`.
  - `docker/training_code/README.md`: idea, ASR (PhoWhisper-large fine-tune, from PR #73's section), LLM (Qwen3-4B-
    Instruct-2507, 4.02B, 16,384-token context, LoRA SFT recipe from `docs/shipped-model.md`), ReAct tool loop,
    data and synthetic copies (masking, leave-app-out), commands, seeds, and the HF URLs of the fine-tuned models.
    Whether those repos become public (the organisers reproduce training with internet) is an open decision for the
    team; the README lists the URLs either way.
- [x] **Step 4: Run** `pytest tests/test_dockerfile.py -v` → PASS; full `pytest` → no new failures.
- [x] **Step 5: Commit** `git commit -m "Build the image on CUDA 12.8 torch with the organiser /code layout"`

### Task 5: Build, smoke-test and upload on GitHub Actions

**Files:**
- Create: `.github/workflows/docker-image.yml`, `scripts/docker_smoke.sh`

**Interfaces:**
- Consumes: `scripts/docker_build.sh`, `configs/smoke.json`, `docker_entry --check-models`.
- Secrets (set once by the owner, never in the repo): `HF_TOKEN`, `HF_DATASET` (team data repo), `RCLONE_CONFIG`
  (contents of `rclone.conf` with a Drive remote named `gdrive`, made with `rclone config` on the owner's machine).

- [x] **Step 1: Write `scripts/docker_smoke.sh <image> <split parent dir>`** (bash, LF): with `--network none`,
  `/data:ro`, CPU only:
  1. `python -m lifemate.docker_entry --check-models` → exit 0;
  2. `LIFEMATE_SHIPPED=/code/configs/smoke.json bash /code/predict.sh`;
  3. `LIFEMATE_SHIPPED=... jupyter nbconvert --to notebook --execute /code/predict_notebook.ipynb --output /tmp/nb.ipynb`;
  4. on the host: both result files have one line per sample, keys `id, tool_calls, state_hash, time`, equal except
     `time`; `zaic_tools --validate` passes on `submission.jsonl`.
  Exits non-zero on any failure.
- [x] **Step 2: Write the workflow** (`workflow_dispatch`, inputs `ref` default `main`, `upload` default `true`;
  `ubuntu-latest`, timeout 180 min): free disk (`jlumbroso/free-disk-space@v1.3.1`) → checkout → `pip install -r
  requirements.txt` + `python -m lifemate.fetch_data --hf-repo "$HF_DATASET"` → `HF_TOKEN=... bash
  scripts/docker_build.sh` → `bash scripts/docker_smoke.sh ... data` → if `upload`: install rclone, write
  `RCLONE_CONFIG` to `~/.config/rclone/rclone.conf`, `rclone copy` the tar.gz to `gdrive:zaic2026/<sha>/`,
  `rclone link` it (anyone with the link) → write the link and MD5 to the job summary. Delete the data folder before
  the job ends.
- [x] **Step 3: Verify:** open a PR, run the workflow on the branch with `upload=false`; expected: build and smoke
  steps green, image size and MD5 in the summary.
- [x] **Step 4: Commit** `git commit -m "Build, smoke-test and upload the image on GitHub Actions"`

### Task 6: First Test Solution run (owner, on the portal)

- [x] **Step 1:** Run the workflow on `main` (after merge) with `upload=true`; copy the Drive link and MD5 from the summary.
- [x] **Step 2:** Portal → LifeMate → *Test Solution*: paste the link and MD5, *Submit Test*. One run per day; up to 5
  retries only for link, checksum or download errors.
- [x] **Step 3:** On failure open *View full log*, fix on a `fix/` branch, repeat the next day. On success record the
  date, commit, image MD5 and size in `docs/shipped-model.md` (PR into `main`).

## Status 10 Nov 2026

All tasks are done in PR #76, and the image passed the portal's Test Solution on 10 Nov ("Test passed. Your solution
runs correctly."; commit `28e209e`, MD5 `51fe14c8721c93178f03ad42af613e7b`, 14.44 GiB). Changes from the plan, found
while building on Actions:

- Base image `pytorch/pytorch:2.15.0-cuda13.2-cudnn9-runtime` instead of 2.7.1 (same torch as development; still
  CUDA ≤ 13.2, kernels sm_75 to sm_120). Its system Python needs `PIP_BREAK_SYSTEM_PACKAGES=1`.
- The runner's `/mnt` is the root disk (52 GB free). The containerd image store kept the weights twice and filled it,
  so the workflow uses the classic overlay2 store and streams `docker save | pigz | rclone rcat` to Drive; the MD5 is
  Drive's own. Each upload gets its own folder (time, commit, run id, attempt, label) with an `info.txt`; Drive access
  and free space are checked before the build; runs queue.
- `--check-models` also loads each config and tokenizer/processor offline: `snapshot_download(local_files_only=True)`
  returns partial snapshots.
- `time` is written only with `infer --time-field` (docker_entry passes it), so ordinary runs stay byte-identical.
- `requirements.txt` applies `docker/constraints.txt`, so molab runs use the image's library versions.
- PRs that change the image build and smoke-test it without upload (`scripts/shipped_fingerprint.py` ignores new
  experiment presets).
- `training_code/` also holds the code at commit `0c7152b` (tag `sft-final-h-code`) that trained the shipped models.

Still open: the GPU check of score and time with the image's libraries (molab runbook C1), the public model repos
question below, and the final upload (label `final`) and submission by 16 Nov 23:59 GMT+7.

## Open decisions for the owner

- **Drive space.** Settled: the owner's Drive has room (322 GB free on 10 Nov).
- **Public model repos** for the training reproduction (Task 4 README); needed by 16 Nov, not for Test Solution. The
  fine-tuned weights are in the image and the base models are public, so training can be reproduced and compared
  without them; making `TripleTrouble/lifemate-sft-final-h` and `pho-large-ft-all` public is still the owner's call.
