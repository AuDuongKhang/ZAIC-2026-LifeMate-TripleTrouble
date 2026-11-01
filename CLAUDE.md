# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Zalo AI Challenge 2026, **LifeMate** track (team TripleTrouble). Organiser specs: `context/*.md`. The data package is in `data/` (gitignored, licensed to registered teams only, never commit it); `data/README.md` and `data/train/zaic_tools/README.md` are the authoritative format docs.

## Commands

```bash
bash scripts/setup.sh                          # macOS/Linux: .venv + pip install -r requirements-dev.txt -e .
                                               # (Windows: scripts\setup.ps1); then activate .venv
python -m lifemate.fetch_data --hf-repo <user>/<dataset>   # team's private HF copy (HF_TOKEN); or --zip / --url
python -m lifemate.hf_upload --repo <user>/<dataset> --zip <file>   # one-off; creates the dataset private
pytest                                         # unit tests + oracle end-to-end (skipped without data/train)
pytest tests/test_scoring.py::test_each_call_matches_at_most_once

python -m lifemate.infer --split train --agent oracle --asr gold --limit 50   # -> runs/<time>_<split>_<agent>_<asr>/
python -m lifemate.infer --split public_test --agent empty --asr none --ids public_0001 public_0002
python -m lifemate.evaluate runs/<run>          # train only: metrics.json + per_sample.jsonl + RESULTS.md row
python -m lifemate.evaluate --row-only runs/<run>
python -m lifemate.sft.records --name gold-pool --subset pool   # SFT records (gold replays) -> train_records/
python -m lifemate.sft.train --records gold-pool --eval-records gold-dev --out checkpoints/sft-v1 --hf-repo <owner>/lifemate-sft-v1
python -m lifemate.sft.merge --adapter checkpoints/sft-v1/last --out merged/sft-v1 --hf-repo <owner>/lifemate-sft-v1
cd data && PYTHONPATH=public_test python -m zaic_tools --split public_test --root . --validate ../runs/<run>/submission.jsonl
```

Without the editable install (e.g. the user's base Anaconda, whose torch/numpy are broken), set `PYTHONPATH=src`. On Windows, set `PYTHONUTF8=1` for Vietnamese output. `LIFEMATE_DATA_ROOT` overrides the data folder (default `data/`). Reference numbers on train: oracle 100.0, empty 7.18.

All repo files (code, comments, docs, notebook UI, commits) are in English; Vietnamese appears only as data. The team mostly uses macOS, so keep `scripts/setup.sh` working (bash, LF line endings via `.gitattributes`).

## Git workflow (team rule)

- **Never commit or push directly to `main`.** molab runs `main` by default, so `main` must always work.
- Branch from an up-to-date `main`: `feat/<topic>`, `fix/<topic>`, or `exp/<topic>` for experiments (for example `exp/qwen3-4b-react`).
- Open a PR into `main` (`gh pr create --base main`). The description says what changed and how it was verified: `pytest`, plus the train score / `RESULTS.md` row when agent or ASR behaviour changed. Merge after a teammate has looked at it.
- To try a branch on molab before merging, put its name in the notebook's *Branch* field.
- Commit messages and PR text are in English. Never commit data, tokens, or data download links (see `.gitignore`). The one exception is the team's HF dataset name, which the notebook fills in by default: the repo is private and the dataset cannot be read without a token.
- On the repo owner's machine `gh`'s active account is not the repo account. Run `gh` with `GH_TOKEN="$(gh auth token --user AuDuongKhang)"` instead of switching accounts.

## Architecture

`src/lifemate/` pipeline: `infer.py` runs, for each sample, ASR → build `Task` → `agent.run(task, env)` and writes `submission.jsonl` (the official file), `trace.jsonl` (transcript, every call with its result, seconds, traceback) and `meta.json` (args, git commit, timing).

- `asr/` and `agents/`: pluggable backends, chosen by name (`--asr`, `--agent`) from the `BACKENDS` / `AGENTS` dicts in each `__init__.py`. A new approach is a new subclass registered there. One instance serves every sample, so load models in `__init__`. `gold` ASR returns train transcripts (upper bound with perfect ASR), `none` returns "". The `oracle` agent replays labels.
- `task.Task`: everything an agent may use. `history` user turns get ASR `text` (`audio_text` is stripped), `tools` are catalog specs for `installed_apps` + `sys.ask_missing_info`. `label` is set on train for oracle/debug only. A real agent must never read it.
- `scoring.py`: local copy of the official metric, with our assumptions in its docstring. `evaluate.py` gets the gold hash by replaying `label.tool_calls` in a fresh ToolEnv, which reproduces `label.final_state` on all 1545 train samples.
- `data.import_zaic_tools`: every split ships its **own** `zaic_tools` under the same module name, so one process handles one split. Steps that touch two splits run as subprocesses (see `tests/test_pipeline.py` and the notebook).
- `sft/`: LoRA SFT. `records` renders each train sample by running `ReActAgent` with `ReplayLLM` (gold calls, then a fixed final text) in a fresh train `ToolEnv`, so training text equals inference text (few-shot examples included); `tokenize` labels only the agent's turns; `train` is plain PyTorch + peft with checkpoint/resume to a private HF repo; `merge` bakes the adapter in. Training deps are in `requirements-train.txt`, installed with `--no-deps`. `notebooks/molab_train.py` runs it on molab.
- `notebooks/molab_run.py`: marimo notebook for molab (GPU). molab can't sync a private repo, so the team uploads this file once. It clones the repo itself (shallow, chosen branch, GitHub token), fetches data with `lifemate.fetch_data` from the team's private HF dataset (dataset name filled in by default, tokens typed in the UI and never stored in the repo; the HF model owner is picked from the token's user and orgs), installs `requirements.txt`, runs infer (+ evaluate on train) in subprocesses, and offers downloads plus the RESULTS.md row. Team results are logged by hand in `RESULTS.md`.

## Rules that constrain the code

- Never read `<split>_environment/` (including `state.json`) from solution code. App data goes only through `ToolEnv`, **exactly one per sample**, created by the runner. Agents get that env and must not create another.
- **Every `env.call` is logged and scored.** Schema-rejected calls count as unmatched (they lower precision) and do not change state. `env.validate()` is free and not logged, so check arguments with it first. Non-schema problems (unknown ID, empty search) are not errors: they return `{"ok": true, "done": false, ...}` or an empty list.
- Score per sample: 0 if the state hash differs, otherwise F1 over calls. Free-text keys (`utils.FREE_TEXT_KEYS`, `x-match: fuzzy` params) are dropped from the hash and scored by token-F1 > 0.6. Strings in the hash are lowercased. Gold `[]` (out of scope) scores only with an empty submission.
- Limits: whole system < 6B params, LLM < 5B with ≤ 16,384-token context; ≤ 30 s per sample including ASR on 1× A30 24 GB (molab's RTX Pro 6000 is faster, so treat molab timings as optimistic); fully offline Docker; fixed seeds (`utils.seed_everything`, called after the models are built) and greedy decoding.
- Deadlines: Docker solution 16 Nov 2026 23:59 GMT+7; private test 20–23 Nov.

## Data facts that shape the design

- train 1545 (act 1217 / ask 217 / out-of-scope 111), public_test 634. Gold calls per sample: 1–6, mostly 1–3. 66% of samples have history (up to 24 turns); assistant turns carry their tool_calls/results, already applied to the state.
- **Apps differ between splits**: train 46 apps, public_test 36, only 25 shared (for example ecommerce, grocery, parking, stocks and train_bus_tickets exist only in public_test). Shared tool schemas are identical. Private test will likely bring unseen apps too, so the agent must work from the schemas in `tools`, not memorised app behaviour.
- Installed-tool specs average ~13k characters per sample (max ~25k), so they need trimming or compressing to fit 16k tokens with history.
- Audio clips average 4.5 s (max 13.5 s), 16 kHz mono. History user turns have their own wav (`<id>_u1.wav`, …).
- `sys.ask_missing_info` often follows reads (for example `contacts.search` finds several "Minh" → ask for `phone.call` `contact_id`), so ask decisions depend on tool results, not only on the transcript.
