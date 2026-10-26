# LifeMate — Team TripleTrouble (Zalo AI Challenge 2026)

An agent that turns a spoken Vietnamese request into tool calls on simulated apps (smart home, banking, ride-hailing,
messaging, …). For each sample it acts (calls the right tools), asks for missing information
(`sys.ask_missing_info`), or does nothing when no installed app can help. The organiser specs are in
[`context/`](context/): task, data, metric, rules and deadlines.

## Setup

Python ≥ 3.10 (3.11 recommended). The setup scripts create `.venv/`, install the project in editable mode and tell
you whether the data is there.

macOS / Linux:

```bash
bash scripts/setup.sh            # needs python3.11/3.12/3.10 (macOS: brew install python@3.11)
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
.venv\Scripts\Activate.ps1
```

### Data

The data is licensed to registered teams only. Never commit it (`data/` is gitignored) and never put the
download link in the repo. Get the link from the competition portal (Data page), then:

```bash
python -m lifemate.fetch_data --url "<train+public_test.zip link>"   # ~880 MB download, ~1.1 GB unpacked
python -m lifemate.fetch_data --zip ~/Downloads/train+public_test.zip # or from a zip you already have
```

This produces `data/train/`, `data/public_test/`, `data/README.md` (the organisers' format docs, worth reading).
Set `LIFEMATE_DATA_ROOT` to use a data folder elsewhere.

## Run and score

```bash
pytest                                                                # should pass, oracle scores 100
python -m lifemate.infer --split train --agent oracle --asr gold --limit 50
python -m lifemate.evaluate runs/<run folder>                         # train only: score + RESULTS.md row
python -m lifemate.infer --split public_test --agent <agent> --asr <asr>   # -> runs/<run>/submission.jsonl
```

Each run writes `runs/<time>_<split>_<agent>_<asr>/`:

| File | What |
|---|---|
| `submission.jsonl` | the file to submit (one line per sample: id, tool_calls, state_hash) |
| `trace.jsonl` | transcript, every call with its result, seconds and traceback per sample, for debugging |
| `meta.json` | arguments, git commit, timing (limit: 30 s per sample on the scoring A30) |
| `metrics.json`, `per_sample.jsonl` | written by `evaluate` (train only) |

Reference scores on train: `oracle` 100.0 (checks the scorer), `empty` 7.18 (only out-of-scope samples score).
Only train has labels, so public_test is scored on the leaderboard (5 submissions per day).

## Running on molab (GPU)

[`notebooks/molab_run.py`](notebooks/molab_run.py) is a marimo notebook for [molab](https://molab.marimo.io). Each press
of **Pull + run** clones or pulls the chosen branch (default `main`), downloads the data if the session does not
have it yet, installs `requirements.txt`, runs `infer` (and `evaluate` on train), then shows the score table,
download buttons and the row to paste into `RESULTS.md`.

1. Open the notebook in molab. Either add it from GitHub (new notebook → paste the file's GitHub URL), or, if
   molab cannot read the private repo, upload `notebooks/molab_run.py` once. It pulls the code itself, so it only
   needs re-uploading when the notebook file changes.
2. Turn on the GPU in *notebook specs*.
3. Fill in a GitHub token (fine-grained, read-only *Contents* on this repo) and the data link. Neither is saved in the
   notebook file. Do not paste them into code cells.

molab notebooks are public by link, so keep secrets and data out of them. molab's GPU (RTX Pro 6000, 96 GB) is much
stronger than the scoring A30 (24 GB) while its CPU is weaker (4 vs 16 cores), so treat molab timings as optimistic.

## Logging results

Add one row per experiment to [`RESULTS.md`](RESULTS.md); `evaluate` and the notebook print it ready to paste.
Fill in the leaderboard score by hand for public_test submissions, and say what changed in *notes*.

## Project layout

```text
src/lifemate/
  infer.py        runner: per sample ASR -> Task -> agent.run(task, env); writes the run folder
  evaluate.py     local metric on train, per-group breakdown, RESULTS.md row
  scoring.py      re-implementation of the official metric (assumptions in its docstring)
  task.py         Task: what an agent sees (request, history, context, tool specs)
  data.py         samples, tool catalog, per-split zaic_tools import
  fetch_data.py   download / unpack the data
  asr/            ASR backends (--asr): gold (train transcripts), none
  agents/         agents (--agent): empty, oracle
notebooks/molab_run.py   molab runner
scripts/setup.sh|ps1     virtualenv setup
tests/                   pytest
context/                 organiser specs (Markdown)
```

### Adding an approach

Subclass `ASR` in `src/lifemate/asr/` or `Agent` in `src/lifemate/agents/`, register it in that package's
`__init__.py`, then run it with `--asr <name>` / `--agent <name>`. Add its dependencies to `requirements.txt`.
An agent receives the sample's only `ToolEnv` and must stick to these rules:

- Call tools only through `env.call(name, arguments)`. Every call is logged and scored, failed ones included, so
  check arguments with `env.validate(name, arguments)` first (free and not logged).
- Never read files under `data/<split>/<split>_environment/` and never create another `ToolEnv`.
- Never use `task.label` (it exists on train only for the oracle).
- Use deterministic decoding (greedy / temperature 0); seeds are fixed by the runner.

## Competition limits (summary)

Whole system < 6B parameters, LLM < 5B with a context ≤ 16,384 tokens; ≤ 30 s per sample on 1× A30, 16-core CPU,
64 GB RAM; fully offline Docker; open models and data released before 26 Oct 2026. The Docker solution is due on
16 Nov 2026 at 23:59 (GMT+7), and the private test runs 20–23 Nov. Details: [`context/RULE.md`](context/RULE.md).
