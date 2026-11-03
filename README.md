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

The data is licensed to registered teams only. Never commit it (`data/` is gitignored), and never put links or
tokens in the repo. The team keeps one copy in a **private** Hugging Face dataset (ask a teammate for its name and
access):

```bash
hf auth login                                                         # once; or export HF_TOKEN=... (read access)
python -m lifemate.fetch_data --hf-repo <user>/<dataset>              # ~880 MB download, ~1.1 GB unpacked
python -m lifemate.fetch_data --zip ~/Downloads/train+public_test.zip # or from the organisers' zip
python -m lifemate.fetch_data --url "<link from the portal>"          # or straight from the organisers
```

Uploading the team copy (once, needs a token with write access; the dataset is created private and public ones are
refused):

```bash
python -m lifemate.hf_upload --repo <user>/<dataset> --zip ~/Downloads/train+public_test.zip
python -m lifemate.hf_upload --repo <user>/<dataset>                  # or re-pack the unpacked data/ folder
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

## Training the agent model (LoRA SFT)

The agent is Qwen3-4B-Instruct-2507 with a LoRA adapter merged into its weights. Training needs a CUDA GPU (molab:
[`notebooks/molab_train.py`](notebooks/molab_train.py) runs the same steps from a UI). Install the training extras with
`pip install --no-deps -r requirements-train.txt` (peft + accelerate; keeps the installed torch). Every step is
deterministic with its default `--seed 0`.

```bash
# 1. Optional ASR-noisy text: transcribe the train clips with an ASR model NOT fine-tuned on them
python -m lifemate.transcribe --split train --asr whisper --asr-kw model=vinai/PhoWhisper-large \
    --name pho-large-fp16 --subset pool                              # -> asr_cache/train/pho-large-fp16.jsonl

# 2. Training records: gold calls replayed through the agent (4 few-shot examples, like inference), plus
#    function-masked copies (--mask) and an ASR-noisy copy (--text cache:...), all in one file
python -m lifemate.sft.records --name v2-pool --subset pool --mask 1 --text gold --text cache:pho-large-fp16
python -m lifemate.sft.records --name gold-dev --subset dev         # eval loss only, never trained on

# 3. LoRA training (r 16, alpha 32, 2 epochs, lr 1e-4, batch 16), checkpoints + resume through a private HF repo
python -m lifemate.sft.train --records v2-pool --eval-records gold-dev --out checkpoints/sft-v2 \
    --hf-repo <owner>/lifemate-sft-v2

# 4. Merge the adapter into the base weights and upload the model; the agent preset points at it
python -m lifemate.sft.merge --adapter checkpoints/sft-v2/last --out merged/sft-v2 --hf-repo <owner>/lifemate-sft-v2
python -m lifemate.infer --split public_test --agent-preset sft-v2 --asr-preset pho-large-ft-b
```

Experiments train on the pool only (`configs/splits.json`), so dev and holdout stay unseen. The shipped model is
retrained with the same recipe on `--subset all` (records and transcription) and no eval records. `RESULTS.md` notes
give the exact command, presets and model revision of the shipped system.

## Running on molab (GPU)

[`notebooks/molab_run.py`](notebooks/molab_run.py) is a marimo notebook for [molab](https://molab.marimo.io). Each press
of **Pull + run** clones or pulls the chosen branch (default `main`), downloads the data if the session does not
have it yet, installs `requirements.txt`, runs `infer` (and `evaluate` on train), then shows the score table,
download buttons and the row to paste into `RESULTS.md`.

1. Upload `notebooks/molab_run.py` to molab once (molab cannot sync from a private repo). It pulls the code itself
   on every run, so it only needs re-uploading when the notebook file itself changes.
2. Turn on the GPU in *notebook specs*.
3. Fill in a GitHub token (fine-grained, read-only *Contents* on this repo), the HF dataset name and an HF read
   token. They live only in the UI fields, not in the notebook file. Never paste them into code cells.

molab notebooks are public by link, so keep secrets and data out of them. molab's GPU (RTX Pro 6000, 96 GB) is much
stronger than the scoring A30 (24 GB) while its CPU is weaker (4 vs 16 cores), so treat molab timings as optimistic.

## Docker (final solution)

The image runs the shipped system offline on one split. `configs/shipped.json` names the agent and ASR presets,
extra kwargs and the pinned model revisions; the build downloads exactly those weights and the entry
(`lifemate.docker_entry`) loads them with the Hugging Face offline flags set.

```bash
HF_TOKEN=... bash scripts/docker_build.sh            # BuildKit secret; writes lifemate_<sha>.tar.gz + .sha256
docker run --rm --gpus all --network none \
    -v $PWD/data/public_test:/data:ro -v $PWD/out:/app/result lifemate:<sha>   # -> out/submission.jsonl, out/meta.json
```

`/data` is the split folder itself; the split name comes from its `<split>_tools_catalog.json`. The image carries
`data/train/train.jsonl` and the train catalog for the few-shot bank, never audio or environments. The entry exits
with an error if the submission does not have one line per sample. Pass `--build-arg BASE=pytorch/pytorch:<tag>` to
match molab's torch version (CUDA 12.x).

## Contributing

Never commit to `main` directly: molab runs it, so it must always work. Branch off `main` (`feat/…`, `fix/…`,
`exp/…`), push, and open a pull request. Say what changed and how you checked it (`pytest`, and the train
score when the agent or ASR changed). Test a branch on molab first by typing its name in the notebook's *Branch* field.

## Logging results

Add one row per experiment to [`RESULTS.md`](RESULTS.md); `evaluate` and the notebook print it ready to paste.
Fill in the leaderboard score by hand for public_test submissions, and say what changed in *notes*.

## Project layout

```text
src/lifemate/
  infer.py        runner: per sample ASR -> Task -> agent.run(task, env); writes the run folder
  docker_entry.py Docker entry: offline run of configs/shipped.json on the split mounted at /data
  evaluate.py     local metric on train, per-group breakdown, RESULTS.md row
  scoring.py      re-implementation of the official metric (assumptions in its docstring)
  task.py         Task: what an agent sees (request, history, context, tool specs)
  data.py         samples, tool catalog, per-split zaic_tools import
  fetch_data.py   download (HF / URL) or unpack the data
  hf_upload.py    one-off upload of the data to the team's private HF dataset
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
