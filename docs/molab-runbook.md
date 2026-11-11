# Molab runbook: ten experiments (7-8 Nov 2026)

Everything that has to run on molab (GPU), step by step, for the plan in `docs/superpowers/plans/2026-11-06-ten-experiments.md`.
All code is merged into `main` except the logit distillation (PR #61), which also lives on branch `exp/sft-q1` (= `main` plus
distillation). Times marked *(est.)* are estimates, the others were measured on earlier molab runs. Numbers in this file
are the plan's thresholds: nobody relaxes them, and a result that is close to a threshold goes to the owner of the plan.

## 0. Before anything else

1. **Upload the current `notebooks/molab_train.py` from `main` to molab again** (and `molab_run.py` if the copy on molab is old).
   The new fields (*leave-app-out*, *schema paraphrase table*) are part of the notebook file; the repo clone does not bring them.
2. Fill in, in every notebook session: GitHub token (repo is private), HF token. Use an HF token **with write access** in
   `molab_run.py` for every step that uploads (convert, recovery, remix, synth) and in `molab_train.py` (second token field).
   The label of the `molab_run.py` field says "read"; it is only passed on as `HF_TOKEN`, so a write token works.
3. In each notebook press **Pull repo** (again whenever you change the branch; never while training runs), then **Fetch data**
   once per session.
4. **Branch to type in the *Branch* field:**

   | Run | Branch |
   |---|---|
   | everything | `main` |
   | `sft-kd1`, `sft-t1`, `sft-kd2` (distillation) and `sft-q1` | `exp/sft-q1` |

5. The *Command* field of `molab_run.py` ("or command instead of infer") runs `python -m lifemate.<command>` from the work
   folder, so `sft.remix --name r1` means `python -m lifemate.sft.remix --name r1`. Press **Run** (the button next to the preset
   dropdowns) to execute it; with the field empty, Run does inference. Commands that write a model or a file upload it to the Hub
   themselves (`--hf-repo`): there is no upload field, and later sessions read it from the Hub by name.
6. Models and data go to the private Hub repos of `TripleTrouble`. Never put a token or data in the repo.

## 1. Recipes and conventions

**sft-v2 recipe** (`molab_train.py`), used by every training run below. Only *Run name*, *HF model repo*, *Branch* and the
fields named in each run change:

| Field | Value |
|---|---|
| Train records | `pool (experiments)` |
| Eval loss records | `dev` |
| few-shot examples in the records | `4` |
| function-masked copies per record | `1` |
| perturb masked schemas | off |
| weekday table in the prompt | off |
| records with the gold apps left out, % | `0` |
| ASR-noisy copy from cache | `pho-large-fp16` |
| ASR model that writes that cache | `vinai/PhoWhisper-large` (off-the-shelf; the notebook transcribes the pool clips in about 15 minutes if the session has no cache) |
| synthetic samples / reasoning sentences / schema paraphrase table | empty |
| resume | off (tick it only to continue a stopped run, with the same name, fields and extra args) |
| `--max-steps` | `0` |
| extra train args | per run |

2 epochs is the default. A training run took about 3 hours on molab for 474 steps (measured on sft-v3d).
Press **Start training**, watch the log, and when it has finished press **Merge and upload** (needs the *HF model repo* filled in).
The merged model lands at `TripleTrouble/lifemate-sft-<name>`.

**Gate** (ASR text, ASR preset `pho-large-ft-b`), for every trained model except the teacher `sft-t1`:

- holdout diff >= +0.5 with the 95% CI above 0, **and**
- dev diff >= -0.3, **and**
- the holdout ask column drops by at most 5 points.
- Baselines: holdout 41.37 (ask 33.06), dev 66.86 (sft-v2 + check_refs, ASR `pho-large-ft-b`).

**How to run the gate for a model named `<name>`** (`molab_run.py`, branch `main`): split `train`, agent preset `sft-v2-refs`,
ASR preset `pho-large-ft-b`, extra infer args `--agent-kw model=TripleTrouble/lifemate-sft-<name>`; run subset `holdout`, then
`dev` (each about 17 minutes *(est.)*: ~300 samples at ~3.4 s). The run folder is named after the preset `sft-v2-refs`, so write
down which run belongs to which model. For each run send back: the RESULTS row the notebook prints, plus the files
`per_sample.jsonl`, `metrics.json` and `meta.json` (download buttons) so the diff and CI can be computed against the baselines.

**Hub storage.** Private repos hold about 40 GB now and every merged model adds about 8.5 GB. After a model fails its gate, delete
its repo on the Hub at once (log the RESULTS row first). Do not make repos public: they hold weights trained on licensed data.

## 2. Split between the two accounts

**K (account 1)**

1. Hub cleanup is done (`lifemate-sft-v2e1` and `lifemate-sft-r1` deleted).
2. Step A3 (convert Qwen3.5, once) then A4 (probes), then the distillation chain (B1) and `sft-q1` (B3).
3. If K finishes first: take `sft-v6w`.

**P (account 2)**, in this order: A2 (diagnostic), A6 (rollout + recovery file), A7 (remix table), then the training queue
`sft-v5a` (only if A2 says so) -> `sft-v5r` -> `sft-v6w` -> `sft-v6d`. Stop when the evening of 8 Nov is near: a run that cannot be
trained and gated by then is skipped. Do not run a generation job with the 27B teacher while a training run is active on the same
account (GPU memory).

Task 4 (`sft-v4`, section A8 and B6) is optional and only worth starting if an account is idle: pilot plus bulk take hours
*(est. 4 h for the bulk)* before a 3-hour training run.

## A. Inference and data steps (`notebooks/molab_run.py`)

### A1. Distillation report on sft-v2 (Task 0 Step 3) - K, notebook `molab_train.py`, about 30 min

Branch `exp/sft-q1` (or `exp/logit-kd`). Run name `sft-kd1`, *HF model repo* **empty**, sft-v2 recipe fields, extra train args:
`--kd-teachers TripleTrouble/lifemate-sft-v2 --kd-report-only`. Start training; it stops after writing
`lifemate_work/checkpoints/sft-kd1/kd_report.json`. Send the file.
**Decision 1:** if `share_top1_below_0.9` is under 1% **or** `share_top1_not_gold` is under 0.1%, the teacher adds almost nothing:
skip B1 step 1. (Judgement thresholds from the plan.) If a `sft-kd1` run is already training, send its `kd_report.json` and log
instead of repeating this.

### A2. Leave-app-out diagnostic (Task 3, 15 min) - P

Branch `main`. Split `train`, subset `dev`, agent preset `sft-v2-refs-lao-diag`, ASR preset `gold`. Compare against the gold dev
baseline `sft-v2 + check_refs` = 72.86. **If the diff is <= -3 with the 95% CI below 0**, the model depends on same-app examples:
train `sft-v5a` (B4). Otherwise skip `sft-v5a` and send the row.

### A3. Convert Qwen3.5-4B to a text-only model (Task 2 Step 1, once) - K, about 15 min

Branch `main`. Command field:

```
sft.text_only --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --chat-template repo/configs/chat_templates/qwen_json_nothink.jinja --out merged/qwen35-4b-text --hf-repo TripleTrouble/qwen35-4b-text
```

Press Run (HF write token). It prints `<X>B parameters -> <Y>B text-only` and uploads the private repo
`TripleTrouble/qwen35-4b-text`. Expected Y is about 4.21; **above 4.44 (the limit with the 1.55B ASR) stop this candidate.** If it
fails with an unknown `qwen3_5` architecture, the transformers version on molab is too old: tell the owner.

### A4. Base model probes, no SFT (Task 2 Step 2) - K, holdout, gold text

Branch `main`, split `train`, subset `holdout`, agent preset `react-fewshot-k4`, ASR preset `gold`, *Command* empty. Baseline
(untrained Qwen3-4B) = 29.69. Two runs, extra infer args:

- Student A: `--agent-kw model=TripleTrouble/qwen35-4b-text` (Qwen3.5 may run slowly: the log of earlier runs shows the
  linear-attention layers falling back to plain PyTorch).
- Student B: `--agent-kw model=Qwen/Qwen3-4B-Thinking-2507 --agent-kw max_new_tokens=1024`. Also send `trace.jsonl` and
  `meta.json`: the share of `max_steps`, `time` or empty stops above 10% means repetition or cut-off thinking.

**Decision:** below **34.69** the candidate is closed (Qwen3-4B-Instruct stays). Student A at or above 34.69: train `sft-q1` (B3).
Student B at or above 34.69: tell the owner (it needs a follow-up plan, probably out of time).

### A5. Beam search (Task 2 Step 4, optional) - K, about 35 min

Branch `main`, split `train`, agent preset `sft-v2-refs-beam3`, ASR `pho-large-ft-b`, subset `holdout`, then `dev` if holdout is
not a loss. Apply the gate. **Stop if the max seconds per sample is above 15.**

### A6. Recovery rollout and file (Task 7) - P

1. Branch `main`, split `train`, subset `pool`, agent preset `react-fewshot-k4`, ASR preset `gold`, extra empty (about 40 minutes
   *(est.)*). This is the untrained model on purpose: sft-v2 has seen the pool and makes too few mistakes on it. Note the run folder name.
2. Command: `sft.recovery --run runs/<exact run folder name> --name r1 --max 200 --hf-repo TripleTrouble/zaic2026-lifemate-data`.
   It replays every candidate and uploads `synth/recovery-r1.jsonl`. **If `usable` is below 100, stop and report.**

### A7. Schema paraphrase table (Task 5 Steps 3 and 5) - P, about 30-40 min *(est.)*

Branch `main`, Command:

```
sft.remix --name r1 --revision fc05daec18b0a78c049392ed2e771dde82bdf654 --hf-repo TripleTrouble/zaic2026-lifemate-data
```

(The revision is the Qwen3.5-27B commit used for the reasoning sentences; it needs about 60 GB of GPU memory.) It writes the table,
prints how many apps passed the check and uploads `remix/r1.json`. **Stop and report if fewer than 80% of the apps pass, or if,
after reading 10 apps in `remix/r1.json` by hand, 2 of 10 change a tool's meaning.** Holdout apps are never written.

### A8. Act/ask scenarios (Task 4, optional) - whoever has an idle account

Branch `main`, three Commands one after the other (same session):

```
sft.synth pack --name p1 --subset pool --limit 40 --per-prompt 4
sft.synth answer --name p1 --teacher Qwen/Qwen3.5-27B --revision fc05daec18b0a78c049392ed2e771dde82bdf654
sft.synth ingest --name p1 --hf-repo TripleTrouble/zaic2026-lifemate-data
```

Read every accepted scenario of `synth/p1.jsonl` by hand. **Stop if fewer than 50% of the scenarios are accepted, or if asks are
below 30% of the accepted.** Then the bulk (about 4 h *(est.)*), again three Commands:

```
sft.synth pack --name v1 --subset pool --offset 40 --limit 908 --per-prompt 4
sft.synth answer --name v1 --teacher Qwen/Qwen3.5-27B --revision fc05daec18b0a78c049392ed2e771dde82bdf654
sft.synth ingest --name v1 --hf-repo TripleTrouble/zaic2026-lifemate-data
```

`answer` skips prompts that already have an answer, so an interrupted run resumes by repeating the same command. The last command
uploads `synth/v1.jsonl`.

## B. Training runs (`notebooks/molab_train.py`, recipe of section 1)

### B1. Distillation (Task 0 Steps 4-6) - K

Branch `exp/sft-q1`. Teachers must share the student's vocabulary (checked: Qwen3-14B and Qwen3-30B-A3B do).

1. **`sft-kd1`** (only if A1 did not say skip): run `sft-kd1`, model repo `TripleTrouble/lifemate-sft-kd1`, extra
   `--kd-teachers TripleTrouble/lifemate-sft-v2`. Do it in the **same session** as A1 so `kd_targets.pt` is reused (otherwise the
   teacher runs again, about 30 minutes). Merge, then run the gate.
2. **`sft-t1`, the large teacher** (not recommended): extra `--model Qwen/Qwen3-14B --lora-targets q_proj,k_proj,v_proj,o_proj`,
   model repo `TripleTrouble/lifemate-sft-t1`, time box 8 hours ending 8 Nov 12:00. A 14B model needs about 3.5 times the time of the
   4B *(est. from the parameter ratio)*, so the box is probably too short. Evaluate on holdout, ASR preset `gold`, branch `exp/sft-q1`,
   agent preset `sft-t1-refs`. **Decision 2:** continue only at **56.0** or above (sft-v2 50.96 + 5).
3. **`sft-kd2`** (only after step 2 passed): extra `--kd-teachers TripleTrouble/lifemate-sft-t1`, repo
   `TripleTrouble/lifemate-sft-kd2`; gate.
4. **Decision 3:** the best passing model joins the combined run; if none passes, log the rows and stop distillation.

### B2. `sft-v6w` - decision weight (Task 6) - P (or K)

Branch `main`, run `sft-v6w`, repo `TripleTrouble/lifemate-sft-v6w`, extra `--decision-weight 3` (one value only). No other dependency.
Merge, gate.

### B3. `sft-q1` - Qwen3.5 student (Task 2 Step 3) - K, only if A4 Student A >= 34.69

Branch `exp/sft-q1`, run `sft-q1`, repo `TripleTrouble/lifemate-sft-q1`, extra:

```
--model TripleTrouble/qwen35-4b-text --lora-targets q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,gate_proj,up_proj,down_proj
```

Qwen3.5 has three linear-attention blocks for each full-attention block; the list above puts adapters on both kinds (the default
list would only reach the full-attention blocks and the MLPs). Check the "trainable params" line in the log before leaving it
running. Merge, gate (the gate command uses `--agent-kw model=TripleTrouble/lifemate-sft-q1`). If it passes, later runs use this base.

### B4. `sft-v5a` - leave-app-out examples (Task 3) - P, only if A2 says so

Branch `main`, run `sft-v5a`, repo `TripleTrouble/lifemate-sft-v5a`, field *records with the gold apps left out, %* = `50`.
The records name must read `gold-pool-m1-asr-pho-large-fp16-lao50`. Merge, gate.

### B5. `sft-v5r` and `sft-v6d` - P

- **`sft-v5r`** (needs A7): branch `main`, run `sft-v5r`, repo `TripleTrouble/lifemate-sft-v5r`, field *schema paraphrase table* =
  `remix/r1.json` (records name ends in `-rx`). Merge, gate.
- **`sft-v6d`** (needs A6): branch `main`, run `sft-v6d`, repo `TripleTrouble/lifemate-sft-v6d`, field *synthetic samples* =
  `synth/recovery-r1.jsonl`. Merge, gate.

### B6. `sft-v4` (Task 4, optional, needs A8)

Branch `main`, run `sft-v4`, repo `TripleTrouble/lifemate-sft-v4`, field *synthetic samples* = `synth/v1.jsonl`. Merge, gate.

## C. After the gates

Send the owner (the person who keeps the plan) every RESULTS row, the three files per gate run, and a note of which run folder is
which model. The owner then applies the thresholds, adds the rows to `RESULTS.md`, and decides what enters the combined run:

- **8 Nov 12:00:** re-estimate the target from the best holdout score.
- **Combined `sft-v7`** (Task 12): only if two or more training changes passed; one run with all of them on the pool, gated
  against the best single run. The owner prepares the options and will send the exact fields.
- **9 Nov: `sft-final2`**: the winning recipe on all train (*Train records* = `all train (final retrain only)`), about 4.5 hours,
  then one `public_test` run (split `public_test`, ASR preset `pho-large-ft-all-beam5`), check `over_time_limit` is 0 and keep it
  only if it beats 57.5 (sft-final-h, `docs/shipped-model.md`). The owner sends the exact fields.
- Docker: the image of sft-final-h passed the portal's *Test Solution* on 10 Nov (`docs/shipped-model.md`, "Docker image").
  Deadline 16 Nov 23:59 GMT+7; Test Solution is open until 15 Nov 23:59, one run per day. A model that beats 57.5 is
  shipped by changing `configs/shipped.json` in a PR (the PR builds and smoke-tests the image), then one upload run.

### C1. Image check on GPU (score and time with the image's libraries)

`requirements.txt` applies `docker/constraints.txt`, so a molab run installs the image's transformers and
huggingface_hub (torch stays molab's). Branch `main` once the Docker PR is merged, split `public_test`, agent preset
`sft-final-h`, ASR preset `pho-large-ft-all-beam5`, HF model owner `TripleTrouble`, extra infer args:

```
--agent-kw revision="e3a659cb166907460f564c31a01e613786a835b6" --asr-kw revision="73bb56a73e1e1470bbda0fad99353b6f1ed6c338" --time-field
```

Send `meta.json` (`max_seconds`, `over_time_limit` must be 0) and `submission.jsonl` (compared with the 7 Nov
submission, or submitted to the public leaderboard against 57.5).

### C2. `sft-q1b` gate

Preset `sft-q1b-refs` (PR #77; before it is merged, branch `feat/preset-sft-q1b`), split `train`, subset `holdout`, ASR
preset `pho-large-ft-b`. Compare with the sft-v2 + check_refs baseline rescored with the current scorer: **42.77**
(41.37 under the old scorer), gate +0.5 with the CI above 0 (`python -m lifemate.compare`).

### C3. Empty retry, weekday fix and sft-q1c

Plan: `docs/superpowers/plans/2026-11-11-empty-retry-weekday-q1c.md` (Tasks 4 to 6). Presets (all `react`, `examples` 4,
`check_refs`, `id_fix`, `spoken_email`): `sft-q1b-refs-fixes-retry` (+ `empty_retry`), `sft-q1b-refs-fixes-wd`
(+ `weekday_fix`), `sft-q1b-refs-fixes-all` (both), `sft-q1c-refs-fixes` (sft-q1c, no new option),
`sft-q1c-refs-fixes-all` (sft-q1c, both). Branch `feat/empty-retry` until it is merged, then `main`.

**Training `sft-q1c`** (`notebooks/molab_train.py`): every field of the sft-q1b run (its `run_config.json` next to the
weights in `TripleTrouble/lifemate-sft-q1b` wins over this list), with three changed:

| Field | Value |
|---|---|
| Branch | the branch of the sft-q1b run (`main`) |
| Run name | `sft-q1c` |
| HF model repo | `TripleTrouble/lifemate-sft-q1c` |
| Train records | `pool (experiments)` |
| Eval loss records | `dev` |
| few-shot examples in the records | `4` |
| function-masked copies per record | `1` |
| perturb / weekday table / synthetic / reasoning / paraphrase | off, empty |
| **records with the gold apps left out, %** | **`50`** |
| ASR-noisy copy from cache | `pho-large-fp16` |
| extra train args | `--model TripleTrouble/qwen35-4b-text --lora-targets q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,gate_proj,up_proj,down_proj` |

Check: the records name reads `gold-pool-m1-asr-pho-large-fp16-lao50`, "trainable params" matches the sft-q1b log, 356
steps. Then merge the adapter as for sft-q1b and note the HF commit sha of the merged model.

**Gate runs** (`notebooks/molab_run.py`): Branch `feat/empty-retry`, Split `train`, ASR preset `pho-large-ft-b`, HF model
owner `TripleTrouble`, extra infer args empty.

| Run | Agent preset | Train subset |
|---|---|---|
| G1 | `sft-q1b-refs-fixes-retry` | `holdout` |
| G2 | `sft-q1b-refs-fixes-retry` | `dev` |
| G3 | `sft-q1c-refs-fixes` (after the sft-q1c merge) | `holdout` |
| G4 | `sft-q1c-refs-fixes` (after the sft-q1c merge) | `dev` |

Send the run folders (`trace.jsonl`, `submission.jsonl`, `meta.json`). Baselines: `sft-q1b-refs-fixes` with
`pho-large-ft-b`, holdout 44.83 (`runs/20261110-161452_dc97998_train_sft-q1b-refs-fixes_pho-large-ft-b_holdout`) and dev
75.04 (`runs/20261110-163834_dc97998_train_sft-q1b-refs-fixes_pho-large-ft-b_dev`); gate +0.5 on holdout with the CI above
0 and dev >= -0.3 (`python -m lifemate.compare <baseline> <run>`).

- `empty_retry` (G1, G2): gate passed, it stays in the `-all` presets; below the gate but not negative, only the public run
  decides; negative on either split, it is dropped from the shipped preset.
- `sft-q1c` (G3, G4): pooled difference over the 597 samples below 0 closes it; at or above 0 it gets a public run.
- `weekday_fix` has no train gate (three samples change in the simulator); the public run decides.

**Public runs**: Branch `feat/empty-retry`, Split `public_test`, ASR preset `pho-large-ft-all-beam5`, HF model owner
`TripleTrouble`, extra infer args `--time-field`. In this order: `sft-q1b-refs-fixes-all` (`sft-q1b-refs-fixes-wd` if
`empty_retry` was dropped), `sft-q1c-refs-fixes-all` (only if sft-q1c was kept), `sft-q1b-refs-fixes` (only if the first
scores below 61.05). Validate `submission.jsonl`, check `over_time_limit` is 0, submit; the baseline to beat is 61.05 and a
tie goes to sft-q1b with the fewer options.

## D. Closed, nothing to run

Experiments 7 (few-shot selection by tool structure: +1.04 points of hit rate against a 10-point bar), 8 (date correction: simulation
shows no gain; the GPU step was declined), 10 (vote across models: +0.76 [-1.50, +3.04] against a +1 bar). Results are in `RESULTS.md`.
