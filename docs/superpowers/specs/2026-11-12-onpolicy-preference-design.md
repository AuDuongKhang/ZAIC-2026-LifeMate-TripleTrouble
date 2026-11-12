# On-policy preference training for unseen apps — design

Date: 12 Nov 2026. Status: draft for the owner's approval. Implementation plan: to be written from this document.

## Intent

The shipped agent (`sft-q1c-refs-fixes-all-clock`, public 63.44) loses most of its remaining points on decisions, not on
details a rule can patch. On the 12 Nov holdout gate run (G3, ASR, 49.25) the loss report (`lifemate.loss_report`) shows:

| Failure | Samples | Points lost |
|---|---|---|
| Other tools than gold (wrong tool set) | 49 | 16.50 |
| State right, calls differ (reads, free text) | 99 | 19.10 |
| Ask/act flip | 22 | 7.41 |
| Wrong write arguments (dates, numbers, ids) | ~20 | ~7 |

Holdout (apps absent from the training pool) scores 25 points below dev. Rules such as `clock_fix` move the score by
about one sample each. The owner wants the direction with the largest potential on the private test, which is expected
to look like the public test (a mix of train apps and apps never seen in training).

The idea: let the model learn from **its own mistakes on apps it does not know**. sft-q1c is sampled on pool samples
rendered as if their apps were unseen (renamed tools, gold apps left out of the few-shot examples). Wherever a sampled
next step disagrees with gold at a decision (which tool, ask or act, stop or continue), the gold step and the model's
step form a preference pair. A preference loss (DPO with an SFT term) then pushes the model toward the gold decision
and away from its own error. Rounds can repeat on the new model, which makes this an offline approximation of
on-policy RL with a verifiable reward, built on the SFT infrastructure the repo already has.

## Why this direction (comparison made in the 12 Nov session)

| Option | Potential on unseen apps | Time to a gated result | Notes |
|---|---|---|---|
| **Step-level preference on the model's own errors (chosen)** | High | ~2 days for round 1 | Targets the two largest decision losses; reuses records, masking, tokenizer, train loop |
| GRPO / PPO with `ToolEnv` rollouts | Highest | 4+ days, uncertain | Needs a multi-turn rollout loop inside training; no tested result possible before 16 Nov |
| Synthetic apps + generic simulator | High | 3+ days | Data distribution risk; no simulator exists |
| RFT (keep only full-score samples) | Medium | ~1.5 days | Still plain SFT on the pool, which lost on holdout six times |
| WiSE-FT (interpolate sft-q1c with its base) | Medium | hours | Kept as a side track (below) |
| Logit KD, decision weight, recovery records, beam, prompt vote | — | — | Already tried and lost (`RESULTS.md`) |

What differs from the failed `sft-v6w` (decision weight) and `sft-v6d` (recovery records): those only reweighted or
added gold text. Here the negatives are the model's own outputs, sampled under unseen-app conditions, so the loss acts
exactly where the model is wrong, and the reference model keeps it close to sft-q1c everywhere else.

## Goals

- A model `lifemate-sft-q1d` (sft-q1c + preference LoRA, merged) that beats sft-q1c on the holdout gate and on public.
- Ship it in the Docker image before 16 Nov 2026 23:59 GMT+7 if, and only if, it wins; otherwise the image stays at 63.44.
- The pipeline (sampling, pairing, loss) is reusable for a second round on q1d.

## Non-goals

- Online RL (GRPO/PPO) and synthetic apps: not before 16 Nov.
- Any change at inference: decoding stays greedy, prompts, rules and presets stay as they are; only the weights change.
- Retraining from the base model.

## Anti-goals (a failure even if it technically works)

- Training on holdout, dev, public or any public catalog content. Holdout must stay the clean measure of unseen apps.
- Teaching the model that a valid alternative is wrong (a read in another order, an equivalent search query). Pairs
  must only encode decisions where gold is the single right answer.
- A model that wins on dev or public but loses on holdout: that is memorisation, not generalisation.
- Breaking the shipped image or its reproducibility (`training_code/` must reproduce the shipped weights).

## Constraints

- Team rules (CLAUDE.md): no paid or external services, no LLM APIs; local open models only; LLM < 5B, whole system
  < 6B; ≤ 30 s per sample on an A30; fixed seeds; greedy decoding at inference; offline Docker.
- Temperature sampling is allowed for producing training data only (as K5a in the 2 Nov plan already stated).
- Compute: molab GPU, available continuously until 16 Nov 23:59.
- Git: never on `main`; design and plans on `docs/*` branches, code on `feat/*` or `exp/*`, each with a PR.
- Reproducibility (Docker guide 6.2): the organisers must be able to reproduce the shipped weights from
  `training_code/`. Sampling on a GPU is not bit-for-bit repeatable, so the pair file itself ships in `training_code/`
  (the image already carries `train.jsonl`, so no new data leaves the team), sampling uses fixed seeds, and
  `records_commit` in `configs/shipped.json` points at the code that made the pairs and the training.

## How it works (decided)

### 1. Prompts to sample from

- Source: the **pool** subset only (train minus dev and holdout), the same samples sft-q1c was trained on.
- Each sample is rendered as an SFT record by the existing replay (`sft.records`), with function masking
  (`sft.mask`, fresh random names, not the names used in sft-q1c's records) and with the gold apps left out of the
  few-shot examples. This puts the model in unseen-app conditions on samples whose gold is known.
- Text: half of the records with gold text, half with the ASR-noisy text of the `pho-large-fp16` cache, as in sft-q1c.
- Every agent turn of a record defines a decision point: the conversation up to that turn is the prefix, the gold turn
  is the gold step.

### 2. Sampling

- At each decision point, sft-q1c generates K = 4 continuations with temperature 0.8 (top-p 0.95), plus the greedy
  continuation. Fixed seed per (sample, turn).
- A sampled step is parsed with the agent's parser (`agents.parsing.parse_tool_calls`).

### 3. Which disagreements become pairs

A sampled step is compared to the gold step on the decision only:

| Gold step | Model step | Pair? |
|---|---|---|
| a write call to tool X | a call to another tool, an ask, or a plain answer | yes |
| `sys.ask_missing_info` | any tool call or a plain answer | yes |
| a plain answer (done, or out of scope) | any tool call | yes, if the call is a write or an ask |
| a read call to tool X | another read tool | **no** (reads in another order or another search can be valid) |
| a read call | a write call or an ask | yes |
| same tool | other arguments | **no** in round 1 (argument conventions are inconsistent in gold) |

Tool identity is compared on the real tool names (the masking map is inverted before comparing). The rejected text is
the model's sampled step exactly as generated; the chosen text is the gold step as the record renders it.

Each decision point contributes at most one pair (the most frequent wrong decision among the K samples), so a point the
model gets wrong 4 times out of 4 does not outweigh others. A point where the greedy step is already gold and only
sampled steps disagree is kept, but marked, so the plan can compare "greedy-wrong only" against "all" pairs.

### 4. Loss and training

- Model: a new LoRA (same targets and rank as sft-q1c) on top of the merged sft-q1c weights at its pinned revision.
  The reference model is the same network with the adapter disabled, so no second copy is loaded.
- Loss per pair: DPO on the step's tokens (β = 0.1) plus an SFT (NLL) term on the chosen step with weight 1 (the
  "RPO" form), so the gold step's likelihood cannot drop while the margin grows.
- Mixed in: the plain SFT records of the pool at a small share, so behaviour away from decision points does not drift.
- One epoch, learning rate 5e-6 to 2e-5 (the plan picks the value with the pilot), same tokenizer and chat template,
  the same max length as the records.
- Implemented as an option of `lifemate.sft.train` (new loss path, pair records with `prompt`, `chosen`, `rejected`),
  not as a new trainer, so checkpointing and HF upload are unchanged.

### 5. Gates and decisions

- **Pilot first** (before the full run): sample 100 pool samples. Proceed only if at least 15% of decision points give
  a pair; below that the model is too sure on pool and the plan must change the prompt conditions before training.
- **Train gate** (`notebooks/molab_run.py`, ASR `pho-large-ft-b`, the shipped agent preset with the new model):
  holdout at least +1.0 over sft-q1c with the CI above 0, and dev at least -0.3. The bar is higher than +0.5 because
  the change is to the weights, which every past retrain failed on holdout.
- **Public run** only after the train gate; ship only if the leaderboard is at or above 63.44.
- **Round 2** (sample again with q1d, train q1e) only if round 1 passed both gates and time remains before 15 Nov
  23:59 (last Test Solution).

### 6. Side track: WiSE-FT

Independent of the main track and cheap: merge `θ = α·θ_q1c + (1-α)·θ_base` for α in {0.7, 0.85}, run the holdout/dev
gate, and keep the best if it wins by the same bar. If both tracks win, the plan may apply the interpolation to q1d.

## Schedule (GMT+7)

| When | What |
|---|---|
| 12–13 Nov | Plan; sampling script with a sampling option for `HFChatLLM` (data generation only), pair builder, preference loss in `sft.train`; unit tests on CPU |
| 13 Nov evening | Pilot on molab (100 pool samples): pair rate, pair types, a few read by hand |
| 14 Nov | Full sampling on the pool; train q1d; merge; holdout/dev gate. WiSE-FT gate in parallel |
| 15 Nov | Public run; ship PR + image; Test Solution (last day it is offered) |
| 16 Nov | Buffer; submit link + MD5 before 23:59 |

## Left to the builder (the implementation plan)

- File layout of the sampler and pair builder, the record format of a pair, and how it is tokenized (chosen and
  rejected share the prefix; labels only on the step).
- Batching and speed of sampling on molab (the pool has ~950 samples and ~2.5 agent turns each; with K = 4 that is
  about 10,000 generations, which must fit in a few hours).
- The exact learning rate and the SFT-record share, from the pilot.
- How the pair file is versioned and copied into `training_code/`.
- `sft.mask` must expose the renaming map of a masked record so sampled calls can be mapped back to real names.
- Tests: pairing table (section 3) as unit tests, loss on a toy model, masking inversion, deterministic sampling seeds.

## Risks

- **The model is too sure on pool samples** (it trained on them), so few disagreements appear: the pilot measures this
  before any training. Masking with fresh names and leaving the gold apps out are meant to prevent it.
- **Pairs teach a convention instead of a decision**: the pairing table excludes reads and argument differences; the
  pilot includes reading 30 pairs by hand.
- **Holdout noise** (CI about ±3 points): a real but small gain may not pass +1.0; then it is not shipped, by design.
- **Time**: if round 1 is not gated by 15 Nov 12:00, it is not shipped; the image stays at 63.44.

## Calls made by the designer (not discussed with the owner; check these)

- K = 4 samples at temperature 0.8 / top-p 0.95 per decision point, plus the greedy step.
- The pairing table of section 3, in particular: no pairs for read-vs-read or for argument differences in round 1.
- DPO β = 0.1 with an SFT term of weight 1 (RPO form), one epoch, a new LoRA on the merged sft-q1c.
- The pilot threshold (15% of decision points give a pair) and the train gate (holdout +1.0 with CI above 0, dev ≥ -0.3).
- The pair file ships inside `training_code/` for reproducibility.
- WiSE-FT runs as a side track with α in {0.7, 0.85}.
