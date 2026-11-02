# Final Sprint Plan: beyond SFT v1 (2 Nov – 14 Nov)

**Goal:** raise the private-test score from the current public 47.85 (SFT v1 + `pho-large-ft-a`) with two people working in parallel, one on the agent/LLM side and one on the ASR side, and freeze one shippable system on 14 Nov.

**Owners:** **K** = agent/LLM track (Khang). **P** = ASR track (partner). Tasks name their owner; shared tasks name both.

**Merges two plans:** `2026-10-29-agent-track.md` (Phases 3–4 stay as written there; this plan adds to them and does not replace them) and `2026-10-28-asr-models-and-data.md` (B1/B2 are done; this plan covers what is left after the fine-tuned ASR).

**Execution details:** `2026-11-02-sprint-execution.md` turns K1, K3, K5b and K6 into step-by-step tasks and adds schema perturbation and the Docker package.

## Where the points are lost (sft-v1, gold transcripts)

| Subset | act | ask | out-of-scope (empty) | overall |
|---|---|---|---|---|
| dev | 63.08 | 60.23 | 76.0 | 63.74 |
| holdout | 45.34 | **13.24** | 85.71 | 42.30 |

Public test: 47.85 with `pho-large-ft-a` (ASR WER 0.096 on dev + holdout). Unknown so far: how many points the remaining ASR errors cost. S1 measures it. Everything else is ordered by evidence from this table, not by guess. Holdout (apps unseen in training) is the best stand-in for private test, so it is the number to move.

## Rules that apply to every task

- **Never label, pseudo-label, transcribe-to-train or tune on public_test / private_test** (`context/RULE.md`: "Labelling public_test or private_test (manually or with external services or models) to train or tune a solution is not allowed"). Public_test is used for one thing: reading the leaderboard score of a submission.
- Experiments are measured on **dev and holdout only**, with `compare` against the current best. Pool samples are for training; the final retrain is the only job that touches dev and holdout (Task 14 of the agent plan).
- Synthetic data may be generated with other models or services during development, never at inference (`RULE.md`). Inference is offline, < 6B params total, ≤ 30 s per sample on an A30, deterministic.
- Train data is licensed to registered teams only. Before any train text goes to an external API, K and P agree it is acceptable (Question Q2). No tokens, no data, no model links in the repo.
- Branch per task from an up-to-date `main`, PR with `pytest` output and the `compare` line, review by the other owner, then merge. Every behaviour change gets a `RESULTS.md` row.
- New agent behaviour is opt-in through `--agent-kw` and a preset until a run beats the current best on dev **and** holdout.

## Timeline

| Dates | K (agent / LLM) | P (ASR) | Submission |
|---|---|---|---|
| 2–5 Nov | K1 merge PR #14; K2 diagnose; start K3 | P1 ship `pho-large-ft-b`; **P2 pool ASR cache for SFT v2**; P3 | none |
| 5–8 Nov | K1 `sft-v2` run + gate; K3 guards; K4 | P3 result; P4 beam / n-best | `sft-v2` + best ASR |
| 8–11 Nov | K5 distillation (gated); K3 tuning | P5 final ASR retrain | best system |
| 11–14 Nov | K6 final SFT retrain, Task 15 | P6 timing and determinism of ASR; both: hand-off | freeze 14 Nov |

Docker (deadline 16 Nov 23:59 GMT+7) is planned in `2026-11-02-sprint-execution.md` (Tasks 9–10, owner K). Synthetic data (K5b) moved to P on 2 Nov; see the owner note there.

---

## P: ASR track

### P1: Ship `pho-large-ft-b` (today)
- [ ] Upload `final` to a private HF repo (`molab_asr.py`, Upload; weights + processor only), then transcribe the eval clips as cache `pho-large-ft-b` and run `asr_report`.
- [ ] Add the ASR preset `pho-large-ft-b` and the `RESULTS.md` ASR row. Reference: `pho-large-ft-a` WER 0.0960 / CER 0.0564 on 2108 clips (that repo was lost; this run is its replacement, best 400-clip eval WER 0.1057 vs 0.1067). Keep a copy of `model.safetensors` outside the org repo (storage limit).

### P2: Pool ASR cache for SFT v2 (blocks K1, due 5 Nov)
- [ ] `transcribe --split train --asr whisper --asr-kw model=vinai/PhoWhisper-large --name pho-large-fp16 --subset pool`, then upload `pho-large-fp16.jsonl` to the team's private place and tell K.
- Use the **untuned** model on purpose: the fine-tuned model has seen the pool, so its transcripts are almost clean and the SFT would learn nothing about noise. Optional later: a second cache from a model fine-tuned on a disjoint half of the pool, to get realistic noise at ASR-quality WER.

### P3: How many points does ASR still cost? (free, answers where to spend the rest)
- [ ] Run `sft-v1` on dev and holdout with ASR `pho-large-ft-b`; `compare` against K's gold runs (dev 63.74, holdout 42.30).
- [ ] Decision: gap < 5 points → ASR is good enough, P helps on the agent side (P4 is skipped; P5 is kept for safety). Gap > 10 points → run `asr_errors` on the losing samples and prioritise P4.

### P4: Better decoding (only if P3 says ASR still costs points)
- [ ] Beam search (`num_beams` 5) against greedy: WER on eval and p95 seconds. Keep it only if WER improves and p95 stays far below the 30 s budget (ASR + LLM together).
- [ ] N-best: pass the top-2 hypotheses to the agent as one request text ("A / or: B") and measure on dev and holdout. This is agent-visible, so agree the format with K first.

### P5: Final ASR retrain (8–10 Nov)
- [ ] Same hyper-parameters as `pho-large-ft-b`, `--subset all` clips (pool + dev + holdout, about 5k clips), then `final`. There is no held-out set left, so accept it only if the public score is not lower than with `pho-large-ft-b` (one submission). Keep `pho-large-ft-b` as the fallback.
- [ ] Optional if time: VietSpeech subset (Run B in the ASR plan), only if licence question 1 in `docs/asr_audit.md` is answered.

### P6: Timing and determinism of the ASR half (11–14 Nov)
- [ ] Per-clip seconds on molab for the chosen ASR (p95 from `asr_report`) and a repeat-run check (same input → identical text). Hand the numbers to the system-level check (agent plan Task 15).

---

## K: Agent / LLM track

### K1: SFT v2 (agent plan Task 14, already planned)
- [ ] Merge PR #14. Train `v2-pool` = gold + `--mask 1` + the ASR copy built from P2's cache. Gate as in the agent plan: keep v2 only if holdout improves and the ask column does not drop more than 5 points.

### K2: Diagnose where sft-v1 and sft-v2 lose points (free)
- [ ] `diagnose runs/<run>` on the dev and holdout gold runs: missing calls, wrong arguments, extra optional keys, stop reasons. Write the top three failure patterns with 3–5 example ids each into the PR. Everything below is chosen from this, not from the table in this plan.

### K3: Deterministic guards in the agent (no training, opt-in kwargs)
Each is one `--agent-kw` flag with its own preset and its own dev + holdout `compare`:
- [ ] **Ask rule.** Holdout ask is 13.24. When a read returns several candidates and the next call needs exactly one id, force `sys.ask_missing_info` instead of letting the model pick.
- [ ] **Out-of-scope guard.** A gold `[]` scores only with an empty submission, and any read call already breaks it (seen in the `RESULTS.md` notes). Add a short "is this request covered by an installed app?" check before the first tool call, and compare empty-sample accuracy before and after.
- [ ] **Argument validation with `env.validate()`.** It is free and not logged. Fix only what the schema decides (types, enums, required keys). Do **not** strip optional keys by a fixed rule: gold uses optional parameters in 40% of reads and 58% of writes that have them (agent plan evidence).

### K4: Name fallback for `contacts.search`
- [ ] About 27% of proper names are misheard (`asr_errors` report). If a search by a name returns nothing, retry by a partial name or by the relation; then fetch the address book and take the closest name by folded-string similarity; then search again with the exact name. Count every extra `search` call in the score (reads are scored), so measure the net effect on dev and holdout, not just recall.

### K5: Distillation and other training (gated)
Run only if, after K1 and K3, holdout ask is still below about 40 or holdout overall is below about 50. Cheapest first:
- [ ] **K5a, self-distillation (rejection sampling), no API.** Sample several greedy-with-different-prompt or low-temperature runs of `sft-v1` on **pool** samples, keep only trajectories that score full F1 against the gold calls, add them to the records, and retrain. This teaches the model the correct behaviours it only reaches sometimes. Temperature sampling is a training-time tool only; inference stays greedy.
- [ ] **K5b, teacher-generated synthetic data for unseen apps.** A larger model (an API, development only) writes new user requests and gold-style trajectories for tool schemas of apps that have no or few train examples. Verify each trajectory by replaying it in a fresh train `ToolEnv` (calls valid, no schema rejections, state change as intended) and drop the rest. Include **function masking** (renamed tools, already in Task 12) so the model keeps reading the schema. Budget: set a cap on tokens and generated records before starting.
- [ ] **K5c, preference or RL-style training** (DPO, GRPO) is out of scope for this sprint (agent plan, Tier 3). Revisit only if K1–K5b are done by 10 Nov.
- Teachers must work from **train** schemas and train requests only. Nothing from public_test, not even the request texts, goes into a prompt.

### K6: Final retrain and hand-off (agent plan Task 14 Step 3 and Task 15)
- [ ] Final retrain on all train with the winning recipe, then twice-run determinism (`cmp` on the two `submission.jsonl`), mean/max seconds, the pinned preset with `revision`, and the training commands in `README.md`.

---

## Shared: decisions to settle in the next sync

- **Q1.** Is K5b (an API as teacher) wanted, and who pays? If not, K5a is the only distillation we do.
- **Q2.** Is sending train **text** (not audio) to an external API acceptable under the data licence? If unsure, ask the organisers, or restrict K5b to synthetic prompts built from the catalog schemas only, with no real train request text.
- **Q3.** N-best format between the ASR and the agent (P4): one string with alternatives, or a second field on `Task`? This touches `task.Task`, which K owns.
- **Q4.** Who owns the Docker package from 12 Nov?
- **Q5.** One person owns `RESULTS.md` merges each day, to avoid the conflicts we already had.

## Success criteria

- 5 Nov: `sft-v2` records built from P2's cache and training started.
- 8 Nov: one submission with `sft-v2` + the best ASR; ASR cost known from P3.
- 14 Nov: one frozen model pair (LLM revision + ASR), twice-run deterministic, with its timing in `RESULTS.md`. Anything not finished by then stays out of the shipped system.
