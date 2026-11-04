# Score Target Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise the public score from 56.01 towards the team's 80-90 target by 10 Nov 2026, and hand the best gated system to Docker Task 9 on 11 Nov whatever the score is.

**Architecture:** Spend GPU time on the model first, because guards and rules have added almost nothing so far and SFT added the most. Each training change is gated on holdout, and a small offline tool shows which error buckets each new model moved. ASR work runs in parallel behind a WER gate that needs no agent run.

**Tech Stack:** Python, PyTorch + peft on molab, HF Whisper pipeline, `lifemate.compare`, `lifemate.sim`.

**Spec:** `docs/agent_tuning.md`, `docs/hybrid_rules.md`, `docs/error_analysis.md`, and the row and simulation tables in `RESULTS.md`.

## Where we stand

Same agent (sft-v2 + check_refs) on gold text and on ASR text:

| Subset | Lost by the agent (gold text) | Lost to ASR | Score on ASR |
|---|---|---|---|
| dev | 27.14 | 6.00 | 66.86 |
| holdout | 49.04 | 9.59 | 41.37 |

Each bucket replaced by gold, alone, on the ASR baselines (they overlap, so they do not add up):

| Bucket | holdout | dev |
|---|---|---|
| Ask/act (gold asks / agent asks) | +13.47 (+6.40 / +7.07) | +5.00 (+2.33 / +2.67) |
| Reads (wrong args / missing / extra) | +11.95 (+6.17 / +3.76 / +1.18) | +3.40 (+2.40 / +0.41 / +0.37) |
| Write set | +11.43 | +7.13 |
| Write args, all | +16.85 | +13.58 |
| of which free text / number / date / other string / id | +3.75 / +2.97 / +2.60 / +1.92 / +1.82 | +2.49 / +1.37 / +1.40 / +2.10 / +4.46 |

- ASR alone cannot reach the target: perfect text adds 6-10 points on train. The agent must gain at least 15-25.
- No known lever is shown to reach 80-90. Treat the target as the direction, and 8 Nov as the point where we re-estimate it.

## Calendar

| Date | K | P |
|---|---|---|
| 4-5 Nov | Tasks 1 and 2 | Task 3 code |
| 5-6 Nov | Task 2 decision, Task 4, Task 6 only if its trigger fires | Task 3 gate |
| 6-8 Nov | Review P's work | Task 5 (`sft-v4`) |
| 8 Nov | Re-estimate the target from the best holdout score | same |
| 9 Nov | Task 7 retrain and submission | hand over the final ASR preset |
| 10 Nov | Buffer, one more submission if a gated change is left | buffer |
| 11 Nov | **Start Docker Task 9 with the best system, whatever the score** | review |

Docker deadline: 16 Nov 2026 23:59 GMT+7. The floor system is `sft-final` + `pho-large-ft-all-beam5` (56.01).

## Global Constraints

- LLM < 5B params, context <= 16,384 tokens, whole system < 6B; <= 30 s per sample on 1x A30 with ASR included; offline at inference; fixed seeds; deterministic decoding (`context/RULE.md`).
- No paid or external services and no LLM APIs in the repo, dev tools or data pipeline. Free tiers and local open models only (`CLAUDE.md`).
- Never label public_test or private_test to train or tune. Public is only for reading the leaderboard score of a submission.
- Experiments train on the pool only. Only Task 7 uses dev and holdout for training. `sft-final` and `pho-large-ft-all` have seen dev and holdout, so never gate on them.
- **Gate for every change (ASR text, `pho-large-ft-b`):** holdout diff >= +0.5 with the 95% CI above 0, dev diff >= -0.3, holdout ask column drops by at most 5 points. Baselines: `runs/20261102-104252_3c4fd00_train_sft-v2_pho-large-ft-b_holdout_check_refs` (41.37, ask 33.06), `runs/20261102-101746_3c4fd00_train_sft-v2_pho-large-ft-b_dev_check_refs` (66.86). From Task 4 on, also compare against the best model that passed.
- New behaviour is opt-in (kwarg, flag, preset). Each GPU run gets a `RESULTS.md` row with its compare line.
- Do not repeat what is already measured: `freetext`, `minimal_reads`, `ask_guard`, `leak_guard`, `refusal_check`, prompt vote, `date_hints` without retraining, an intent router, an app filter.
- English in the repo. Branch per task from an up-to-date `main`, PR into `main`, with `pytest` output and compare lines. Never commit data, tokens, synthetic samples or `train_records/`. On Windows set `PYTHONUTF8=1`.

## Not in this plan

- **LLM rewrite of the request:** it has no audio, cannot restore lost names or digits, can damage correct text, and shifts the agent's input. Same point ceiling as Task 3 with more risk.
- **N-best reranking:** only worth building if Task 3 shows that context helps but the prompt is not enough, and only with time left after 8 Nov.
- **Larger LoRA or more epochs:** sft-v2's eval loss stopped improving near step 200 (0.040) while train loss kept falling, which points to data variety, not capacity. Eval loss was on dev, so this is suggestive only.
- **Date resolver rewrite:** ceiling +2.60 on holdout with about half coverage. Task 4 targets the same errors first.

## Review Focus

1. **Train labels leaking into the Whisper prompt:** on train, `sample` still holds `audio_text` and `label`, so using them would fake a WER win. Test in Task 3.
2. **Non-deterministic prompts:** building the vocabulary through a `set` changes the order between processes, and a re-run then differs from the submission. Test in Task 3.
3. **Prompt longer than the decoder allows:** long history must be cut, not crash a clip. Test in Task 3.
4. **Out-of-scope samples in the ceilings:** a gold `[]` sample with calls must be counted, and its fix is an empty call list. Test in Task 1.
5. **Trained with date hints, served without them** (or the reverse): the preset must carry the flag. Test in Task 4.

---

### Task 1: `lifemate.ceilings`, the error bucket report (K, no GPU)

**Files:**
- Create: `src/lifemate/ceilings.py`
- Test: `tests/test_ceilings.py`

**Interfaces:**
- Consumes: `lifemate.sim.rewrite_run(run, data_root, name, rewrite, out_dir)`.
- Produces: `fix_rewrite(samples: dict[str, dict], arg_types: set[str] = frozenset(), reads: str | None = None, ask: str | None = None, write_set: bool = False, out_of_scope: bool = False)` returning a `lifemate.sim` rewrite; `arg_type(spec: dict | None, key: str, value) -> str` returning one of `free, id, date, num, enum_bool, string`; `BUCKETS: dict[str, dict]` (bucket name to `fix_rewrite` kwargs); CLI `python -m lifemate.ceilings runs/<run>` writing `<run>/ceilings/<bucket>/` and `<run>/ceilings.md`.

Rules: `arg_type` checks in order `x-match` -> free, `x-ref` (or `items.x-ref`) -> id, ISO date or `HH:MM` value or `format` date/date-time/time -> date, boolean or enum -> enum_bool, number -> num, otherwise string. Pred calls align to gold calls by name, greedily, best argument overlap first. `arg_types` applies only to write calls. `reads` is `drop_extra`, `add_missing`, `fix_args` or `all`. `ask` is `gold_asks`, `agent_asks` or `all` and replaces the whole sample with gold. Buckets: `free_text, id, date, num, string, enum_bool, all_args, reads, reads_drop_extra, reads_add_missing, reads_fix_args, ask_act, ask_gold_asks, ask_agent_asks, write_set, out_of_scope, all_args+reads, all_args+reads+ask, everything`.

- [ ] **Step 1: Write the failing tests** in `tests/test_ceilings.py` (skip without `data/train`, using the fake-run fixture pattern of `tests/test_guard_sim.py`):
  - `test_arg_type_reads_the_value_when_the_schema_has_no_format`: a plain string parameter with value `"2026-11-14"` -> `date`.
  - `test_fix_rewrite_returns_none_when_nothing_changes`: a sample whose calls equal gold, under every bucket -> `None`.
  - `test_all_buckets_score_at_least_the_run`: every bucket's `worse` is 0 except `write_set` and `everything`, which may reorder writes.
  - `test_out_of_scope_fix_is_an_empty_call_list`: gold `[]` with one pred call -> `[]`, and that sample scores 1.
- [ ] **Step 2:** Run `pytest tests/test_ceilings.py -v`. Expected: FAIL, module missing.
- [ ] **Step 3:** Implement `src/lifemate/ceilings.py`. Use `lifemate.scoring` only through `rewrite_run`. Refuse a run whose `meta.json` split is not `train`.
- [ ] **Step 4:** Run `pytest tests/test_ceilings.py -v`. Expected: PASS.
- [ ] **Step 5: Check against the numbers above.** Run `python -m lifemate.ceilings runs/20261102-104252_3c4fd00_train_sft-v2_pho-large-ft-b_holdout_check_refs`. Expected: `ask_act` +13.47, `reads` +11.95, `reads_fix_args` +6.17, `free_text` +3.75, `date` +2.60, `id` +1.82.
- [ ] **Step 6:** Run `pytest`, commit `Add lifemate.ceilings, score ceilings per error bucket` on `feat/ceilings`, open a PR.

### Task 2: Train and gate `sft-v3` (K, GPU)

**Files:**
- Modify: `configs/presets.json`, `tests/test_presets.py`, `RESULTS.md`

**Interfaces:**
- Consumes: `lifemate.ceilings` from Task 1.
- Produces: preset `sft-v3-refs` (`{"model": "${HF_MODEL_OWNER}/lifemate-sft-v3", "examples": 4, "check_refs": true}`) and **the winning recipe**, which Tasks 4, 5 and 7 reuse.

- [ ] **Step 1: Train** with `notebooks/molab_train.py`, branch `main`: run name `sft-v3`, model repo `TripleTrouble/lifemate-sft-v3`, records pool, eval dev, masked copies 2, perturb on, ASR copy `pho-large-fp16`. Expected: about 3 hours; records named with `-m2-p`.
- [ ] **Step 2:** Merge the adapter with the notebook's merge step and check that `TripleTrouble/lifemate-sft-v3` loads.
- [ ] **Step 3:** Add the preset and a `test_sft_v3_preset` asserting `resolve("agent", "sft-v3-refs", {"HF_MODEL_OWNER": "team"}) == ("react", {"model": "team/lifemate-sft-v3", "examples": 4, "check_refs": True})`. Run `pytest tests/test_presets.py -v`. Expected: PASS.
- [ ] **Step 4:** On `notebooks/molab_run.py`, run agent preset `sft-v3-refs` on holdout, then dev, with ASR preset `pho-large-ft-b`, and on holdout with ASR `gold`. Evaluate each.
- [ ] **Step 5:** Run `python -m lifemate.compare` against both ASR baselines, `python -m lifemate.ceilings` on the holdout ASR run, and add three `RESULTS.md` rows with the compare lines.
- [ ] **Step 6: Decide.**
  - Gate passed: the winning recipe is masked 2 + perturb. Otherwise it stays masked 1, no perturb (`sft-v2`).
  - Holdout diff below +3: trigger Task 6.
  - Holdout diff +3 or more: also run the same recipe with masked copies 4 (run name `sft-v3m4`, about 4.5 hours) and gate it against `sft-v3`.
- [ ] **Step 7:** PR `Add the sft-v3 preset and results`.

### Task 3: Context prompt for Whisper (P, GPU for the gate)

**Files:**
- Create: `src/lifemate/asr/context.py`
- Modify: `src/lifemate/asr/whisper.py`, `src/lifemate/infer.py:94`, `src/lifemate/transcribe.py`, `configs/presets.json`, `RESULTS.md`
- Test: `tests/test_asr_context.py`, `tests/test_asr_backends.py`

**Interfaces:**
- Produces: `context_prompt_text(sample: dict, apps: dict[str, dict], max_chars: int = 600) -> str`; `WhisperASR(..., context_prompt: bool = False, split: str | None = None, data_root: Path | None = None)`; presets `pho-large-ft-b-ctx`, `pho-large-ft-b-beam5-ctx`, `pho-large-ft-all-beam5-ctx`.

Prompt order, each part deduplicated by first occurrence: `context.user_profile.name`; assistant `text` and string values of `tool_results` from history, newest turn first, up to 250 characters; `name_vi` and `description` of each installed app (`apps` comes from the split's catalog) until `max_chars`. `split` and `data_root` reach the backend through the existing `infer.with_split`, in both `infer.py` and `transcribe.py`. Per clip, pass `prompt_ids = tokenizer.get_prompt_ids(text, return_tensors="pt")`, cut to 200 tokens, in `generate_kwargs`. An empty text passes no `prompt_ids`.

- [ ] **Step 1: Write the failing tests:**
  - `test_prompt_never_uses_audio_text_or_label`: put a unique marker in `audio_text` of the sample and of each history turn and in `label`; the marker is absent from the output.
  - `test_prompt_is_deterministic`: two calls, and a call in a subprocess with a different `PYTHONHASHSEED`, give equal strings.
  - `test_prompt_respects_max_chars`: `len(output) <= max_chars` with 24 long history turns.
  - `test_prompt_without_history_or_apps_is_empty`: a sample with no history, no apps and no profile name -> `""`.
  - In `tests/test_asr_backends.py`, with the existing fake pipeline: `test_context_prompt_off_by_default` (`prompt_ids` not in the kwargs of the generate call), `test_context_prompt_caps_tokens` (a fake tokenizer returning 500 ids gives 200).
- [ ] **Step 2:** Run `pytest tests/test_asr_context.py tests/test_asr_backends.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement as above. Confirm on molab that the real pipeline accepts `prompt_ids`; if it does not, call `model.generate` directly inside `transcribe`.
- [ ] **Step 4:** Run `pytest`. Expected: PASS. Commit on `feat/asr-context-prompt`, open a PR.
- [ ] **Step 5: WER gate**, 2108 eval clips: `python -m lifemate.transcribe --split train --asr whisper --asr-kw model=TripleTrouble/pho-large-ft-b --asr-kw context_prompt=true --name pho-large-ft-b-ctx --subset eval`, the same with `num_beams=5`, then `python -m lifemate.asr_report --split train <cache>` and `python -m lifemate.asr_errors <cache>`. Pass: WER below 0.0962 (greedy) and 0.0911 (beam 5), name error rate not worse, and no more than 2 extra clips with a transcript 1.5x longer than gold (prompt echo). Add rows to the ASR table.
- [ ] **Step 6: Agent gate**, only if Step 5 passes: `sft-v2-refs` with ASR `cached:pho-large-ft-b-ctx` on holdout and dev, global gate. Fail: drop the feature.
- [ ] **Step 7: Public**, only if Step 6 passes: best agent + `pho-large-ft-all-beam5-ctx`, one submission. Keep it only if it beats the best score so far.

### Task 4: Date hints in training (K, GPU)

Date arguments cost up to +2.60 on holdout, and on public a wrong date can end a whole sample. The hints lost 2 points on a model that never trained on them, so this tests them with a retrain.

**Files:**
- Modify: `notebooks/molab_train.py`, `configs/presets.json`, `tests/test_presets.py`, `RESULTS.md`

**Interfaces:**
- Consumes: the winning recipe from Task 2; `records --date-hints`.
- Produces: preset `sft-v3d-refs` (`{"model": "${HF_MODEL_OWNER}/lifemate-sft-v3d", "examples": 4, "check_refs": true, "date_hints": true}`).

- [ ] **Step 1:** Add `date_hints = mo.ui.checkbox(value=False, label="weekday table in the prompt (--date-hints)")` to `molab_train.py`. Pass `--date-hints` to `records`, and add `-d` to the train and eval record names when ticked, so a records file without hints is never reused.
- [ ] **Step 2:** Add the preset and `test_sft_v3d_preset_serves_with_date_hints`, asserting `date_hints` is `True` in the resolved kwargs. Run `pytest`. Expected: PASS.
- [ ] **Step 3:** Train run name `sft-v3d`, winning recipe, hints ticked. Merge.
- [ ] **Step 4:** Run holdout and dev on ASR `pho-large-ft-b`. Apply the global gate against the baselines and against the winner of Task 2. Add `RESULTS.md` rows. If it passes, the winning recipe gains "date hints".
- [ ] **Step 5:** PR `Train with date hints`.

### Task 5: Synthetic data aimed at the largest buckets, `sft-v4` (P, GPU)

Ask/act and read arguments are the largest holdout buckets, and the holdout gap comes from new scenarios as well as new apps.

**Files:**
- Modify: `src/lifemate/sft/synth.py` (writer prompt text only), `tests/test_sft_synth.py`, `configs/presets.json`, `RESULTS.md`

**Interfaces:**
- Consumes: the winning recipe; `python -m lifemate.sft.synth pack|ingest`; the notebook's *synthetic samples* field.
- Produces: `synth/synth-v1.jsonl` in the team's private HF dataset (never in git); preset `sft-v4-refs`.

- [ ] **Step 1:** Add to the writer prompt: for each brief, write one pair with the same intent, one kind `act` where a detail in the request picks exactly one record, and one kind `ask` where several records fit or a required value is missing. Add: read calls pass only the arguments the user stated. Add a test asserting that both sentences appear in a packed prompt. Run `pytest tests/test_sft_synth.py -v`. Expected: PASS.
- [ ] **Step 2: Pilot:** `pack --name p1 --subset pool --limit 40 --per-prompt 4`, answer with a free chat model, `ingest --name p1`. Read every accepted scenario by hand. Stop if fewer than 50% are accepted or if ask scenarios are below 30% of those accepted, and tell K.
- [ ] **Step 3: Bulk** with the same prompt across the pool, as far as free quotas allow by 7 Nov. Upload as `synth/synth-v1.jsonl`. Record the accepted count by kind in the PR.
- [ ] **Step 4:** Train `sft-v4`: winning recipe + the synthetic file, records pool, eval dev. Merge and add the preset.
- [ ] **Step 5:** Run holdout and dev on ASR `pho-large-ft-b`, apply the global gate against the current best, run `python -m lifemate.ceilings`, and add `RESULTS.md` rows. If it passes, build `synth-all` for Task 7 with `--subset all`.
- [ ] **Step 6:** PR `Synthetic scenarios for ask/act contrast`.

### Task 6: Base model probe (K, GPU, only if `sft-v3` gains less than +3 on holdout)

The base was fixed on 29 Oct without a measured comparison. This is a time-boxed check for 6 Nov, no longer than half a day.

**Files:**
- Modify: `RESULTS.md`

- [ ] **Step 1:** Pick up to 2 open models with native tool calling, released before 26 Oct 2026, with a compatible licence. Print `sum(p.numel() for p in model.parameters())` for each and drop any at 5B or above, vision tower included. Start with `Qwen/Qwen3.5-4B`, which the spec rejected as about 5B with thinking on by default.
- [ ] **Step 2:** Run each on holdout with no SFT: `python -m lifemate.infer --split train --subset holdout --agent react --asr gold --agent-kw model=<id> --agent-kw examples=4`. Baseline: `react-fewshot-k4` 29.69. Stop for a model if its tool calls do not parse with `agents/parsing.py`.
- [ ] **Step 3: Decide.** If one reaches at least 34.69 (+5) and 7 Nov is still free, train the winning recipe on it with `--model <id>` and gate it. Otherwise close this task with a `RESULTS.md` row and keep Qwen3-4B.

### Task 7: Final retrain and submission (K, GPU, 9 Nov)

**Files:**
- Modify: `configs/presets.json`, `tests/test_presets.py`, `RESULTS.md`

**Interfaces:**
- Consumes: the winning recipe, `synth-all` if Task 5 passed, the final ASR preset from Task 3 or `pho-large-ft-all-beam5`.
- Produces: preset `sft-final2` (same kwargs as the winning preset) and a short list of the best system for Docker Task 9: agent preset, ASR preset, both model revisions.

- [ ] **Step 1:** If no model passed its gate by 9 Nov, skip to Step 5 with the floor system.
- [ ] **Step 2:** Train run name `sft-final2`, records all, eval none, winning recipe. Expected: about 4.5 hours with masked 2, more with synthetic data. Merge.
- [ ] **Step 3:** Add the preset and extend `test_sft_presets_name_their_merged_model_and_four_examples` with `sft-final2` by giving it the model name `lifemate-sft-final2`. Run `pytest`. Expected: PASS.
- [ ] **Step 4:** Run public_test with the final ASR preset, validate with `zaic_tools --validate`, submit once, and add the row. Check `over_time_limit` is 0 and record max seconds. Keep it only if it beats the best score so far.
- [ ] **Step 5:** Write the best system list in the PR. On 11 Nov, Docker Task 9 starts from it.
