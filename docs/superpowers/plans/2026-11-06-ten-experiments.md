# Ten Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test logit distillation first, then run the other time-boxed experiments on the agent, and hand the best gated recipe to the `sft-final2` retrain on 9 Nov 2026.

**Architecture:** K asked for distillation to run first, so Task 0 takes the first training slots. Elsewhere cheap checks come first, and each experiment has a stop rule, so a dead idea costs hours and not a training slot. No-GPU simulations and single inference runs decide on 6-7 Nov. Training slots on 7-8 Nov go only to changes that train differently, one change per run, so a result can be attributed. Winners are combined once, then retrained on all train.

**Tech Stack:** Python, pytest, PyTorch + peft on molab, `lifemate.sim`, `lifemate.compare`, `lifemate.ceilings`, marimo notebooks.

**Spec:** No separate spec. The evidence is below. Two earlier plans stay in force and are referenced, not repeated: `docs/superpowers/plans/2026-11-05-unseen-examples-schema.md` (leave-app-out) and Task 5 of `docs/superpowers/plans/2026-11-04-score-target.md` (act/ask pairs).

## Evidence

| Measurement | Value |
|---|---|
| Score ceilings on the holdout ASR baseline (41.37), overlapping | args +16.85, ask/act +13.47, reads +11.95, write set +11.43 |
| Share of public samples calling a public-only app | 268 of 634 (42%) |
| sft-v2 + check_refs, gold text | dev 72.86, holdout 50.96 |
| Pool-trained teachers still on HF | Only `lifemate-sft-v2`. `sft-v3` and `sft-v3d` were deleted (14cfcfc). `sft-final` has seen holdout and `sft-r1` lost its gate, so neither can teach an experiment |
| Largest gains so far | SFT: public 30.19 -> 47.85. Guards and rules: about 0 |
| `sft-r1` (reasoning sentences) vs sft-v2, ASR | holdout -6.87 [95% CI -11.43, -2.35], dev -1.38 [-5.77, +3.50] |
| `sft-v3` (mask 2) vs sft-v2, ASR | holdout -1.96 [-5.64, +1.60]. The Task 6 probe trigger (below +3) has fired, and no probe row exists |
| Records with every gold tool in their 4 examples | pool 74.1%, holdout 13.8% |
| Parameters read from the safetensors headers | `Qwen3.5-4B` 4.66B: text 4.21B, vision 0.33B, MTP 0.12B. `gemma-4-E4B-it` 8.0B: out. `Qwen3-4B-Thinking-2507` 4.02B |
| Chat templates and vocab | Qwen3.5-4B, Qwen3.5-27B and Qwen3.6-27B: `<function=...>` tool calls, vocab 248320, thinking on unless `enable_thinking=false`. Qwen3-4B and Qwen3-30B-A3B: JSON tool calls, vocab 151936. The Thinking template always opens `<think>` |

None of these shows that an experiment below gains points. Only its gate can.

## The ten experiments

| # | Experiment | Task | Kind | Owner | Stop rule |
|---|---|---|---|---|---|
| KD | Logit distillation, run first | 0 | Report, then train | K | Its three decision points |
| 1 | Base model probe (Qwen3.5-4B, Thinking) | 1, 2 | Inference, then 1 train | K | Holdout gold without SFT below 34.69 |
| 2 | Leave-app-out few-shot in training | 3 | Train | K | Diagnostic or gate in the 5 Nov plan |
| 3 | Schema paraphrase by a local teacher | 5 | Generate, train | K | Under 80% of apps pass the check, or 2 of 10 read wrong |
| 4 | Act/ask contrast pairs (`sft-v4`) | 4 | Generate, train | K | Under 50% accepted or asks under 30% of accepted |
| 5 | Recovery records (DAgger) | 7 | Rollout, train | K | Under 100 usable records |
| 6 | Extra loss weight on decision tokens | 6 | Train | K | Gate |
| 7 | Few-shot selection by tool structure | 8 | Offline, then inference | K | Analog hit rate gains under 10 points |
| 8 | Date correction, arithmetic triage | 9 | Simulation, then inference | K | Simulation worse than 0 on holdout |
| 9 | LLM beam search | 2 | Inference | K | Gate, or max seconds above 15 on molab |
| 10 | Vote across models | 10 | Simulation | K | Holdout diff below +1 |

K owns all of them. If training slots run short, the order is KD, 1, 2, 4, 3, 6, 5.

## Calendar

| Date | No GPU | GPU |
|---|---|---|
| 7 Nov | Task 0 code | Task 0: report on `sft-v2`, large teacher probe, `sft-kd1`, `sft-t1` |
| 7-8 Nov | Tasks 1, 8 (offline part), 9, 10 | Gate `sft-kd1`. `sft-kd2` if the teacher passes. Task 2 probes |
| 8 Nov | Read gates, re-estimate the target at 12:00 | With the slots left, in order: `sft-q1`, `sft-v5a`, `sft-v4`, `sft-v5r`, `sft-v6w`, `sft-v6d`. `sft-v7` combination by evening |
| 9 Nov | | Task 12: `sft-final2` (about 4.5 hours) and one submission |
| 10 Nov | Buffer | One more submission if a gated change is left |
| 11 Nov | **Docker Task 9 starts with the best system, whatever the score** | |

Docker deadline: 16 Nov 2026 23:59 GMT+7. The floor system is `sft-final` + `pho-large-ft-all-beam5` (56.01). A training run takes about 3 hours. Two molab accounts run in parallel.

## Global Constraints

- LLM < 5B params, context <= 16,384 tokens, whole system < 6B, <= 30 s per sample on 1x A30 with ASR included, offline at inference, fixed seeds, deterministic decoding (`context/RULE.md`). ASR `pho-large-*` is 1.55B, so the LLM may be at most 4.44B.
- Only open models released before 26 Oct 2026, with a compatible licence.
- No paid or external services and no LLM APIs in the repo, dev tools or data pipeline. Teachers run locally on molab, in development only, and nothing under `agents/`, `infer.py` or `docker_entry.py` imports them.
- Never label public_test or private_test to train or tune. Do not generate data from the public catalog. Keep holdout apps out of generated data and experiment training data.
- Experiments train on the pool only. `sft-final` has seen dev and holdout, so never gate on it.
- **Gate (ASR text, `pho-large-ft-b`):** holdout diff >= +0.5 with the 95% CI above 0, dev diff >= -0.3, holdout ask column drops by at most 5 points. Baselines: `runs/20261102-104252_3c4fd00_train_sft-v2_pho-large-ft-b_holdout_check_refs` (41.37, ask 33.06) and `runs/20261102-101746_3c4fd00_train_sft-v2_pho-large-ft-b_dev_check_refs` (66.86). Gold-text reference: `runs/20261102-092303_3c4fd00_train_sft-v2_gold_holdout_check_refs` (50.96).
- Unless a task says otherwise, a trained experiment uses the sft-v2 recipe: records pool, eval dev, examples 4, masked copies 1, no perturb, ASR copy `pho-large-fp16`, 2 epochs, plus its one change.
- New behaviour is opt-in. With every new option off, prompts, records and loss stay identical to today's. Each GPU run gets a `RESULTS.md` row with its compare line.
- Do not repeat what is measured: `freetext`, `minimal_reads`, `ask_guard`, `leak_guard`, `refusal_check`, prompt vote, `date_hints`, `empty_hint`, mask 2, an intent router, an app filter.
- English in the repo. Docs and code on separate branches, one PR per task into `main`, merged after a teammate looks. Never commit data, tokens, `train_records/` or `synth/`. On Windows set `PYTHONUTF8=1`. Tests run with `PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python -m pytest`.
- **Prerequisite:** `sft/thoughts.py` (`Teacher`, used by Tasks 4 and 5) exists only on `feat/reasoning-lines`. Merge that branch into `main` before Tasks 4 and 5, or cut them from it. Everything on it is opt-in.

## Review Focus

1. **A multimodal checkpoint loaded with `AutoModelForCausalLM`:** it builds a text model with other parameter names and does not raise, so a probe or training run would use random weights. It must fail or be converted, never run silently. Test in Task 1.
2. **A `<tool_call>` written inside the thinking text:** it must not reach `env.call`, because every call is scored. Thinking cut off by the token limit counts as no output. Test in Task 1.
3. **"Mai" as a name:** "gọi cô Mai" must not be read as "tomorrow" and change a date. Test in Task 9.
4. **Loss on the model's own wrong read in a recovery record:** that would teach the mistake. The prefix turns must carry no labels. Test in Task 7.
5. **Holdout apps in teacher-written schemas:** that would invalidate the gate. The writer refuses them, and records keep their original names. Test in Task 5.

---

### Task 0: Logit distillation, run first (GPU)

The trainer computes the soft targets itself at the start of a run, so no file moves between notebooks. A teacher must share the student's vocabulary, which the trainer checks. For Qwen3-4B that rules out Qwen3.5 and Qwen3.6 teachers.

**Risk to read first:** `sft-v2` trained on these same records (final train loss 0.029, eval loss 0.041), so its outputs on them may be close to the hard labels. If so, distilling from it adds nothing. The report in Step 3 measures this before a training slot is spent. A larger teacher fine-tuned on the same records can overfit the same way.

**Files:**
- Create: `src/lifemate/sft/kd.py`
- Modify: `src/lifemate/sft/train.py`, `configs/presets.json`, `tests/test_presets.py`, `RESULTS.md`
- Test: `tests/test_sft_kd.py`, `tests/test_sft_train.py`

**Interfaces:**
- Produces: `soft_targets(logits, temp, top_k)`, `merge_targets(targets, top_k)`, `kd_sum(student_logits, ids, probs, temp)`, `target_report(targets, gold)` in `kd.py`. `train --kd-teachers A,B --kd-alpha 0.5 --kd-temp 2 --kd-top-k 20 --kd-report-only --lora-targets ...`. Loss: `(1 - alpha) * CE + alpha * temp^2 * KL`, on the teacher's top-k plus one tail bin. Logged `loss` and `eval_loss` stay plain cross-entropy. Presets `sft-kd1-refs`, `sft-t1-refs`, `sft-kd2-refs`.

- [ ] **Step 1:** Implement and test the above on `exp/logit-kd`. Run `pytest`. Expected: PASS. Commit and push the branch for molab.
- [ ] **Step 2: Large teacher probe, no code.** On `notebooks/molab_run.py`, branch `exp/logit-kd`, split train, subset holdout, ASR preset `gold`, agent preset `react-fewshot-k4`, extra infer args `--agent-kw model=Qwen/Qwen3-30B-A3B-Instruct-2507`. The notebook evaluates the run. Baselines: untrained 4B 29.69, sft-v2 50.96. If it does not fit in GPU memory, use `Qwen/Qwen3-14B`.
- [ ] **Step 3: Report on `sft-v2`, about 30 minutes.** On `notebooks/molab_train.py`, sft-v2 recipe, run name `sft-kd1`, extra train args `--kd-teachers TripleTrouble/lifemate-sft-v2 --kd-report-only`. Read `kd_report.json`. **Decision 1:** with `share_top1_below_0.9` under 1%, or `share_top1_not_gold` under 0.1%, the teacher adds almost nothing: skip Step 4. These thresholds are judgment calls.
- [ ] **Step 4: Train `sft-kd1`:** same settings without `--kd-report-only`. The targets are reused from `kd_targets.pt`. Merge, add the preset, run holdout and dev with ASR `pho-large-ft-b`, apply the gate.
- [ ] **Step 5: Train the teacher `sft-t1`,** on the second account, at the same time as Step 4: sft-v2 recipe, extra train args `--model <teacher from Step 2> --lora-targets q_proj,k_proj,v_proj,o_proj`. Attention layers only, because a mixture-of-experts model would otherwise get an adapter on every expert. Time box: 8 hours, ended by 8 Nov 12:00. Merge, run holdout on gold text. **Decision 2:** continue only at 56.0 or above (sft-v2 50.96 + 5). Otherwise close the large-teacher path with a row.
- [ ] **Step 6: Train `sft-kd2`:** sft-v2 recipe, `--kd-teachers TripleTrouble/lifemate-sft-t1`. Merge, gate. **Decision 3:** the best passing model joins Task 12. If neither passes, log the rows and stop distillation.
- [ ] **Step 7:** Add the `RESULTS.md` rows, with `kd_report.json` values in the notes, and open a PR.

### Task 1: LLM options and the text-only Qwen3.5 checkpoint

**Files:**
- Create: `src/lifemate/sft/text_only.py`, `configs/chat_templates/qwen_json_nothink.jinja`
- Modify: `src/lifemate/agents/llm.py`, `src/lifemate/agents/react.py`, `configs/presets.json`
- Test: `tests/test_llm.py`, `tests/test_text_only.py`, `tests/test_presets.py`

**Interfaces:**
- Produces:
  - `finish_text(text: str, stop_strings: set[str], open_think: bool) -> str` in `llm.py`: removes the stop strings; with `</think>` present keeps only what follows the last one; with `open_think` true and no `</think>` returns `""`. `open_think` is true when the rendered prompt ends with `<think>\n`.
  - `HFChatLLM(..., num_beams: int = 1)`, and `ReActAgent(..., max_new_tokens: int = 384, num_beams: int = 1)`, forwarded when the agent builds the LLM and recorded in `self.info`.
  - `python -m lifemate.sft.text_only --model <id> --revision <sha> --chat-template <file> --out <dir> --hf-repo <repo>`: loads with `AutoModelForImageTextToText`, copies `model.language_model.*` and `lm_head` into the matching causal-LM class, drops vision and MTP, stores the template in the tokenizer, prints the parameter count. The rest of the pipeline then uses `AutoModelForCausalLM` unchanged.
  - `qwen_json_nothink.jinja`: the Qwen3-4B-Instruct template, with `<think>\n\n</think>\n\n` opening the generation prompt and every assistant turn after the last user message. Tool calls stay `<tool_call>` JSON, so `parsing.py`, `render_examples`, `replay.py` and `mask.py` do not change.
  - Presets: `sft-v2-refs-beam3` (`sft-v2-refs` + `"num_beams": 3`).

- [ ] **Step 1: Write the failing tests:**

```python
# tests/test_llm.py
def test_finish_text_drops_a_call_written_inside_thinking():
    call = '<tool_call>{"name":"x","arguments":{}}</tool_call>'
    assert finish_text(f"draft {call}</think>\n\n{call}<|im_end|>", {"<|im_end|>"}, True) == call

def test_finish_text_with_cut_off_thinking_is_empty():
    assert finish_text('reasoning <tool_call>{"name":"x"', set(), True) == ""

def test_finish_text_without_thinking_is_unchanged():
    assert finish_text("Đã xong.", set(), False) == "Đã xong."
```

  In `tests/test_text_only.py`, with `pytest.importorskip` on the Qwen3.5 classes and a tiny random config: `test_text_only_keeps_logits` (same text input gives equal logits before and after conversion, and loading the output with `AutoModelForCausalLM` reports no missing keys) and `test_text_only_has_no_vision_params`. In `tests/test_llm.py`, using the cached Qwen3-4B tokenizer with the new template: `test_json_template_renders_prefixes` (`encode_record` raises nothing on a two-call record) and `test_json_template_prompt_has_empty_think`. In `tests/test_presets.py`: `test_beam_preset_passes_num_beams`.
- [ ] **Step 2:** Run `pytest tests/test_llm.py tests/test_text_only.py tests/test_presets.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement as above. `generate` passes `num_beams` to `model.generate` and uses `finish_text`.
- [ ] **Step 4:** Run `pytest`. Expected: PASS.
- [ ] **Step 5:** Commit `Add LLM beam and thinking options, and the text-only Qwen3.5 converter` on `exp/base-model-probe`, open a PR.

### Task 2: Probes and beam search, experiments 1 and 9 (GPU)

**Files:**
- Modify: `RESULTS.md`

**Interfaces:**
- Consumes: Task 1.
- Produces: the base model decision, the KD trigger, and a `RESULTS.md` row per run.

- [ ] **Step 1:** Using the notebook's command field, convert with `lifemate.sft.text_only` and pinned revisions: `Qwen/Qwen3.5-4B` -> `TripleTrouble/qwen35-4b-text`, private. Expected: about 4.21B parameters, under the 4.44B limit. Otherwise stop this candidate. A Qwen3.5 student would need a teacher with its own vocabulary, such as Qwen3.6-27B (vocab 248320), converted the same way.
- [ ] **Step 2:** Run each on holdout with gold text, no SFT, then evaluate. Base command: `python -m lifemate.infer --split train --subset holdout --agent react --asr gold --agent-kw examples=4 --agent-kw model=<id>`. Baseline: `react-fewshot-k4` 29.69.

| Candidate | `<id>` and extra options |
|---|---|
| Student A | `TripleTrouble/qwen35-4b-text` |
| Student B | `Qwen/Qwen3-4B-Thinking-2507`, `--agent-kw max_new_tokens=1024` |

  For Student B, also record the counts of `stop` values in `trace.jsonl`, and max seconds. A share of `max_steps`, `time` or empty outputs above 10% means repetition or cut-off thinking; say so in the row.
- [ ] **Step 3: Decide the student.**
  - Below 34.69: close the candidate with its row. Qwen3-4B-Instruct stays.
  - Student A at 34.69 or above: train `sft-q1` with the sft-v2 recipe and `--model TripleTrouble/qwen35-4b-text`. Print the trainable LoRA parameter count first, and check that `LORA_TARGETS` reached the attention layers of every block; if some blocks have other module names, extend the targets by a train-time option. Merge, gate against sft-v2. If it passes, later runs use this base.
  - Student B at 34.69 or above: write a short follow-up plan for reasoning from its own rollouts, kept only when they match gold. It needs a new records path and may not fit the calendar.
- [ ] **Step 4: Beam.** Run preset `sft-v2-refs-beam3` with ASR `pho-large-ft-b` on holdout, then dev if holdout is not a loss. Apply the gate. Stop if max seconds exceeds 15, since the A30 is slower than molab.
- [ ] **Step 5:** Add the rows, PR `Log the base model probe and beam runs`.

### Task 3: Leave-app-out few-shot, experiment 2 (GPU)

**Files:** as in `docs/superpowers/plans/2026-11-05-unseen-examples-schema.md`.

**Interfaces:**
- Consumes: Tasks 1, 2, 4, 5 and 6 of that plan, for `sft-v5a` only.
- Produces: preset `sft-v5a-refs`.

- [ ] **Step 1:** Implement its Tasks 1, 2 and 4, following that plan's tests. I found no branch or run for them on 6 Nov, so confirm with the team first.
- [ ] **Step 2:** Run its Task 5 diagnostic, and stop if that plan's rule says so.
- [ ] **Step 3:** Train and gate `sft-v5a` per its Task 6. `sft-v5b` (schema notes) stays in that plan and only runs if a slot is free.

### Task 4: Act/ask pairs written by a local teacher, experiment 4 (GPU)

**Files:**
- Modify: `src/lifemate/sft/synth.py`, `tests/test_sft_synth.py`, `configs/presets.json`, `RESULTS.md`

**Interfaces:**
- Consumes: `sft.thoughts.Teacher(model, revision, max_new_tokens)`, Task 5 of `docs/superpowers/plans/2026-11-04-score-target.md`.
- Produces: `python -m lifemate.sft.synth answer --name <n> --teacher <id> --revision <sha>`, which writes `scenarios/prompt_NN.json` for each packed prompt without an answer. `synth.main(argv, llm=None)` takes a fake LLM in tests. Preset `sft-v4-refs`.

- [ ] **Step 1:** Check `origin/feat/synth` and the private HF dataset for work already done on Task 5. Skip what is done.
- [ ] **Step 2: Write the failing tests:** `test_answer_writes_one_file_per_prompt` (fake LLM, 3 prompts -> 3 files holding its text) and `test_answer_skips_answered_prompts` (a second run makes 0 LLM calls). Run them. Expected: FAIL.
- [ ] **Step 3:** Implement the mode, with `max_new_tokens=2048`. The teacher is imported only inside this mode.
- [ ] **Step 4:** Run `pytest tests/test_sft_synth.py -v`. Expected: PASS. Commit on `feat/synth-local-teacher`, open a PR.
- [ ] **Step 5:** Follow the Task 5 steps of that plan, using `answer` in place of pasting into a chat model: writer prompt, pilot of 40 read by hand with its stop rule, bulk, train `sft-v4`, gate, rows.

### Task 5: Schema paraphrase, experiment 3 (GPU)

Masking swaps names for random words and keeps the description text. This copy gives the model other wording for the same tools. Rewriting requests and tool results into a new domain is out of scope: it breaks the agreement between request, results and arguments, and could not be verified in time.

**Files:**
- Create: `src/lifemate/sft/remix.py`
- Modify: `src/lifemate/sft/mask.py`, `src/lifemate/sft/records.py`, `notebooks/molab_train.py`, `configs/presets.json`, `RESULTS.md`
- Test: `tests/test_sft_remix.py`, `tests/test_molab_train.py`

**Interfaces:**
- Consumes: `sft.thoughts.Teacher`, `load_splits()["holdout_apps"]`, the train catalog.
- Produces:
  - File `train_records/remix/<name>.json`: `{"params": {old: new}, "apps": {app: {"name": new, "tools": {suffix: {"name": new, "description": text}}}}}`. `params` is one catalog-wide synonym table, matching the single parameter map of `mask.py`.
  - `check_app(app_spec: dict, answer: dict) -> str | None`: every tool present, identifiers valid, new and unique, descriptions non-empty. Returns the reason, or None.
  - `remix_record(record: dict, table: dict, rng: random.Random) -> dict`: copy with `source="remixed"`, tools shuffled, built on `mask.py`'s `_Renamer`, `_rename_call` and `_rename_example_calls`. Apps missing from the table keep their names and text. The `Provide one of:` suffix is kept after the new description.
  - `python -m lifemate.sft.remix --name <n> --revision <sha> --hf-repo <dataset>` (molab). `records --remix FILE` adds one copy after the masked ones. Record names get `-rx`. Preset `sft-v5r-refs`.

- [ ] **Step 1: Write the failing tests:**
  - `test_remix_refuses_holdout_apps`: the writer's app list, and `records --remix` given a table naming `news`, both exclude or reject it.
  - `test_remix_renames_everywhere`: no old tool or parameter name is left in the schemas, calls or example calls of a copy, for apps in the table.
  - `test_remix_keeps_absent_apps`: an app missing from the table stays byte-identical in the tool list.
  - `test_remix_replay_maps_back`: renaming a copy's calls back gives the original calls.
  - `test_check_app_rejects_missing_tool` and `test_records_without_remix_unchanged`.
- [ ] **Step 2:** Run `pytest tests/test_sft_remix.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement. Teacher `Qwen/Qwen3.5-27B` at the pinned revision used for `thoughts`, `max_new_tokens=2048`, one prompt per app; a failed app gets one retry and is then left out.
- [ ] **Step 4:** Add the notebook field in the pattern of `thoughts`. Run `pytest`. Expected: PASS. Commit on `exp/schema-paraphrase`, open a PR.
- [ ] **Step 5: Pilot.** Write the table, read 10 apps by hand. Stop and report if under 80% of apps pass `check_app` or if 2 of 10 change a tool's meaning.
- [ ] **Step 6:** Train `sft-v5r`: the sft-v2 recipe plus `--remix`. Merge, gate, add rows, PR.

### Task 6: Loss weight on decision tokens, experiment 6 (GPU)

Mean cross-entropy over tokens gives a long free-text value more weight than the single tokens that choose ask or act, or the tool.

**Files:**
- Modify: `src/lifemate/sft/tokenize.py`, `src/lifemate/sft/train.py`, `configs/presets.json`, `RESULTS.md`
- Test: `tests/test_sft_tokenize.py`, `tests/test_sft_train.py`

**Interfaces:**
- Produces: `decision_positions(turn_ids: list[int], tokenizer) -> set[int]`, the turn's first token, the tokens of the tool name value, and the tokens of top-level argument keys. `encode_record(record, tokenizer, max_len=MAX_LEN, decision_weight=1.0)` adds `"weights"` only when the weight differs from 1. `train --decision-weight` (default 1.0), in `RUN_KEYS`. The training loss is the weighted sum over the weight sum. Eval loss stays unweighted so curves compare. Preset `sft-v6w-refs`.

- [ ] **Step 1: Write the failing tests:**
  - `test_decision_positions_cover_name_and_keys`: for call `{"name": "a.b", "arguments": {"k": "dài"}}`, decoding the positions gives a text containing `a.b` and `k`, not `dài`.
  - `test_first_token_of_plain_answer_is_a_decision`: a plain final text marks only its first token.
  - `test_no_weights_by_default`: `"weights" not in encode_record(record, tok)`.
  - `test_weight_one_keeps_the_loss`: on the tiny trainer model, loss at weight 1.0 equals today's.
- [ ] **Step 2:** Run `pytest tests/test_sft_tokenize.py tests/test_sft_train.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement as above. Run `pytest`. Expected: PASS. Commit on `exp/decision-weight`, open a PR.
- [ ] **Step 4:** Train `sft-v6w`: the sft-v2 recipe, extra train args `--decision-weight 3`. One value only. Merge, gate, add rows, PR.

### Task 7: Recovery records, experiment 5 (GPU)

Gold replays only show on-path states. After one wrong read, the model is somewhere it never trained on.

**Files:**
- Create: `src/lifemate/sft/recovery.py`
- Modify: `src/lifemate/sft/replay.py`, `src/lifemate/sft/tokenize.py`, `configs/presets.json`, `RESULTS.md`
- Test: `tests/test_sft_recovery.py`, `tests/test_sft_tokenize.py`, `tests/test_sft_replay.py`

**Interfaces:**
- Consumes: a scored pool run's `trace.jsonl`, `--extra-samples` and the notebook's synthetic samples field, `synth.env_id`.
- Produces:
  - `recovery_sample(sample: dict, trace: dict, catalog: dict, n: int) -> dict | None`: the sample with `env_id`, `id="<id>~r<n>"` and `prefix_calls`, the longest leading run of read calls in the trace when at least one differs from gold at its position. None if there is no such prefix.
  - `replay(...)` reads `sample.get("prefix_calls")`: the LLM emits the prefix, then the gold calls the prefix did not already make. The expected call list is that sequence, and the final state hash must equal the gold replay's. The record gets `"context_turns": <prefix length>`.
  - `encode_record` leaves the first `context_turns` agent turns unlabelled.
  - `python -m lifemate.sft.recovery --run runs/<run> --name <n> --max 200` -> `synth/recovery-<n>.jsonl`. Preset `sft-v6d-refs`.

- [ ] **Step 1: Write the failing tests:**
  - `test_prefix_turns_have_no_labels`: all -100 over the prefix turn, labels on the turns after it.
  - `test_recovery_stops_prefix_at_a_write`: a write in the trace ends the prefix.
  - `test_recovery_none_when_on_path`: a trace equal to gold -> None.
  - `test_replay_rejects_a_changed_state`: a prefix that changes the state hash -> None.
  - `test_replay_without_prefix_unchanged`.
- [ ] **Step 2:** Run `pytest tests/test_sft_recovery.py tests/test_sft_tokenize.py tests/test_sft_replay.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement. Run `pytest`. Expected: PASS. Commit on `exp/recovery-records`, open a PR.
- [ ] **Step 4:** Run `react-fewshot-k4`, gold text, `--subset pool` (about 40 minutes). This is the untrained model: sft-v2 has seen the pool and makes too few mistakes on it. Write the file. Stop and report if under 100 records are usable.
- [ ] **Step 5:** Train `sft-v6d`: the sft-v2 recipe plus the file as synthetic samples. Merge, gate, add rows, PR.

### Task 8: Few-shot selection by tool structure, experiment 7

For an unseen app, BM25 on the request finds examples with similar words. It does not find a tool of the same shape on another app, such as `end_ride` for `end_session`.

**Files:**
- Modify: `src/lifemate/agents/fewshot.py`, `src/lifemate/agents/react.py`, `configs/presets.json`, `RESULTS.md`
- Create: `src/lifemate/fewshot_report.py`
- Test: `tests/test_fewshot.py`

**Interfaces:**
- Produces:
  - `tool_tokens(spec: dict) -> set[str]`: words of the name suffix and parameter names, split on `_`, plus `kind`.
  - `ExampleBank.select(..., tools: list[dict] | None = None, analog_weight: float = 0.0)`. Score is BM25 + `analog_weight` x the mean, over the example's gold tools, of the best Jaccard of `tool_tokens` against the 3 installed tools of apps absent from the bank that share the most words with the request (ties by name). No such tool -> the score is plain BM25, so seen-app samples keep their examples.
  - `ReActAgent(..., analog_weight: float = 0.0)`, recorded in `self.info`.
  - `python -m lifemate.fewshot_report --subset holdout --analog-weight W`: the share of samples whose examples hold a tool analogous to a gold tool (same suffix, or Jaccard >= 0.5). It reads labels for this report only.

- [ ] **Step 1: Write the failing tests:**
  - `test_analog_weight_zero_is_bm25`: same picks as today.
  - `test_analog_prefers_same_shape_tool`: an unseen `parking.end_session` ranks an `end_*` example above a word-matched one.
  - `test_seen_app_sample_is_unchanged`: all installed apps in the bank -> same picks at any weight.
  - `test_analog_is_deterministic`: same order under another `PYTHONHASHSEED`.
- [ ] **Step 2:** Run `pytest tests/test_fewshot.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement. The bank needs the train catalog for example tools. Run `pytest`. Expected: PASS. Commit on `exp/fewshot-structure`, open a PR.
- [ ] **Step 4:** Run the report at weights 0, 1, 2, 4. Stop if the best weight gains under 10 points of hit rate over weight 0.
- [ ] **Step 5:** Run `sft-v2-refs` with the best weight and ASR `pho-large-ft-b` on holdout, then dev. Apply the gate. Note: sft-v2 trained on BM25 examples, and earlier changes to the examples at inference lost points, so a miss here does not rule out training with this selection.

### Task 9: Date correction and arithmetic triage, experiment 8 (no GPU first)

`docs/hybrid_rules.md` asks whether correcting a model's date argument is acceptable. This task assumes yes, because it changes an argument of a call the model sent and sends no new call. K confirms before Step 5.

**Files:**
- Create: `src/lifemate/agents/date_fix.py`
- Modify: `src/lifemate/sim.py`, `src/lifemate/agents/react.py`, `configs/presets.json`, `RESULTS.md`
- Test: `tests/test_date_fix.py`, `tests/test_sim.py`

**Interfaces:**
- Consumes: `agents.dates.WEEKDAYS`, `sim.REWRITES`.
- Produces:
  - `resolve_date(request: str, current_time: str) -> str | None`: the ISO date when the request holds exactly one date phrase of: hôm nay, ngày mai, sáng/trưa/chiều/tối mai, ngày mốt, thứ X (này), thứ X tuần sau, chủ nhật (này/tuần sau). None otherwise. No schema or app name is used.
  - `fix_call_dates(call: dict, spec: dict | None, request: str, current_time: str) -> dict`: replaces the date part of the single date-valued string argument and keeps any time part. A call with two date arguments is left as is, and so is an argument whose description names a past range.
  - Rewrite `date_fix` in `sim.REWRITES`. `ReActAgent(..., date_fix: bool = False)`, applied before `env.validate`. Preset `sft-v2-refs-datefix`.

- [ ] **Step 1: Write the failing tests:**

```python
@pytest.mark.parametrize("now, text, date", [
    ("2026-10-01T09:00:00", "đặt lịch thứ ba tuần sau", "2026-10-06"),
    ("2026-10-05T09:00:00", "thứ sáu tuần sau họp", "2026-10-16"),
    ("2026-12-04T09:00:00", "thứ sáu tuần sau họp", "2026-12-11"),
    ("2026-09-24T09:00:00", "vé máy bay thứ bảy tuần sau", "2026-10-03"),
])
def test_resolve_date_next_week(now, text, date):
    assert resolve_date(text, now) == date

def test_mai_as_a_name_is_not_a_date():
    assert resolve_date("gọi cho cô Mai", "2026-10-01T09:00:00") is None

def test_two_phrases_are_left_alone():
    assert resolve_date("hôm nay xem giá, sáng mai nhắc tôi", "2026-10-01T09:00:00") is None
```

  Also `test_fix_keeps_time_part` and `test_fix_skips_two_date_arguments`.
- [ ] **Step 2:** Run `pytest tests/test_date_fix.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement. Run `pytest`. Expected: PASS. Commit on `exp/date-fix`, open a PR.
- [ ] **Step 4: Simulate:** `python -m lifemate.sim date_fix <run>` on dev, tune phrases on dev only, then holdout, both ASR baselines. Add the row to the Offline simulations table. Stop if holdout is below 0. The simulation keeps later calls as recorded, so it understates a search that now finds results.
- [ ] **Step 5:** One GPU run of the preset on holdout and dev. Apply the gate.
- [ ] **Step 6: Arithmetic triage, no code.** From `ceilings/num` on both ASR baselines, read every fired sample and classify it: computed from tool values or the request, or plain extraction or ASR. If at least half are computed, write a follow-up task. Otherwise close it with a line in `RESULTS.md`.

### Task 10: Vote across models, experiment 10 (no GPU)

Each sample gets exactly one ToolEnv, and every call on it is scored, so two full runs cannot be compared at inference. The per-sample best of `sft-v2` and `sft-r1` (48.69 on holdout) is therefore not reachable. Any vote must work on proposals before they are sent, as the prompt vote does. This simulation only estimates the room for one.

**Files:**
- Modify: `src/lifemate/sim.py`, `RESULTS.md`
- Test: `tests/test_sim.py`

**Interfaces:**
- Produces: rewrite `run_vote` in `sim.REWRITES`, with `--kw others='["runs/a","runs/b"]'`. It keys each run's write and ask calls per sample. The majority's call list wins, and with no majority the first run stands.

- [ ] **Step 1: Write the failing tests:** `test_run_vote_takes_the_majority`, `test_run_vote_keeps_first_without_majority`. Run. Expected: FAIL.
- [ ] **Step 2:** Implement. Run `pytest tests/test_sim.py -v`. Expected: PASS. Commit on `exp/run-vote`, open a PR.
- [ ] **Step 3:** Simulate on ASR holdout and dev: sft-v2 baseline first, others the sft-v3 and sft-v3d runs (commits 580cad8, 7bbb1d8), which trained on the pool. Add the row. `sft-final` and `sft-r1` stay out: the first has seen these samples, the second lost.
- [ ] **Step 4: Decide.** Below +1 on holdout, or a CI touching 0: close it. Otherwise measure the parameters of three LoRA adapters against the 6B limit (0.43B left with Qwen3-4B) and write a follow-up task for a vote among adapters. The vote runs 3 models per step, so check time too.

### Task 12: Combine and hand off

**Files:**
- Modify: `configs/presets.json`, `tests/test_presets.py`, `RESULTS.md`

**Interfaces:**
- Consumes: every change that passed its gate. Follows Task 7 of `docs/superpowers/plans/2026-11-04-score-target.md`.
- Produces: preset `sft-final2` and the best system list for Docker Task 9.

- [ ] **Step 1:** List the experiments that passed, with their diffs, in the PR.
- [ ] **Step 2:** With two or more training changes passing: train `sft-v7` with all of them on the pool, and gate it against the best single run. With one: skip to Step 3. With none: the floor system goes to Docker.
- [ ] **Step 3:** 8 Nov 12:00: re-estimate the target from the best holdout score, and write it down.
- [ ] **Step 4:** 9 Nov: train `sft-final2` on all train with the winning recipe, plus inference options that passed. Run public_test, validate with `zaic_tools --validate`, submit once, add the row. Check `over_time_limit` is 0. Keep it only if it beats 56.01.
- [ ] **Step 5:** Write the agent preset, ASR preset, both revisions and the parameter total (< 6B, LLM < 5B) in the PR. Docker Task 9 starts from it on 11 Nov.
