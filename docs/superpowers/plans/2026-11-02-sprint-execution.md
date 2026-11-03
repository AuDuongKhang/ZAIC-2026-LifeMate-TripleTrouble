# Final Sprint Execution Plan (2–16 Nov)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** turn the sprint plan (`2026-11-02-final-sprint.md`) into executable tasks: ship `sft-final`, then try the three changes most likely to raise the **private-test** score (schema perturbation, synthetic scenarios, a narrow ask guard), and ship one frozen, offline, deterministic Docker image by 14 Nov.

**Architecture:** each change is opt-in (a records flag, an agent kwarg or a preset) and enters the shipped system only after a paired `compare` on holdout and dev. Synthetic data is produced by a dev-time teacher (Claude API), verified by replay in a fresh train `ToolEnv` and stored as extra train samples that `lifemate.sft.records` renders like real ones. The Docker image runs `lifemate.infer` through a small entry module with the shipped presets, pinned model revisions and no network.

**Tech Stack:** Python 3.11, transformers, peft (training), `anthropic` SDK (dev only, never imported on the inference path), marimo/molab, Docker with an NVIDIA PyTorch base image.

**Spec:** `docs/superpowers/specs/2026-10-27-agent-roadmap-design.md` (goals, limits, Docker constraints) and the umbrella plan `docs/superpowers/plans/2026-11-02-final-sprint.md` (owners, rules, questions Q1–Q5). This plan details K1, K3, K5b and K6 of that plan, adds schema perturbation and the Docker package, and replaces none of its P tasks.

## Evidence this plan is built on (2 Nov)

| Run | dev | holdout | holdout ask | public |
|---|---|---|---|---|
| sft-v1 | 63.74 | 42.30 | 13.24 | 47.85 (ft-a) |
| sft-v2 (pool, mask 1, ASR copy) | 72.86 | 50.36 | 32.34 | 50.53 (ft-b) |
| sft-v2 + `check_refs` | not run | 50.96 (+0.60 [+0.03, +1.47], 4 better / 0 worse) | 32.34 | – |

- **Holdout predicts public; dev does not** (50.36 vs 50.53, dev 72.86). Holdout samples that use only pool apps also score 52.6, so the gap is new *scenarios*, not only new apps. Holdout is the gate.
- **Private test will bring unseen apps.** 410 of 634 public samples (65%) install at least one app that train does not have (`ecommerce, event_tickets, grocery, habit_tracker, pantry, parking, public_transit, recipes, stocks, train_bus_tickets, translator`). History share and length match train (66%, ~4 turns).
- **Where sft-v2 loses holdout points** (147 of 297): same write tools with wrong arguments 11.8 pts, extra/different optional or free-text arguments with a correct state 10.9, wrong or extra write tools 8.8, acting where gold asks 6.4, wrong read tool 5.6, asking where gold acts 3.5. Wrong dates: 7 samples (~2 pts).
- **A naive ask rule is harmful:** "the id was picked from a list of ≥ 2 candidates → ask" touches 96 holdout act samples worth 51.7 points to rescue at most 9 ask samples.
- **No extra model is justified:** no ASR hallucination in the 634 public transcripts, context never overflows, ~0.43B parameters of headroom stay unused.

## Global Constraints

- LLM < 5B params, context ≤ 16,384 tokens, whole system < 6B; ≤ 30 s per sample on 1× A30 with ASR included; offline at inference; fixed seeds; greedy decoding (`context/RULE.md`).
- Never read `<split>_environment/` from solution code; exactly one `ToolEnv` per sample at inference. Dev-time tools (synthetic data, simulators) reach app data only through `ToolEnv` too.
- **Never label or tune on public_test / private_test**, and put nothing from public_test (requests, schemas, app names) into a teacher prompt. Public_test is only for reading the leaderboard score of a submission.
- **Leakage:** experiments train on pool samples only; synthetic samples for experiments are built only on pool environments. Only the final retrain uses dev and holdout.
- **Gate for every change:** paired `python -m lifemate.compare` on holdout shows a positive difference whose 95% interval excludes 0, the dev difference's interval is not entirely below 0, and the holdout ask column drops by no more than 5 points. Same agent kwargs on both sides of a compare.
- New behaviour is opt-in (`--agent-kw`, a records flag, a preset) until it passes the gate. Defaults stay as they are.
- `requirements.txt` gains nothing that pulls torch (`tests/test_requirements.py`), and nothing dev-only: `anthropic` is imported lazily inside `lifemate.sft.teacher` and listed in `requirements-dev.txt`.
- English in the repo; chat with the team in Vietnamese. Branch per task from an up-to-date `main` (`feat/…`, `exp/…`, `docs/…`), PR with `pytest` output and the `compare` lines, review by the other owner. Never commit data, tokens, synthetic samples or `train_records/`.
- Every behaviour change gets a `RESULTS.md` row with the `compare` line in its notes. On Windows set `PYTHONUTF8=1`.

## Review Focus

1. **Synthetic leakage:** a synthetic sample built on a dev or holdout environment inflates the gate. `records` must refuse extra samples whose `env_id` is outside the pool unless `--subset all` (test in Task 5).
2. **Perturbation that breaks the training text:** dropping an optional parameter that a history call, the gold calls, an ask's `params` or a few-shot example line uses makes the record schema-invalid (test in Task 2).
3. **Docker tries the network:** with `--network none` and no token, any `from_pretrained` without the offline flags hangs or fails. The entry module sets the offline variables before importing transformers (test in Task 9).
4. **Docker data layout:** `/data` is the split folder itself (no `<split>/` parent), and the few-shot bank still needs `train/train.jsonl` and the train catalog inside the image (test in Task 9).
5. **The ask guard firing on act samples** whose request picks the candidate by a field other than its name (station, time, price). The guard scores every string field of the item, and a test pins "request names the station → no ask" (Task 7).

---

## Sprint schedule

Owners: **K** = agent/LLM + shipping (Khang), **P** = ASR + synthetic data (partner). **Every task has exactly one owner.** Changed on 2 Nov: P had already answered Q1/Q2 and started generating synthetic data and writing its code, so Tasks 3–6 moved to P and Tasks 9–10 (Docker) to K. P6 (ASR timing) is still the same A30 run as Task 10, which K runs with P's ASR settings.

## Progress (2 Nov)

| Task | Status |
|---|---|
| 1 | waiting for the `sft-final` retrain |
| 2 | code in PR #18 (perturbation, one-of parameters kept after review); `sft-v3` run and gate pending |
| 3–6 | P, in progress |
| 7 | done, PR #19: the guard loses at every setting (dev −14.93 at best, holdout −6.64), not shipped |
| 8 | dropped (Task 7's decision rule) |
| 9 | K, started on `feat/docker` |
| 10 | K, after Task 9 and the final models |

| Dates | K | P | Submission |
|---|---|---|---|
| 2–3 Nov | Task 1: final retrain, dev `check_refs`, presets PR; Task 9 skeleton (`empty` agent in Docker) | P3 with `sft-v2` (below); Tasks 3–5 code | `sft-final` + `pho-large-ft-b` |
| 3–5 Nov | Task 2 `sft-v3` run and gate; Task 9 with real models | Task 6 pilot and bulk; P4 only if P3 says ASR costs > 5 pts | – |
| 4–8 Nov | Task 9 done (PR merged); review P's synthetic PR | Task 6 `sft-v4` run and gate; P5 final ASR retrain | best gated system |
| 9–11 Nov | Final retrain with the winning recipe (Task 1 Steps 1–2 again, run name `sft-final2`); Task 10 Steps 1–3 on a rented A30 (covers P6) | hand K the final ASR preset and revision | final candidate |
| 12–14 Nov | Task 10 Steps 4–6: freeze, checksum, upload | review K's Docker PR; fixes only | freeze 14 Nov 23:59 |
| 15–16 Nov | submit the Docker form | buffer | deadline 16 Nov 23:59 |

**P's work in this sprint:** P3, P4, P5 of the umbrella plan, plus Tasks 3–6 of this plan. P3 now runs with `sft-v2` instead of `sft-v1`: `notebooks/molab_run.py`, split `train`, extra `--subset holdout`, agent preset `sft-v2`, ASR preset `pho-large-ft-b`, then `python -m lifemate.compare runs/20261102-080138_263ebdc_train_sft-v2_gold_holdout runs/<new run>`. Loss < 5 points → skip P4; > 10 points → P4 first, and tell K to add the `contacts.search` name fallback (K4 of the umbrella plan). P's synthetic code may differ from Tasks 3–5 in shape; K's review checks it against Review Focus 1 and the Global Constraints (pool-only environments, replay verification, nothing from public_test, no data committed, rendering through `records`, the same gate).

**K's work in this sprint:** Tasks 1, 2, 7 (done), 9 and 10 of this plan.

Anything not gated by 11 Nov stays out of the shipped system.

## Questions to settle at the next sync

- **Q1/Q2** (umbrella plan): answered by P, who is running Tasks 3–6. Record the answers (teacher model, cost cap, whether train text is sent) in P's PR.
- **Q4:** settled: K owns Docker (Tasks 9–10).
- **Q6 (new):** the few-shot bank reads `train/train.jsonl` at inference, so the image carries train text, and the organisers ask for a public Google Drive link. The guidance asks for "training code, inference code, and data", which suggests this is expected; confirm with the organisers if anyone doubts it.
- **Q7 (new):** budget for ~3 hours of a rented A30 (or the closest 24 GB Ampere card) for Task 10 timing. molab's GPU is several times faster, so molab timings cannot prove the 30 s limit.

---

### Task 1: Ship `sft-final` and adopt `check_refs` (K)

**Files:**
- Modify: `configs/presets.json`, `RESULTS.md`
- Test: `tests/test_presets.py`

**Interfaces:**
- Produces: agent presets `sft-v2-refs` = `{"agent": "react", "agent_kw": {"model": "${HF_MODEL_OWNER}/lifemate-sft-v2", "examples": 4, "check_refs": true}}`; `sft-final` gains `"check_refs": true` only if Step 3 passes.

- [ ] **Step 1: Final retrain on molab** (`notebooks/molab_train.py`, branch `main`): run name `sft-final`, model repo `TripleTrouble/lifemate-sft-final`, *Train records* = all train, *Eval loss records* = none, few-shot 4, masked copies 1, ASR-noisy copy `pho-large-fp16`, ASR model `vinai/PhoWhisper-large`, extra args empty, max-steps 0. Expected: about 1.6× the 2 h 15 min of `sft-v2`; the log ends at epoch 2.0 with no eval loss. If the session dies, tick *resume* and press Start with identical fields.
- [ ] **Step 2: Merge and record the revision.** Press *Merge and upload*. Read the commit hash with `python -c "from huggingface_hub import HfApi; print(HfApi().model_info('TripleTrouble/lifemate-sft-final').sha)"` (HF_TOKEN set). Expected: a 40-character hash; write it down for Task 10.
- [ ] **Step 3: Dev gate for `check_refs`** (`notebooks/molab_run.py`): split `train`, extra `--subset dev --agent-kw check_refs=true`, agent preset `sft-v2`, ASR preset `gold`. Then `python -m lifemate.compare runs/20261102-074734_263ebdc_train_sft-v2_gold_dev runs/<new dev run>`. Pass: the interval's upper end ≥ 0 (dev must not get worse; holdout already passed).
- [ ] **Step 4: Write the failing preset test** in `tests/test_presets.py`:

```python
def test_check_refs_presets():
    env = {"HF_MODEL_OWNER": "team"}
    assert resolve("agent", "sft-v2-refs", env) == (
        "react", {"model": "team/lifemate-sft-v2", "examples": 4, "check_refs": True})
    assert resolve("agent", "sft-final", env)[1].get("check_refs") is CHECK_REFS_IN_FINAL
```

  with `CHECK_REFS_IN_FINAL = True` if Step 3 passed, else `None`.
- [ ] **Step 5:** `pytest tests/test_presets.py -v`. Expected: FAIL (unknown preset `sft-v2-refs`).
- [ ] **Step 6:** Add the presets. Move the uncommitted `RESULTS.md` edit onto branch `exp/check-refs` (`git stash`, `git switch -c exp/check-refs origin/main`, `git stash pop`), fix the note "prompt v1rompt v1", add the Step 3 row (and P3's row if P has it) with the compare lines.
- [ ] **Step 7:** `pytest`. Expected: all pass.
- [ ] **Step 8: Public submission:** split `public_test`, agent preset `sft-final`, ASR preset `pho-large-ft-b` (plus extra `--agent-kw check_refs=true` if Step 3 passed and the PR is not merged yet). Submit `submission.jsonl`, add the leaderboard score to its `RESULTS.md` row.
- [ ] **Step 9: Commit and PR**

```bash
git add configs/presets.json tests/test_presets.py RESULTS.md
git commit -m "Add check_refs presets and the sft-final results"
GH_TOKEN="$(gh auth token --user AuDuongKhang)" gh pr create --base main --title "check_refs presets and sft-final results"
```

---

### Task 2: Schema perturbation of masked copies, then `sft-v3` (K)

Holdout loses most points on unseen *schemas and scenarios*. Masked copies already rename apps, tools and parameters; this task also varies the schema's shape, so the model reads what is there instead of what it memorised.

**Files:**
- Create: `src/lifemate/sft/perturb.py`
- Modify: `src/lifemate/sft/records.py` (flag `--perturb`), `notebooks/molab_train.py` (checkbox, record name suffix)
- Test: `tests/test_sft_perturb.py`

**Interfaces:**
- Consumes: records as written by `records.py` (`{"id", "messages", "tools", "source", ...}`; tools are `{"type": "function", "function": {"name", "description", "parameters"}}`); `lifemate.agents.parsing.BLOCK` to read few-shot `<tool_call>` lines.
- Produces: `perturb_record(record: dict, rng: random.Random, drop_p: float = 0.5) -> dict` returning a copy with perturbed `tools` and `"perturbed": True`; messages unchanged.

Behaviour of `perturb_record`:
- For each tool except `sys.ask_missing_info`: shuffle the order of `parameters.properties`.
- Drop each optional property (not in `required`) with probability `drop_p`, **unless it is used**. Used = an argument key of any `tool_calls` entry in the messages (history and gold), a name in the `params` of a `sys.ask_missing_info` call whose `tool_name` is this tool, or an argument key of a `<tool_call>` line for this tool in the system message (few-shot examples).
- Never touch `required`, enums, descriptions or names (masking already renames).

- [ ] **Step 1: Write the failing tests** in `tests/test_sft_perturb.py` (use the `act_record` / `ask_record` / `examples_record` fixture pattern of `tests/test_sft_mask.py`):
  - `test_perturb_keeps_every_used_parameter`: for every call in `perturb_record(rec, Random(0))["messages"]`, `set(call.arguments) <= set(properties of that tool)`; the same for the few-shot lines of `examples_record`.
  - `test_perturb_keeps_required_and_messages`: `required` lists equal as sets per tool; `messages` identical.
  - `test_perturb_drops_some_unused_optional_parameter`: over `Random(0..9)`, at least one copy has fewer properties than the original for some tool.
  - `test_perturb_is_deterministic`: equal output for equal seeds.
  - `test_records_cli_perturbs_only_masked_copies`: `records.main([... "--mask", "1", "--perturb", "--examples", "0", "--limit", "3", "--tokenizer", "none", "--out-dir", tmp])` writes rows whose `gold` records have no `perturbed` key and whose `masked` records have `perturbed: True`.
- [ ] **Step 2:** `pytest tests/test_sft_perturb.py -v`. Expected: FAIL (`lifemate.sft.perturb` missing).
- [ ] **Step 3:** Implement `perturb_record` in `src/lifemate/sft/perturb.py`. In `records.py`, add `--perturb` (store_true, help "perturb the schemas of masked copies (lifemate.sft.perturb)") and apply `perturb_record(masked, random.Random(f"{args.seed}:{sample['id']}:{k}:p"))` after `mask_record`. Gold and ASR copies stay unperturbed, so one copy of each sample keeps the exact inference text.
- [ ] **Step 4:** `pytest tests/test_sft_perturb.py tests/test_sft_mask.py -v`. Expected: PASS.
- [ ] **Step 5:** In `molab_train.py`, add `perturb = mo.ui.checkbox(value=False, label="perturb masked schemas (--perturb)")`, pass `--perturb` with the masked flags, and add `-p` to the train record name in `_name` when ticked. Update the notebook's how-to text and the `sft/` bullet in `CLAUDE.md`.
- [ ] **Step 6:** `pytest`. Commit `Perturb the schemas of masked SFT copies` on `feat/schema-perturb`, push and open a PR.
- [ ] **Step 7: Experiment `sft-v3`** on molab (branch `feat/schema-perturb`): run name `sft-v3`, model repo `TripleTrouble/lifemate-sft-v3`, records pool, eval dev, masked copies **2**, perturb ticked, ASR copy `pho-large-fp16`. Expected time about 1.3× `sft-v2`. Merge, add an `sft-v3` preset (same shape as `sft-v2`, plus `check_refs` if Task 1 adopted it).
- [ ] **Step 8: Gate.** Run dev and holdout with ASR `gold` and the same agent kwargs as the baseline runs; compare both against `sft-v2` (with `check_refs` if adopted). Write both rows and compare lines to `RESULTS.md`. If it passes, the winning recipe for the final retrain becomes "masked 2 + perturb".

---

### Task 3: Synthetic scenarios: probe, verify, write (P)

The teacher (Task 4) needs to see the sample's app data, and its output must be proven executable before anything is trained on it. This task holds the parts that need no API.

**Files:**
- Create: `src/lifemate/sft/synth.py`
- Test: `tests/test_sft_synth.py`

**Interfaces:**
- Consumes: `lifemate.data.import_zaic_tools`, `load_catalog`; `lifemate.agents.refs.scalar_strings`, `unseen_refs`; catalog specs carry `"kind": "read" | "write"`.
- Produces:
  - `probe(sample: dict, catalog: dict[str, dict], zaic_tools, data_root: Path) -> list[dict]`: one fresh `ToolEnv("train", env_id(sample))`; calls every installed tool with `kind == "read"` and an empty `required` list with `{}` arguments; returns `[{"name", "arguments", "result"}]`. Context for the teacher only, never part of a synthetic label.
  - `verify(sample: dict, calls: list[dict], kind: str, catalog: dict[str, dict], zaic_tools, data_root: Path) -> str | None`: `None` when the calls are acceptable for a scenario of that kind (`act`, `ask`, `empty`), otherwise the reason.
  - `synthetic_sample(base: dict, request: str, calls: list[dict], n: int) -> dict`.
  - `env_id(sample: dict) -> str` returns `sample.get("env_id", sample["id"])`.

`verify` rules, checked in order in a fresh `ToolEnv` (the reason strings are fixed, the report counts them):
1. `kind == "empty"`: accept only `calls == []` (reason `"calls on empty"`). Otherwise `"no calls"` if `calls == []`, and `"kind mismatch"` if `kind == "ask"` and the last call is not `sys.ask_missing_info` (or `kind == "act"` and any call is).
2. `"not installed: <name>"` if a call's tool is not installed.
3. `"schema: <error>"` if `env.validate` fails.
4. `"unseen id: <param>=<value>"` if an x-ref value was never seen in the history's tool results or an earlier call's result (`unseen_refs` with a growing `seen` set).
5. `"no effect: <name>"` if a call returns `{"ok": true, "done": false}` (an unknown id or a no-op: wasteful, so not gold-like).
6. `"ask not last"` if `sys.ask_missing_info` appears anywhere but last.
7. `"repeat"` if the same (name, arguments) appears twice.

`synthetic_sample` returns `{"id": f"synth_{base['id']}_{n}", "env_id": base["id"], "audio": base["audio"], "audio_text": request, "context": base["context"], "history": base["history"], "label": {"tool_calls": calls}, "source": "synth"}`. The history is kept because the environment state already includes it; the teacher is shown that history.

- [ ] **Step 1: Write the failing tests** (skipped without `data/train`, like the other data tests):
  - `test_verify_accepts_the_gold_calls_of_pool_samples`: for the first 20 pool act samples, `verify(s, s["label"]["tool_calls"], "act", ...) is None`.
  - `test_verify_rejects_a_bad_enum`: a gold call with one enum value replaced by `"zzz"` → reason starts with `"schema:"`.
  - `test_verify_rejects_an_invented_id`: a write call whose x-ref argument is set to `"id_does_not_exist_999"` → reason starts with `"unseen id:"`.
  - `test_verify_rejects_ask_not_last`: an ask followed by another call → `"ask not last"`.
  - `test_probe_only_calls_reads_without_required_params`: every probe `name` has `kind == "read"` and an empty `required` list.
  - `test_synthetic_sample_points_at_its_base_environment`: `env_id(synthetic_sample(base, "x", [], 0)) == base["id"]`, `id == f"synth_{base['id']}_0"`.
- [ ] **Step 2:** `pytest tests/test_sft_synth.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement `probe`, `verify`, `synthetic_sample`, `env_id` in `src/lifemate/sft/synth.py`.
- [ ] **Step 4:** `pytest tests/test_sft_synth.py -v`. Expected: PASS.
- [ ] **Step 5:** Commit `Probe and verify synthetic SFT scenarios` on `feat/synth`.

---

### Task 4: Claude teacher and the batch CLI (P)

**Files:**
- Create: `src/lifemate/sft/teacher.py`
- Modify: `src/lifemate/sft/synth.py` (CLI), `requirements-dev.txt` (`anthropic`)
- Test: `tests/test_sft_teacher.py`

**Interfaces:**
- Consumes: Task 3's `probe`, `verify`, `synthetic_sample`; `lifemate.splits.load_splits`; `lifemate.infer.build_task` + `GoldTranscriptASR` to render tools the way the agent sees them (`lifemate.agents.prompting.tool_schema`).
- Produces:
  - `MODEL = "claude-opus-5-5"`; `OUTPUT_SCHEMA` (below).
  - `build_prompt(sample: dict, tools: list[dict], probes: list[dict], n: int, train_text: bool) -> tuple[str, str]` → `(system, user)`.
  - `parse_scenarios(text: str) -> list[dict]` → `[{"request": str, "kind": "act"|"ask"|"empty", "calls": [{"name", "arguments": dict}]}]`; malformed scenarios are skipped, never raised.
  - `submit_batch(client, jobs: list[tuple[str, str, str]]) -> str` (jobs are `(custom_id, system, user)`; returns the batch id).
  - `collect_batch(client, batch_id: str) -> dict[str, str]` (`custom_id → text`; errored, expired, canceled and `stop_reason == "refusal"` results are left out and counted on stdout).
  - `ask_one(client, system: str, user: str) -> str` (synchronous, for the pilot).

`OUTPUT_SCHEMA` (arguments travel as a JSON string, because structured outputs need `additionalProperties: false` everywhere and tool arguments are free-form):

```python
OUTPUT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["scenarios"], "properties": {
    "scenarios": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["request", "kind", "calls"], "properties": {
            "request": {"type": "string"},
            "kind": {"type": "string", "enum": ["act", "ask", "empty"]},
            "calls": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                "required": ["name", "arguments_json"],
                "properties": {"name": {"type": "string"}, "arguments_json": {"type": "string"}}}}}}}}}
```

API usage (from the claude-api reference; never guess other parameters):
- Batch params per job: `model=MODEL`, `max_tokens=16000`, `system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]`, `messages=[{"role": "user", "content": user}]`, `output_config={"effort": "high", "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}}`, wrapped in `Request(custom_id=..., params=MessageCreateParamsNonStreaming(...))` and sent with `client.messages.batches.create(requests=[...])`. Results come back in any order: key them by `custom_id`.
- `ask_one` uses `client.beta.messages.create(..., betas=["server-side-fallback-2026-07-01"], fallbacks="default")` with the same system/messages/output_config (fallbacks are not accepted on the Batches API). Check `stop_reason` before reading content; the JSON is the first `text` block.
- Credentials come from the environment (`ANTHROPIC_API_KEY` or an `ant auth login` profile). Never in the repo or the notebook file.

Prompt content (`build_prompt`):
- **System (fixed, cached):** the task (write `n` new spoken Vietnamese requests a user of these apps could say *now*, each with the exact tool calls a perfect assistant would make); conventions copied from `data/train/zaic_tools/README.md` and `context/EVALUATION.md` (every call is scored, so only needed reads; ids only from results; `sys.ask_missing_info(tool_name, params)` when a required value is missing or several candidates fit; `empty` when no installed app can do it; time arguments as ISO with the context's timezone); a target mix per prompt of about 60% act with 2–4 calls, 25% ask, 15% empty; spoken style (numbers and e-mail addresses as speech, regional words, fillers, self-corrections), varied across scenarios.
- **User:** `context` (current_time, location, user_profile), the installed tool schemas as the agent sees them, the history turns, the probe results. With `train_text=True` it also gets the base sample's own request and gold calls as one example of the argument conventions; with `False` (Q2 = no) nothing from train text or labels.

CLI in `synth.py`:

```
python -m lifemate.sft.synth pilot  --name synth-pilot --subset pool --limit 20 --scenarios 3 [--no-train-text]
python -m lifemate.sft.synth submit --name synth-v1 --subset pool --scenarios 3 [--limit N] [--no-train-text]
python -m lifemate.sft.synth collect --name synth-v1 --batch <batch id>
```

`pilot` and `collect` parse, run `verify` on each scenario, and write accepted ones with `synthetic_sample` to `train_records/synth/<name>.jsonl` (git-ignored), plus `train_records/synth/<name>.report.json` with accepted/rejected counts by reason and by kind. `submit` writes `train_records/synth/<name>.batch` holding the batch id. `--subset` accepts `pool` and `all` only; base samples are those of the subset.

- [ ] **Step 1: Write the failing tests** (no network: pass a fake client object):
  - `test_submit_batch_sends_model_schema_and_cache`: a fake `client.messages.batches.create` records its `requests`; assert each has `params["model"] == "claude-opus-5-5"`, `params["output_config"]["format"]["schema"] == OUTPUT_SCHEMA`, and `params["system"][0]["cache_control"] == {"type": "ephemeral"}`.
  - `test_parse_scenarios_reads_arguments_json`: a text with one scenario whose `arguments_json` is `"{\"value\": 3}"` → `calls[0]["arguments"] == {"value": 3}`.
  - `test_parse_scenarios_skips_bad_arguments`: `arguments_json` = `"{not json"` → that scenario is dropped, the next one kept.
  - `test_collect_batch_skips_errors_and_refusals`: fake results (`SimpleNamespace`) with types `succeeded`, `errored`, `expired`, plus a succeeded one with `stop_reason == "refusal"` → only the first id is returned.
  - `test_prompt_without_train_text_has_no_train_request`: `build_prompt(s, tools, probes, 3, train_text=False)` user text does not contain `s["audio_text"]`.
  - `test_teacher_is_not_imported_by_inference`: after `import lifemate.infer`, `"anthropic" not in sys.modules`.
- [ ] **Step 2:** `pytest tests/test_sft_teacher.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `teacher.py` (import `anthropic` inside the functions that need it) and the CLI in `synth.py`.
- [ ] **Step 4:** `pytest`. Expected: PASS.
- [ ] **Step 5:** Commit `Generate synthetic scenarios with a Claude teacher` on `feat/synth`.

---

### Task 5: Render synthetic samples as SFT records (P)

**Files:**
- Modify: `src/lifemate/sft/records.py`, `notebooks/molab_train.py`
- Test: `tests/test_sft_synth.py`

**Interfaces:**
- Consumes: Task 3's `env_id`; synthetic sample files from Task 4.
- Produces: `records --extra-samples PATH` (repeatable).

Behaviour:
- Extra samples are rendered after the subset's samples, with the same `--mask` / `--perturb` copies. Their text is always `gold` (there is no audio); `cache:` sources skip them and the summary counts `asr skipped`.
- The `ToolEnv` of every record is `ToolEnv("train", env_id(sample))` (real samples are unchanged: `env_id` falls back to the id).
- If `--subset` is not `all`, every extra sample's `env_id` must be a pool id, else `ap.error("extra sample <id> uses a non-pool environment ...")`.
- The few-shot bank is unchanged: synthetic samples are never examples.

- [ ] **Step 1: Write the failing tests:**
  - `test_records_render_extra_samples`: write a file with `synthetic_sample(pool_sample, pool_sample["audio_text"], pool_sample["label"]["tool_calls"], 0)`; run records with `--subset pool --limit 1 --mask 1 --extra-samples <file> --examples 0 --tokenizer none`; the output has two `gold` and two `masked` records, and one record id starts with `synth_`.
  - `test_records_refuse_synthetic_samples_on_eval_environments`: the same with a dev sample as base → `SystemExit`.
  - `test_records_skip_the_asr_source_for_extra_samples`: with `--text gold --text cache:<tmp cache>` (build the cache with `_write_cache` from `tests/test_sft_replay.py`) the synthetic sample yields only a `gold` record.
- [ ] **Step 2:** `pytest tests/test_sft_synth.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement in `records.py`.
- [ ] **Step 4:** In `molab_train.py`, add a text field *synthetic samples (HF dataset path)*, e.g. `synth/synth-v1.jsonl`. When set, download the file with `huggingface_hub.hf_hub_download(repo_id=hf_repo.value, repo_type="dataset", filename=..., token=hf_token.value, local_dir=work)`, pass `--extra-samples`, and add `-<file stem>` to the train record name.
- [ ] **Step 5:** `pytest`. Commit `Train on verified synthetic samples`, push `feat/synth`, open the PR (Tasks 3–5 together).

---

### Task 6: Synthetic run and `sft-v4` (P)

- [ ] **Step 1: Pilot:** `python -m lifemate.sft.synth pilot --name synth-pilot --subset pool --limit 20 --scenarios 3` (add `--no-train-text` if Q2 = no). Expected: a report with accept rate; aim for ≥ 50% of scenarios accepted.
- [ ] **Step 2: Read all accepted pilot scenarios by hand** (P, 15 minutes; K reads them too if time allows) with this checklist: the request sounds spoken and Vietnamese; the calls are what you would expect; asks are asked for a real reason; empty requests really have no app; argument style matches train (names, quantities, ISO times). If more than 3 of 20 fail the checklist, change the system prompt and repeat Step 1. Do not go to Step 3 until it passes.
- [ ] **Step 3: Bulk:** `submit --name synth-v1 --subset pool --scenarios 3` (≈ 2,800 scenarios), then `collect` once the batch has ended. Cost cap: check `usage` on the pilot's responses, multiply by 948/20, halve for the batch discount; stop and ask the team if the estimate is over the budget agreed in Q1.
- [ ] **Step 4:** Upload `train_records/synth/synth-v1.jsonl` to the team's private HF dataset under `synth/` (`huggingface_hub.upload_file`, repo type dataset). Never commit it.
- [ ] **Step 5: Train `sft-v4`:** the best recipe so far (Task 2 result) plus the synthetic file, pool records, eval dev. Merge, add the preset.
- [ ] **Step 6: Gate** on dev and holdout against the current best, with `RESULTS.md` rows. If it passes, the final retrain uses a second synthetic batch built with `--subset all` (Step 3 again with `--name synth-all`) plus the real all-train records.

---

### Task 7: Ask-guard simulator (K)

Decide the ask rule from data before touching the agent. The simulator replays the *existing* traces with the guard applied and scores the result exactly. After an ask the agent stops, so the counterfactual run is "the calls before the guarded write, then the ask".

**Files:**
- Create: `src/lifemate/agents/ask_guard.py`, `src/lifemate/guard_sim.py`
- Test: `tests/test_ask_guard.py`

**Interfaces:**
- Consumes: run folders with `trace.jsonl` (per sample: `request`, `calls` with `name`, `arguments`, `result`); catalog specs (`kind`, x-ref properties); `lifemate.evaluate.evaluate(pred_path, "train", data_root)`; `lifemate.compare`.
- Produces:
  - `fold(text: str) -> list[str]`: lowercase, strip Vietnamese diacritics (NFD, drop combining marks, `đ → d`), split on non-alphanumerics.
  - `match_score(request: str, item: dict) -> float`: max over the item's string values (except `id`) of |tokens(value) ∩ tokens(request)| / |tokens(value)|.
  - `ask_instead(request: str, call: dict, spec: dict, candidates: dict[str, list[dict]], min_score: float = 0.5, margin: float = 0.2) -> dict | None`. `candidates` maps each id seen in a result list of ≥ 2 dicts with an `id` to that list. It returns `{"name": "sys.ask_missing_info", "arguments": {"tool_name": call["name"], "params": [param]}}` for the first x-ref argument of a `kind == "write"` call whose value is in `candidates`, when the chosen item's score is below `min_score` or beats the best other candidate by less than `margin`; otherwise `None`.
  - `python -m lifemate.guard_sim runs/<run> [--min-score 0.5 --margin 0.2 | --grid]`: writes `runs/<run>/guard_sim/<setting>/submission.jsonl` by replaying the guarded call list in a fresh train `ToolEnv` (`env.get_tool_calls()`, `env.get_state_hash()`), scores it with `evaluate`, and prints the `compare` line against the run's own `per_sample.jsonl`. `--grid` tries `min_score` in {0.3, 0.5, 0.7} × `margin` in {0.1, 0.2, 0.3}.

- [ ] **Step 1: Write the failing tests** in `tests/test_ask_guard.py` (pure data, no `data/train` needed):
  - `test_no_ask_when_the_request_names_the_station`: bikes `[{"id": "b1", "station": "Trạm Biển Đông"}, {"id": "b2", "station": "Trạm Cầu Rồng"}]`, request "giữ xe ở trạm Biển Đông", call `reserve(bike_id="b1")` → `None`.
  - `test_ask_when_nothing_in_the_request_picks_one`: same bikes, request "giữ giùm một chiếc xe", call `reserve(bike_id="b1")` → the ask with `params == ["bike_id"]`.
  - `test_no_ask_for_a_single_candidate`: one bike → `None`.
  - `test_no_ask_for_read_calls`: spec `kind == "read"` → `None`.
  - `test_fold_strips_diacritics`: `fold("Đường Lò Đúc") == ["duong", "lo", "duc"]`.
- [ ] **Step 2:** `pytest tests/test_ask_guard.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `ask_guard.py`, then `guard_sim.py` (it builds `candidates` while walking a trace's calls in order, exactly as the agent will).
- [ ] **Step 4:** `pytest tests/test_ask_guard.py -v`. Expected: PASS.
- [ ] **Step 5: Simulate:** `python -m lifemate.guard_sim runs/20261102-074734_263ebdc_train_sft-v2_gold_dev --grid`. Pick the setting with the best dev difference, then run only that setting on the holdout run. Decision: go to Task 8 only if holdout is ≥ +1.0 with an interval excluding 0 and dev is not worse. Otherwise write the numbers in the PR and stop here.
- [ ] **Step 6:** Commit `Simulate an ask guard on recorded traces` on `exp/ask-guard`, push, open the PR with the grid table.

### Task 8: Ask guard in the agent (K, only if Task 7 says so)

**Files:**
- Modify: `src/lifemate/agents/react.py`, `configs/presets.json`
- Test: `tests/test_react_agent.py`

**Interfaces:**
- Consumes: `ask_instead` from Task 7.
- Produces: `ReActAgent(..., ask_guard: bool = False, ask_min_score: float = 0.5, ask_margin: float = 0.2)`, recorded in `self.info`; preset `<best>-ask` with the chosen thresholds.

- [ ] **Step 1: Write the failing tests** with the existing fake LLM/env helpers in `tests/test_react_agent.py`:
  - `test_ask_guard_replaces_an_unsupported_pick`: the LLM reads two candidates, then picks one the request does not name; the env receives `sys.ask_missing_info` instead of the write, and `trace["stop"] == "asked"`.
  - `test_ask_guard_off_by_default`: the same script without the kwarg sends the write.
- [ ] **Step 2:** `pytest tests/test_react_agent.py -v`. Expected: FAIL.
- [ ] **Step 3:** In `run`, keep `candidates` updated after every `env.call` result. Before `env.call` of a validated call, if `self.ask_guard` and `ask_instead(...)` returns an ask, call `env.call` with that ask, set `trace["stop"] = "asked"` and return.
- [ ] **Step 4:** `pytest`. Expected: PASS.
- [ ] **Step 5: Gate on molab** (dev and holdout, the same model as the baseline), then `RESULTS.md` rows. The real runs must confirm the simulator within about 1 point; if they do not, do not ship the guard.
- [ ] **Step 6:** Commit `Add the ask guard to the ReAct agent`, PR.

---

### Task 9: Docker package (K)

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `configs/shipped.json`, `src/lifemate/docker_entry.py`, `scripts/docker_build.sh`
- Modify: `src/lifemate/asr/whisper.py` (`revision`), `README.md` (Docker section)
- Test: `tests/test_docker_entry.py`, `tests/test_asr_backends.py`

**Interfaces:**
- Consumes: `lifemate.infer.main(argv) -> Path` (the run folder), presets, `HF_MODEL_OWNER`.
- Produces:
  - `WhisperASR(model, device="cuda", dtype="float16", revision: str | None = None)`, passing `revision` to the pipeline's model and processor loading.
  - `configs/shipped.json`: `{"agent_preset": ..., "asr_preset": ..., "agent_kw": {...}, "asr_kw": {...}, "hf_model_owner": "TripleTrouble"}`, where `agent_kw` / `asr_kw` hold at least `revision` (the hashes from Task 1 Step 2 and the ASR repo). One file, read by the build and the entry, so the two cannot disagree.
  - `load_config(path: Path = Path("configs/shipped.json")) -> dict`.
  - `find_split(data_dir: Path) -> str`: the `<split>` of the single `<split>_tools_catalog.json` in `data_dir`; raises `SystemExit` naming the files found when there are none or several.
  - `fetch_models(config: dict) -> None`: `snapshot_download` of every model the two presets name, at the pinned revisions, into `HF_HOME` (build time only).
  - `main(argv: list[str] | None = None) -> Path`: `--data /data --out /app/result/submission.jsonl --config configs/shipped.json --work /tmp/lifemate`. It sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `HF_MODEL_OWNER` **before** importing `lifemate.infer`, builds `<work>/data/<split>` → `/data` and `<work>/data/train` → `/app/data/train` as symlinks, runs infer with `--split <split> --data-root <work>/data --agent-preset ... --asr-preset ...` plus `--agent-kw` / `--asr-kw` from the config, copies `submission.jsonl` to `--out`, and exits non-zero if its line count differs from the split's sample count.

Image layout:
- Base: an official `pytorch/pytorch:<torch>-cuda12.4-cudnn9-runtime` tag matching the torch version used on molab (record `torch.__version__` there first). CUDA 12.4 is chosen because it runs on the A30 drivers in common use; avoid images that need a newer driver than the scoring host may have.
- `WORKDIR /app`; copy the repo (`.dockerignore` excludes `runs/`, `checkpoints/`, `train_records/`, `data/`, `.venv/`, `.git/`); copy only `data/train/train.jsonl` and `data/train/train_tools_catalog.json` to `/app/data/train/` (few-shot bank; see Q6); `pip install -r requirements.txt` and `-e .`.
- `ENV HF_HOME=/models`; `RUN --mount=type=secret,id=hf_token HF_TOKEN=$(cat /run/secrets/hf_token) python -c "from lifemate.docker_entry import fetch_models, load_config; fetch_models(load_config())"`. The token is a build secret, never a layer or an `ARG`.
- `ENTRYPOINT ["python", "-m", "lifemate.docker_entry"]`. Also write to `/app/result/` because the rules say `result/submission.jsonl`; switch to whatever the organisers' guideline says once it is published.
- `scripts/docker_build.sh`: `DOCKER_BUILDKIT=1 docker build --secret id=hf_token,env=HF_TOKEN -t lifemate:<git short sha> .` then `docker save | gzip > lifemate_<sha>.tar.gz` and `sha256sum`.

- [ ] **Step 1: Write the failing tests:**
  - `tests/test_docker_entry.py::test_find_split_reads_the_catalog_name`: a tmp dir with `private_test_tools_catalog.json` → `"private_test"`; with none → `SystemExit`.
  - `test_entry_sets_offline_flags_before_infer`: monkeypatch `lifemate.infer.main` to record `os.environ` and return a tmp run folder with a one-line `submission.jsonl`; call `docker_entry.main([...])` on a tmp split folder with one sample → the recorded env has `HF_HUB_OFFLINE == "1"`, and `--out` holds that line. Skip on `OSError` from symlinks (Windows without developer mode).
  - `test_entry_fails_on_a_short_submission`: the fake infer writes 0 lines for 1 sample → `SystemExit` with a non-zero code.
  - `tests/test_asr_backends.py::test_whisper_passes_the_revision`: monkeypatch `transformers.pipeline` to capture kwargs → the revision reaches it.
- [ ] **Step 2:** `pytest tests/test_docker_entry.py tests/test_asr_backends.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `docker_entry.py`, the Whisper `revision`, `configs/shipped.json` (start with `empty` / `none` presets so the skeleton builds before the models are final), `Dockerfile`, `.dockerignore`, `scripts/docker_build.sh`.
- [ ] **Step 4:** `pytest`. Expected: PASS.
- [ ] **Step 5: Skeleton smoke** on any machine with Docker: build, then `docker run --rm --network none -v $PWD/data/public_test:/data:ro -v $PWD/out:/app/result lifemate:<sha>`. Expected: `out/submission.jsonl` with 634 lines that passes `cd data && PYTHONPATH=public_test python -m zaic_tools --split public_test --root . --validate ../out/submission.jsonl`.
- [ ] **Step 6:** Commit `Add the Docker package` on `feat/docker`, PR. After the models are chosen, Task 10 fills in `configs/shipped.json`.

---

### Task 10: A30 rehearsal, freeze and hand-off (K)

- [ ] **Step 1: Fill `configs/shipped.json`** with the values from the final retrain (Task 1 Step 2) and P's final ASR preset and revision: the final agent and ASR presets, `check_refs` / guard kwargs if they passed, and both revisions. Rebuild.
- [ ] **Step 2: A30 run** (Q7): on a rented A30 (or the nearest 24 GB Ampere card; write down which), `docker load`, then run the image on public_test with `--gpus all --network none`, **twice**. Expected: `cmp` finds the two files identical; `meta.json` shows max seconds per sample < 25 (5 s of margin under 30). If max ≥ 25: lower `time_budget_s` in the shipped `agent_kw` (it counts from the sample start, ASR included) and re-run; never ship over 25.
- [ ] **Step 3: Cross-check** the A30 submission against the molab run of the same config: count samples whose `tool_calls` differ. bf16 numerics can differ between GPUs; more than ~2% different samples means something else differs (revision, preset, data). Find out what before freezing.
- [ ] **Step 4: Freeze (by 14 Nov 23:59):** tag the commit `final-<date>`; write in `RESULTS.md` notes the image tag, sha256, both model revisions, the presets and the A30 timing; add the Docker run command and the training commands (records, synth, train, merge, with seeds) to `README.md`.
- [ ] **Step 5: Submit (15–16 Nov):** upload the `.tar.gz` to Google Drive with link access, submit the form with the link and the sha256. Keep the previous image as the fallback (2 submissions allowed, the last one counts).
- [ ] **Step 6:** Commit `Document the shipped system` on a `docs/` branch, PR.
