# Score Push (after sft-final 53.58) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** raise the public and private score above 53.58 with the fixes from `docs/error_analysis.md`: cheap offline rewrites first, then opt-in agent guards and a prompt vote, handed to the sprint plan's final retrain and Docker image as a list of gated kwargs and one records flag by 10 Nov.

**Architecture:** Three layers. (1) `lifemate.sim` replays recorded train traces with a rewrite applied and scores them exactly, without a GPU, so post-hoc argument rewrites are decided before any GPU run. (2) Every agent change is an opt-in `ReActAgent` kwarg (default off, recorded in `self.info`) with its own preset, measured on dev and holdout with `lifemate.compare`. (3) The one change that alters the prompt (the date table) is also a records flag, so the final retrain of the sprint plan can render it into the training text.

**Tech Stack:** Python 3.10+, pytest, HF transformers + peft (molab GPU for runs and training), the split's own `zaic_tools`.

**Spec:** `docs/error_analysis.md` (findings A–H, section 4 order). Background: `docs/superpowers/plans/2026-11-02-final-sprint.md` (K3, K4).

## Relation to the sprint plans

Owner: **K only.** This plan is K's score work and adds nothing to P's tasks.

- It **replaces K3 and K4** of `2026-11-02-final-sprint.md` (guards, contact fallback): the ask rule and the out-of-scope check are dropped with the evidence below, the validate idea becomes Task 3, the contact fallback becomes Task 8.
- It **does not touch** `2026-11-02-sprint-execution.md`: P's ASR work (P4 beam / n-best, P5) and synthetic data (Tasks 3–6, `sft-v4`) stay P's; the `sft-v3` perturbation run (Task 2), the final retrain `sft-final2` and Docker (Tasks 9–10) stay where they are. This plan names no new model and builds no image.
- **Hand-off** (Task 10): the gated kwargs go into a preset that the final retrain and the Docker task of the sprint plan pick up, and `--date-hints` is added to that retrain's records command.
- **Shared files:** Task 4 edits `sft/replay.py` and `sft/records.py`, which P's synthetic rendering (sprint Task 5) also edits. Rebase on `main` before the PR and keep the flag additive.

## Global Constraints

- Never commit or push to `main`; implementation branches `exp/<topic>` or `feat/<topic>`, one PR each, merged after a teammate looks.
- All repo text in English; Vietnamese only as data (weekday names in the date table are data).
- Never read `<split>_environment/` from solution code; one `ToolEnv` per sample, created by the runner.
- Every `env.call` is scored; `env.validate` is free. A rewrite or guard must never send a call the model would not have sent unless the task says so (Task 8 fallback is the only one, and it is gated).
- Whole system < 6B params, LLM < 5B, ≤ 16,384 tokens, ≤ 30 s per sample on an A30, greedy decoding, fixed seeds, offline Docker.
- **Evaluation model:** sft-final was trained on all of train, so dev/holdout scores with it are contaminated. Every dev/holdout comparison uses **sft-v2 + check_refs** (pool-trained). Baselines, already recorded:
  - dev gold `runs/20261102-093954_3c4fd00_train_sft-v2_gold_dev_check_refs` (72.86)
  - holdout gold `runs/20261102-092303_3c4fd00_train_sft-v2_gold_holdout_check_refs` (50.96)
  - dev ASR `runs/20261102-101746_3c4fd00_train_sft-v2_pho-large-ft-b_dev_check_refs` (66.86)
  - holdout ASR `runs/20261102-104252_3c4fd00_train_sft-v2_pho-large-ft-b_holdout_check_refs` (41.37)
- **Default gate** (unless a task says otherwise): holdout diff ≥ +0.5 with the 95% CI above 0, and dev diff ≥ −0.3, on the ASR runs. Write the `compare` line into the PR and a `RESULTS.md` row for every GPU run.
- Deadlines: hand-off (Task 10) by 10 Nov, so the sprint plan's final retrain (9–11 Nov) can use it; Docker freeze 14 Nov and the 16 Nov deadline belong to the sprint plan.

## Review Focus

- A free-text value that already equals or is longer than the matching result string must stay unchanged (Task 2 test `test_value_already_full_is_kept`).
- A read whose optional value is stated in the request by sound but folded differently (diacritics, case) must keep the value (Task 3 test `test_folded_match_keeps_the_value`).
- The refusal check must fire at most once per sample and never after a call was sent (Task 6 tests `test_refusal_check_fires_once`, `test_no_refusal_check_after_a_call`).
- The date table on a Sunday must put the next day in "tuần sau" (Task 4 test `test_sunday_rolls_to_next_week`).
- An empty-result hint must not change what is sent: same calls, only the tool message text differs (Task 5 test `test_empty_hint_changes_only_the_message`).

## Not in this plan (decided, with evidence)

| Idea | Why not |
|---|---|
| Hard ask rule (several candidates → always ask) | Simulated in PR #19: loses at every setting (dev −14.93 at best, holdout −6.64). Ask errors go both ways (16 vs 20 on holdout ASR). |
| Out-of-scope classifier before the first call | That bucket costs only 0.34–2.0 points; wrong refusals on unseen apps cost more. Task 6 checks refusals instead. |
| Strip optional keys by a fixed rule on writes | Gold uses optional parameters in 58% of writes that have them. Reads only, and only after simulation (Task 3). |
| Constrained JSON decoding | 0–1 `invalid JSON` per run across all recorded runs; rejections are enum values and `already called`, which `validate` already catches for free. |
| Tool-description compression | No run ever stopped on `tokens`; SFT was trained on full tool lists, so trimming shifts the input distribution. |
| Few-shot examples chosen by tool descriptions | No evidence it helps, and SFT learned with request-similar examples, so a new choice shifts the prompt the model was trained on. Revisit only if time is left after Task 9. |
| Other agent architectures: tree search (LATS), Reflexion, best-of-N with execution | Each explores the environment and goes back. Here there is one `ToolEnv` per sample, every call is scored and writes cannot be undone, so exploring is only possible before a call. Task 9 votes before a call instead. |
| Plan-then-execute (ReWOO, LLMCompiler) | Writes need ids from read results, and ask decisions depend on those results, so a plan made up front cannot fill them. Its gain is fewer LLM calls, and time is not the bottleneck (about 3 s of 30 s). |
| Router or multi-agent (act/ask/out-of-scope classifier, then an executor) | Ask depends on read results, so it cannot be decided before the reads (same lesson as PR #19). |
| CodeAct (the model writes Python that calls the tools) | Needs all SFT data rebuilt; chains are 1–3 calls, so there is little logic to gain. |
| Reasoning before each call (thinking mode, distilled traces) | Gold has no reasoning, so it needs a teacher model, new records and a retrain, plus more tokens and time on the A30. Too large for the time left. |
| Ask/act by sequence likelihood | Greedy decoding already follows the model's preference; a threshold cannot fix errors that go both ways, and it is unclear whether scoring alternatives counts as greedy. |

## Timeline

| Dates | Tasks |
|---|---|
| 3–5 Nov | 1, 2, 3 (offline, no GPU) |
| 4–7 Nov | 4, 5, 6, 7, 8 (agent kwargs, one sft-v2 GPU run each on dev + holdout ASR) |
| 6–9 Nov | 9 (prompt vote at the decision step: disagreement count, then the vote run) |
| 9–10 Nov | 10 (combined preset, one public submission, hand-off to the sprint plan) |

---

### Task 1: Generic offline trace rewriter (`lifemate.sim`)

**Files:**
- Create: `src/lifemate/sim.py`
- Modify: `src/lifemate/guard_sim.py` (use `rewrite_run`; CLI unchanged)
- Test: `tests/test_sim.py`, existing `tests/test_guard_sim.py` must still pass

**Interfaces:**
- Produces:
  - `Rewrite = Callable[[dict, dict[str, dict]], list[dict] | None]`: takes one trace line (`{"id", "request", "calls": [{"name","arguments","result"}], ...}`) and the train catalog `{tool name: spec}`, returns the new call list `[{"name","arguments"}]` or `None` (sample unchanged).
  - `rewrite_run(run: Path, data_root: Path, name: str, rewrite: Rewrite) -> dict`: writes `runs/<run>/sim/<name>/submission.jsonl` (unchanged samples copy their recorded line; changed ones replay the new calls in a fresh train `ToolEnv` and take `get_tool_calls()` + `get_state_hash()`), runs `evaluate`, returns `compare(...)` of run vs rewritten plus `fired: int` and `out: Path`.
  - `REWRITES: dict[str, Callable[..., Rewrite]]` (factories taking kwargs), filled by Tasks 2 and 3.
  - CLI: `python -m lifemate.sim <rewrite> runs/<run> [--kw key=value ...]` (`utils.parse_kw`), prints the same line format as `guard_sim`.

- [ ] **Step 1: Write the failing tests** (skip without `data/train`, as `test_guard_sim.py` does; reuse its fake-run fixture pattern of gold traces):
  - `test_identity_rewrite_keeps_every_score`: `rewrite_run(run, root, "id", lambda t, c: None)` → `fired == 0`, `diff == 0.0`.
  - `test_replaying_the_same_calls_keeps_every_score`: rewrite returns the trace's own calls → `fired == 6`, `diff == 0.0`.
  - `test_dropping_all_calls_scores_zero_on_act_samples`: rewrite returns `[]` → mean of the rewritten run is 0.0 on a run of act samples.
- [ ] **Step 2:** `pytest tests/test_sim.py -v` → FAIL (module missing).
- [ ] **Step 3:** Implement `rewrite_run` and the CLI in `sim.py`; move the replay/score loop out of `guard_sim.simulate`, which becomes `rewrite_run(run, root, f"s{min_score}_m{margin}", <guard rewrite>)` with output still under `runs/<run>/guard_sim/` (keep its folder via an `out_dir` keyword on `rewrite_run`, default `run / "sim"`).
- [ ] **Step 4:** `pytest tests/test_sim.py tests/test_guard_sim.py -v` → PASS.
- [ ] **Step 5:** Commit on `exp/offline-rewrites`: `Add lifemate.sim, a generic offline trace rewriter`.

### Task 2: Free-text completion from tool results (finding A)

**Files:**
- Create: `src/lifemate/agents/freetext.py`
- Modify: `src/lifemate/sim.py` (register `freetext`), `src/lifemate/agents/react.py` (kwarg), `configs/presets.json`
- Test: `tests/test_freetext.py`, `tests/test_react_agent.py`

**Interfaces:**
- Consumes: `rewrite_run`, `REWRITES` (Task 1); `ask_guard.fold`; `refs.scalar_strings`.
- Produces:
  - `complete_free_text(value: str, texts: Iterable[str], min_tokens: int = 2) -> str`
  - `complete_call(call: dict, spec: dict | None, texts: Iterable[str], min_tokens: int = 2) -> dict`: applies `complete_free_text` to every `x-match: fuzzy` string argument; returns a new dict, never mutates.
  - `ReActAgent(..., complete_free_text: bool = False, free_text_min_tokens: int = 2)`.

Rule (the only one): if the last `n ≥ min_tokens` folded tokens of `value` equal the first `n` folded tokens of a result string `t` that has more tokens than `n`, return `value` up to the start of that overlap + `t` (original casing and diacritics of `t`). Take the longest `n`; among equal `n`, the first `t` in result order. No match → `value` unchanged.

- [ ] **Step 1: Write the failing tests** in `tests/test_freetext.py`:
  - `test_prefix_is_completed`: `complete_free_text("Đọc bài logistics xanh", ["Logistics xanh: kho hàng Bắc Ninh giảm phát thải nhờ xe tải điện"]) == "Đọc bài Logistics xanh: kho hàng Bắc Ninh giảm phát thải nhờ xe tải điện"`.
  - `test_value_already_full_is_kept`: value equal to `"Đọc bài " + title` with the same title in `texts` → unchanged.
  - `test_one_token_overlap_is_not_enough`: `complete_free_text("Gọi Minh", ["Minh Anh"]) == "Gọi Minh"`.
  - `test_no_overlap_is_kept`: `"Đọc bài mới nhất về điện mặt trời"` with `["Mái nhà dùng pin mặt trời"]` → unchanged.
  - `test_only_fuzzy_params_change`: `complete_call` on a call with a fuzzy `title` and an exact `list_id` changes only `title`.
- [ ] **Step 2:** `pytest tests/test_freetext.py -v` → FAIL.
- [ ] **Step 3:** Implement `freetext.py` (tokenize with spans so the cut keeps the original text; compare folded tokens). Register `REWRITES["freetext"]`: for each trace, collect `scalar_strings` of the results seen before each call (history results included, from the trace's history if recorded, otherwise results only) and apply `complete_call`; return `None` when nothing changed.
- [ ] **Step 4:** `pytest tests/test_freetext.py -v` → PASS.
- [ ] **Step 5: Simulate** on the four baseline runs with `--kw min_tokens=2` and `min_tokens=3`. Free-text keys are outside the state hash, so the result is exact. Gate: default gate on the ASR runs. Not met → put the table in the PR and stop at Step 9 without Steps 6–8.
- [ ] **Step 6: Agent test** `test_free_text_is_completed_before_sending` in `tests/test_react_agent.py`: FakeEnv result `{"items": [{"id": "n1", "title": "Logistics xanh: kho hàng"}]}`, the model then calls `tasks.add {"title": "Đọc bài logistics xanh"}` with a fuzzy `title` spec; `env.called[-1][1]["title"] == "Đọc bài Logistics xanh: kho hàng"` and the assistant message in the conversation carries the completed value. Default off: `test_free_text_completion_is_off_by_default`.
- [ ] **Step 7:** Implement the kwarg in `react.py` (apply before `validate`, record `"complete_free_text"` in `self.info`), preset `sft-v2-refs-ft` and `sft-final-ft`; `pytest` → PASS.
- [ ] **Step 8:** One GPU run of `sft-v2-refs-ft` on holdout ASR to confirm the simulated diff (within ±0.3 of it).
- [ ] **Step 9:** Commit `Complete free-text arguments from tool results`, PR with the simulation table.

### Task 3: Minimal read arguments (finding B, the user's "validate" idea narrowed to reads)

**Files:**
- Create: `src/lifemate/agents/minimal_reads.py`
- Modify: `src/lifemate/sim.py` (register `minimal_reads`), `src/lifemate/agents/react.py`, `configs/presets.json`
- Test: `tests/test_minimal_reads.py`, `tests/test_react_agent.py`

**Interfaces:**
- Consumes: Task 1; `ask_guard.fold`.
- Produces: `drop_unstated_optionals(call: dict, spec: dict | None, said: str) -> dict` (only when `spec["kind"] == "read"`: drop each optional string argument, enum values included, whose folded value tokens are not all in the folded `said`; required arguments, non-string values and every argument of non-read tools are kept. Enum values are often English codes the user never says, so they are dropped unless stated; the simulation decides whether that is right); `ReActAgent(..., minimal_reads: bool = False)`, where `said` = request + every history user text.

- [ ] **Step 1: Failing tests:**
  - `test_unstated_filter_is_dropped`: `appliances.list {"type": "dishwasher"}`, request "máy rửa chén còn bao lâu xong" → `{}`.
  - `test_stated_query_is_kept`: `contacts.search {"query": "Sếp Dũng"}`, request "nhắn sếp dũng" → unchanged.
  - `test_folded_match_keeps_the_value`: `{"query": "Đà Nẵng"}`, request "vé đi da nang" → unchanged.
  - `test_writes_are_untouched`: a write spec → unchanged.
  - `test_required_args_are_kept`.
- [ ] **Step 2:** run → FAIL. **Step 3:** implement + register. **Step 4:** run → PASS.
- [ ] **Step 5: Simulate** on the four baselines. Later calls are kept as recorded although fewer filters change the read result, so this is an optimistic estimate: stricter gate, holdout ≥ +1.0 with CI above 0 and dev ≥ 0. Not met → PR with the table, stop.
- [ ] **Step 6:** Agent kwarg (applied before `validate`), test `test_minimal_reads_drops_before_sending`, preset `sft-v2-refs-mr`; GPU run on dev + holdout ASR; keep only if the real run passes the default gate.
- [ ] **Step 7:** Commit `Drop unstated optional arguments on reads`, PR.

### Task 4: Date table in the context (finding C)

**Files:**
- Create: `src/lifemate/agents/dates.py`
- Modify: `src/lifemate/agents/prompting.py` (`system_prompt`, `build_messages`), `src/lifemate/agents/react.py`, `src/lifemate/sft/replay.py` + `src/lifemate/sft/records.py` (`--date-hints`), `configs/presets.json`
- Test: `tests/test_dates.py`, `tests/test_prompting.py`

**Interfaces:**
- Produces: `date_table(current_time: str) -> str`; `system_prompt(context, version, date_hints: bool = False)`; `build_messages(..., date_hints: bool = False)`; `ReActAgent(..., date_hints: bool = False)`; `replay(..., date_hints: bool = False)`; records flag `--date-hints`.

Exact format (weeks start on Monday, matching train gold: Thu 2026-10-01 "thứ ba tuần sau" = 2026-10-06; Fri 2026-12-04 "thứ sáu tuần sau" = 2026-12-11; Sun 2026-09-20 "thứ ba tới" = 2026-09-22). One line per day from today to the Sunday of next week:

```
Calendar (weeks start on Monday):
2026-10-01 thứ năm (hôm nay, tuần này)
2026-10-02 thứ sáu (ngày mai, tuần này)
2026-10-03 thứ bảy (ngày mốt, tuần này)
2026-10-04 chủ nhật (tuần này)
2026-10-05 thứ hai (tuần sau)
...
2026-10-11 chủ nhật (tuần sau)
```

Weekday names: `thứ hai, thứ ba, thứ tư, thứ năm, thứ sáu, thứ bảy, chủ nhật`. Appended to the system prompt after the `Context:` line.

- [ ] **Step 1: Failing tests** in `tests/test_dates.py`:
  - `test_thursday_table`: `date_table("2026-10-01T09:00:00+07:00")` has first line after the header `"2026-10-01 thứ năm (hôm nay, tuần này)"`, contains `"2026-10-06 thứ ba (tuần sau)"`, last line `"2026-10-11 chủ nhật (tuần sau)"`, 11 day lines.
  - `test_friday_next_week`: `"2026-12-11 thứ sáu (tuần sau)"` in `date_table("2026-12-04T08:00:00+07:00")`.
  - `test_sunday_rolls_to_next_week`: for `"2026-09-20T10:00:00+07:00"`, lines `"2026-09-20 chủ nhật (hôm nay, tuần này)"` and `"2026-09-21 thứ hai (ngày mai, tuần sau)"`, last line `"2026-09-27 chủ nhật (tuần sau)"`.
  - In `tests/test_prompting.py`: `test_date_hints_append_the_table` and `test_date_hints_off_by_default` (system text unchanged without the flag).
- [ ] **Step 2:** run → FAIL. **Step 3:** implement `dates.py` and thread the flag through `system_prompt` → `build_messages` → `ReActAgent` (recorded in `self.info`) → `replay` → records `--date-hints`. **Step 4:** `pytest` → PASS.
- [ ] **Step 5:** Preset `sft-v2-refs-dates`; GPU run on dev + holdout ASR **without retraining** (the model never saw the table, so this is a lower bound). Record the rows; `--date-hints` goes into the hand-off (Task 10) regardless of this result unless it is worse than −1.0 on both subsets.
- [ ] **Step 6:** Commit `Add a weekday table to the agent context` on `feat/date-hints`, PR.

### Task 5: Hint after an empty read (finding D)

**Files:**
- Modify: `src/lifemate/agents/react.py`, `configs/presets.json`
- Test: `tests/test_react_agent.py`

**Interfaces:**
- Produces: `is_empty_result(result) -> bool` in `react.py` (`[]`, or a dict with at least one list value where every list value is empty); `EMPTY_HINT` constant; `ReActAgent(..., empty_hint: bool = False)`.

Exact texts:
- After a sent call whose result is empty, the tool message is the JSON result + `"\nNo results. Try once more with other values: another date (up to 7 days away), fewer optional filters or a shorter query. If that also finds nothing, reply in one short sentence without calling any tool."`
- For a repeated call whose earlier result was empty: `"Already called and it returned nothing. Change the arguments as suggested, or stop."` (other repeats keep today's text).

- [ ] **Step 1: Failing tests:**
  - `test_empty_result_gets_a_hint`: FakeEnv result `{"ok": True, "flights": []}` for `flights.search`; the next prompt's last tool message ends with the hint.
  - `test_empty_hint_changes_only_the_message`: the same model outputs with and without the flag → identical `env.called`.
  - `test_repeat_of_an_empty_call_gets_the_empty_text`.
  - `test_empty_hint_off_by_default`.
- [ ] **Step 2:** run → FAIL. **Step 3:** implement. **Step 4:** `pytest` → PASS.
- [ ] **Step 5:** Preset `sft-v2-refs-empty`; GPU run on dev + holdout ASR; default gate. Also count `stop == "invalid"` on a public run with the flag (6 with sft-final today).
- [ ] **Step 6:** Commit `Hint the model after an empty read` on `feat/agent-guards`, PR.

### Task 6: Second look before a refusal (finding E)

**Files:**
- Create: `src/lifemate/agents/refusal.py`
- Modify: `src/lifemate/agents/react.py`, `configs/presets.json`
- Test: `tests/test_refusal.py`, `tests/test_react_agent.py`

**Interfaces:**
- Consumes: `ask_guard.fold`.
- Produces: `matching_tools(request: str, tools: list[dict], min_hits: int = 1, k: int = 2) -> list[str]` (score = number of distinct folded word bigrams of the request found in the folded tool name parts + description; return up to `k` names with score ≥ `min_hits`, best first, ties by name; `sys.*` tools excluded); `ReActAgent(..., refusal_check: bool = False, refusal_min_hits: int = 1)`; trace key `"refusal_check": [names]`.

Behaviour: only when the model's **first** output has no calls and no parse errors, and `matching_tools` is non-empty: append the plain answer as an assistant message and a user message `"Check again: the installed tool(s) {', '.join(names)} may handle this request. If one does, call it. If the request is out of scope or the user cancelled it, reply in one short sentence without calling any tool."`, then continue the loop. At most once per sample. Not used when rendering SFT records (ReplayLLM has no second answer).

- [ ] **Step 1: Failing tests:**
  - `test_parking_request_matches_end_session`: spec `parking.end_session` with description "Kết thúc phiên gửi xe" and request "kết thúc phiên gửi xe ở toà nhà" → `["parking.end_session"]`.
  - `test_unrelated_request_matches_nothing`.
  - `test_refusal_check_fires_once`: FakeLLM refuses twice → two generations, no calls, `trace["stop"] == "no_call"`.
  - `test_no_refusal_check_after_a_call`: a call then a plain answer → no extra user message.
  - `test_refusal_check_off_by_default`.
- [ ] **Step 2:** run → FAIL. **Step 3:** implement. **Step 4:** `pytest` → PASS.
- [ ] **Step 5:** GPU runs on dev + holdout ASR with `refusal_min_hits` 1 and 2 (presets `sft-v2-refs-refuse1/2`). Gate: default gate **and** the `empty` column (out-of-scope accuracy) not lower than the baseline on either subset. Check public_0590, 0609, 0662, 0674, 0265, 0194, 0466, 0598, 0654 on a public run with the flag.
- [ ] **Step 6:** Commit `Take a second look before refusing` on `feat/agent-guards`, PR.

### Task 7: Closing text never sent as content (finding H)

**Files:**
- Modify: `src/lifemate/agents/react.py`, `src/lifemate/agents/prompts.py`, `src/lifemate/sft/replay.py`, `configs/presets.json`
- Test: `tests/test_react_agent.py`

**Interfaces:**
- Consumes: `FINAL_TEXT`, `REFUSAL_TEXT`, moved from `sft/replay.py` to `agents/prompts.py` (`replay.py` imports them from there; `react.py` cannot import `sft.replay`, which imports `react`), `ask_guard.fold`.
- Produces: `ReActAgent(..., leak_guard: bool = False)`; rejection text `"Not sent: {param} repeats your closing reply. Write the content the user asked for."`

- [ ] **Step 1: Failing tests:** `test_closing_text_as_content_is_rejected` (`sms.send {"content": "Đã xong."}` with a fuzzy `content` spec → not called, rejection in trace), `test_real_content_is_sent`, `test_leak_guard_off_by_default`.
- [ ] **Step 2:** run → FAIL. **Step 3:** implement (folded equality with either constant, fuzzy string params only). **Step 4:** `pytest` → PASS.
- [ ] **Step 5:** No dedicated GPU run: it rides on the combined preset of Task 10. Check public_0327, 0441, 0578 on the next public run.
- [ ] **Step 6:** Commit `Reject the closing text as message content` on `feat/agent-guards`, PR (together with Tasks 5 and 6 is fine).

### Task 8: Contact name fallback (K4, gated by a count first)

**Files:**
- Modify: `src/lifemate/agents/react.py`, `configs/presets.json`
- Test: `tests/test_react_agent.py`

**Interfaces:**
- Produces: `ReActAgent(..., contact_fallback: bool = False)`; trace key `"contact_fallback": query`.

Behaviour: when a sent `contacts.search {"query": q}` returns empty and `q` has ≥ 2 words, the agent itself sends `contacts.search {"query": <last word of q>}` once per sample and gives the model both results (the second prefixed with `"Fallback search by the last word of the name:"`).

- [ ] **Step 1: Count first** (scratch script, not committed): in the dev + holdout ASR baseline traces, samples where a `contacts.search` returned empty while gold has a `contacts.search`. Fewer than 5 → write the count into the PR of Task 5 and skip the rest of this task (the empty-result hint already covers it).
- [ ] **Step 2: Failing tests:** `test_empty_contact_search_retries_by_last_word`, `test_one_word_query_has_no_fallback`, `test_fallback_runs_once`, `test_contact_fallback_off_by_default`.
- [ ] **Step 3:** implement. **Step 4:** `pytest` → PASS.
- [ ] **Step 5:** GPU run on dev + holdout ASR (preset `sft-v2-refs-cf`); default gate (the extra read lowers precision on samples it does not save).
- [ ] **Step 6:** Commit `Retry an empty contact search by the given name`, PR.

### Task 9: Prompt vote at the decision step (self-consistency without sampling)

Greedy decoding stays: the K proposals come from K deterministic prompts that differ only in their few-shot examples, all in the format SFT was trained on (4 examples). Only the winning proposal is sent, so the vote adds no scored call. It targets ask/act errors, which go both ways (16 vs 20 on holdout ASR); a vote does not push one way, which is where the hard ask rule failed.

**Files:**
- Modify: `src/lifemate/agents/fewshot.py` (`select(..., offset)`), `src/lifemate/agents/react.py`, `configs/presets.json`
- Test: `tests/test_fewshot.py`, `tests/test_react_agent.py`

**Interfaces:**
- Consumes: `ExampleBank.select`, `render_examples`, `build_messages`; the agent options of Tasks 2–8 apply to the winning call as to any other.
- Produces:
  - `ExampleBank.select(request, installed_apps, k, exclude_id=None, offset: int = 0) -> list[dict]`: the ranked list sliced `[offset:offset + k]`.
  - `vote_key(call: dict, spec: dict | None) -> tuple`: `(name, sorted arguments without x-match: fuzzy keys)`; for `sys.ask_missing_info`, `(name, tool_name, sorted params)`. JSON-dumped values so dicts and lists compare.
  - `ReActAgent(..., vote_prompts: int = 0, vote_apply: bool = True)`: `0` = off. With `K = vote_prompts`, variant `i` uses examples `select(..., k=self.examples, offset=i * self.examples)`; variant 0 is today's prompt.
  - Trace key `"vote": {"step": int, "keys": [str, ...], "winner": int, "applied": bool}`.

Behaviour:
- The vote runs once per sample, at the first step whose variant-0 output contains a write call (`spec["kind"] == "write"`) or `sys.ask_missing_info`. Reads are never voted on.
- The other variants regenerate that step on the same conversation, with only the system message swapped.
- Each output's key is the tuple of `vote_key` over its parsed calls (unparseable output → key `("invalid",)`).
- The winner is the key with the most votes. It needs at least 2 votes, and variant 0 wins every tie. The output sent is that of the lowest-index variant with the winning key, so its free text is used unchanged.
- `vote_apply=False` records the vote but always sends variant 0 (a disagreement measurement that leaves the submission unchanged).
- Skipped (trace `"vote": null`) when the remaining time is below `vote_prompts * min_step_s`.

- [ ] **Step 1: Failing tests:**
  - `test_select_offset_returns_the_next_examples` (`tests/test_fewshot.py`): `select(..., k=2, offset=2)` returns ranks 3 and 4 of `select(..., k=4)`.
  - `test_vote_majority_replaces_the_pick`: variant 0 proposes `alarm.delete {"alarm_id": "a1"}`, variants 1 and 2 propose `{"alarm_id": "a2"}` → `env.called[-1] == ("alarm.delete", {"alarm_id": "a2"})`, `trace["vote"]["winner"] == 1`.
  - `test_vote_tie_keeps_the_first_prompt`: three different proposals → variant 0 is sent.
  - `test_vote_ignores_free_text_differences`: same tool and ids, different fuzzy `title` → one key, variant 0's title sent.
  - `test_ask_and_act_are_different_keys`: two variants ask, one writes → the ask is sent and `trace["stop"] == "asked"`.
  - `test_reads_are_not_voted`: a read step makes one generate call.
  - `test_vote_runs_once_per_sample`: two write steps → `K + 1` generate calls for them, not `2K`.
  - `test_vote_record_only_sends_the_original`: `vote_apply=False` with a majority against variant 0 → variant 0 sent, `trace["vote"]["applied"] is False`.
  - `test_vote_skipped_when_time_is_short`, `test_vote_off_by_default`.
- [ ] **Step 2:** `pytest tests/test_fewshot.py tests/test_react_agent.py -v` → FAIL.
- [ ] **Step 3:** Implement `select(offset)`, `vote_key` and the vote in `react.py` (record `vote_prompts` and `vote_apply` in `self.info`). StubBank in the tests gains the `offset` argument.
- [ ] **Step 4:** `pytest` → PASS.
- [ ] **Step 5: Disagreement count.** Preset `sft-v2-refs-vote3-dry` (`vote_prompts: 3, vote_apply: false`) on holdout ASR. The submission must equal the baseline's except where the other kwargs differ. Count the samples whose vote keys are not all equal and those where the majority differs from variant 0. If the majority differs from variant 0 on fewer than 3% of samples (about 9 on holdout), stop: PR with the counts, the vote does not ship.
- [ ] **Step 6: Vote run.** Preset `sft-v2-refs-vote3` (`vote_prompts: 3`) on dev and holdout ASR, combined with the kwargs that already passed in Tasks 2–8. Gate: the default gate, the `ask` column not lower on either subset, and timing on molab mean ≤ 5 s and max ≤ 15 s per sample (the A30 is slower; 30 s is the hard cap).
- [ ] **Step 7:** Commit `Vote over few-shot prompts at the decision step` on `exp/prompt-vote`, PR with both runs and the disagreement counts.

### Task 10: Combined preset and hand-off

**Files:**
- Modify: `configs/presets.json`, `RESULTS.md`

**Interfaces:**
- Consumes: every kwarg of Tasks 2–9 that passed its gate.
- Produces: presets `sft-v2-guards` (sft-v2 + check_refs + the passing kwargs, for dev/holdout) and `sft-final-guards` (the same kwargs on sft-final, for public), and the list the sprint plan's final retrain and Docker task use.

- [ ] **Step 1:** `sft-v2-guards` on dev and holdout ASR against the sft-v2 + check_refs baselines. The kwargs passed one by one, so this run checks that they do not cancel each other: the combined holdout diff must be at least the largest single diff. If not, drop kwargs one at a time, smallest single gain first, until it is.
- [ ] **Step 2:** `sft-final-guards` on public_test, one submission. Keep it only if it beats 53.58; the leaderboard score goes into its `RESULTS.md` row.
- [ ] **Step 3:** Hand-off: list the passing kwargs, and `--date-hints` for the records of `sft-final2`, in the PR description and in this plan's checkbox, so the sprint plan's final retrain and Docker task pick them up. The sprint plan files are not edited. The vote's timing (Task 9 Step 6) goes with it for the A30 rehearsal.
- [ ] **Step 4:** PR `Combine the gated agent options` with both compare lines and the public score.
