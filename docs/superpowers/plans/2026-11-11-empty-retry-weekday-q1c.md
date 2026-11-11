# Empty-Read Retry, Weekday Fix and sft-q1c Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the points sft-q1b loses after an empty read and on "thứ X tuần sau" dates, and test one retrain (`sft-q1c`, examples of the gold apps left out for half of the records), in time to ship before 16 Nov 2026.

**Architecture:** Two opt-in `ReActAgent` options in the style of `date_fix` / `id_fix`. `weekday_fix` rewrites the date of a call from the request's single weekday phrase (a narrower, safer form of `date_fix`, scored offline with `lifemate.sim`). `empty_retry` re-sends a read that came back empty without its optional filters and shows the model that result; it needs the model, so it is gated on molab. The retrain changes one field of the sft-q1b recipe and is gated on the same holdout and dev runs.

**Tech Stack:** Python 3.11, pytest, existing `lifemate.agents` / `lifemate.sim` / `configs/presets.json`, `notebooks/molab_run.py` and `notebooks/molab_train.py` on molab.

**Spec:** the loss analysis of 11 Nov 2026 (section "Evidence" below). It builds on `docs/superpowers/plans/2026-11-10-sft-q1b-loss-fixes.md` (PR #79) and on the code of PR #80 (`id_fix`, `spoken_email`).

## Evidence

Runs: `sft-q1b-refs` on gold text (holdout 59.47, dev 80.19), `sft-q1b-refs-fixes` with ASR (holdout 44.83, dev 75.04), `sft-q1b-refs` on public_test (61.05). Counts come from the recorded traces; no GPU run was made for this plan.

- **Empty reads.** A gold read returns an empty list in 4 of 1,545 train samples (10 of 3,661 history calls). The agent gets one in 40 of 634 public samples (24 then ask, 10 stop there), 27 of 297 holdout and 13 of 300 dev samples (ASR runs). Those samples average about 0.1 on holdout and 0.06 on dev. In 19 public, 16 holdout and 8 dev samples the first empty read carries an optional filter (`contacts.search {"query": "Sếp Nguyễn"}` when the contact is "Sếp Dũng", `loyalty.get_points {"program": "Cine Việt"}`, `event_tickets.search {"query": ...}`); the rest are reads whose arguments are all required (`flights.search` with a wrong date).
- **`empty_hint` is a different thing.** It only appended a text hint to the tool message and lost on sft-v2 (holdout -0.84, 0 better / 6 worse). An automatic retry has not been tested.
- **Weekday dates.** On public, sft-q1b writes the calendar date of "thứ X tuần sau" in 2 of 9 samples; 4 of the 7 wrong ones end in an empty flight or train search and an ask. Train gold follows the Monday-week rule of `agents/date_fix.py` in 8 of 9 samples. Dev and holdout hold 3 such samples, all at score 0, so the train gate cannot show this fix; the public leaderboard can.
- **Prototype of `weekday_fix`** (rules of Task 1, on recorded calls): 0 calls changed on the gold calls of all 1,545 train samples (`date_fix` changes 2, both from "hôm nay" / "sáng mai"); on the sft-q1b train runs 3 calls changed, all to the gold date (train_0698, train_1356, train_1750); on public 9 samples changed (public_0081, 0187, 0199, 0213, 0287, 0296, 0331, 0355, 0553).
- **What rules cannot fix.** Ceilings on gold-text holdout / dev: ask/act +9.2 / +3.1, read arguments +5.5 / +2.1, dates +3.9 / +2.8, free text +3.6 / +1.7, numbers +3.3 / +1.2. Time arithmetic (16:00 minus 27 min written as 16:33) and condition branches are computed wrongly by the model.
- **Training.** 272 of 634 public samples call an app that train does not have, and none calls a holdout app. Every pool record of sft-q1b shows examples of its own gold apps, a condition those 272 samples never meet. Hiding the gold apps' examples for half of the samples is the only data change with a public gain so far (Qwen3, all train: 56.01 without, 57.5 with). It has never been trained on the pool, so it has no holdout gate yet. An all-train retrain alone gave +0.13 on public (57.37 to 57.5).

## Global Constraints

- No paid or external services, no LLM APIs; nothing reads `<split>_environment/` or `Task.label` (team rules, CLAUDE.md).
- Every new option is off by default, so `configs/shipped.json` and SFT record rendering (`sft.replay` builds `ReActAgent` with defaults) do not change.
- No app or tool name decides anything; only the schema (`kind`, `required`) and the values in the request and tool results do.
- One deliberate exception to "a rewrite never adds a call": `empty_retry` sends one extra read. It only fires after a read that found nothing, where the sample averages 0.1, and only for a read tool.
- Repo text is English; Vietnamese only as test data. Files keep LF endings.
- Git: never commit to `main`. Code on `feat/empty-retry`, cut from `origin/feat/loss-fixes` while PR #80 is open (from `origin/main` once it is merged). This plan goes on `docs/empty-retry-plan` with its own PR. Result rows go on a `docs/results-*` branch.
- Run tests with `PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python -m pytest` (Windows; `.venv/bin/python` elsewhere).
- Gate for a rule (runbook): holdout diff >= +0.5 with the 95% CI above 0, dev diff >= -0.3 (`python -m lifemate.compare <baseline> <new>`).
- Steps marked **HUMAN (molab)** are run by the repo owner. The implementer prepares everything they need, stops, and continues when the run folders are back.

## Review Focus

- A read that is empty for a real reason, filtered only by a date (`calendar.list_events` for one day), must not be retried. Test in Task 2.
- A write tool, `sys.ask_missing_info`, or a relaxed call that was already sent must never trigger a second call. Test in Task 2.
- An id that appears only in the retry's result must pass `check_refs` and feed `id_fix`. Test in Task 2.
- "thứ hai giờ chiều" (a misheard clock time) and ordinals ("quán thứ hai") must not be read as a weekday. Test in Task 1.
- A call whose start and end fall on different days (parking for three days) must stay as sent. Test in Task 1.

## Schedule (GMT+7)

| When | Who | What |
|---|---|---|
| Wed 11 Nov, evening | owner | Task 4 Step 1: start the `sft-q1c` training (about 10 h, needs no new code) |
| Wed 11 Nov | implementer | Tasks 1 to 3, PR |
| Thu 12 Nov | owner | Task 5 runs G1, G2 (rules) and Task 4 runs G3, G4 (`sft-q1c`), then the public runs of Task 6 |
| Thu 12 Nov | implementer | score the runs, apply the decision rules, log `RESULTS.md` rows |
| Fri 13 Nov | both | Task 7: ship PR, image build, Test Solution (one run per day, until 15 Nov 23:59) |
| Sat 14 to Sun 15 Nov | both | buffer: a second Test Solution run if the first fails |
| Mon 16 Nov 23:59 | owner | submit link + MD5 |

---

### Task 1: `weekday_fix` — the date of a weekday phrase

**Files:**
- Modify: `src/lifemate/agents/date_fix.py`
- Modify: `src/lifemate/agents/react.py` (option, docstring, info, call site after the `date_fix` block)
- Modify: `src/lifemate/sim.py` (factory `weekday_fix`, registered in `REWRITES`)
- Test: `tests/test_date_fix.py`, `tests/test_react_agent.py`, `tests/test_sim.py`

**Interfaces:**
- Produces: `resolve_date(request: str, current_time: str, weekdays_only: bool = False) -> str | None`
- Produces: `fix_call_dates(call, spec, request, current_time, weekdays_only: bool = False) -> dict`
- Produces: `fix_calls(calls, specs, request, current_time, weekdays_only: bool = False) -> list[dict]`
- Produces: `ReActAgent(..., weekday_fix: bool = False)`, `info["weekday_fix"]`, trace key `weekday_fixes` = list of `{"from": args, "to": args}`
- Produces: `sim.weekday_fix(times: dict[str, str] | None = None, data_root: Path | None = None) -> Rewrite`, name `"weekday_fix"` in `REWRITES`

With `weekdays_only=False` every function behaves exactly as today; the existing tests stay unchanged and green. With `weekdays_only=True`:

1. A day phrase (`hôm nay`, `ngày mai`, `ngày mốt`, `sáng/trưa/chiều/tối mai`) resolves to `None`. It still counts as a phrase, so a request with a day phrase and a weekday phrase stays `None`.
2. A weekday phrase directly followed by `giờ`, `rưỡi`, `phút` or `tiếng` resolves to `None` (ASR hears "lúc hai giờ" as "thứ hai giờ").
3. When the phrase has a qualifier (`tuần sau`, `tuần này`, `này`), `hôm đó` and `hôm ấy` after the phrase are not another time reference. A bare weekday keeps today's rule.
4. `fix_call_dates` fixes every date-valued string argument when they all share one date part (a `start` and `end` on the same day), instead of requiring exactly one. Arguments with different date parts, or any that names a past range, leave the call alone.

- [ ] **Step 1: Write the failing tests** in `tests/test_date_fix.py`

```python
THU = "2026-09-24T09:00:00+07:00"   # a Thursday


def test_weekday_next_week():
    assert resolve_date("Đặt cho tôi một ghế chuyến RV21 đi Đà Nẵng thứ bảy tuần sau", THU, weekdays_only=True) == "2026-10-03"


def test_same_day_reference_after_a_qualified_weekday_is_allowed():
    text = "Thứ tư tuần sau anh đi Hải Phòng, giữ một chỗ gửi xe từ năm giờ sáng hôm đó"
    assert resolve_date(text, THU, weekdays_only=True) == "2026-09-30"
    assert resolve_date(text, THU) is None            # the old rule is unchanged


def test_same_day_reference_after_a_bare_weekday_still_blocks():
    assert resolve_date("Thứ hai họp, nhắc anh sáng hôm đó", THU, weekdays_only=True) is None


def test_weekday_followed_by_a_clock_word_is_not_a_weekday():
    assert resolve_date("ghi vào lịch thứ hai giờ chiều", "2026-10-05T09:00:00+07:00", weekdays_only=True) is None


def test_day_phrases_are_left_to_the_model():
    assert resolve_date("hủy hẹn sáng mai", "2026-10-20T21:10:00+07:00", weekdays_only=True) is None
    assert resolve_date("hủy hẹn sáng mai", "2026-10-20T21:10:00+07:00") == "2026-10-21"


def test_ordinal_is_not_a_weekday():
    assert resolve_date("mở quán thứ hai trong danh sách", THU, weekdays_only=True) is None


EVENT = "tạo lịch Bàn giao lúc mười giờ sáng thứ tư tuần sau"
NOW = "2026-10-01T09:00:00+07:00"


def test_start_and_end_on_one_day_are_fixed_together():
    call = {"name": "calendar.create_event", "arguments": {
        "title": "Bàn giao", "start": "2026-10-08T10:00:00+07:00", "end": "2026-10-08T11:00:00+07:00"}}
    fixed = fix_call_dates(call, None, EVENT, NOW, weekdays_only=True)
    assert fixed["arguments"]["start"] == "2026-10-07T10:00:00+07:00"
    assert fixed["arguments"]["end"] == "2026-10-07T11:00:00+07:00"
    assert fix_call_dates(call, None, EVENT, NOW) is call          # the old rule needs a single dated argument


def test_dates_on_different_days_are_left_alone():
    call = {"name": "parking.reserve", "arguments": {"start": "2026-10-08T05:00:00+07:00", "end": "2026-10-11T05:00:00+07:00"}}
    assert fix_call_dates(call, None, EVENT, NOW, weekdays_only=True) is call


def test_a_right_date_returns_the_same_call():
    call = {"name": "tasks.add", "arguments": {"title": "x", "due": "2026-10-07T10:00:00+07:00"}}
    assert fix_call_dates(call, None, EVENT, NOW, weekdays_only=True) is call
```

- [ ] **Step 2: Run them**: `pytest tests/test_date_fix.py -q`. Expected: the new tests fail with `TypeError: ... unexpected keyword argument 'weekdays_only'`.
- [ ] **Step 3: Implement** the three signatures in `date_fix.py`. `fix_calls` and `fix_call_dates` pass `weekdays_only` down. Update the module docstring with the four rules.
- [ ] **Step 4: Run** `pytest tests/test_date_fix.py -q`. Expected: all pass, old tests included.
- [ ] **Step 5: Agent tests** in `tests/test_react_agent.py`, next to the `date_fix` tests and in their style:
  - `test_weekday_fix_corrects_the_date_before_validation`: request `EVENT`, `current_time` `NOW`, the model calls `tasks.add` with `due` `2026-10-08T10:00:00+07:00`; `env.called` holds `due` `2026-10-07T10:00:00+07:00`, the assistant message the model reads back holds the same, and `agent.last_trace["weekday_fixes"]` has one entry.
  - `test_weekday_fix_is_off_by_default`: the same run without the option sends `2026-10-08`.
  - `test_weekday_fix_leaves_day_phrases_alone`: request "nhắc tôi sáng mai", option on, the call is sent as the model wrote it.
  - Extend `test_agent_info_records_prompt_and_options` to expect `weekday_fix` in `info`.
- [ ] **Step 6: Implement the option.** After the `date_fix` block of `run`: `fix_calls(calls, specs, task.request, task.context["current_time"], weekdays_only=True)`, recorded under `weekday_fixes`. Docstring line: what it does, and that it never sends a call.
- [ ] **Step 7: Simulator.** `sim.weekday_fix` mirrors `sim.date_fix` but applies `fix_calls([call], ..., weekdays_only=True)` to each recorded call (the agent fixes one model turn at a time, and a trace does not keep turn boundaries). Tests in `tests/test_sim.py` in the style of the `date_fix` ones: one trace whose call gets the weekday date, one that returns `None`.
- [ ] **Step 8: Verify on recorded runs** (needs `data/train`; run `lifemate.evaluate` first if a run has no `per_sample.jsonl`):

```bash
python -m lifemate.sim weekday_fix runs/check_oracle
python -m lifemate.sim weekday_fix runs/20261110-161452_dc97998_train_sft-q1b-refs-fixes_pho-large-ft-b_holdout
python -m lifemate.sim weekday_fix runs/20261110-163834_dc97998_train_sft-q1b-refs-fixes_pho-large-ft-b_dev
python -m lifemate.sim weekday_fix runs/20261110-153836_dc97998_train_sft-q1b-refs_gold_holdout
```

  Expected: oracle fires on 0 samples and stays at 100.0; ASR holdout fires on train_1750; ASR dev on none; gold holdout on train_0698 and train_1356; no sample gets worse. If the oracle run changes any call, stop and report the sample ids before going on.
- [ ] **Step 9: Run** `pytest -q`, then commit: `Add weekday_fix: resolve the date of a weekday phrase by calendar maths`.

### Task 2: `empty_retry` — re-read without optional filters

**Files:**
- Create: `src/lifemate/agents/empty_retry.py`
- Modify: `src/lifemate/agents/react.py` (option, docstring, info, call loop)
- Test: `tests/test_empty_retry.py`, `tests/test_react_agent.py`

**Interfaces:**
- Produces: `relaxed_call(call: dict, spec: dict | None) -> dict | None`
- Produces: `ReActAgent(..., empty_retry: bool = False)`, `info["empty_retry"]`, trace key `empty_retries` = list of `{"name": str, "from": args, "to": args, "found": bool}`
- Consumes: `is_empty_result` (already in `react.py`), `env.validate`, `scalar_strings`, `entity_names`

`relaxed_call` returns `None` unless `spec["kind"] == "read"`. It drops every argument that is not in `parameters.required` and is not a date- or time-valued string (`^\d{4}-\d{2}-\d{2}` or `^\d{2}:\d{2}`). Nothing dropped: `None`. Otherwise a copy of the call with the remaining arguments.

Agent behaviour, in the call loop right after `result = env.call(...)` and only when the call is not `sys.ask_missing_info` and `is_empty_result(result)`:

1. `retry = relaxed_call(call, specs.get(call["name"]))`. Skip when it is `None`, when its key is already in `done`, or when `env.validate` rejects it.
2. Send it with `env.call`, add its key to `done`, pass its result through `scalar_strings` (and `entity_names` when `id_fix` is on), and record it in `trace["empty_retries"]`.
3. Rewrite the transcript so the model reads the relaxed call and its result: the arguments of this call's entry in the assistant message already appended for the turn become the relaxed arguments, and the tool message carries the retry's result. This holds whether or not the retry found anything, so the transcript always shows one call and its real result, as in training.
4. `empty_hint` text is appended only when the result the model sees is still empty.

- [ ] **Step 1: Write the failing unit tests** in `tests/test_empty_retry.py`

```python
from lifemate.agents.empty_retry import relaxed_call

SEARCH = {"name": "contacts.search", "kind": "read",
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": []}}
FLIGHTS = {"name": "flights.search", "kind": "read", "parameters": {"type": "object", "required": ["origin", "destination", "date"],
           "properties": {"origin": {}, "destination": {}, "date": {}, "passengers": {"type": "integer"}}}}
EVENTS = {"name": "calendar.list_events", "kind": "read", "parameters": {"type": "object", "required": [],
          "properties": {"date_from": {}, "date_to": {}, "query": {}}}}
PLACE = {"name": "maps.search_place", "kind": "read",
         "parameters": {"type": "object", "properties": {"query": {}, "near": {}}, "required": ["query"]}}
SEND = {"name": "sms.send", "kind": "write", "parameters": {"type": "object", "properties": {"note": {}}, "required": []}}


def test_optional_filter_is_dropped():
    assert relaxed_call({"name": "contacts.search", "arguments": {"query": "Sếp Nguyễn"}}, SEARCH) == {
        "name": "contacts.search", "arguments": {}}


def test_required_arguments_stay():
    assert relaxed_call({"name": "maps.search_place", "arguments": {"query": "hiệu thuốc", "near": "Hàng Bài"}}, PLACE) == {
        "name": "maps.search_place", "arguments": {"query": "hiệu thuốc"}}


def test_optional_number_is_dropped():
    call = {"name": "flights.search", "arguments": {"origin": "HAN", "destination": "DAD", "date": "2026-10-03", "passengers": 2}}
    assert relaxed_call(call, FLIGHTS)["arguments"] == {"origin": "HAN", "destination": "DAD", "date": "2026-10-03"}


def test_nothing_to_drop_is_none():
    assert relaxed_call({"name": "flights.search", "arguments": {"origin": "HAN", "destination": "DAD", "date": "2026-10-03"}}, FLIGHTS) is None
    assert relaxed_call({"name": "contacts.search", "arguments": {}}, SEARCH) is None


def test_a_date_filter_is_kept():
    day = {"date_from": "2026-11-22", "date_to": "2026-11-22"}
    assert relaxed_call({"name": "calendar.list_events", "arguments": day}, EVENTS) is None
    assert relaxed_call({"name": "calendar.list_events", "arguments": {**day, "query": "họp"}}, EVENTS)["arguments"] == day


def test_only_read_tools():
    assert relaxed_call({"name": "sms.send", "arguments": {"note": "x"}}, SEND) is None
    assert relaxed_call({"name": "contacts.search", "arguments": {"query": "x"}}, None) is None


def test_the_call_is_not_mutated():
    call = {"name": "contacts.search", "arguments": {"query": "Lan"}}
    relaxed_call(call, SEARCH)
    assert call["arguments"] == {"query": "Lan"}
```

- [ ] **Step 2: Run** `pytest tests/test_empty_retry.py -q`. Expected: `ModuleNotFoundError: lifemate.agents.empty_retry`.
- [ ] **Step 3: Implement `relaxed_call`**, with a module docstring that gives the rule and the evidence line (4 of 1,545 gold samples against 40 of 634 public samples).
- [ ] **Step 4: Run** `pytest tests/test_empty_retry.py -q`. Expected: pass.
- [ ] **Step 5: Agent tests** in `tests/test_react_agent.py`. They need an env whose result depends on the arguments: add a small `ArgEnv(FakeEnv)` in the test file whose `call` looks up `(name, json.dumps(args, sort_keys=True))` and falls back to `{"ok": True}`. The task carries the specs: `dataclasses.replace(TASK, tools=[SEARCH, ...])` with the specs of Step 1 plus a `description`.
  - `test_empty_retry_resends_the_read_without_its_filter`: the model calls `contacts.search {"query": "Sếp Nguyễn"}` (result `{"ok": True, "contacts": []}`), the env answers `{}` with `{"ok": True, "contacts": [{"id": "c_01", "name": "Sếp Dũng"}]}`. Assert `env.called == [("contacts.search", {"query": "Sếp Nguyễn"}), ("contacts.search", {})]`; in the messages of the second `generate`, the assistant turn's tool call has arguments `{}` and the last tool message contains `Sếp Dũng`; `agent.last_trace["empty_retries"] == [{"name": "contacts.search", "from": {"query": "Sếp Nguyễn"}, "to": {}, "found": True}]`.
  - `test_empty_retry_is_off_by_default`: one call sent, the model reads the empty result.
  - `test_empty_retry_skips_a_read_with_only_required_arguments`: `flights.search`, empty, one call sent.
  - `test_empty_retry_skips_a_dated_read`: `calendar.list_events` for one day, empty, one call sent.
  - `test_empty_retry_never_follows_a_write_or_an_ask`: a write whose result has an empty list, and `sys.ask_missing_info`: no extra call.
  - `test_empty_retry_does_not_repeat_a_call_already_sent`: the model first calls `contacts.search {}`, then `contacts.search {"query": "Lan"}` (empty): two calls in total.
  - `test_empty_retry_skips_a_relaxed_call_the_env_rejects`: give `ArgEnv` a `validate` that rejects empty arguments; one call sent.
  - `test_empty_retry_when_both_are_empty`: both results empty; two calls sent, the transcript shows the relaxed call with the empty result, `found` is `False`.
  - `test_ids_from_the_retry_pass_the_ref_check`: `check_refs=True`, the model then calls a write tool whose `x-ref` argument is `c_01`; the write is sent.
  - Extend `test_agent_info_records_prompt_and_options` to expect `empty_retry` in `info`.
- [ ] **Step 6: Run them.** Expected: fail with `unexpected keyword argument 'empty_retry'`.
- [ ] **Step 7: Implement the option** in `react.py` as specified above. Keep a reference to the assistant message appended for the turn so the entry of the retried call can be rewritten; iterate with `enumerate(calls)` so the index is at hand. Docstring: the rule, that it is the only option that sends a call the model did not write, and why (`empty_hint` only changed the message).
- [ ] **Step 8: Run** `pytest -q`. Expected: all pass.
- [ ] **Step 9: Commit**: `Add empty_retry: re-read without optional filters after an empty read`.

### Task 3: Presets and runbook

**Files:**
- Modify: `configs/presets.json`
- Modify: `docs/molab-runbook.md` (new section "C3. Empty retry, weekday fix and sft-q1c" with the run fields of Tasks 4 to 6)
- Modify: `CLAUDE.md` (one clause each for `weekday_fix` and `empty_retry`, only if the architecture section already lists agent options; otherwise leave it)
- Test: `tests/test_presets.py`

All presets use `"agent": "react"`, `"examples": 4`, `"check_refs": true`, `"id_fix": true`, `"spoken_email": true`:

| Preset | Model | Extra options |
|---|---|---|
| `sft-q1b-refs-fixes-retry` | `${HF_MODEL_OWNER}/lifemate-sft-q1b` | `empty_retry` |
| `sft-q1b-refs-fixes-wd` | `${HF_MODEL_OWNER}/lifemate-sft-q1b` | `weekday_fix` |
| `sft-q1b-refs-fixes-all` | `${HF_MODEL_OWNER}/lifemate-sft-q1b` | `empty_retry`, `weekday_fix` |
| `sft-q1c-refs-fixes` | `${HF_MODEL_OWNER}/lifemate-sft-q1c` | none |
| `sft-q1c-refs-fixes-all` | `${HF_MODEL_OWNER}/lifemate-sft-q1c` | `empty_retry`, `weekday_fix` |

- [ ] **Step 1:** Add a test that resolves each of the five presets with `{"HF_MODEL_OWNER": "me"}` and checks its option flags, in the style of the existing preset tests. Run it; expected: `KeyError`.
- [ ] **Step 2:** Add the presets. Run `pytest tests/test_presets.py -q`; expected: pass.
- [ ] **Step 3:** Write the runbook section from Tasks 4 to 6 (fields, baselines, decision rules).
- [ ] **Step 4:** `pytest -q`, commit (`Add presets for empty_retry, weekday_fix and sft-q1c`), push, open the PR into `main`. The description says it is stacked on PR #80, lists the Task 1 Step 8 simulator lines, and states that `empty_retry` is not yet measured.

### Task 4: `sft-q1c` — the sft-q1b recipe with the gold apps left out of half the examples

No code. It can start before Tasks 1 to 3 are done.

- [ ] **Step 1: HUMAN (molab), `notebooks/molab_train.py`.** Copy every field of the sft-q1b run (its `run_config.json` is next to the weights in `TripleTrouble/lifemate-sft-q1b`) and change three: run name, HF repo and the leave-out share.

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

  Check before leaving it: the records name reads `gold-pool-m1-asr-pho-large-fp16-lao50`, the log's "trainable params" line matches the sft-q1b log, and the step count is 356. For a copied field, `run_config.json` wins when it differs from this table; report the difference. The three changed fields (run name, HF repo, leave-out share `50`) always come from this table, whatever `run_config.json` says.
- [ ] **Step 2: HUMAN (molab), merge** the adapter as for sft-q1b, and note the HF commit sha of the merged model.
- [ ] **Step 3: HUMAN (molab), gate runs** with `notebooks/molab_run.py`, branch `feat/empty-retry`, split `train`, ASR preset `pho-large-ft-b`: G3 = preset `sft-q1c-refs-fixes`, subset `holdout`; G4 = the same, subset `dev`.
- [ ] **Step 4: Score** each run (`python -m lifemate.evaluate runs/<run>`) and compare with the sft-q1b baselines:

```bash
python -m lifemate.compare runs/20261110-161452_dc97998_train_sft-q1b-refs-fixes_pho-large-ft-b_holdout runs/<G3>
python -m lifemate.compare runs/20261110-163834_dc97998_train_sft-q1b-refs-fixes_pho-large-ft-b_dev runs/<G4>
```

  Also report the pooled difference over the 597 samples, the way the sft-q1b row in `RESULTS.md` does, and `mean_seconds` / `max_seconds`.
- [ ] **Step 5: Decide.**
  - Pooled difference below 0: sft-q1c is closed; sft-q1b stays the model.
  - Pooled difference at or above 0: sft-q1c takes part in the public runs of Task 6. The leaderboard decides between the two models, as it did for sft-q1b (which failed the holdout gate and won on public).
  - Holdout gate passed outright **and** the GPU is free before Thu 12 Nov 18:00: one optional all-train run `sft-q1c-final` (same fields, *Train records* = `all train (final retrain only)`, about 16 h). It has no local score; it ships only if its public score beats the pool model's. Past evidence for an all-train retrain is +0.13, so skip it without regret when time is short.
- [ ] **Step 6:** Log the rows in `RESULTS.md` on a `docs/results-*` branch.

### Task 5: Gate `empty_retry`

- [ ] **Step 1: HUMAN (molab).** Branch `feat/empty-retry`, split `train`, ASR preset `pho-large-ft-b`, preset `sft-q1b-refs-fixes-retry`: G1 = subset `holdout`, G2 = subset `dev`.
- [ ] **Step 2: Score and compare** against the same two baselines as Task 4 Step 4.
- [ ] **Step 3: Report from `trace.jsonl`:** samples with `agent.empty_retries`, how many have `found` true, the score of those samples before and after (join with the baseline `per_sample.jsonl` by id), samples that end with `sys.ask_missing_info` before and after, `max_seconds`.
- [ ] **Step 4: Decide.** Gate passed: `empty_retry` stays in the `-all` presets. Holdout or dev difference below the gate but not negative: it stays only if the public runs of Task 6 support it. Negative on either split: drop it from the shipped preset and say which samples got worse.
- [ ] **Step 5:** `RESULTS.md` rows on the results branch.

`weekday_fix` has no molab gate on train: Task 1 Step 8 is its train evidence (three samples). Public decides.

### Task 6: Public runs

- [ ] **Step 1: HUMAN (molab).** Branch `feat/empty-retry`, split `public_test`, ASR preset `pho-large-ft-all-beam5`, extra infer args `--time-field`. Runs, in this order of priority if leaderboard submissions are limited:
  1. `sft-q1b-refs-fixes-all` (use `sft-q1b-refs-fixes-wd` instead if Task 5 dropped `empty_retry`)
  2. `sft-q1c-refs-fixes-all` (only if Task 4 Step 5 kept sft-q1c; same rule options as run 1)
  3. `sft-q1b-refs-fixes` (only if run 1 scores below 61.05, to tell the rules from the model)
- [ ] **Step 2:** Validate each `submission.jsonl` (`cd data && PYTHONPATH=public_test python -m zaic_tools --split public_test --root . --validate ../runs/<run>/submission.jsonl`), check `over_time_limit` is 0, submit, and write the leaderboard score into the `RESULTS.md` row by hand.
- [ ] **Step 3: Report** from the run's trace: samples with `weekday_fixes` (expected 9) and with `empty_retries`, and how many of the 40 empty-read samples of the 61.05 run still end with an ask.
- [ ] **Step 4: Decide.** Ship the preset with the highest public score; the baseline to beat is 61.05. A tie goes to sft-q1b with the fewer options.

### Task 7: Ship

Follow `docs/superpowers/plans/2026-11-10-docker-submission.md` and `docs/shipped-model.md` for the mechanics. Separate branch `feat/ship-<preset>` and PR, after the code PR is merged.

**Files:**
- Modify: `configs/shipped.json` (`agent_preset`, `agent_kw.revision` = HF commit sha of the shipped model, `records_commit` = the repo commit that rendered that model's records, from its `run_config.json`)
- Modify: `docs/shipped-model.md` (model, recipe table, public score, how the records were rendered)
- Modify: `docker/training_code/README.md` (model link and revision, parameter counts, the `code_at_<commit>` folder name, the training commands with `--model` and `--lora-targets`)
- Test: `tests/test_docker_entry.py`, `tests/test_dockerfile.py`, `tests/test_presets.py` (its parametrised list of shipped presets)

- [ ] **Step 1:** Check the limits with the shipped numbers: LLM parameters below 5 B and the whole system below 6 B (the Qwen3.5-4B text-only model plus PhoWhisper-large 1.55 B), `max_seconds` of the public run well under 30 s (the A30 is slower than molab).
- [ ] **Step 2:** Edit the files, run `pytest -q`, open the PR. CI builds the image and runs `scripts/docker_smoke.sh`; confirm the image's transformers 5.21.0 loads the Qwen3.5 model and stops at `<|im_end|>` (the end-of-turn fix of PR #69).
- [ ] **Step 3: HUMAN.** After the merge, tag `docker-*` for the upload, then submit the Drive link and MD5 to the portal's Test Solution. Never commit the Drive link.
- [ ] **Step 4:** Record the Test Solution result and the final system in `docs/shipped-model.md` and `RESULTS.md`.

## Not in this plan

- A cap on repeated empty reads (9 public samples send three or more). On an unseen app a gold trajectory may itself search several items that return nothing, and those samples already score about 0, so the cap can cost recall and gain little.
- Rules for time arithmetic, condition branches, ask/act and free text: no rule was found that the recorded runs support.
- Any training data built from the public_test catalog (team rule: unseen apps are the task).
