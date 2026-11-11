# sft-q1b Loss Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the points sft-q1b loses to fixable call shapes, and measure how much of its holdout loss is ASR, without retraining.

**Architecture:** Two opt-in agent options in the style of `date_fix`: `id_fix` rewrites a write call's argument from an entity's name to that entity's id before the call is validated, and `spoken_email` respells the ASR forms of a spoken email address back to the spoken form the model was trained on. `id_fix` is also a `lifemate.sim` rewrite, so it is scored offline on recorded runs; `spoken_email` changes the prompt and needs a molab run. A gold-text run of the current agent splits the remaining loss into ASR and agent.

**Tech Stack:** Python 3.11, pytest, existing `lifemate.agents` / `lifemate.sim` / `configs/presets.json`.

**Spec:** the sft-q1b loss analysis of 2026-11-10 (below, "Evidence"); it replaces the sft-v2 analysis in the untracked `docs/error_10_11.md`.

## Evidence (sft-q1b-refs, ASR `pho-large-ft-b`; holdout 44.11, dev 75.04)

- Holdout loses 55.89 points: state right but calls differ 18.5, other tools 16.5, ask/act flip 9.1, wrong arguments about 10. Loops (1.5), time (p95 8.3 s) and wrong refusals (2 samples) are not the problem.
- Name instead of id: gold string arguments that point at an entity seen earlier use its id 1136 times and its name 6 times (all six are search `query` arguments). sft-q1b passes the name 10 times on holdout (`loyalty.redeem.program = "Cao Nguyên Rewards"`, gold `prog_815`). A one-off simulation of name -> id on recorded calls: holdout +0.44 [95% CI +0.07, +0.94], better 4 / worse 0; dev +0.00.
- Spoken emails: 8 dev+holdout samples have a gold email spoken in the request; all 8 score 0. ASR hears "a còng" as "Account", "Item", "acong", and "chấm test" as "chấm tết", ".step". Train gold texts say the email as "<local> a còng example chấm test" (10 samples), so that is the form the model learnt to turn into `<local>@example.test`.
- Rules already in the repo do not help sft-q1b (sim / guard_sim on the same runs): ask guard holdout -5.83 to -9.41 and dev -16 to -20 over its 3 x 3 grid; `freetext` -0.08; `minimal_reads` -1.79 (CI below 0); `date_fix` +0.17 (CI includes 0). The ask/act, free-text and time-arithmetic losses therefore need training data, not rules (see "Not in this plan").
- ASR cost, measured on sft-v2 only: holdout gold text 46.19 vs ASR 36.07, -10.12 [95% CI -14.68, -5.59].

## Global Constraints

- No paid or external services, no LLM APIs; nothing reads `<split>_environment/` or `Task.label` (team rules, CLAUDE.md).
- Every new option is off by default, so `configs/shipped.json` and SFT record rendering (`sft.replay` builds `ReActAgent` with defaults) do not change.
- A rewrite changes the arguments of a call the model made; it never adds a call (every `env.call` is scored).
- No app or tool name decides anything; only the schema (`kind`, `x-match`) and the values seen in tool results do (private test brings unseen apps).
- Repo text is English; files keep LF endings.
- Decision rule for turning an option on in a shipped preset (runbook gate): holdout diff >= +0.5 with 95% CI above 0, dev diff >= -0.3.

## Review Focus

- A free-text (`x-match: fuzzy`) argument equal to an entity name (a note titled with the article title) must stay a name. Test in Task 1.
- A name shared by two entities (two programs called "MinhPoint") must not be rewritten. Test in Task 1.
- A read call (`contacts.search {"query": "Minh"}`, `loyalty.get_points {"program": "Cao Nguyên"}`) must stay as sent. Test in Task 1.
- A value that already is a seen id, or a non-string value, must stay. Test in Task 1.
- A request that only mentions "example" in ordinary words, or has no email, must come back unchanged. Test in Task 2.

---

### Task 1: `id_fix` — entity name to id in write calls

**Files:**
- Create: `src/lifemate/agents/id_fix.py`
- Modify: `src/lifemate/agents/react.py` (option, docstring, info, call site next to `date_fix`)
- Modify: `src/lifemate/sim.py` (rewrite factory `id_fix`, registered in `REWRITES`)
- Modify: `configs/presets.json` (`sft-q1b-refs-idfix`)
- Test: `tests/test_id_fix.py`, `tests/test_react_agent.py`, `tests/test_sim.py`, `tests/test_presets.py`

**Interfaces:**
- Produces: `entity_names(obj, out: dict[str, set[str]]) -> dict[str, set[str]]` — every dict with a string `id` inside `obj` adds its `name`, `title` and `label` strings (key: `casefold()` of the stripped string) mapped to that id, and the id itself under the key `""` (never a name, since names are stripped and non-empty); returns `out`.
- Produces: `fix_call_ids(call: dict, spec: dict | None, names: dict[str, set[str]]) -> dict` — `call` itself when nothing changes, else a copy with new `arguments`.
- Produces: `fix_calls_ids(calls: list[dict], specs: dict[str, dict], names) -> list[dict]` — `calls` itself when no call changes (same contract as `date_fix.fix_calls`).
- Produces: `ReActAgent(..., id_fix: bool = False)`, `info["id_fix"]`, trace key `id_fixes` = list of `{"from": args, "to": args}`.
- Produces: `sim.id_fix(data_root: Path | None = None) -> Rewrite`, name `"id_fix"` in `REWRITES`.

Rule of `fix_call_ids`: only when `spec["kind"] == "write"`; for each argument whose value is a `str`, whose property is not `x-match: fuzzy`, and whose value is not the id of any entity seen so far (`names[""]`; not `seen`, which also holds the names this rule converts): if `names[value.strip().casefold()]` holds exactly one id, use that id.

The rule is not limited to `x-ref` parameters: on train, 42 gold write arguments that carry a seen entity's id are plain strings (`loyalty.redeem.program` and `.reward_id`, the case this fix is for), against 1 non-fuzzy write argument whose gold value is an entity name.

- [ ] **Step 1: Write the failing tests** in `tests/test_id_fix.py`

```python
REDEEM = {"name": "loyalty.redeem", "kind": "write", "parameters": {"properties": {"program": {"type": "string"}, "reward_id": {"type": "string"}}}}
NOTE = {"name": "notes.create", "kind": "write", "parameters": {"properties": {"title": {"type": "string", "x-match": "fuzzy"}}}}
POINTS = {"name": "loyalty.get_points", "kind": "read", "parameters": {"properties": {"program": {"type": "string"}}}}
RESULT = {"ok": True, "programs": [{"id": "prog_815", "name": "Cao Nguyên Rewards", "rewards": [{"id": "rw_02_1", "name": "Ly phin"}]}]}

def test_entity_names_maps_folded_names_titles_labels_to_ids():
    assert entity_names(RESULT, {}) == {"cao nguyên rewards": {"prog_815"}, "ly phin": {"rw_02_1"}, "": {"prog_815", "rw_02_1"}}

def test_id_of_one_entity_that_names_another_stays():
    names = entity_names([{"id": "c1", "name": "x"}, {"id": "c2", "name": "c1"}], {})
    call = {"name": "loyalty.redeem", "arguments": {"program": "c1", "reward_id": "r"}}
    assert fix_call_ids(call, REDEEM, names) is call

def test_name_of_a_seen_entity_becomes_its_id():
    call = {"name": "loyalty.redeem", "arguments": {"program": " cao nguyên REWARDS", "reward_id": "rw_02_1"}}
    fixed = fix_call_ids(call, REDEEM, entity_names(RESULT, {}))
    assert fixed["arguments"] == {"program": "prog_815", "reward_id": "rw_02_1"} and call["arguments"]["program"] == " cao nguyên REWARDS"

def test_seen_id_free_text_read_and_non_strings_are_left_alone():
    names = entity_names(RESULT, {})
    for spec, call in [(REDEEM, {"name": "loyalty.redeem", "arguments": {"program": "prog_815", "reward_id": "rw_02_1"}}),
                       (NOTE, {"name": "notes.create", "arguments": {"title": "Cao Nguyên Rewards"}}),
                       (POINTS, {"name": "loyalty.get_points", "arguments": {"program": "Cao Nguyên Rewards"}}),
                       (REDEEM, {"name": "loyalty.redeem", "arguments": {"program": 5, "reward_id": "rw_02_1"}})]:
        assert fix_call_ids(call, spec, names) is call

def test_a_name_shared_by_two_entities_is_left_alone():
    names = entity_names([{"id": "p1", "name": "MinhPoint"}, {"id": "p2", "name": "minhpoint"}], {})
    call = {"name": "loyalty.redeem", "arguments": {"program": "MinhPoint", "reward_id": "r"}}
    assert fix_call_ids(call, REDEEM, names) is call

def test_fix_calls_ids_returns_the_same_list_when_nothing_changes():
    calls = [{"name": "loyalty.redeem", "arguments": {"program": "prog_815", "reward_id": "rw_02_1"}}]
    assert fix_calls_ids(calls, {"loyalty.redeem": REDEEM}, entity_names(RESULT, {})) is calls
```

- [ ] **Step 2: Run** `pytest tests/test_id_fix.py -q` — expected: ImportError (module missing).
- [ ] **Step 3: Implement** `src/lifemate/agents/id_fix.py` with the three functions above (module docstring states the 1136:6 evidence and the rule).
- [ ] **Step 4: Run** `pytest tests/test_id_fix.py -q` — expected: 5 passed.
- [ ] **Step 5: Agent test** in `tests/test_react_agent.py`: with `id_fix=True`, a `FakeLLM` that first reads (result holding `{"id": "prog_815", "name": "Cao Nguyên Rewards"}`) and then redeems by name sends `program="prog_815"` to the env, records one `id_fixes` entry, and `info["id_fix"] is True`; with the default, the env gets the name. Names from `task.history` tool results count too (second test: entity only in history).
- [ ] **Step 6: Implement in `react.py`**: `names = entity_names([t.get("tool_results") for t in task.history], {})` next to `seen`; after the `date_fix` block, `fixed = fix_calls_ids(calls, specs, names)` when `self.id_fix`, trace like `date_fixes`; after each `env.call`, `entity_names(result, names)` next to `scalar_strings(result, seen)`.
- [ ] **Step 7: Sim rewrite** `sim.id_fix(data_root=None)`: loads the train samples for history results, walks `trace["calls"]` in order (names grow from each recorded `result`), returns `None` when nothing changes. Test in `tests/test_sim.py` with a hand-made trace line and `times`-style injected samples (factory takes `histories: dict[str, list] | None` for tests, like `date_fix(times=...)`).
- [ ] **Step 8: Preset** `"sft-q1b-refs-idfix"` = `sft-q1b-refs` + `"id_fix": true`; test `resolve("agent", "sft-q1b-refs-idfix", env) == (name, {**kw, "id_fix": True})`.
- [ ] **Step 9: Run** `pytest tests/test_id_fix.py tests/test_react_agent.py tests/test_sim.py tests/test_presets.py -q` — expected: all pass.
- [ ] **Step 10: Offline score** `python -m lifemate.sim id_fix runs/<sft-q1b-refs holdout>` and the dev run — expected about holdout +0.44 [CI above 0], dev +0.00. If dev is below 0, stop and report.
- [ ] **Step 11: Commit** `Add id_fix: pass an entity's id where a write call names it`.

### Task 2: `spoken_email` — respell misheard spoken emails

**Files:**
- Create: `src/lifemate/agents/spoken.py`
- Modify: `src/lifemate/agents/react.py` (option, docstring, info; respelled task used for the few-shot query and the prompt)
- Modify: `configs/presets.json` (`sft-q1b-refs-fixes` = `sft-q1b-refs` + `id_fix` + `spoken_email`)
- Test: `tests/test_spoken.py`, `tests/test_react_agent.py`, `tests/test_presets.py`

**Interfaces:**
- Produces: `respell_emails(text: str) -> str` — case-insensitive, on NFC text:
  - `(a còng|a cong|acong|account|item)` followed by whitespace, the word `example`, then `.` or `chấm` becomes `a còng` (the dot is the email evidence: "account example" alone stays);
  - `example` followed by optional spaces, `.` or `chấm`, optional spaces, then `test|tết|tét|step` as a whole word becomes `example chấm test`.
  Anything else is returned unchanged.
- Produces: `ReActAgent(..., spoken_email: bool = False)`, `info["spoken_email"]`; when on, `run` works on `dataclasses.replace(task, request=respell_emails(task.request), history=[user turns with respelled "text", other turns as they are])`.

- [ ] **Step 1: Failing tests** in `tests/test_spoken.py`, from the recorded ASR transcripts:

```python
@pytest.mark.parametrize("heard, said", [
    ("gửi Family Account Example chấm tết báo là", "gửi Family a còng example chấm test báo là"),
    ("gửi Family Item Example chấm test, báo", "gửi Family a còng example chấm test, báo"),
    ("địa chỉ vendor acong example.step nhé", "địa chỉ vendor a còng example chấm test nhé"),
    ("gửi nhóm bảy mươi ba a còng example chấm test báo", "gửi nhóm bảy mươi ba a còng example chấm test báo"),
])
def test_misheard_email_parts_get_the_spoken_form(heard, said):
    assert respell_emails(heard) == said

@pytest.mark.parametrize("text", ["Cho anh xem một example về cách nấu phở", "Kiểm tra account ngân hàng", "show me an account example", "Đặt báo thức sáu giờ", ""])
def test_text_without_an_email_is_unchanged(text):
    assert respell_emails(text) == text
```

- [ ] **Step 2: Run** `pytest tests/test_spoken.py -q` — expected: ImportError.
- [ ] **Step 3: Implement** `respell_emails` in `src/lifemate/agents/spoken.py` (two compiled regexes; module docstring cites the 8 samples and the train spoken form).
- [ ] **Step 4: Run** `pytest tests/test_spoken.py -q` — expected: 8 passed.
- [ ] **Step 5: Agent tests** in `tests/test_react_agent.py`: with `spoken_email=True` the system/user messages hold `a còng example chấm test` for a request heard as `Account Example chấm tết`, a history user turn is respelled too, the bank is asked with the respelled request (`StubBank.asked`), and `info["spoken_email"] is True`; default leaves the text as heard.
- [ ] **Step 6: Implement** in `react.py` at the top of `run` (before the bank and `build_messages`; `task.request` is also used by `date_fix` and refusal matching, which then see the respelled text).
- [ ] **Step 7: Preset** `sft-q1b-refs-fixes` + preset test; run `pytest tests/test_spoken.py tests/test_react_agent.py tests/test_presets.py -q` — all pass.
- [ ] **Step 8: Count firing** on every recorded request in `runs/*/trace.jsonl` (one-off script): list each request `respell_emails` changes. Expected: only requests with a spoken email; report the count.
- [ ] **Step 9: Commit** `Add spoken_email: respell misheard spoken email addresses`.

### Task 3: Full suite, push, molab measurement

- [ ] **Step 1:** `pytest -q` with `LIFEMATE_DATA_ROOT` pointing at the data — expected: all pass (the Docker notebook test needs a `data/` folder in the checkout).
- [ ] **Step 2:** push the branch; open the PR only after the molab numbers are in.
- [ ] **Step 3 (molab, branch of this plan, split `train`):**

| run | agent preset | ASR preset | subsets | compared with |
|---|---|---|---|---|
| A: ASR share | `sft-q1b-refs` | `gold` | holdout, dev | 44.11 / 75.04 (same agent on `pho-large-ft-b`) |
| B: both fixes | `sft-q1b-refs-fixes` | `pho-large-ft-b` | holdout, dev | 44.11 / 75.04 |

- [ ] **Step 4:** `python -m lifemate.compare` B against the baselines; `spoken_email`'s share is B minus the `id_fix` sim result of Task 1 Step 10. Apply the decision rule; if B passes, a follow-up PR points `configs/shipped.json` at it (shipping sft-q1b itself is a separate PR).

## Not in this plan

The ask/act (9.1 points), free-text wording (5.6) and time-arithmetic losses need training data: every rule tried on sft-q1b is flat or negative (Evidence). That work is a synthetic-data and retrain plan (`lifemate.sft.synth` with a local teacher on molab, 3 h per retrain, gate), to be written after run A shows how much of the holdout loss is the agent's rather than ASR's.
