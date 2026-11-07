# Leave-App-Out Examples and Schema Notes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test two training changes aimed at apps the model never trained on: few-shot prompts without the sample's own apps, and tool schemas that keep the catalog's `kind`, `returns` and app text. Hand any change that passes the gate to `sft-final2` by 9 Nov 2026.

**Architecture:** Both changes are opt-in and render into SFT records through the real agent, so training text stays equal to inference text. A short diagnostic run on the existing `sft-v2` decides whether the leave-app-out training is worth its GPU time. Each trained model then goes through the existing holdout and dev gate.

**Tech Stack:** Python, pytest, PyTorch + peft on molab, `lifemate.sft.records`, `lifemate.compare`, marimo notebooks.

**Spec:** No separate spec. The evidence is below and comes from the sft-v2 + check_refs baselines. This plan feeds Task 7 of `docs/superpowers/plans/2026-11-04-score-target.md`.

## Evidence

| Measurement | Value |
|---|---|
| Records with every gold tool in their 4 examples: pool / holdout | 74.1% / 13.8% |
| Records with at least one gold tool in their 4 examples: pool / holdout | 92.6% / 51.0% |
| Same-app examples per prompt, out of 4: pool / holdout | 3.01 / 1.46 |
| Holdout, gold text: real examples of the app added to the bank, on a model that never trained on it | +6.41 [95% CI +2.58, +10.56] |
| Catalog text that `tool_schema` drops | `kind`, `returns` (non-empty on 70 of 147 train tools, 62 of 122 public tools), app `name_vi` and `description` |

These measurements do not show that either change gains points. Only the gated runs below can show that.

## Global Constraints

- LLM < 5B params, context <= 16,384 tokens, whole system < 6B, <= 30 s per sample on 1x A30 with ASR included, offline at inference, fixed seeds, greedy decoding (`context/RULE.md`).
- No paid or external services and no LLM APIs in the repo, dev tools or data pipeline.
- Never label public_test or private_test to train or tune. Experiments train on the pool only. `sft-final` has seen dev and holdout, so never gate on it.
- Do not generate data from the public catalog, and keep holdout apps out of experiment training data.
- A real agent never reads `Task.label`. The one exception is the `leave_gold_apps_out` flag, used only for SFT rendering and the train-only diagnostic.
- With every new option off, prompts and records stay byte-identical to today's.
- **Gate (ASR text, `pho-large-ft-b`):** holdout diff >= +0.5 with the 95% CI above 0, dev diff >= -0.3, holdout ask column drops by at most 5 points. Baselines: `runs/20261102-104252_3c4fd00_train_sft-v2_pho-large-ft-b_holdout_check_refs` (41.37, ask 33.06) and `runs/20261102-101746_3c4fd00_train_sft-v2_pho-large-ft-b_dev_check_refs` (66.86).
- English in the repo. Docs and code on separate branches, one PR each into `main`, merged after a teammate looks. Never commit data, tokens or `train_records/`. On Windows set `PYTHONUTF8=1`.

## Calendar

| Date | Work |
|---|---|
| 5 Nov | Tasks 1-4, code PR. Task 5 diagnostic once the branch can run on molab |
| 6 Nov | Task 6: train `sft-v5a` and `sft-v5b`, about 2.5 hours each. Two molab accounts run them in parallel; one account runs them in sequence inside one 12-hour session |
| 7 Nov | Task 6: gate both. Train `sft-v5c` if both pass |
| 8 Nov | Task 7: hand-off and re-estimate of the target |
| 9 Nov | Score-target Task 7 retrains `sft-final2` with the winning recipe |
| 11 Nov | Docker Task 9 starts with the best system, whatever the score |

## Not in this plan

Synthetic scenarios (score-target Task 5), the base model probe (score-target Task 6), the tool-name plan line and further ASR work stay outside this plan.

## Review Focus

1. **Label read on a split without labels:** `leave_gold_apps_out` on a task with `label=None` must raise, not silently run with all examples. Test in Task 1.
2. **A sample whose installed apps are nearly all gold apps:** selection returns fewer than k examples, or none, and the record still renders. Test in Task 2.
3. **Ask samples:** the app the ask is about counts as a gold app, so its examples are left out as well. Test in Task 1.
4. **Trained with schema notes, served without them, or the reverse:** the notebook's suggested preset follows the run's records name, not the live checkbox. Test in Task 4.
5. **Masking and perturbing a noted schema:** `Provide one of:` stays at the end of the description so `perturb.ONE_OF` still matches, and masked tools show no original app prefix. Test in Task 3.

---

### Task 1: Leave the gold apps out of the examples (bank and agent)

**Files:**
- Modify: `src/lifemate/agents/fewshot.py`, `src/lifemate/agents/react.py`, `configs/presets.json`
- Test: `tests/test_fewshot.py`, `tests/test_react_agent.py`, `tests/test_presets.py`

**Interfaces:**
- Produces:
  - `gold_apps(label: dict) -> set[str]` in `fewshot.py`, replacing `_gold_apps(sample)`, with the same rule: a call's app, or the app of the tool an ask is about.
  - `ExampleBank.select(request, installed_apps, k, exclude_id=None, offset=0, exclude_apps: frozenset[str] | set[str] = frozenset())`: skips docs whose gold apps intersect `exclude_apps`. Out-of-scope docs have no gold apps, so they still qualify.
  - `ReActAgent(..., leave_gold_apps_out: bool = False)`, recorded in `self.info`. When on, every `bank.select` call, the prompt vote included, passes `exclude_apps=gold_apps(task.label)`. When off, the keyword is not passed, so the existing `StubBank` stays valid. A `None` label raises `ValueError`.
  - Preset `sft-v2-refs-lao-diag`: `{"model": "${HF_MODEL_OWNER}/lifemate-sft-v2", "examples": 4, "check_refs": true, "leave_gold_apps_out": true}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fewshot.py
def test_select_leaves_out_excluded_apps(train_samples):
    bank = ExampleBank(train_samples, pool_ids={s["id"] for s in train_samples})
    s = next(x for x in train_samples if x["label"]["tool_calls"] and len(x["context"]["installed_apps"]) > 3)
    apps = gold_apps(s["label"])
    picked = bank.select(s["audio_text"], set(s["context"]["installed_apps"]), k=4, exclude_id=s["id"], exclude_apps=apps)
    assert all(not (gold_apps(p["label"]) & apps) for p in picked)

def test_ask_target_counts_as_a_gold_app():
    label = {"tool_calls": [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "sms.send", "params": ["contact_id"]}}]}
    assert gold_apps(label) == {"sms"}

def test_out_of_scope_examples_survive_any_exclusion(train_samples):
    bank = ExampleBank(train_samples, pool_ids={s["id"] for s in train_samples})
    installed = set(train_samples[0]["context"]["installed_apps"])
    picked = bank.select("kể chuyện cười", installed, k=20, exclude_apps=installed)
    assert picked and all(not p["label"]["tool_calls"] for p in picked)

# tests/test_react_agent.py
def test_leave_gold_apps_out_passes_the_label_apps_to_the_bank():
    seen = {}
    class Bank:
        def __len__(self): return 1
        def select(self, *args, **kw):
            seen.update(kw)
            return []
    label = {"tool_calls": [{"name": "alarm.delete", "arguments": {"alarm_id": "a1"}}]}
    task = dataclasses.replace(TASK, label=label, context={**TASK.context, "installed_apps": {"alarm": {"tools": []}}})
    ReActAgent(llm=FakeLLM(["Xong."]), examples=2, bank=Bank(), leave_gold_apps_out=True).run(task, FakeEnv())
    assert seen["exclude_apps"] == {"alarm"}

def test_leave_gold_apps_out_refuses_a_task_without_a_label():
    task = dataclasses.replace(TASK, context={**TASK.context, "installed_apps": {}})
    agent = ReActAgent(llm=FakeLLM(["Xong."]), examples=2, bank=StubBank("EX"), leave_gold_apps_out=True)
    with pytest.raises(ValueError, match="label"):
        agent.run(task, FakeEnv())

# tests/test_presets.py
def test_only_diagnostic_presets_read_labels():
    readers = {k for k, v in load_presets()["agent"].items() if v["agent_kw"].get("leave_gold_apps_out")}
    assert readers == {"sft-v2-refs-lao-diag"}
```

- [ ] **Step 2: Run them.** Run: `PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python -m pytest tests/test_fewshot.py tests/test_react_agent.py tests/test_presets.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** the interfaces above. Update the existing callers of `_gold_apps`.
- [ ] **Step 4: Run them again.** Same command. Expected: PASS.
- [ ] **Step 5: Commit** on `feat/unseen-examples-schema`: `Add an opt-in that leaves the gold apps out of the examples`.

### Task 2: Render leave-app-out records

**Files:**
- Modify: `src/lifemate/sft/replay.py`, `src/lifemate/sft/records.py`
- Test: `tests/test_sft_replay.py`

**Interfaces:**
- Consumes: `ReActAgent(leave_gold_apps_out=...)` from Task 1.
- Produces:
  - `replay(..., leave_gold_apps_out: bool = False)`.
  - CLI `records --leave-app-out P`, a float in `[0, 1]`, default 0; anything else is an `ap.error`.
  - For each `(sample, source index)`, the record leaves the gold apps out when `random.Random(f"{args.seed}:{sample['id']}:{i}:lao").random() < P`.
  - Every record gets a `leave_app_out` boolean key. Masked and perturbed copies inherit it, because they are built from the first source's record.
  - The final print adds the number of records with the flag.

- [ ] **Step 1: Write the failing tests**

```python
def _example_tools(rec):
    lines = rec["messages"][0]["content"].split("\n")
    return [json.loads(l[len("<tool_call>"):-len("</tool_call>")])["name"] for l in lines if l.startswith("<tool_call>")]

def _read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def _lao(tmp_path, p, name="t", limit="40"):
    return _read(records.main(["--name", name, "--subset", "pool", "--limit", limit, "--tokenizer", "none",
                               "--out-dir", str(tmp_path), "--leave-app-out", p]))

def test_leave_app_out_removes_gold_app_examples(train_samples, tmp_path):
    by_id = {s["id"]: s for s in train_samples}
    rows = _lao(tmp_path, "1")
    assert rows and all(r["leave_app_out"] for r in rows)
    for r in rows:
        apps = gold_apps(by_id[r["id"]]["label"])
        assert not any(t.split(".")[0] in apps for t in _example_tools(r))

def test_leave_app_out_zero_keeps_the_same_messages(train_samples, tmp_path):
    plain = _read(records.main(["--name", "a", "--subset", "pool", "--limit", "10", "--tokenizer", "none", "--out-dir", str(tmp_path)]))
    zero = _lao(tmp_path, "0", name="b", limit="10")
    assert [r["messages"] for r in plain] == [r["messages"] for r in zero]
    assert not any(r["leave_app_out"] for r in zero)

def test_leave_app_out_share_is_deterministic(train_samples, tmp_path):
    first, again = _lao(tmp_path, "0.5", name="a"), _lao(tmp_path, "0.5", name="b")
    assert first == again and 0 < sum(r["leave_app_out"] for r in first) < len(first)

def test_leave_app_out_keeps_a_sample_with_few_other_apps(train_samples, tmp_path):
    fewest = min((s for s in train_samples if s["label"]["tool_calls"] and s["id"] in set(load_splits()["pool"])),
                 key=lambda s: len(s["context"]["installed_apps"]))
    assert replay_sample(fewest, examples=4, leave_gold_apps_out=True) is not None

def test_leave_app_out_rejects_a_share_outside_zero_to_one(train_samples, tmp_path):
    with pytest.raises(SystemExit):
        _lao(tmp_path, "1.5")
```

Add `leave_gold_apps_out=False` to `replay_sample` in `tests/conftest.py`.

- [ ] **Step 2: Run them.** Run: `PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python -m pytest tests/test_sft_replay.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** the interfaces above and describe the flag in the `records.py` docstring.
- [ ] **Step 4: Run them again.** Same command. Expected: PASS.
- [ ] **Step 5: Check the real records.** Run: `PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python -m lifemate.sft.records --name lao-check --subset pool --leave-app-out 0.5 --tokenizer none`. Expected: 0 replays dropped beyond a run without the flag, and about half the records flagged. Delete the file afterwards.
- [ ] **Step 6: Commit:** `Render SFT records that leave the gold apps out of the examples`.

### Task 3: Schema notes

**Files:**
- Modify: `src/lifemate/data.py`, `src/lifemate/agents/prompting.py`, `src/lifemate/agents/react.py`, `src/lifemate/sft/replay.py`, `src/lifemate/sft/records.py`
- Test: `tests/test_prompting.py`, `tests/test_sft_replay.py`, `tests/test_sft_mask.py`, `tests/test_sft_perturb.py`

**Interfaces:**
- Produces:
  - `load_catalog` adds `app_name_vi` and `app_description` to each spec. `tool_schema` ignores them unless notes are on.
  - `tool_schemas(specs: list[dict], schema_notes: bool = False) -> list[dict]` in `prompting.py`. With notes off it equals `[tool_schema(s) for s in specs]`.
  - `tool_schema(spec, notes: str = "", returns: bool = False)`. `notes` goes before the description, `returns` adds the `Returns:` sentence after it, and `Provide one of:` stays last.
  - `ReActAgent(..., schema_notes: bool = False)`, recorded in `self.info`. `run` uses `tool_schemas`.
  - `replay(..., schema_notes: bool = False)` and CLI `records --schema-notes`.

Exact description text, for tools of kind `read` or `write` only. `sys.ask_missing_info` has kind `meta` and stays unchanged:

```
First tool of an app in the list:  [read] App Nhà thông minh (Điều khiển đèn, ...). <description> Returns: devices[]. Provide one of: a | b.
Later tools of that app:           [write] App Nhà thông minh. <description> Provide one of: a | b.
```

`Returns:` appears only when `returns` is non-empty. Measured growth is about 11% of the tool text, or roughly 1,360 characters per sample. The app description appears once per app because repeating it on every tool would cost about 10% more.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prompting.py
SPECS = [
    {"name": "lock.list", "kind": "read", "app": "lock", "app_name_vi": "Khoá", "app_description": "Khoá cửa", "returns": "locks[]",
     "description": "Danh sách", "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "lock.set", "kind": "write", "app": "lock", "app_name_vi": "Khoá", "app_description": "Khoá cửa", "returns": "",
     "description": "Đặt", "parameters": {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                                          "required": [], "x-one-of": [["a"], ["b"]]}},
    {"name": "sys.ask_missing_info", "kind": "meta", "app": "sys", "app_name_vi": "Hệ thống", "app_description": "x", "returns": "",
     "description": "Ask", "parameters": {"type": "object", "properties": {}, "required": []}},
]

def _descriptions(**kw):
    return [t["function"]["description"] for t in tool_schemas(SPECS, **kw)]

def test_schema_notes_exact_text():
    assert _descriptions(schema_notes=True) == [
        "[read] App Khoá (Khoá cửa). Danh sách Returns: locks[].",
        "[write] App Khoá. Đặt Provide one of: a | b.",
        "Ask",
    ]

def test_schema_notes_off_is_unchanged():
    assert tool_schemas(SPECS) == [tool_schema(s) for s in SPECS]
    assert _descriptions() == ["Danh sách", "Đặt Provide one of: a | b.", "Ask"]

# tests/test_sft_perturb.py
def test_one_of_is_still_found_with_schema_notes():
    fn = tool_schemas(SPECS, schema_notes=True)[1]["function"]
    assert _one_of_params(fn["description"]) == {"a", "b"}

# tests/test_sft_mask.py
def test_masked_notes_show_no_original_app_prefix(train_samples):
    s = next(x for x in train_samples if x["label"]["tool_calls"])
    def prefixes(rec):
        text = json.dumps(mask_record(rec, random.Random(0))["tools"], ensure_ascii=False)
        return {app for app in s["context"]["installed_apps"] if f"{app}." in text}, text
    plain, _ = prefixes(replay_sample(s))
    noted, text = prefixes(replay_sample(s, schema_notes=True))
    assert "App " in text and noted <= plain  # the notes add no original app prefix

# tests/test_sft_replay.py
def test_records_cli_schema_notes_change_only_the_tools(train_samples, tmp_path):
    plain = _read(records.main(["--name", "a", "--subset", "pool", "--limit", "5", "--tokenizer", "none", "--out-dir", str(tmp_path)]))
    noted = _read(records.main(["--name", "b", "--subset", "pool", "--limit", "5", "--tokenizer", "none", "--out-dir", str(tmp_path),
                                "--schema-notes"]))
    assert [r["messages"] for r in plain] == [r["messages"] for r in noted]
    assert plain[0]["tools"] != noted[0]["tools"] and "App " in json.dumps(noted[0]["tools"], ensure_ascii=False)
```

Add `schema_notes=False` to `replay_sample` in `tests/conftest.py`.

- [ ] **Step 2: Run them.** Run: `PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python -m pytest tests/test_prompting.py tests/test_sft_perturb.py tests/test_sft_mask.py tests/test_sft_replay.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** the interfaces above.
- [ ] **Step 4: Run them again.** Same command. Expected: PASS.
- [ ] **Step 5: Check the token length.** Run: `PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python -m lifemate.sft.records --name notes-check --subset pool --schema-notes`, with the default tokenizer so it prints the token report. Expected: the printed max stays under `MAX_LEN` (12,288), so no record is skipped in training. If it does not, put the count of records over the limit in the PR. Delete the file afterwards.
- [ ] **Step 6: Commit:** `Add opt-in schema notes: kind, returns and app text`.

### Task 4: Training notebook and docs

**Files:**
- Modify: `notebooks/molab_train.py`, `CLAUDE.md`
- Test: `tests/test_molab_train.py`

**Interfaces:**
- Consumes: `records --leave-app-out`, `records --schema-notes`.
- Produces:
  - UI `leave_out = mo.ui.number(start=0, stop=100, value=0, label="records with the gold apps left out of the examples, % (--leave-app-out)")`.
  - UI `schema_notes = mo.ui.checkbox(value=False, label="kind, returns and app text in the tool schemas (--schema-notes)")`.
  - Record names keep this order: `gold-<subset><mask><perturb><asr><synth>-lao<N>-schema-d-k<k>`. `-lao<N>` goes on train records only. `-schema` goes on train and eval records. `--leave-app-out` receives `N/100`.
  - The merge cell reads notes from the saved records name with `r"-schema(-d)?(-k\d+)?$"` and adds `"schema_notes": true` to the suggested preset. Without a saved `run_config.json` it falls back to the checkbox, as `date_hints` does.

- [ ] **Step 1: Write the failing tests.** Give `_commands` the keyword parameters `leave_out=0` and `notes=False`, and add `leave_out` and `schema_notes` to its `fakes`. Add `schema_notes` (unticked) to the `fakes` of `_merge_suggestion`. Then add:

```python
def test_new_options_name_the_records_and_pass_the_flags(tmp_path):
    records, train = _commands(tmp_path, False, leave_out=50, notes=True)
    assert [_after(c, "--name") for c in records] == ["gold-pool-m2-p-lao50-schema", "gold-dev-schema"]
    assert _after(records[0], "--leave-app-out") == "0.5" and "--leave-app-out" not in records[1]
    assert all("--schema-notes" in c for c in records) and _after(train, "--records") == "gold-pool-m2-p-lao50-schema"

def test_new_options_off_keep_the_commands(tmp_path):
    records, _ = _commands(tmp_path, False)
    assert [_after(c, "--name") for c in records] == ["gold-pool-m2-p", "gold-dev"]
    assert not any("--schema-notes" in c or "--leave-app-out" in c for c in records)

@pytest.mark.parametrize("saved, expected", [
    ("gold-pool-m1-asr-x-lao50-schema", True), ("gold-pool-m1-schema-d-k2", True), ("gold-pool-m1-lao50", False), ("gold-schema-m2", False),
])
def test_merge_serves_schema_notes_as_trained(tmp_path, saved, expected):
    assert ('"schema_notes": true' in _merge_suggestion(tmp_path, False, saved)) is expected

def test_date_hints_are_still_found_after_the_schema_suffix(tmp_path):
    assert '"date_hints": true' in _merge_suggestion(tmp_path, False, "gold-pool-m1-lao50-schema-d")
```

- [ ] **Step 2: Run them.** Run: `PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python -m pytest tests/test_molab_train.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** the interfaces above and add both options to step 4 of the notebook's instructions. Add one sentence on both flags to the `sft/` bullet of `CLAUDE.md`.
- [ ] **Step 4: Run the full suite.** Run: `PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python -m pytest`. Expected: PASS.
- [ ] **Step 5: Commit and open the PR.** Commit `Add the leave-app-out and schema note options to the training notebook`, push `feat/unseen-examples-schema`, and open a PR into `main` with the `pytest` output. On the owner's machine use `GH_TOKEN="$(gh auth token --user AuDuongKhang)"`.

### Task 5: Diagnostic on `sft-v2` (K, GPU, about 15 minutes)

This asks whether a model trained with same-app examples depends on them. It trains nothing and uses labels only to choose which examples to hide.

**Files:**
- Modify: `RESULTS.md`

- [ ] **Step 1: Run.** On `notebooks/molab_run.py`, branch `feat/unseen-examples-schema`, split train, subset dev, agent preset `sft-v2-refs-lao-diag`, ASR preset `gold`. Evaluate it. Expected: 300 samples, 0 errors.
- [ ] **Step 2: Compare.** Run: `python -m lifemate.compare runs/20261102-093954_3c4fd00_train_sft-v2_gold_dev_check_refs runs/<diag run>`. Add a `RESULTS.md` row with the compare line, marked as a diagnostic.
- [ ] **Step 3: Decide.**
  - Diff <= -3 with the 95% CI below 0: the model depends on same-app examples, so train `sft-v5a` in Task 6.
  - Otherwise: skip `sft-v5a` and note why in the row. `sft-v5b` still runs, because it does not depend on this result.

### Task 6: Train and gate (K, GPU)

**Files:**
- Modify: `configs/presets.json`, `tests/test_presets.py`, `RESULTS.md`

**Interfaces:**
- Consumes: the notebook options from Task 4 and the decision from Task 5.
- Produces: only for models that were trained and merged (an HF repo must exist before its preset):

| Preset | Model | Agent kwargs besides `model`, `examples: 4`, `check_refs: true` |
|---|---|---|
| `sft-v5a-refs` | `${HF_MODEL_OWNER}/lifemate-sft-v5a` | none |
| `sft-v5b-refs` | `${HF_MODEL_OWNER}/lifemate-sft-v5b` | `schema_notes: true` |
| `sft-v5c-refs` | `${HF_MODEL_OWNER}/lifemate-sft-v5c` | `schema_notes: true` |

All three start from the `sft-v2` recipe: records pool, eval dev, masked copies 1, perturb off, ASR copy `pho-large-fp16`, 2 epochs.

- [ ] **Step 1: Train `sft-v5a`** with `notebooks/molab_train.py`: leave-out 50, schema notes off, repo `TripleTrouble/lifemate-sft-v5a`. Expected records name: `gold-pool-m1-asr-pho-large-fp16-lao50`. Merge it. Skip this step if Task 5 said so.
- [ ] **Step 2: Train `sft-v5b`:** leave-out 0, schema notes on, repo `TripleTrouble/lifemate-sft-v5b`. Expected records name: `gold-pool-m1-asr-pho-large-fp16-schema`. Merge it.
- [ ] **Step 3: Add each preset** with an exact `resolve` test, for example `resolve("agent", "sft-v5b-refs", {"HF_MODEL_OWNER": "team"}) == ("react", {"model": "team/lifemate-sft-v5b", "examples": 4, "check_refs": True, "schema_notes": True})`. Run: `PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python -m pytest tests/test_presets.py -v`. Expected: PASS.
- [ ] **Step 4: Run each model** on holdout and dev with ASR preset `pho-large-ft-b`, and on holdout with ASR `gold`. Evaluate each. Expected: `over_time_limit` 0 and max seconds noted.
- [ ] **Step 5: Compare** each ASR run against its Global Constraints baseline with `python -m lifemate.compare`, run `python -m lifemate.ceilings` on the holdout ASR run, and add `RESULTS.md` rows with the compare lines.
- [ ] **Step 6: Decide.**
  - Both pass: train `sft-v5c` with leave-out 50 and schema notes on, repo `TripleTrouble/lifemate-sft-v5c`. Gate it against the baselines and the better single model. Keep it only if its holdout mean beats that model and it passes the gate.
  - One passes: it is the winning recipe, and `sft-v5c` is not trained.
  - None passes: the recipe stays `sft-v2`, and `RESULTS.md` records that.
- [ ] **Step 7: PR** `Add the sft-v5 presets and results`, with `pytest` output and every compare line.

### Task 7: Hand-off (K)

**Files:**
- Modify: `RESULTS.md`

- [ ] **Step 1:** List in the Task 6 PR the winning flags for `sft-final2`, for example `--leave-app-out 0.5 --schema-notes`, and whether its preset needs `schema_notes: true`. Score-target Task 7 retrains from this list on 9 Nov. Add synthetic data to it only if score-target Task 5 passed its own gate.
- [ ] **Step 2:** On 8 Nov, re-estimate the 80-90 target from the best holdout score and write it in `RESULTS.md` as a dated note, as the score-target calendar asks.
- [ ] **Step 3:** After the final retrain, check the Docker hand-off. `configs/shipped.json` must point at a preset whose `schema_notes` matches how the model was trained, and no shipped preset may set `leave_gold_apps_out`.
