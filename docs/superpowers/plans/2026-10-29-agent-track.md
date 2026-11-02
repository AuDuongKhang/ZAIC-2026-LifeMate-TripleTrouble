# Agent Track Implementation Plan (29 Oct – 14 Nov)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the LLM side from the Tier 0 baseline (train 16.58 with gold transcripts, public 18.95) to a LoRA-SFT agent that is robust to unseen apps, with a submittable system at the end of every phase.

**Architecture:**
- Phase 1 builds what every experiment needs: fixed splits, paired comparison, a per-step agent trace with a failure breakdown, and dynamic few-shot.
- Phase 2 renders gold trajectories **by running the real `ReActAgent` with an LLM that replays gold calls**, so training text is identical to inference text by construction. A small custom LoRA trainer (peft only, no trl/datasets) trains on it, checkpoints to a private HF repo, and merges the adapter so inference stays on the existing `HFChatLLM` path.
- Phase 3 adds Hammer-style function masking and ASR-noisy transcripts, then a final retrain on all train data.

**Tech Stack:** Python ≥ 3.10, pytest, transformers (tokenizer runs locally without torch), torch + peft on molab only.

**Spec:** `docs/superpowers/specs/2026-10-27-agent-roadmap-design.md` (sections Tiers, Evaluation, Experiments and "Update 29 Oct"). This plan replaces the Track A part of `2026-10-27-experiments-batch1.md`:
- batch-1 Tasks 1–5 (shared infrastructure) were built by the ASR owner on `feat/experiments-infra`;
- batch-1 Tasks 6–7 (few-shot) are Tasks 5–6 here, amended;
- batch-1 Tasks 8–9 (ASR comparison) and `2026-10-28-asr-models-and-data.md` belong to the ASR owner.

## Progress (as of `origin/main` ede28f0, 2 Nov)

| Phase | Tasks | Status | Result |
|---|---|---|---|
| 1 | 1–6 | Done (PR #9, #11) | few-shot k4 chosen: dev 21.02 -> 38.35, holdout 27.65 -> 29.69; public_test 21.46 (whisper), 30.19 with `pho-large-ft-a` |
| 2 | 7–11 | Done (PR #12) | `sft-v1`: dev 63.74, holdout 42.30 (+12.61, Gate 2 passed), public_test 47.85 with `pho-large-ft-a` |
| 3 | 12–14 | Code in PR #14 (`feat/sft-v2`); molab runs pending | Tasks 12–13 done (masking, ASR-noisy records, notebook fields); Task 14 presets added, training/compare/final retrain not run yet |
| 4 | 15 | Partly (PR #14) | `revision` kwarg and README training commands done; determinism check, timing and the pinned preset wait for the chosen model |

Remaining weak spots from the `sft-v1` rows: holdout ask is 13.24 (dev 60.23), so apps unseen in training are the gap that Tasks 12–13 target. Next: merge PR #14, then Task 14 on molab (`sft-v2` with masked copies 1 and ASR cache `pho-large-fp16`). Freeze 14 Nov.

## Why this order (evidence from the 27 Oct runs)

Error breakdown of the train-100 gold-transcript run (275 gold calls):
- 67 matched (24%).
- 110 gold calls never attempted: 50 reads, 60 writes or asks. 37 of 94 act/ask samples made fewer calls than gold.
- 98 attempted with wrong arguments. **74 of those added an optional key the gold call omits** (`list_bills.type` 14, `repeat` 21, `alarm.create.label` 10, `get_forecast.location` 8, `list_events.date_from/to` 14, `get_status.room/floor` 11, `ride_hailing.book.payment` 5). A read with an invented filter then returns `[]`, and the agent stops (train_0075).
- Gold uses optional parameters in 40% of reads and 58% of writes that have them, so no fixed stripping rule works. This is a convention the model has to learn from examples, which is what few-shot and SFT give and prompt rules did not (v2 lost to v1).
- Prompts are 5.0k tokens on average, 7.3k at most (Qwen3 tokenizer, first 100 train samples, all tools and full history). **No tool retrieval or history trimming is needed**, and few-shot has ~8k tokens of headroom.
- The trace does not record the LLM output, rejected calls or why the agent stopped, so the 37 under-calling samples cannot be diagnosed yet.

Decisions that follow:
- Base model frozen at **Qwen3-4B-Instruct-2507**. Spec A2 (LLM comparison) is dropped, since SFT needs a fixed base now.
- Spec A3 (prompt v3) is dropped. Its behaviours (ask after several matches, check a condition first, minimal arguments) are taught through few-shot examples and SFT data instead.
- SFT starts in Phase 2 (31 Oct), not after the 2 Nov gate. It is the largest expected gain, and the team has never trained a model, so the pipeline has to be de-risked early.
- GRPO (Tier 3) is out of this plan. Write a separate plan only if SFT v2 is done by 10 Nov.

## Timeline

| Dates | Phase | Leaderboard submission at the end |
|---|---|---|
| 29–31 Oct | 1: Tasks 1–6 (Task 4 = presets + notebook dropdowns) | best few-shot k, bank = all train, best ASR from the ASR owner |
| 31 Oct – 5 Nov | 2: Tasks 7–11 | SFT v1 |
| 5–10 Nov | 3: Tasks 12–14 | SFT v2, then the final retrain |
| 10–14 Nov | 4: Task 15 | none new. Hand the chosen model to Docker packaging; freeze 14 Nov |

## Global Constraints

- LLM < 5B params, context ≤ 16,384 tokens, whole system < 6B; ≤ 30 s per sample on 1× A30 with ASR included; offline, fixed seeds, greedy decoding.
- Never read `<split>_environment/`. Exactly one `ToolEnv` per sample at inference. Training-time replays create a fresh train `ToolEnv` per record.
- **Leakage rule:** few-shot retrieval and every training job use pool samples only, except the final retrain (Task 14). Experiment numbers come from dev or holdout, never from the first 100 or from the pool.
- On `public_test` and private test, the few-shot bank uses **all** train samples.
- `requirements.txt` must not gain anything that pulls torch (`tests/test_requirements.py`). Training-only packages go in `requirements-train.txt`, installed on molab with `pip install --no-deps -r requirements-train.txt`.
  - That file is **shared with ASR fine-tuning**, which needs `accelerate` and `datasets`. Whoever creates it first owns the header comment; the other adds lines. Do not create a second file.
  - Pin nothing that conflicts with the other track without asking.
- Defaults stay at the best measured setup until a run beats it: `examples=0`, `prompt_version="v1"`, `check_refs=False`. New behaviour is opt-in through `--agent-kw`.
- Never edit a released prompt version. Never commit data, tokens, HF repo names, checkpoints or `train_records/`; add `train_records/` and `checkpoints/` to `.gitignore`.
- English everywhere in the repo. Branch `feat/agent-infra` for Phase 1, `feat/sft` for Phases 2–3, each cut from an up-to-date `main`, with a PR into `main`. The PR text gives `pytest` output and the dev/holdout `compare` lines.
- Every run that changes agent behaviour gets a `RESULTS.md` row, plus the `compare` line against the current best in its notes.

## Review Focus

1. **Holdout leakage through history:** a sample whose *history* uses a holdout app is a holdout sample. Tested in Task 1.
2. **Training text equals inference text:** a record whose replay through the agent does not reproduce exactly the gold calls is dropped and counted, never trained on with partial calls. Tested in Task 7.
3. **Loss mask:** only assistant tokens count, and the chat template must render every earlier turn identically when a turn is appended, or the mask shifts. Tested with the real Qwen3 tokenizer in Task 8.
4. **Masking must rename consistently everywhere:** tool schemas, history `tool_calls`, gold calls, and the `tool_name`/`params` values inside `sys.ask_missing_info` arguments. Tested in Task 12.
5. **Resuming a checkpoint continues the same data order** and does not repeat or skip examples. Tested in Task 9.

---

## Phase 1: infrastructure and few-shot (branch `feat/agent-infra`)

### Tasks 1 and 2: already built by the ASR owner (`feat/experiments-infra`, 28 Oct)

Batch-1 Tasks 1–5 are implemented on `origin/feat/experiments-infra`:
- `lifemate.splits` and `configs/splits.json`, with `infer --subset`;
- `lifemate.compare`;
- number-normalized WER/CER;
- the ASR cache: `lifemate.transcribe`, `lifemate.asr.cached.CachedASR`, `lifemate.asr_report`;
- progress output, the HF token and a command mode in the molab notebook.

Do not reimplement them.

- [x] **Step 1:** Check whether `feat/experiments-infra` is merged into `main` (`git log origin/main --oneline | grep "paired bootstrap"`).
  - If it is, cut `feat/agent-infra` from `main`.
  - If not, cut it from `origin/feat/experiments-infra`, and rebase onto `main` once that PR merges. The agent PR then says it depends on that PR.
- [x] **Step 2:** `pytest -v` → all pass on the new branch before any change.
- [x] **Step 3:** Check the interfaces used below exist with these names:
  - `lifemate.splits.load_splits() -> {"holdout_apps", "holdout", "dev", "pool", ...}`;
  - `lifemate.splits.sample_apps(sample)`;
  - `infer --subset`;
  - `python -m lifemate.compare runs/<a> runs/<b>`.

  If a name differs, use the existing one, and note it in the PR rather than renaming it.

### Task 3: Agent step trace and failure breakdown

**Files:**
- Modify: `src/lifemate/agents/base.py` (`Agent.last_trace`), `src/lifemate/agents/react.py`, `src/lifemate/infer.py`
- Create: `src/lifemate/diagnose.py`
- Test: `tests/test_react_agent.py` (extend), `tests/test_diagnose.py`

**Interfaces:**
- Produces:
  - `Agent.last_trace: dict | None`, reset at the start of every `run`. `ReActAgent` fills `{"steps": [{"output": str (first 2000 chars), "rejected": [str, ...]}], "stop": str}`.
    - Each rejected entry is `f"{name}: {reason}"`, where the reason is the validate error, `already called`, `unknown id ...` (check_refs) or `invalid JSON`.
    - `stop` is one of `"no_call"`, `"asked"`, `"time"`, `"tokens"`, `"invalid"`, `"max_steps"`.
  - `infer` writes `"agent": agent.last_trace` into each `trace.jsonl` line.
  - `diagnose(run_dir: Path, data_root: Path) -> dict`, plus CLI `python -m lifemate.diagnose runs/<run>`. The CLI prints the breakdown and writes `diagnose.json` next to `per_sample.jsonl`. Train only.
- Breakdown, over gold calls of act/ask samples:
  - `matched`, using `scoring.call_matches`;
  - `missing_read`, `missing_write`, `missing_ask`, split by catalog `kind`, for gold calls with no predicted call of the same name;
  - for same-name calls that do not match: `extra_key` and `missing_key` counted per `tool.param`, and `wrong_value` per param;
  - `extra_pred_calls`: predicted calls whose name is not in gold;
  - per sample, `fewer_calls` / `more_calls` than gold;
  - `stop` reasons counted from the trace when present.

- [x] **Step 1: Write the failing tests**

```python
def test_trace_records_rejections_and_stop_reason():
    llm = FakeLLM([CALL("alarm.delete", {}), "Xong."])
    agent = ReActAgent(llm=llm)
    agent.run(TASK, FakeEnv(bad={"alarm.delete"}))
    assert agent.last_trace["steps"][0]["rejected"] == ["alarm.delete: missing contact_id"]
    assert agent.last_trace["stop"] == "no_call"

def test_ask_stops_with_asked():
    agent = ReActAgent(llm=FakeLLM([CALL("sys.ask_missing_info", {"tool_name": "sms.send", "params": ["contact_id"]})]))
    agent.run(TASK, FakeEnv())
    assert agent.last_trace["stop"] == "asked"
```

  In `tests/test_diagnose.py`, build a two-sample fake run folder in `tmp_path`, using train sample ids (skip without data). Its submission holds one exact gold copy and one copy with an added optional key. Assert `matched` and `extra_key[<tool>.<param>] == 1`.
- [x] **Step 2:** `pytest tests/test_react_agent.py tests/test_diagnose.py -v` → FAIL.
- [x] **Step 3:** Implement the code. Every `return` in `ReActAgent.run` sets `stop` first.
- [x] **Step 4:** `pytest -v` → all pass.
- [x] **Step 5:** Run `python -m lifemate.diagnose runs/20261027-045012_5a7ce91_train_react_gold` locally. Expect matched 67, extra_key total 74. No stop reasons, since that run predates the trace. Paste the output into the PR.
- [x] **Step 6:** Commit `Record agent steps and stop reason; add failure breakdown`.

### Task 4: Named experiment presets and notebook dropdowns

Typing `--agent-kw` by hand makes runs hard to repeat and easy to get wrong. A preset is a named agent or ASR configuration. One name works in the CLI, the molab notebook and Docker, and `meta.json` records it. A new architecture or setting is one line in `configs/presets.json` on the branch, with no notebook change.

`notebooks/molab_run.py` was just changed on `feat/experiments-infra`. Do this task only on a branch that already contains that change (see Tasks 1 and 2), and tell the ASR owner that the notebook moved to presets.

**Files:**
- Create: `src/lifemate/presets.py`, `configs/presets.json`
- Modify: `src/lifemate/infer.py`, `notebooks/molab_run.py`
- Test: `tests/test_presets.py`

**Interfaces:**
- `configs/presets.json`: `{"agent": {NAME: {"agent": str, "agent_kw": {...}}}, "asr": {NAME: {"asr": str, "asr_kw": {...}}}}`.
  - A string value may contain `${VAR}`, filled from the environment at resolve time. Private HF repo owners are never committed. For example, the model of `sft-v1` is `"${HF_MODEL_OWNER}/lifemate-sft-v1"`.
  - Initial content:
    - agent: `empty`, `oracle`, `react-v1` (defaults), `react-v2-refs` (`prompt_version=v2`, `check_refs=true`);
    - asr: `gold`, `none`, `pho-medium` (`model=vinai/PhoWhisper-medium`), `pho-large` (`model=vinai/PhoWhisper-large`), `pho-large-fp32` (with `dtype=float32`).
- `presets.load_presets(path: Path = REPO_ROOT / "configs/presets.json") -> dict`.
- `presets.resolve(kind: str, name: str, environ: Mapping[str, str] = os.environ, path: Path = <committed file>) -> tuple[str, dict]` returns `(backend name, kwargs)`, where `kind` is `"agent"` or `"asr"`.
  - An ASR name `cached:<cache>` needs no entry and resolves to `("cached", {"name": "<cache>"})`.
  - An unknown name raises `KeyError` listing the known names.
  - An unset `${VAR}` raises `ValueError` naming the variable.
- CLI `python -m lifemate.presets --list` prints `{"agent": [names], "asr": [names]}` as JSON, for the notebook.
- `infer --agent-preset NAME --asr-preset NAME`:
  - The preset fills `--agent` / `--asr` and its kwargs. `--agent-kw` / `--asr-kw` given on top override single keys.
  - Giving both `--agent` and `--agent-preset` is an error (same for ASR).
  - `meta.json` gains `"agent_preset"` and `"asr_preset"` (null when unused).
- Notebook, in two steps:
  1. **Pull** clones, installs and fetches data, as today.
  2. A new cell runs `python -m lifemate.presets --list` in the pulled repo and shows two dropdowns:
     - agent preset;
     - ASR preset, plus one `cached:<name>` entry for each `lifemate_work/asr_cache/<split>/<name>.jsonl` present in the session.

  A new text field "HF model owner" is exported as `HF_MODEL_OWNER` to every step. **Run** calls `infer --agent-preset ... --asr-preset ...`, plus the existing extra-args field. Keep the split, limit, extra args and command mode as they are. The run folder name uses the preset names.

- [x] **Step 1: Write the failing tests**

```python
def test_resolve_fills_env_vars(tmp_presets):
    assert resolve("agent", "sft-test", {"HF_MODEL_OWNER": "me"}, tmp_presets) == ("react", {"model": "me/lifemate-sft-test"})

def test_unset_env_var_is_named(tmp_presets):
    with pytest.raises(ValueError, match="HF_MODEL_OWNER"):
        resolve("agent", "sft-test", {}, tmp_presets)

def test_cached_asr_needs_no_entry():
    assert resolve("asr", "cached:pho-large-fp16") == ("cached", {"name": "pho-large-fp16"})

def test_every_committed_preset_names_a_registered_backend():
    p = load_presets()
    assert all(v["agent"] in AGENTS for v in p["agent"].values())
    assert all(v["asr"] in BACKENDS for v in p["asr"].values())

def test_infer_records_presets(tmp_path):  # skip without data/train
    out = infer.main(["--split", "train", "--agent-preset", "oracle", "--asr-preset", "gold", "--limit", "2", "--out", str(tmp_path / "r")])
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["agent_preset"] == "oracle" and meta["agent"] == "oracle" and meta["asr_preset"] == "gold"
```

  `tmp_presets` is a fixture: a temporary presets file with one agent entry, `"sft-test": {"agent": "react", "agent_kw": {"model": "${HF_MODEL_OWNER}/lifemate-sft-test"}}`.
- [x] **Step 2:** `pytest tests/test_presets.py -v` → FAIL.
- [x] **Step 3:** Implement `presets.py` and the `infer` flags. Then update the notebook.
- [x] **Step 4:** `pytest -v` → all pass. Run `marimo check notebooks/molab_run.py` if marimo is installed.
- [x] **Step 5:** On molab, run preset `oracle` + `gold` with `--limit 5` on train. Expected score 100.0, and `meta.json` shows both preset names.
- [x] **Step 6:** Commit `Add named experiment presets and notebook dropdowns`. Tell the team to re-upload the notebook to molab.

From here on, every task that adds an agent or ASR setting also adds its preset, and every molab run in this plan uses presets.

### Task 5: Dynamic few-shot examples (A1)

Implement batch-1 Task 6 (`src/lifemate/agents/fewshot.py`, `examples=k` on `ReActAgent`, `examples_text` in `build_messages`, its tests and the token-budget check), with these changes:
- `ExampleBank.from_data(data_root: Path, split: str) -> ExampleBank`. The pool is `load_splits()["pool"]` when `split == "train"`, and every train id otherwise. `ReActAgent` gains `split: str = "train"`. `infer` does `agent_kw.setdefault("split", args.split)` when `"split"` is a parameter of the agent's `__init__` (`inspect.signature`). Add a test that `from_data(root, "public_test")` indexes all 1545 train samples.
- `info["examples"] = k` and `info["bank_size"] = len(pool)` go into meta.json.
- The token-budget check (batch-1 Task 6 Step 5) runs on train **and** public_test with `examples=4`. Record both maxima in the PR.
- Add the presets `react-fewshot-k2` and `react-fewshot-k4` (`examples=2` / `4`).

- [x] Steps 1–6 as in batch-1 Task 6, plus the test above.

### Task 6: Phase 1 runs on molab

Branch `feat/agent-infra`, with presets chosen in the notebook. Download every run folder into `runs/`, then run `evaluate` and `diagnose` locally.

- [x] **Step 1:** Dev baseline: split `train`, agent preset `react-v1`, ASR preset `gold`, extra args `--subset dev`. Expected: 300 samples, 0 errors.
- [x] **Step 2:** The same with `react-fewshot-k2`, then `react-fewshot-k4`. Run `compare` of k0 against k2 and of k0 against k4. Put the `RESULTS.md` rows and both compare lines in the PR. Max seconds must stay < 15.
- [x] **Step 3:** For the best k, rerun on `--subset holdout` against k=0 on holdout. This checks the gain holds for apps that have no examples in the pool.
- [x] **Step 4:** If a k wins on dev, with the CI above 0, and does not lose on holdout, run full public_test with that k and the ASR owner's current best ASR. Validate with `zaic_tools --validate` and submit. Log the leaderboard score.
- [x] **Step 5:** Open the PR. In the description, give the decision: the k used from now on, or 0.

---

## Phase 2: SFT v1 (branch `feat/sft`)

### Task 7: Training records rendered by replaying gold calls through the agent

**Files:**
- Create: `src/lifemate/sft/__init__.py`, `src/lifemate/sft/replay.py`, `src/lifemate/sft/records.py`
- Modify: `.gitignore` (`train_records/`, `checkpoints/`)
- Test: `tests/test_sft_replay.py`

**Interfaces:**
- Consumes: `infer.build_task`, `ReActAgent`, `load_splits` (Task 1), `GoldTranscriptASR`, train `zaic_tools.ToolEnv`.
- Produces:
  - `class ReplayLLM` (implements `agents.llm.ChatLLM`). `__init__(gold_calls: list[dict], final_text: str)`. Each `generate` returns the next gold call as one `<tool_call>{json}</tool_call>`, then `final_text`. It records `(messages, tools, output)` for every call. `count_tokens` returns 1.
  - `replay(sample: dict, task: Task, env) -> dict | None` runs `ReActAgent(llm=ReplayLLM(...), prompt_version=..., clock=<frozen clock>)` on the task and returns `{"id", "messages", "tools"}`. `messages` is the last recorded prompt plus one assistant message for the last output: `{"role": "assistant", "content": "", "tool_calls": [...]}` for a call, `{"role": "assistant", "content": text}` for text. It returns `None` when `env.get_tool_calls()` differs from the gold calls.
  - Final text: `"Đã xong."` for act samples; for out-of-scope samples, `"Xin lỗi, tôi không làm được việc này."`; none for ask samples, which end on the ask call. Exact copy, stored as `FINAL_TEXT` and `REFUSAL_TEXT` in `replay.py`.
  - CLI `python -m lifemate.sft.records --name NAME --subset pool|all [--text gold|cache:<asr_name>] [--mask K] [--seed 0] [--prompt v1]` writes `train_records/NAME.jsonl`, one replayed conversation per line with `"source"` set to `gold`, `asr` or `masked`. It prints how many records were written, how many replays were dropped, and the max token count. `--text` and `--mask` are added in Tasks 12–13. In this task only `gold` and 0 work, and the others exit with a clear message.
- One gold call per assistant step, in label order. This matches what the agent can execute at inference: one call, then the result, then the next.

- [x] **Step 1: Write the failing tests** (skip without data). Put the helper `replay_sample(sample) -> dict | None` in `tests/conftest.py`: it builds the task with gold ASR and a fresh train `ToolEnv`, then calls `replay`. Tasks 8 and 12 build their record fixtures with it.

```python
def test_replay_reproduces_gold_calls_for_every_kind(train_samples):
    for kind in ("act", "ask", "empty"):
        s = next(x for x in train_samples if label_kind(x["label"]["tool_calls"]) == kind)
        rec = replay_sample(s)  # test helper: builds task + fresh env, calls replay
        assert rec is not None
        calls = [c["function"] for m in rec["messages"] if m["role"] == "assistant" for c in m.get("tool_calls", [])]
        assert [(c["name"], c["arguments"]) for c in calls] == [(g["name"], g["arguments"]) for g in s["label"]["tool_calls"]]

def test_replay_messages_match_what_the_agent_sends(train_samples):
    s = next(x for x in train_samples if len(x["label"]["tool_calls"]) >= 2)
    rec = replay_sample(s)
    assert rec["messages"][0]["role"] == "system" and rec["messages"][0]["content"] == system_prompt(s["context"], "v1")
    assert [m["role"] for m in rec["messages"]].count("tool") == len(s["label"]["tool_calls"])

def test_ask_ends_on_the_ask_call_and_act_on_final_text(train_samples):
    ask = next(x for x in train_samples if label_kind(x["label"]["tool_calls"]) == "ask")
    act = next(x for x in train_samples if label_kind(x["label"]["tool_calls"]) == "act")
    assert replay_sample(ask)["messages"][-1]["tool_calls"][0]["function"]["name"] == "sys.ask_missing_info"
    assert replay_sample(act)["messages"][-1] == {"role": "assistant", "content": FINAL_TEXT}

def test_records_cli_uses_pool_only(tmp_path, monkeypatch):
    # run records.main(["--name", "t", "--subset", "pool"]) with --limit 30; every id is in load_splits()["pool"]
    ...
```

- [x] **Step 2:** `pytest tests/test_sft_replay.py -v` → FAIL.
- [x] **Step 3:** Implement `replay.py` and `records.py`. The CLI imports train `zaic_tools` once (`--split` is always train).
- [x] **Step 4:** `pytest -v` → all pass. Then run `python -m lifemate.sft.records --name gold-pool --subset pool` locally. Record in the PR: records written, replays dropped (expected 0; investigate any), max tokens.
- [x] **Step 5:** Commit `Render SFT records by replaying gold calls through the agent`.

### Task 8: Tokenization with an assistant-only loss mask

**Files:**
- Create: `src/lifemate/sft/tokenize.py`
- Test: `tests/test_sft_tokenize.py`

**Interfaces:**
- Produces: `encode_record(record: dict, tokenizer, max_len: int = 12288) -> dict | None` returns `{"input_ids": list[int], "labels": list[int]}`, with labels −100 outside assistant turns. It returns `None` when the record is longer than `max_len`. The caller counts those, and they are never truncated.
- Method: render with `tokenizer.apply_chat_template(messages[:i+1], tools=record["tools"], tokenize=True)` for every prefix that ends at an assistant message. The tokens added over the previous prefix, which was rendered with `add_generation_prompt=True`, are that assistant turn. Assert that each shorter render is a token prefix of the longer one, and raise `ValueError` naming the record id if it is not.

- [x] **Step 1: Write the failing tests.** Use the real `Qwen/Qwen3-4B-Instruct-2507` tokenizer from the local HF cache, and skip if it is not cached.

```python
def test_only_assistant_tokens_are_labelled(tok, two_call_record):
    enc = encode_record(two_call_record, tok)
    labelled = tok.decode([t for t, l in zip(enc["input_ids"], enc["labels"]) if l != -100])
    assert labelled.count("<tool_call>") == 2 and "Đã xong." in labelled
    assert "<tool_response>" not in labelled and "You are a voice assistant" not in labelled

def test_labels_end_with_end_of_turn(tok, two_call_record):
    enc = encode_record(two_call_record, tok)
    last = [t for t, l in zip(enc["input_ids"], enc["labels"]) if l != -100][-3:]
    assert tok.convert_tokens_to_ids("<|im_end|>") in last

def test_too_long_record_is_skipped(tok, two_call_record):
    assert encode_record(two_call_record, tok, max_len=100) is None
```

- [x] **Step 2:** `pytest tests/test_sft_tokenize.py -v` → FAIL.
- [x] **Step 3:** Implement it.
- [x] **Step 4:** `pytest -v` → all pass. Then encode all of `train_records/gold-pool.jsonl` locally and record in the PR: labelled tokens per record (mean) and total tokens per epoch.
- [x] **Step 5:** Commit `Tokenize SFT records with an assistant-only loss mask`.

### Task 9: LoRA trainer with checkpoint/resume, training log, plots and merge

**Files:**
- Create: `src/lifemate/sft/train.py`, `src/lifemate/sft/state.py`, `src/lifemate/sft/plots.py`, `src/lifemate/sft/merge.py`, `requirements-train.txt` (`peft`, `matplotlib`, each with a comment)
- Test: `tests/test_sft_state.py`

**Interfaces:**
- Consumes: `encode_record` (Task 8).
- Produces:
  - `state.epoch_order(n: int, epoch: int, seed: int) -> list[int]`: a permutation from `random.Random(seed * 1000 + epoch)`.
  - `state.TrainState` (dataclass): `epoch`, `pos` (index into that epoch's order), `step`, `seed`, plus `to_json()` / `from_json()`.
  - `state.batches(n, state, micro_bs, epochs) -> Iterator[list[int]]` walks `epoch_order` epoch after epoch from `(state.epoch, state.pos)` up to `epochs`, and advances `state` in place before each yield. A checkpoint is written only at optimizer-step boundaries (every `grad_accum` micro-batches), so the saved `state` points at the first micro-batch not yet trained on.
  - CLI `python -m lifemate.sft.train --records NAME [--eval-records NAME] --out checkpoints/RUN --hf-repo USER/REPO [--resume] [--max-steps N]`. Defaults:
    - LoRA r 16, alpha 32, dropout 0.05, on `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`;
    - lr 1e-4, cosine schedule, 3% warmup;
    - 2 epochs, micro-batch 1, gradient accumulation 16;
    - bf16, gradient checkpointing, max_len 12288, seed 0.
  - **Checkpoint** every 30 min and at the end. It holds the adapter, optimizer, scheduler, torch/cuda/python RNG state and `TrainState`. It is uploaded to the private HF model repo under `checkpoints/RUN/last/`. `--resume` downloads it and continues.
  - **Log** `train_log.jsonl`, one line per optimizer step: `step, epoch, loss, lr, grad_norm, tokens, seconds`. Eval loss on `--eval-records` every 50 steps. Uploaded with each checkpoint.
  - `plots.plot_log(log_path: Path, out_dir: Path)` writes `loss.png`, `lr.png` and `grad_norm.png`.
  - `python -m lifemate.sft.merge --adapter checkpoints/RUN/last --out merged/RUN --hf-repo USER/REPO` merges into bf16 base weights, saves the tokenizer alongside, and uploads. Inference then uses `--agent-kw model=USER/REPO` (or a local path) without peft.

- [x] **Step 1: Write the failing tests** for the pure-Python parts:

```python
def test_resume_continues_the_same_order():
    full = list(batches(10, TrainState(epoch=0, pos=0, step=0, seed=0), micro_bs=1, epochs=2))
    assert len(full) == 20
    st = TrainState(epoch=0, pos=0, step=0, seed=0)
    first = []
    for b in batches(10, st, micro_bs=1, epochs=2):
        first.append(b)
        if len(first) == 13:  # stop inside epoch 1
            break
    resumed = TrainState.from_json(st.to_json())
    assert first + list(batches(10, resumed, micro_bs=1, epochs=2)) == full

def test_epochs_use_different_orders():
    assert epoch_order(50, 0, 0) != epoch_order(50, 1, 0) and sorted(epoch_order(50, 1, 0)) == list(range(50))

def test_plot_log_writes_pngs(tmp_path):
    # write 5 fake log lines, call plot_log, assert the three PNGs exist (skip if matplotlib missing)
    ...
```

- [x] **Step 2:** `pytest tests/test_sft_state.py -v` → FAIL.
- [x] **Step 3:** Implement `state.py` and `plots.py`. Then implement `train.py` and `merge.py`, importing torch and peft inside `main()` so tests and the registry never import them. The training loop is plain PyTorch: loss = mean cross-entropy over labelled tokens per micro-batch, gradient clipping 1.0, `seed_everything(seed)` before building the model.
- [x] **Step 4:** `pytest -v` → all pass, `test_requirements` included.
- [x] **Step 5:** Commit `Add LoRA SFT trainer with checkpoint/resume, training log, plots and merge`.

### Task 10: molab training notebook

**Files:**
- Create: `notebooks/molab_train.py` (marimo)

The notebook copies the clone/fetch-data cells of `notebooks/molab_run.py`. Fields:
- Branch;
- GitHub token;
- HF dataset and HF token (read);
- HF model repo and HF token (write);
- run name;
- records name;
- `--resume` checkbox;
- `--max-steps` (0 = full).

It then:
1. installs `requirements.txt`, then `requirements-train.txt` with `--no-deps`, and prints `torch.__file__` to show it is the system torch;
2. runs `sft.records` (pool), then `sft.train` as a subprocess with streamed output;
3. redraws the three plots from `train_log.jsonl` every 60 s while training runs;
4. offers `sft.merge` as a button once training ends.

- [x] **Step 1:** Write the notebook. `marimo check notebooks/molab_train.py` passes (if marimo is installed locally), and `pytest -v` still passes.
- [x] **Step 2:** Commit `Add molab training notebook`. Tell the team to upload it to molab.

### Task 11: SFT v1 runs on molab

- [x] **Step 1: Smoke test**, `--max-steps 20`. Check that the loss goes down, then stop the run and restart it with `--resume`. The log continues at step 21 with no repeated step. Record tokens/s and the projected time for 2 epochs. If the projection is over 10 h, drop to 1 epoch.
- [x] **Step 2: Full run** on `gold-pool`, with eval records from dev (`--subset dev` records, used only for eval loss). Then merge to the HF repo `<owner>/lifemate-sft-v1`, and add the agent preset `sft-v1` (`model=${HF_MODEL_OWNER}/lifemate-sft-v1`, plus the Phase 1 winner's `examples` if any) to `configs/presets.json`.
- [x] **Step 3:** Run agent preset `sft-v1` with ASR preset `gold`, extra args `--subset dev`, then the same on holdout. Run `compare` against the best Phase 1 run on each subset. Run `diagnose` on the dev run. Watch three numbers: the `extra_key` total (expected to drop sharply), the ask column, and max seconds (< 15).
- [x] **Step 4:** If SFT v1 beats Phase 1 on dev and on holdout, run full public_test with it and the best ASR, validate it and submit. Add the `RESULTS.md` rows and open the PR. It records Gate 2 of the spec.

---

## Phase 3: robustness and final model (branch `feat/sft`, continued)

### Task 12: Function masking (Hammer)

**Files:**
- Create: `src/lifemate/sft/mask.py`
- Modify: `src/lifemate/sft/records.py` (`--mask K`)
- Test: `tests/test_sft_mask.py`

**Interfaces:**
- Produces: `mask_record(record: dict, rng: random.Random) -> dict`. It returns a new record with `source="masked"` in which:
  - each app prefix, tool name suffix and parameter name is replaced by a plausible identifier (English word lists in the module; `sys.ask_missing_info` keeps its name);
  - the same mapping is applied to:
    - the `tools` schemas: property keys and `required`;
    - every assistant `tool_calls` entry in `messages`;
    - the `tool_name` and `params` values of `sys.ask_missing_info` calls;
    - every whole-word occurrence of an original tool name or parameter name inside description strings, such as the `Provide one of: a + b` suffix and the `(id from <tool>)` notes written by `prompting._clean_property`;
  - all other description text, enum values, argument values, user and system text, and tool results are unchanged;
  - the tool order is shuffled.
- `--mask K` writes K masked copies per gold record, in addition to the original.

- [x] **Step 1: Write the failing tests**

```python
def test_mask_renames_consistently(ask_record):
    m = mask_record(ask_record, random.Random(0))
    names = {t["function"]["name"] for t in m["tools"]}
    for msg in m["messages"]:
        for c in msg.get("tool_calls", []):
            assert c["function"]["name"] in names
    ask = m["messages"][-1]["tool_calls"][0]["function"]["arguments"]
    assert ask["tool_name"] in names
    target = next(t for t in m["tools"] if t["function"]["name"] == ask["tool_name"])
    assert set(ask["params"]) <= set(target["function"]["parameters"]["properties"])

def test_mask_leaves_no_original_names_in_tools(act_record):
    m = mask_record(act_record, random.Random(0))
    text = json.dumps(m["tools"], ensure_ascii=False)
    for t in act_record["tools"]:
        assert t["function"]["name"] not in text or t["function"]["name"] == "sys.ask_missing_info"

def test_mask_keeps_results_and_user_text(act_record):
    m = mask_record(act_record, random.Random(0))
    keep = lambda r: [(x["role"], x["content"]) for x in r["messages"] if x["role"] in ("tool", "user", "system")]
    assert keep(m) == keep(act_record)

def test_mask_is_deterministic(act_record):
    assert mask_record(act_record, random.Random(3)) == mask_record(act_record, random.Random(3))
```

- [x] **Step 2:** `pytest tests/test_sft_mask.py -v` → FAIL.
- [x] **Step 3:** Implement it.
- [x] **Step 4:** `pytest -v` → all pass.
- [x] **Step 5:** Commit `Add function masking for SFT records`.

### Task 13: ASR-noisy records

Depends on an ASR cache from the ASR owner, `asr_cache/train/<name>.jsonl` (written by `lifemate.transcribe --subset pool`), covering the pool's final and history clips. If that cache is not ready by 6 Nov, skip this task and note it in the PR.

**The cache must come from an ASR model that was not fine-tuned on pool clips**: the B1 winner off the shelf, for example `pho-large-fp16`. B2 fine-tunes on exactly these clips, so its transcripts of them would be unrealistically clean, and the LLM would not learn real ASR errors. Ask the ASR owner for this cache early. It is one transcribe run of about 4,000 clips.

**Files:**
- Modify: `src/lifemate/sft/records.py` (`--text cache:<name>`)
- Test: `tests/test_sft_replay.py` (extend)

**Interfaces:**
- `--text cache:<name>` builds tasks with the existing `lifemate.asr.cached.CachedASR(name, split="train")` in place of `GoldTranscriptASR`. It writes records with `source="asr"`, and their gold calls and results are unchanged. A missing clip raises `CachedASR`'s `KeyError`, which names the file.

- [x] **Step 1:** Write two tests. With a two-line temporary cache, the request text in the record comes from the cache. A missing clip raises `KeyError` with the file name.
- [x] **Step 2:** Run them and watch them fail. Then implement.
- [x] **Step 3:** `pytest -v` → all pass.
- [x] **Step 4:** Commit `Build SFT records from cached ASR transcripts`.

### Task 14: SFT v2 and the final retrain

- [ ] **Step 1:** Build the records `v2-pool` = gold + `--mask 1` + (if Task 13 is done) the ASR copy. Train them as in Task 11, then merge to `<owner>/lifemate-sft-v2`. Add the preset `sft-v2`; the final retrain gets `sft-final`.
- [ ] **Step 2:** Run dev and holdout, with gold text and with the best ASR cache, and `compare` against SFT v1. Keep v2 only if holdout improves and the ask column does not drop by more than 5 points. The Calibration paper warns that masking can hurt asking.
- [ ] **Step 3: Final retrain.** Same recipe as the winner, `--subset all` (the leakage exemption in Global Constraints), same seed and hyperparameters. It has no held-out score, so accept it only if public_test is not lower than the gated model's. One submission.
- [ ] **Step 4:** Log the `RESULTS.md` rows, then open the PR.

## Phase 4: hand-off to Docker

### Task 15: Timing and determinism check of the chosen system

- [ ] **Step 1:** Run full public_test twice with the chosen model, ASR and settings. The two `submission.jsonl` files must be byte-identical (`cmp`). Record mean and max seconds; max must be < 15 on molab.
- [ ] **Step 2:** Write into `RESULTS.md` notes the exact inference command (agent and ASR preset names) and the model repo revision (commit hash). In `configs/presets.json`, pin that revision through a `revision` kwarg in the shipped preset: add `revision: str | None = None` to `HFChatLLM` and `ReActAgent` and pass it to `from_pretrained`. Give them to the Docker owner. Add the training commands (records, train, merge, with seeds) to `README.md`, because the Docker package must ship training code.
- [ ] **Step 3:** Commit `Document the shipped model and how to retrain it` on a `docs/` branch with a PR.
