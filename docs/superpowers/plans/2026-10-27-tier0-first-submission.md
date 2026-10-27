# Tier 0 First Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get a training-free tool-use agent and a real Vietnamese ASR running end to end, so that 1–2 public_test results can be submitted today (27 Oct 2026).

**Architecture:** A new `whisper` ASR backend (PhoWhisper via HF transformers) and a new `react` agent: the off-the-shelf instruct LLM is prompted in its native tool-call chat format, and every proposed call is checked with `env.validate()` before `env.call()`. The loop stops when the model emits no call, after `sys.ask_missing_info`, or on the step or time budget. Prompt building, output parsing and the loop are pure Python with an injected LLM, so they are unit-tested locally with fakes. The model itself only runs on molab.

**Tech Stack:** Python ≥ 3.10, `transformers`, `accelerate`, `soundfile` (torch comes preinstalled on molab), pytest.

**Spec:** `docs/superpowers/specs/2026-10-27-agent-roadmap-design.md`. This plan covers Tier 0 only, without tool retrieval (see Global Constraints for why that is safe today).

## Global Constraints

- LLM < 5B params, prompt + generation ≤ 16,384 tokens per call. Whole system < 6B. Defaults: LLM `Qwen/Qwen3-4B-Instruct-2507` (4.0B, non-thinking) and ASR `vinai/PhoWhisper-medium` (0.77B).
- ≤ 30 s per sample including ASR. The agent's own budget defaults to `time_budget_s=20`.
- Greedy decoding only (`do_sample=False`, `num_beams=1`). Call `seed_everything` after model build; the runner already does this.
- The agent never creates a `ToolEnv` and never reads `<split>_environment/`. A call that fails `env.validate()` is **never** passed to `env.call()`.
- `torch` and `transformers` are imported lazily inside backend constructors, so `pytest` keeps working on machines without torch.
- All code, comments, prompts and commits in English. Branch `feat/agent-roadmap`, PR into `main`, never push to `main`.
- No tool retrieval today: installed specs are ≤ ~25k characters (about 10k tokens), which fits the budget once the oldest history turns are dropped. Retrieval comes in a later plan.

## Review Focus

1. **Malformed model output** (broken JSON inside `<tool_call>`, unknown tool name): no crash, no `env.call`. The error is fed back once, then the loop stops. Tested in Task 4 (parser) and Task 6 (loop).
2. **A model that repeats the same call**: an identical (name, arguments) call is not sent twice. The loop ends at `max_steps`. Tested in Task 6.
3. **Long history plus big specs over the token budget**: the oldest history turns are dropped first, and tool specs and the final request are never dropped. Tested in Task 3.
4. **Time budget hit mid-loop**: stop cleanly, and calls already made stay in the submission. Tested in Task 6 with a fake clock.
5. **`sys.ask_missing_info`**: after the agent calls it, it makes no more calls. Tested in Task 6.

---

### Task 1: Pass constructor options to agents and ASR from the CLI

**Files:**
- Modify: `src/lifemate/infer.py` (argparse + `build_asr`/`build_agent` calls)
- Modify: `src/lifemate/utils.py` (add `parse_kw`)
- Test: `tests/test_utils.py` (create)

**Interfaces:**
- Produces: `parse_kw(items: list[str] | None) -> dict[str, object]` in `lifemate.utils`. It parses `key=value`; each value goes through `json.loads` and falls back to the raw string. Also produces the CLI flags `--agent-kw KEY=VALUE` and `--asr-kw KEY=VALUE` (repeatable, `action="append"`), passed as `build_agent(args.agent, **parse_kw(args.agent_kw))` and likewise for ASR. Both flags are recorded in `meta.json`.

- [ ] **Step 1: Write the failing test**

```python
from lifemate.utils import parse_kw

def test_parse_kw_types():
    assert parse_kw(["model=Qwen/Qwen3-4B-Instruct-2507", "max_steps=6", "time_budget_s=12.5", "flag=true"]) == {
        "model": "Qwen/Qwen3-4B-Instruct-2507", "max_steps": 6, "time_budget_s": 12.5, "flag": True}

def test_parse_kw_empty():
    assert parse_kw(None) == {}
```

- [ ] **Step 2:** `pytest tests/test_utils.py -v` → FAIL (ImportError).
- [ ] **Step 3:** Implement `parse_kw` and wire up both flags in `infer.py`. Split on the first `=` only.
- [ ] **Step 4:** `pytest -v` → all pass, including the oracle end-to-end.
- [ ] **Step 5:** Commit `Pass agent/ASR constructor options from the infer CLI`.

### Task 2: WER in evaluate

**Files:**
- Create: `src/lifemate/asr/wer.py`
- Modify: `src/lifemate/evaluate.py` (add `asr_wer` to `metrics.json` and to the `RESULTS.md` notes)
- Test: `tests/test_wer.py`

**Interfaces:**
- Produces: `normalize(text: str) -> list[str]` (NFC, lowercase, strip punctuation, split on whitespace) and `wer(ref: str, hyp: str) -> float` (word-level Levenshtein distance / number of ref words. When ref is empty: 0.0 if hyp is empty too, otherwise 1.0).
- `evaluate()` adds `metrics["asr_wer"]` (mean WER of the final-turn `request` in `trace.jsonl` against train `audio_text`, rounded to 4 decimals) only when `trace.jsonl` exists, the split is train and the run's ASR is not `gold` or `none` (read from `meta.json`).

- [ ] **Step 1: Write the failing tests**

```python
from lifemate.asr.wer import wer

def test_wer_ignores_case_and_punctuation():
    assert wer("Tắt đèn phòng khách.", "tắt đèn phòng khách") == 0.0

def test_wer_counts_substitution_and_deletion():
    assert wer("chuyển cho mẹ năm trăm nghìn", "chuyển cho mẹ năm nghìn") == 1 / 6

def test_wer_nfc():
    assert wer("hoà", "hoà") == 0.0
```

- [ ] **Step 2:** `pytest tests/test_wer.py -v` → FAIL.
- [ ] **Step 3:** Implement `wer.py`, then the `evaluate.py` addition.
- [ ] **Step 4:** `pytest -v` → pass.
- [ ] **Step 5:** Commit `Report ASR word error rate in evaluate`.

### Task 3: Prompt building

**Files:**
- Create: `src/lifemate/agents/prompting.py`
- Test: `tests/test_prompting.py`

**Interfaces:**
- Consumes: `Task` (`lifemate.task`).
- Produces:
  - `tool_schema(spec: dict) -> dict`: returns `{"type": "function", "function": {"name", "description", "parameters"}}`. If `parameters` has `x-one-of`, append `" Provide one of: a | b."` (groups joined with `" or "` inside a group, `" | "` between groups) to the description. Drop catalog-only keys (`kind`, `returns`, `app`) and every `x-*` key inside parameters, except that `x-match: fuzzy` becomes `" (free text)"` appended to that property's description.
  - `system_prompt(context: dict) -> str`: the rules below, plus `current_time`, `timezone`, `location`, `user_profile` as compact JSON.
  - `history_messages(history: list[dict]) -> list[dict]`: user turns become `{"role": "user", "content": text}`. An assistant turn with calls becomes one `{"role": "assistant", "content": text, "tool_calls": [{"type": "function", "function": {"name", "arguments": dict}}]}` followed by one `{"role": "tool", "content": json.dumps(result, ensure_ascii=False)}` per result. An assistant turn without calls becomes plain assistant content.
  - `build_messages(task: Task, count_tokens: Callable[[list[dict]], int], budget: int) -> list[dict]`: returns `[system, *history, user(request)]`. While `count_tokens(messages) > budget`, it drops the oldest history turn: a user turn together with the assistant/tool messages that follow it. If nothing is left to drop, it returns as is.
- System prompt rules (exact meaning, wording free):
  1. You are a voice assistant that acts through the given tools. The user's words come from speech recognition and may contain recognition errors.
  2. Handle only the user's **last** message. Earlier turns are context, and their actions are already done.
  3. Call a read tool only when you need its data (an id, a current value, whether something exists) to decide or to fill a parameter.
  4. Use ids and values exactly as tools return them. Dates and times in ISO 8601 with the context's timezone offset.
  5. If a required parameter cannot be determined from the request, context, history or tool results, call `sys.ask_missing_info` with that tool's name and the missing parameter names, then stop.
  6. If none of the tools can do the request, reply in one short sentence without calling any tool.
  7. When the request is done, reply in one short sentence without calling any tool.

- [ ] **Step 1: Write the failing tests**

```python
def test_tool_schema_one_of_and_fuzzy():
    spec = {"name": "sms.send", "kind": "write", "app": "sms", "returns": "", "description": "Send SMS",
            "parameters": {"type": "object", "properties": {
                "contact_id": {"type": "string", "x-ref": "contacts.contacts"},
                "phone_number": {"type": "string"},
                "content": {"type": "string", "x-match": "fuzzy", "description": "Body"}},
                "required": ["content"], "x-one-of": [["contact_id"], ["phone_number"]]}}
    fn = tool_schema(spec)["function"]
    assert fn["name"] == "sms.send" and fn["description"].endswith("Provide one of: contact_id | phone_number.")
    assert "x-one-of" not in fn["parameters"] and "x-ref" not in fn["parameters"]["properties"]["contact_id"]
    assert fn["parameters"]["properties"]["content"]["description"] == "Body (free text)"

def test_history_messages_pairs_calls_with_results():
    msgs = history_messages([{"role": "user", "text": "báo thức mai?"},
        {"role": "assistant", "text": "Có 1.", "tool_calls": [{"name": "alarm.list", "arguments": {}}], "tool_results": [{"ok": True}]}])
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool"]
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "alarm.list"

def test_build_messages_drops_oldest_history_first():
    task = Task(id="t", request="tắt đèn", context=CTX, tools=[], history=[
        {"role": "user", "text": "u1"}, {"role": "assistant", "text": "a1"},
        {"role": "user", "text": "u2"}, {"role": "assistant", "text": "a2"}])
    msgs = build_messages(task, count_tokens=len, budget=4)   # 1 token per message
    assert [m.get("content") for m in msgs[1:]] == ["u2", "a2", "tắt đèn"]
    assert msgs[0]["role"] == "system"
```

(`CTX` is a module-level dict with `current_time`, `timezone`, `location`, `user_profile`, `installed_apps`.)

- [ ] **Step 2:** `pytest tests/test_prompting.py -v` → FAIL.
- [ ] **Step 3:** Implement `prompting.py`.
- [ ] **Step 4:** `pytest tests/test_prompting.py -v` → PASS.
- [ ] **Step 5:** Commit `Add prompt building for the tool-use agent`.

### Task 4: Parse tool calls from model output

**Files:**
- Create: `src/lifemate/agents/parsing.py`
- Test: `tests/test_parsing.py`

**Interfaces:**
- Produces: `parse_tool_calls(text: str) -> tuple[list[dict], list[str]]`. It returns `(calls, errors)`. It finds every `<tool_call>…</tool_call>` block (the Qwen native format), `json.loads` the inside and expects `{"name": str, "arguments": dict}`; `arguments` given as a JSON string is decoded too. A block that fails to parse adds a short message to `errors` and no call. Text without blocks gives `([], [])`.

- [ ] **Step 1: Write the failing tests**

```python
def test_two_calls():
    t = '<tool_call>\n{"name": "alarm.list", "arguments": {}}\n</tool_call>\n<tool_call>{"name": "a.b", "arguments": "{\\"x\\": 1}"}</tool_call>'
    assert parse_tool_calls(t) == ([{"name": "alarm.list", "arguments": {}}, {"name": "a.b", "arguments": {"x": 1}}], [])

def test_plain_text_means_no_call():
    assert parse_tool_calls("Đã tắt đèn.") == ([], [])

def test_broken_json_is_an_error_not_a_call():
    calls, errors = parse_tool_calls('<tool_call>{"name": "alarm.list", "arguments": {</tool_call>')
    assert calls == [] and len(errors) == 1
```

- [ ] **Step 2:** `pytest tests/test_parsing.py -v` → FAIL.
- [ ] **Step 3:** Implement with a non-greedy regex over `<tool_call>(.*?)</tool_call>` with DOTALL.
- [ ] **Step 4:** `pytest tests/test_parsing.py -v` → PASS.
- [ ] **Step 5:** Commit `Parse native tool-call blocks from model output`.

### Task 5: LLM and ASR backends (GPU code, verified on molab)

**Files:**
- Create: `src/lifemate/agents/llm.py`
- Create: `src/lifemate/asr/whisper.py`
- Modify: `src/lifemate/asr/__init__.py` (register `"whisper": WhisperASR`)
- Modify: `requirements.txt` (add `transformers>=4.51`, `accelerate`, `soundfile`; leave torch to molab's preinstall)
- Test: `tests/test_registry.py`

**Interfaces:**
- Produces:
  - `class ChatLLM(Protocol)`: `generate(messages: list[dict], tools: list[dict]) -> str` and `count_tokens(messages: list[dict], tools: list[dict]) -> int`.
  - `class HFChatLLM`: `__init__(model: str = "Qwen/Qwen3-4B-Instruct-2507", max_new_tokens: int = 384, dtype: str = "bfloat16")`. It uses `tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True)`, `model.generate(do_sample=False, num_beams=1, max_new_tokens=…)`, and decodes only the new tokens, keeping special tokens out but keeping the `<tool_call>` text. Exposes `context_limit = 16384`.
  - `class WhisperASR(ASR)` with `name = "whisper"`: `__init__(model: str = "vinai/PhoWhisper-medium", device: str = "cuda")`. Reads the wav with `soundfile`, resamples only if it is not 16 kHz, and calls the HF `automatic-speech-recognition` pipeline with `generate_kwargs={"num_beams": 1, "do_sample": False}`. Returns stripped text.

- [ ] **Step 1: Write the failing test** (registry only; the model code is checked on molab in Task 7)

```python
def test_backends_registered_without_importing_torch():
    import sys
    from lifemate.asr import BACKENDS
    from lifemate.agents import AGENTS
    assert "whisper" in BACKENDS and "react" in AGENTS
    assert "torch" not in sys.modules
```

- [ ] **Step 2:** `pytest tests/test_registry.py -v` → FAIL. It keeps failing on `react` until Task 6 registers it, and that is expected.
- [ ] **Step 3:** Implement `llm.py` and `whisper.py`, with `torch`/`transformers` imported inside `__init__` only, and register `whisper`.
- [ ] **Step 4:** `pytest -v` → everything passes except `test_registry` (waiting on `react`).
- [ ] **Step 5:** Commit `Add HF chat LLM and PhoWhisper ASR backends`.

### Task 6: The `react` agent loop

**Files:**
- Create: `src/lifemate/agents/react.py`
- Modify: `src/lifemate/agents/__init__.py` (register `"react": ReActAgent`)
- Test: `tests/test_react_agent.py`

**Interfaces:**
- Consumes: `build_messages`, `tool_schema` (Task 3), `parse_tool_calls` (Task 4), `ChatLLM`/`HFChatLLM` (Task 5).
- Produces: `class ReActAgent(Agent)` with `name = "react"`, `__init__(llm: ChatLLM | None = None, model: str = "Qwen/Qwen3-4B-Instruct-2507", max_steps: int = 8, max_invalid: int = 2, time_budget_s: float = 20.0, clock: Callable[[], float] = time.perf_counter)`. It builds `HFChatLLM(model)` only when `llm` is None.
- `run(task, env)` behaviour:
  - Before every generation it checks the clock and stops if `clock() - start > time_budget_s`. The prompt budget is `context_limit - max_new_tokens`, measured with `llm.count_tokens`.
  - For each parsed call, in order:
    - `sys.ask_missing_info`: validate, call it, then stop the whole run.
    - An exact repeat of an already-executed `(name, canonical-JSON arguments)`: skip it. Feed back `"Already called; reuse the earlier result."` as the tool message.
    - Otherwise `env.validate`. If that fails, feed back `"Rejected: <error>. Fix the call."` as the tool message and add 1 to the invalid count. Otherwise run `env.call` and feed back the JSON result.
  - The assistant message (with its `tool_calls`) and its tool messages go into the conversation before the next generation.
  - Parse errors are fed back as a user message `"Your tool call was not valid JSON: …"` and count as invalid.
  - Stop when: no calls and no parse errors, the invalid count exceeds `max_invalid`, `max_steps` generations are reached, or the time budget is hit.

- [ ] **Step 1: Write the failing tests** with `FakeLLM(outputs: list[str])` (returns the outputs in order, `count_tokens` = 1, `context_limit = 16384`, `max_new_tokens = 0`), and `FakeEnv` (records `validated` and `called`; `validate` rejects names in a given `bad` set, and `call` returns `{"ok": True}`).

```python
CALL = lambda n, a: f'<tool_call>{{"name": "{n}", "arguments": {json.dumps(a)}}}</tool_call>'

def test_plain_answer_makes_no_calls():
    env = FakeEnv(); ReActAgent(llm=FakeLLM(["Không làm được."])).run(TASK, env)
    assert env.called == []

def test_invalid_call_is_never_executed():
    env = FakeEnv(bad={"sms.send"})
    ReActAgent(llm=FakeLLM([CALL("sms.send", {}), "Xong."])).run(TASK, env)
    assert env.called == [] and env.validated == ["sms.send"]

def test_stops_after_ask_missing_info():
    env = FakeEnv()
    ReActAgent(llm=FakeLLM([CALL("sys.ask_missing_info", {"tool_name": "phone.call", "params": ["contact_id"]}),
                            CALL("phone.call", {"contact_id": "c1"})])).run(TASK, env)
    assert [c[0] for c in env.called] == ["sys.ask_missing_info"]

def test_identical_call_is_not_repeated():
    env = FakeEnv(); step = CALL("alarm.list", {})
    ReActAgent(llm=FakeLLM([step] * 10), max_steps=8).run(TASK, env)
    assert [c[0] for c in env.called] == ["alarm.list"]

def test_time_budget_stops_but_keeps_calls():
    ticks = iter([0.0, 1.0, 25.0, 26.0, 27.0])
    env = FakeEnv()
    ReActAgent(llm=FakeLLM([CALL("alarm.list", {}), CALL("alarm.delete", {"alarm_id": "a"})]),
               time_budget_s=20, clock=lambda: next(ticks, 99.0)).run(TASK, env)
    assert [c[0] for c in env.called] == ["alarm.list"]

def test_broken_json_retried_then_gives_up():
    env = FakeEnv()
    ReActAgent(llm=FakeLLM(['<tool_call>{bad</tool_call>'] * 5), max_invalid=2).run(TASK, env)
    assert env.called == []
```

- [ ] **Step 2:** `pytest tests/test_react_agent.py -v` → FAIL.
- [ ] **Step 3:** Implement `react.py` and register it.
- [ ] **Step 4:** `pytest -v` → all pass, including `test_registry`.
- [ ] **Step 5:** Commit `Add training-free ReAct tool-use agent`.

### Task 7: Smoke runs on molab and the first train numbers

Molab needs the branch on GitHub, and `CLAUDE.md` says to use the repo account's token.

- [ ] **Step 1:** Push: `GH_TOKEN="$(gh auth token --user AuDuongKhang)" git push -u origin feat/agent-roadmap` → the branch appears on GitHub.
- [ ] **Step 2 (Lead, molab):** In the notebook, set Branch `feat/agent-roadmap`, split `train`, agent `react`, asr `gold`, extra `--limit 100`. Expected: the run finishes, no `ERROR` lines, and evaluate prints a `RESULTS.md` row. Note the mean and max seconds.
- [ ] **Step 3 (Lead, molab):** Same, with asr `whisper`. Expected: `metrics.json` has `asr_wer`, and the score is lower than or equal to the gold-ASR run. If WER > 0.3 or max seconds > 15, stop and report back before continuing.
- [ ] **Step 4:** Paste both rows into `RESULTS.md`, then commit `Log first Tier 0 train results`.

### Task 8: First public_test submission

- [ ] **Step 1 (Lead, molab):** Run split `public_test`, agent `react`, asr `whisper`, no limit. Expected: 634 lines, 0 errors, `over_time_limit = 0` in `meta.json`.
- [ ] **Step 2:** Validate the downloaded file: `cd data && PYTHONPATH=public_test python -m zaic_tools --split public_test --root . --validate ../runs/<run>/submission.jsonl` → no schema problems reported.
- [ ] **Step 3 (Lead):** Submit on the portal, then add the row to `RESULTS.md` with the leaderboard score filled in by hand.
- [ ] **Step 4 (optional second submission):** Pick one variant based on the Task 7 numbers: either `--asr-kw model=vinai/PhoWhisper-large` if WER is the weak spot, or `--agent-kw max_steps=6` if time is. Same validate-and-submit steps.
- [ ] **Step 5:** Commit the `RESULTS.md` rows, then open the PR: `GH_TOKEN="$(gh auth token --user AuDuongKhang)" gh pr create --base main` with what changed, `pytest` output and the train/public numbers.
