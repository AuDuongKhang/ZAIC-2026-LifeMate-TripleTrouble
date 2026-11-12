# On-Policy Preference Training (sft-q1d) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train `lifemate-sft-q1d`: sft-q1c plus a preference LoRA learned from sft-q1c's own wrong decisions on pool samples rendered as unseen apps, and ship it before 16 Nov 2026 23:59 GMT+7 only if it passes the gates.

**Architecture:** Three new development-only modules under `src/lifemate/sft/`: `sample_steps` (sample sft-q1c at every agent turn of masked, leave-app-out pool records), `pairs` (turn disagreements on decisions into chosen/rejected pairs), `pref` (pair encoding and the DPO + NLL loss). `sft.train` gains a `--pairs` mode with a frozen reference (the same network with the adapter disabled). `sft.interpolate` does the WiSE-FT side track. A thin molab notebook runs the steps through a pure command builder. Nothing on the inference path changes except a new preset that points at the new weights.

**Tech Stack:** Python 3.11, PyTorch, transformers, peft (as in `sft.train`), pytest with the tiny random Qwen3 fixture of `tests/test_sft_train.py`, marimo notebooks on molab.

**Spec:** `docs/superpowers/specs/2026-11-12-onpolicy-preference-design.md` (approved 12 Nov 2026). Read it before any task.

## Global Constraints

- No paid or external services, no LLM APIs; local open models only (CLAUDE.md).
- Inference is unchanged: greedy decoding, same prompts, rules and options. Temperature sampling lives only in `sft/sample_steps.py`, which `agents/`, `infer.py` and `docker_entry.py` never import.
- Only **pool** samples are sampled and paired. dev, holdout, public_test and any public catalog content never enter a pair file.
- LLM < 5B, whole system < 6B; ≤ 30 s per sample on an A30; fixed seeds.
- Starting values (design, "Where the starting values come from"): K = 4, temperature 0.8, top-p 0.95 (the pilot replaces K and temperature); β = 0.1; SFT weight 1.0; one epoch; learning rate 5e-6 to 2e-5; SFT-record share 0.2; held-out pairs 10% by sample; pilot threshold 15%; gate holdout ≥ +1.0 with CI above 0 and dev ≥ -0.3; WiSE-FT α ∈ {0.7, 0.85}.
- sft-q1c: `TripleTrouble/lifemate-sft-q1c` at revision `8cb4336e529f0a4b2d6f10cdcfc63d15c45b651d`; its base `TripleTrouble/qwen35-4b-text` at `300ccc1` (the repo only records the short sha: Task 7 looks the full one up with `HfApi().list_repo_commits` and pins it as `BASE_REVISION` in `pref_commands.py`); LoRA targets `q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,gate_proj,up_proj,down_proj`.
- The pair file ships in `docker/training_code/` at ship time (approved by the owner).
- Repo text is English; Vietnamese only as data. LF endings. Tests: `PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python -m pytest` (Windows).
- Git: this plan and the design stay on `docs/onpolicy-pref-design` (PR #93). Code on `feat/onpolicy-pref`, cut from `origin/main`, one PR. Results on `docs/results-q1d`. Ship on `feat/ship-q1d`. Never commit to `main`; add only the files you changed; run `gh` with `GH_TOKEN="$(gh auth token --user AuDuongKhang)"`.
- Steps marked **HUMAN (molab)** are run by the owner; the implementer prepares the exact fields and stops until the run folders are back.

## Review Focus

- A prefix (or prefix + step) longer than the max length is skipped and counted, never cut: sampler (Task 3) and `encode_pair` (Task 4) each have a test.
- An unparsable sampled output, or one naming a tool the record does not show, is decision `other` and never makes a pair (Task 1 test).
- Records holding a sample id outside the pool make the sampler exit before any generation (Task 3 test).
- A resumed sampling run writes no duplicate (record, turn) row and gives the same rows as an uninterrupted run (Task 3 test).
- The cached reference log-probs are recomputed when the pair file, model or revision changes (Task 5 test).

## File structure

| File | Responsibility |
|---|---|
| `src/lifemate/sft/pairs.py` (new) | decision of one assistant step, pairing table, choice of the rejected step, held-out split, pair file + review sheet + report (CLI) |
| `src/lifemate/sft/sample_steps.py` (new) | decision points of masked records, temperature sampling with fixed seeds, append-only resumable steps file (CLI) |
| `src/lifemate/sft/pref.py` (new) | `encode_pair`, `step_logps`, `rpo_loss` |
| `src/lifemate/sft/train.py` | `--pairs` mode, reference cache, preference logging; `--revision` |
| `src/lifemate/sft/merge.py` | `--revision` for the base model |
| `src/lifemate/sft/interpolate.py` (new) | WiSE-FT weight interpolation (CLI) |
| `src/lifemate/sft/pref_commands.py` (new) | pure builder of the molab commands for each step |
| `notebooks/molab_pref.py` (new) | molab UI around `pref_commands` (clone, fetch, run, upload) |
| `configs/presets.json` | preset `sft-q1d-refs-fixes-all-clock` |
| `docs/molab-runbook.md` | section C5 with the fields of every HUMAN step |

---

### Task 0: Branches

- [ ] **Step 1:** Commit this plan on `docs/onpolicy-pref-design` (worktree `../wt-pref-design`), push; PR #93 now carries design + plan.
- [ ] **Step 2:** `git fetch origin && git worktree add -b feat/onpolicy-pref ../wt-pref origin/main` for Tasks 1 to 8; link `data/` into the worktree (junction) so the data tests run.

### Task 1: Decisions and the pairing table

**Files:** Create `src/lifemate/sft/pairs.py` (functions only in this task); Test `tests/test_sft_pairs.py`

**Interfaces:**
- Produces: `Decision = tuple[str, str | None]`, kinds `"write" | "read" | "ask" | "answer" | "other"`.
- Produces: `decision(message: dict, name_map: dict[str, str], kinds: dict[str, str]) -> Decision`: `message` is an assistant message as records store it (`tool_calls[i]["function"]["name"/"arguments"]`) or as `sft.replay.assistant_message(text)` returns it. First tool call only. Its name is mapped back with `name_map` (new → original; names absent from it are already original). `sys.ask_missing_info` → `("ask", <original name of arguments["tool_name"]>)`. A name not in `kinds` → `("other", None)`. No tool call → `("answer", None)`.
- Produces: `pair_type(gold: Decision, model: Decision) -> str | None`.
- Produces: `args_differ(gold: dict, model: dict, name_map: dict[str, str], fuzzy: dict[str, set[str]]) -> bool`: both assistant messages call the same write tool (after `name_map`); True when some argument key present on both sides (after mapping parameter names back) is not in `fuzzy[tool]` and the values differ (`!=`, exactly as `scoring.call_matches` compares them). An optional key on one side only is not a difference.
- Produces: `tool_kinds(catalog: dict[str, dict]) -> dict[str, str]`: `{name: spec["kind"]}` for every tool of `data.load_catalog(root, "train")` except `sys.ask_missing_info`.

Pairing table (design section 3, plus one ruling): `gold_kind:model_kind`, and `write:other_write` for a write to another tool.

| gold \ model | write same | write other | read | ask | answer | other |
|---|---|---|---|---|---|---|
| write | — | `write:other_write` | `write:read` | `write:ask` | `write:answer` | — |
| ask | `ask:write` | `ask:write` | `ask:read` | — (any target) | `ask:answer` | — |
| answer | `answer:write` | `answer:write` | `answer:read` | `answer:ask` | — | — |
| read | `read:write` | `read:write` | — | `read:ask` | `read:answer` | — |

Added on 12 Nov at the owner's request: `write:args` when gold and model write to the same tool and `args_differ` is True (Task 2 assigns it, since it needs the messages, not only the decisions).

Ruling recorded here: `read:answer` (the model stops where gold still reads) is a pair; the design's table is silent on it but its Intent names "stop or continue" as a decision. `answer:read` (the model reads where gold stops) is also a pair: every call is scored, so the extra read lowers precision, and a gold-empty (out-of-scope) sample scores only with an empty submission (`tests/test_scoring.py::test_out_of_scope_needs_empty_submission`).

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from lifemate.sft.pairs import decision, pair_type

KINDS = {"alarm.create": "write", "alarm.list": "read", "contacts.search": "read", "sms.send": "write"}
MAP = {"orbit.make": "alarm.create", "orbit.find": "alarm.list", "reef.lookup": "contacts.search", "comet.dispatch": "sms.send"}

def msg(name=None, **arguments):
    if name is None:
        return {"role": "assistant", "content": "Đã xong."}
    return {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}]}

def test_decision_maps_names_back_and_reads_the_kind():
    assert decision(msg("orbit.make", t="x"), MAP, KINDS) == ("write", "alarm.create")
    assert decision(msg("reef.lookup"), MAP, KINDS) == ("read", "contacts.search")
    assert decision(msg("sys.ask_missing_info", tool_name="comet.dispatch", params=["to"]), MAP, KINDS) == ("ask", "sms.send")
    assert decision(msg(), MAP, KINDS) == ("answer", None)
    assert decision(msg("nebula.invent"), MAP, KINDS) == ("other", None)

@pytest.mark.parametrize("gold, model, expected", [
    (("write", "alarm.create"), ("write", "alarm.create"), None),
    (("write", "alarm.create"), ("write", "sms.send"), "write:other_write"),
    (("write", "alarm.create"), ("read", "alarm.list"), "write:read"),
    (("write", "alarm.create"), ("ask", "alarm.create"), "write:ask"),
    (("write", "alarm.create"), ("answer", None), "write:answer"),
    (("write", "alarm.create"), ("other", None), None),
    (("ask", "sms.send"), ("ask", "alarm.create"), None),
    (("ask", "sms.send"), ("write", "sms.send"), "ask:write"),
    (("ask", "sms.send"), ("read", "contacts.search"), "ask:read"),
    (("ask", "sms.send"), ("answer", None), "ask:answer"),
    (("answer", None), ("read", "alarm.list"), "answer:read"),
    (("answer", None), ("write", "alarm.create"), "answer:write"),
    (("answer", None), ("ask", "sms.send"), "answer:ask"),
    (("read", "alarm.list"), ("read", "contacts.search"), None),
    (("read", "alarm.list"), ("write", "alarm.create"), "read:write"),
    (("read", "alarm.list"), ("ask", "sms.send"), "read:ask"),
    (("read", "alarm.list"), ("answer", None), "read:answer"),
    (("read", "alarm.list"), ("other", None), None),
])
def test_pairing_table(gold, model, expected):
    assert pair_type(gold, model) == expected

def test_an_unparsable_output_is_other():
    from lifemate.sft.replay import assistant_message
    broken = assistant_message('<tool_call>{"name": "orbit.make", "arguments": {</tool_call>')
    assert decision(broken, MAP, KINDS) == ("other", None)

FUZZY = {"alarm.create": {"label"}, "sms.send": {"content"}}
MAP_P = MAP | {"stamp": "time", "tag": "label"}

@pytest.mark.parametrize("gold_args, model_args, expected", [
    ({"time": "2026-11-06T20:30:00+07:00"}, {"stamp": "2026-11-06T19:30:00+07:00"}, True),   # non-fuzzy value differs
    ({"time": "2026-11-06T20:30:00+07:00"}, {"stamp": "2026-11-06T20:30:00+07:00"}, False),
    ({"time": "t", "label": "Đi chợ"}, {"stamp": "t", "tag": "đi chợ sáng"}, False),         # fuzzy key ignored
    ({"time": "t"}, {"stamp": "t", "tag": "x"}, False),                                        # optional key on one side only
    ({"time": "t", "repeat": "daily"}, {"stamp": "t"}, False),
])
def test_args_differ_on_non_fuzzy_keys_only(gold_args, model_args, expected):
    from lifemate.sft.pairs import args_differ
    gold = msg("alarm.create", **gold_args)
    model = msg("orbit.make", **model_args)
    assert args_differ(gold, model, MAP_P, FUZZY) is expected

def test_args_differ_needs_the_same_write_tool():
    from lifemate.sft.pairs import args_differ
    assert args_differ(msg("alarm.create", time="a"), msg("comet.dispatch", time="b"), MAP_P, FUZZY) is False
```

(`assistant_message` keeps unparsable text as plain `content`, markup included; `decision` must not read that as an answer.)

- [ ] **Step 2:** Run `pytest tests/test_sft_pairs.py -q`. Expected: `ModuleNotFoundError: No module named 'lifemate.sft.pairs'`.
- [ ] **Step 3:** Implement the four functions. `decision` takes `message["tool_calls"][0]` when present; a message whose `content` still holds `<tool_call>` markup (parse failure) → `("other", None)`.
- [ ] **Step 4:** Run the tests. Expected: all pass.
- [ ] **Step 5:** Commit (`sft.pairs: decisions and the pairing table`).

### Task 2: Pair building, held-out split, review sheet (CLI)

**Files:** Modify `src/lifemate/sft/pairs.py`; Test `tests/test_sft_pairs.py`

**Interfaces:**
- Consumes: `decision`, `pair_type`, `tool_kinds` (Task 1); steps rows of Task 3 (format below; this task's tests write them by hand).
- Produces: `choose_rejected(gold_message: dict, candidates: list[str], name_map, kinds, fuzzy) -> tuple[str, str, bool] | None`: candidates are `[greedy] + samples` (texts); a candidate's type is `pair_type(decision(gold), decision(candidate))`, or `write:args` when that is None, gold is a write and `args_differ` is True. Returns `(type, rejected_text, greedy_wrong)` for the most frequent wrong model decision (`write:args` candidates form one group; ties: the earliest candidate), rejected text = earliest candidate of that group; `greedy_wrong` = the greedy candidate has a type. None when no candidate has one. `fuzzy` comes from `data.fuzzy_params(load_catalog(root, "train"))`.
- Produces: `is_heldout(sample_id: str) -> bool`: `zlib.crc32(sample_id.encode()) % 10 == 0`.
- Produces: pair row `{"id", "record", "turn", "type", "greedy_wrong", "messages", "tools", "chosen", "rejected"}`: `messages` = the record's messages before `turn`, `tools` = the record's tools, `chosen` = the record's message at `turn`, `rejected` = `assistant_message(rejected_text)`.
- Produces: CLI `python -m lifemate.sft.pairs --records NAME [--records NAME ...] --steps NAME --out NAME [--greedy-wrong-only] [--records-dir DIR]` writes `<records-dir>/pairs/<out>.jsonl` (train), `<out>-heldout.jsonl` and `<out>-review.md` (30 pairs drawn with `random.Random(0)`: last user message, gold step, rejected step, type), prints and returns `{"points", "with_pair", "rate", "greedy_wrong", "by_type", "train", "heldout"}`.
- Steps row (written by Task 3): `{"records": NAME, "record": int, "id": str, "turn": int, "greedy": str, "samples": [str], "temperature": float, "top_p": float, "seed": int, "model": str, "revision": str}`.

- [ ] **Step 1: Write the failing tests** (add to `tests/test_sft_pairs.py`):
  - `test_choose_rejected_takes_the_most_frequent_wrong_decision`: gold `("write", "alarm.create")`, candidates `[greedy = gold call, read alarm.list, answer, read alarm.list]` → `("write:read", <first read text>, False)`.
  - `test_choose_rejected_breaks_ties_by_order_and_flags_a_wrong_greedy`: candidates `[answer, read]` (one each) → type `"write:answer"`, `greedy_wrong` True.
  - `test_no_pair_when_only_fuzzy_arguments_differ`: candidates that call `alarm.create` with another `label` (fuzzy) → None.
  - `test_write_args_pair_when_a_non_fuzzy_argument_differs`: candidates `[gold call, alarm.create with another time, alarm.create with another time]` → `("write:args", <first such text>, False)`.
  - `test_cli_writes_train_heldout_review_and_report(tmp_path)`: a records file with one masked record of 2 agent turns and a steps file with rows for both turns (one disagreeing); assert files exist, report `points == 2`, `with_pair == 1`, `rate == 0.5`, the pair's `messages` end with the user turn (turn 0) and `chosen` equals the record's message at `turn`; pairs of an id with `is_heldout(id)` True go only to `-heldout.jsonl`.
  - `test_greedy_wrong_only_keeps_only_those_pairs`.
- [ ] **Step 2:** Run them. Expected: `ImportError` / `AttributeError` on the new names.
- [ ] **Step 3:** Implement. Records and steps are joined on `(records, record)`; a steps row whose record index is missing raises `ValueError`.
- [ ] **Step 4:** Run `pytest tests/test_sft_pairs.py -q`. Expected: all pass.
- [ ] **Step 5:** Commit (`sft.pairs: pair files, held-out split and review sheet`).

### Task 3: Step sampler (CLI)

**Files:** Create `src/lifemate/sft/sample_steps.py`; Test `tests/test_sft_sample_steps.py`

**Interfaces:**
- Consumes: records from `sft.records` (masked copies carry `source == "masked"` and `name_map`); `sft.tokenize.render`, `MAX_LEN`; steps row format (Task 2).
- Produces: `decision_points(record: dict) -> list[int]`: indices of the assistant messages after the last user message.
- Produces: `point_seed(seed: int, records: str, record: int, turn: int) -> int`: `zlib.crc32(f"{seed}:{records}:{record}:{turn}".encode())`.
- Produces: `sample_texts(model, tokenizer, messages, tools, k: int, temperature: float, top_p: float, seed: int, max_new_tokens: int = 384) -> tuple[str, list[str]]`: (greedy text, k sampled texts). Prompt = `tokenizer.apply_chat_template(..., add_generation_prompt=True)` exactly as `HFChatLLM._encode`; greedy with `do_sample=False`; samples in one `generate` call with `do_sample=True, num_return_sequences=k` after `torch.manual_seed(seed)`; texts cut with `agents.llm.finish_text` and the stop strings of `HFChatLLM` (reuse, do not copy).
- Produces: CLI `python -m lifemate.sft.sample_steps --records NAME [--records NAME ...] --name NAME --model M --revision SHA [--k 4] [--temperature 0.8] [--top-p 0.95] [--seed 0] [--limit N] [--max-len 12288] [--hf-repo DATASET] [--upload-every 200] [--records-dir DIR]` appends to `<records-dir>/steps/<name>.jsonl`; only records with `source == "masked"`; `--limit N` keeps the records of the first N distinct sample ids in file order.

- [ ] **Step 1: Write the failing tests** (sampling replaced by a fake through `monkeypatch.setattr(sample_steps, "sample_texts", fake)`; the fake returns deterministic strings derived from its `seed`):
  - `test_decision_points_are_the_agent_turns`: a replayed act record (helper `conftest.replay_sample`) → indices of its assistant messages after the last user message.
  - `test_refuses_records_outside_the_pool(tmp_path)`: a masked record whose id is in `splits["holdout"]` (`lifemate.splits.load_splits`) → `SystemExit`, steps file not created.
  - `test_skips_unmasked_records_and_counts_long_prefixes(tmp_path, capsys)`: with `--max-len 50`, every point is skipped and the printed summary says how many.
  - `test_resume_gives_the_rows_of_an_uninterrupted_run(tmp_path)`: run with a fake that raises after 2 points, run again; rows equal those of one clean run, no duplicate `(records, record, turn)`.
  - `test_point_seed_is_stable`: `point_seed(0, "a", 3, 5) == zlib.crc32(b"0:a:3:5")`.
  - `test_sample_texts_on_the_tiny_model` (uses the `tiny` fixture pattern of `tests/test_sft_train.py`, skipped without the cached tokenizer): returns one greedy text and `k` texts; the same seed gives the same texts twice.
- [ ] **Step 2:** Run them. Expected: `ModuleNotFoundError`.
- [ ] **Step 3:** Implement. Model loading: `agents.llm.load_causal_lm(model, torch.bfloat16, revision)`, tokenizer at the same revision; uploads follow `sft/thoughts.py` (`--hf-repo`, `--upload-every`, path `steps/<name>.jsonl` in the dataset). The module docstring states it is development-only and never imported by `agents/`, `infer.py` or `docker_entry.py`.
- [ ] **Step 4:** Run the tests. Expected: all pass.
- [ ] **Step 5:** Add `test_inference_never_imports_the_sampler`: `ast` scan of `src/lifemate/agents/*.py`, `infer.py`, `docker_entry.py` finds no import of `sample_steps`, `pairs` or `pref`. Run, pass, commit (`sft.sample_steps: sample the model at every agent turn of masked pool records`).

### Task 4: Pair encoding and the loss

**Files:** Create `src/lifemate/sft/pref.py`; Test `tests/test_sft_pref.py`

**Interfaces:**
- Consumes: pair row (Task 2), `sft.tokenize.render`, `MAX_LEN`.
- Produces: `encode_pair(pair: dict, tokenizer, max_len: int = MAX_LEN) -> dict | None`: `{"prefix": list[int], "chosen": list[int], "rejected": list[int]}`; `prefix = render(messages, tools, add_generation_prompt=True)`, each step = `render(messages + [step], tools)[len(prefix):]`; `ValueError` when the prefix does not render as a prefix (same check as `encode_record`); None when `len(prefix) + max(len(chosen), len(rejected)) > max_len`.
- Produces: `step_logps(model, prefix: list[int], step: list[int], device) -> tuple[Tensor, Tensor]`: (sum of log-probs of the step tokens, mean NLL of the step tokens), logits only at the step positions (`logits_to_keep`, as `record_logits` in `sft.train`).
- Produces: `rpo_loss(pc, pr, rc, rr, nll_c, beta: float, sft_weight: float) -> tuple[Tensor, Tensor]`: `margin = beta * ((pc - rc) - (pr - rr))`, `loss = -logsigmoid(margin) + sft_weight * nll_c`; returns `(loss, margin)`.

- [ ] **Step 1: Write the failing tests:**
  - `test_rpo_loss_at_the_reference_is_log_two`: `pc=rc=-3`, `pr=rr=-5`, `nll_c=0`, β 0.1 → loss `== pytest.approx(math.log(2))`, margin 0.
  - `test_rpo_loss_adds_the_weighted_nll`: same with `nll_c=2`, `sft_weight=0.5` → `log 2 + 1`.
  - `test_margin_sign_follows_the_preference`: `pc - rc = 1`, `pr - rr = -1`, β 0.1 → margin `0.2`, loss `< log 2`.
  - `test_encode_pair_shares_the_prefix` (tiny tokenizer): chosen and rejected steps are non-empty and `render(messages + [chosen])` equals `prefix + chosen`.
  - `test_encode_pair_skips_a_pair_over_max_len`: `max_len=10` → None.
- [ ] **Step 2:** Run. Expected: `ModuleNotFoundError`.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: all pass.
- [ ] **Step 5:** Commit (`sft.pref: pair encoding and the DPO + NLL loss`).

### Task 5: `--pairs` mode in `sft.train`, `--revision` in train and merge

**Files:** Modify `src/lifemate/sft/train.py`, `src/lifemate/sft/merge.py`; Test `tests/test_sft_train.py`

**Interfaces:**
- Consumes: `encode_pair`, `step_logps`, `rpo_loss` (Task 4); pair files `<records-dir>/pairs/<NAME>.jsonl` (Task 2).
- Produces (train CLI): `--pairs NAME`, `--eval-pairs NAME`, `--beta 0.1`, `--sft-weight 1.0`, `--sft-share 0.2`, `--revision SHA` (passed to `load_causal_lm` and the tokenizer). With `--pairs`, the training items are the encoded pairs plus `round(sft_share / (1 - sft_share) * n_pairs)` encoded records of `--records` drawn with `random.Random(seed)`; each item's loss is `rpo_loss` (pairs) or the mean cross-entropy of its labelled tokens (records); an optimizer step averages the `grad_accum` item losses. Skipped pairs (None) are counted and printed.
- Produces: reference log-probs computed once before the first step under `model.disable_adapter()` for train and eval pairs, cached at `<out>/ref_logps.pt` with key `{"pairs_sha", "eval_pairs_sha", "model", "revision", "max_len"}`; a different key recomputes.
- Produces: log keys per step `loss`, `dpo`, `nll`, `margin`, `pref_acc` (share of the step's pairs with margin > 0); at eval also `eval_pref_acc` over `--eval-pairs` beside the existing `eval_loss`.
- Produces: `RUN_KEYS` gains `pairs`, `beta`, `sft_weight`, `sft_share`, `revision`; `--pairs` with `--kd-teachers` or `--decision-weight` ≠ 1 is refused by `ap.error`.
- Produces (merge CLI): `--revision SHA` for the base model.

- [ ] **Step 1: Write the failing tests** (in `tests/test_sft_train.py`; a module fixture builds a pair file from the `tiny` records: for each record's last agent turn, `chosen` = that message, `rejected` = the same message with its tool name replaced by another tool of the record, or a plain "Đã xong." when it had no call):
  - `test_pairs_run_starts_at_log_two`: `--pairs p --sft-share 0 --sft-weight 0 --max-steps 1` → first logged `dpo == pytest.approx(math.log(2), abs=1e-4)` and `margin == pytest.approx(0, abs=1e-4)` (LoRA starts at zero, so policy = reference).
  - `test_pairs_run_logs_preference_terms_and_eval_accuracy`: with `--eval-pairs p` every eval row has `eval_pref_acc` in [0, 1] and `eval_loss`.
  - `test_pairs_resume_matches_an_uninterrupted_run`: as `test_resumed_run_matches_an_uninterrupted_one` with `--pairs p`.
  - `test_reference_cache_is_rebuilt_when_the_pairs_change`: run, rewrite the pair file with one pair less, run into a new `--out` reusing the cache file → the cache key's `pairs_sha` differs and the stored tensors have the new length.
  - `test_pairs_refuse_kd_and_decision_weight`.
  - `test_merge_accepts_a_revision` (monkeypatch `load_causal_lm` to record its `revision` argument).
- [ ] **Step 2:** Run `pytest tests/test_sft_train.py -q -k "pairs or revision"`. Expected: failures on the unknown options.
- [ ] **Step 3:** Implement. Keep the plain SFT path byte-identical when `--pairs` is absent (the existing tests guard it).
- [ ] **Step 4:** Run `pytest tests/test_sft_train.py -q`. Expected: all pass, the old ones included.
- [ ] **Step 5:** Commit (`sft.train: preference mode with a frozen reference; --revision`).

### Task 6: WiSE-FT interpolation

**Files:** Create `src/lifemate/sft/interpolate.py`; Test `tests/test_sft_interpolate.py`

**Interfaces:**
- Produces: `interpolate_state(a: dict, b: dict, alpha: float) -> dict`: `alpha * a[k] + (1 - alpha) * b[k]` for floating tensors, `a[k]` for the others; `ValueError` naming the first key when the key sets or a shape differ.
- Produces: CLI `python -m lifemate.sft.interpolate --a MODEL --a-revision SHA --b MODEL --b-revision SHA --alpha A --out DIR [--hf-repo REPO]`: saves the model with `a`'s config and tokenizer (chat template included).

- [ ] **Step 1: Failing tests:** `alpha=1` returns `a`'s values; `alpha=0.5` the mean; a missing key and a shape mismatch raise `ValueError`; the CLI on two tiny models (different `torch.manual_seed`) writes a model that loads with `load_causal_lm` and has the mean weights at `alpha=0.5`.
- [ ] **Step 2:** Run. Expected: `ModuleNotFoundError`.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: pass.
- [ ] **Step 5:** Commit (`sft.interpolate: WiSE-FT weight interpolation`).

### Task 7: Preset, molab command builder, notebook, runbook, PR

**Files:** Modify `configs/presets.json`, `docs/molab-runbook.md`; Create `src/lifemate/sft/pref_commands.py`, `notebooks/molab_pref.py`; Test `tests/test_presets.py`, `tests/test_sft_pref_commands.py`, `tests/test_molab_pref.py`

**Interfaces:**
- Produces: preset `sft-q1d-refs-fixes-all-clock` = `sft-q1c-refs-fixes-all-clock` with `model` `${HF_MODEL_OWNER}/lifemate-sft-q1d`; presets `sft-q1c-wise70-refs-fixes-all-clock` and `sft-q1c-wise85-refs-fixes-all-clock` with models `${HF_MODEL_OWNER}/lifemate-sft-q1c-wise70` / `-wise85`.
- Produces: `commands(step: str, f: dict) -> list[list[str]]` for steps `records`, `sample`, `pairs`, `train`, `merge`, `interpolate`; `f` holds `name`, `model`, `revision`, `k`, `temperature`, `top_p`, `limit`, `lr`, `beta`, `sft_weight`, `owner`, `data_repo`. Exact commands (python = `sys.executable -m`):
  - `records`: `lifemate.sft.records --name pref-{name}-gold --subset pool --text gold --mask 1 --leave-app-out 1 --seed 7 --tokenizer none [--limit L]`, the same with `--name pref-{name}-asr --text cache:pho-large-fp16`, and `lifemate.sft.records --name gold-pool-m1-asr-pho-large-fp16-lao50 --subset pool --text gold --text cache:pho-large-fp16 --mask 1 --leave-app-out 0.5 --tokenizer none` (the sft-q1c records, for the SFT share) and `--name gold-dev --subset dev` (eval loss). Seed 7 gives masked names sft-q1c never saw (its records used seed 0).
  - `sample`: `lifemate.sft.sample_steps --records pref-{name}-gold --records pref-{name}-asr --name {name}-t{T} --model {model} --revision {revision} --k {k} --temperature {T} --top-p {top_p} --seed 0 --hf-repo {data_repo} [--limit L]`, one command per temperature in `f["temperature"]` (a comma-separated list).
  - `pairs`: `lifemate.sft.pairs --records pref-{name}-gold --records pref-{name}-asr --steps {name}-t{T} --out {name}-t{T}`.
  - `train`: `lifemate.sft.train --records gold-pool-m1-asr-pho-large-fp16-lao50 --pairs {name}-t{T} --eval-pairs {name}-t{T}-heldout --eval-records gold-dev --model {model} --revision {revision} --out checkpoints/sft-q1d-{name} --hf-repo {owner}/lifemate-sft-q1d --epochs 1 --lr {lr} --beta {beta} --sft-weight {sft_weight} --lora-targets q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,gate_proj,up_proj,down_proj`.
  - `merge`: `lifemate.sft.merge --adapter checkpoints/sft-q1d-{name}/last --out merged/sft-q1d-{name} --hf-repo {owner}/lifemate-sft-q1d --model {model} --revision {revision}`.
  - `interpolate`: `lifemate.sft.interpolate --a {model} --a-revision {revision} --b {owner}/qwen35-4b-text --b-revision {BASE_REVISION} --alpha {a} --out merged/sft-q1c-wise{100a} --hf-repo {owner}/lifemate-sft-q1c-wise{100a}` for α 0.7 and 0.85.
- Notebook `notebooks/molab_pref.py`: the clone / pull / fetch / pip cells of `molab_thoughts.py` (same fields and defaults), then one form (dropdown `step`, the fields of `f`, Start, Stop) whose start cell runs `commands(step, f)` in order in the background with the HF write token in the environment, and shows the last report (`pairs` prints it) and the tail of the log.

- [ ] **Step 1: Failing tests:** `test_q1d_and_wise_presets_are_the_clock_preset_with_another_model` (`resolve` each against `sft-q1c-refs-fixes-all-clock`, only `model` differs); `test_clock_fix_is_off_in_every_other_preset` and `test_new_options_are_off_in_every_other_preset` updated to include the three new presets (they inherit `clock_fix`, `empty_retry` and `weekday_fix`); `tests/test_sft_pref_commands.py` asserts each command above for `name="pilot"`, `temperature="0.7,1.0"`, `limit=100`; `tests/test_molab_pref.py` runs the start cell with fakes as `tests/test_molab_thoughts.py` does and checks it runs exactly `commands(...)`.
- [ ] **Step 2:** Run them. Expected: failures (unknown preset, missing module, missing notebook).
- [ ] **Step 3:** Implement the presets, `pref_commands.py` and the notebook.
- [ ] **Step 4:** Write runbook section C5 "Preference training (sft-q1d)" with the fields of Tasks 8 to 11.
- [ ] **Step 5:** `pytest -q` (whole suite, data linked). Expected: all pass. Commit (`Preference training: presets, molab commands and notebook, runbook C5`), push `feat/onpolicy-pref`, open the PR (description: what each module does, that inference is unchanged, test counts, that no GPU run has been made).

### Task 8: Pilot (**HUMAN (molab)**, then implementer)

- [ ] **Step 1: HUMAN.** Upload `notebooks/molab_pref.py` to molab once. Branch `feat/onpolicy-pref`; name `pilot`; model `TripleTrouble/lifemate-sft-q1c`; revision `8cb4336e529f0a4b2d6f10cdcfc63d15c45b651d`; limit 100; K 8; temperature `0.7,1.0`; top-p 0.95. Steps in order: `records`, `sample`, `pairs`. Send `train_records/pairs/pilot-t0.7*` and `pilot-t1.0*` (the review sheets and the printed reports).
- [ ] **Step 2: Implementer.** For each temperature: the rate, `greedy_wrong`, `by_type`. Read the 30 pairs of each review sheet and mark each "decision" or "convention" (a pair where gold is not the single right answer). Also recompute the rate for K = 4 by keeping only `samples[:4]` (re-run `pairs` on a copy of the steps file truncated to 4 samples).
- [ ] **Step 3: Decide** and write it in the runbook C5 and the PR:
  - Pick the smallest K in {4, 8} and the lower temperature that give a rate ≥ 15%.
  - Neither temperature reaches 15% at K = 8: stop; report the rate by type to the owner (the design's risk "model too sure on pool").
  - More than 10 of 30 hand-read pairs at the chosen setting are conventions: stop; name the pair types concerned; the owner decides whether to drop those types (a `--types` filter is then added to `pairs` with a test).

### Task 9: Full sampling, cheap selection, q1d (**HUMAN (molab)**, then implementer)

- [ ] **Step 1: HUMAN.** Same notebook, name `r1`, limit 0 (whole pool), K and temperature from Task 8. Steps `records`, `sample`, `pairs`. Then `train` three times with `lr / beta / sft_weight` = `1e-5 / 0.1 / 1.0`, `5e-6 / 0.1 / 1.0`, `1e-5 / 0.3 / 1.0`, each with its own name suffix (`r1-a`, `r1-b`, `r1-c`, changing only `--out` and the HF path). If the full pair file holds at least 300 pairs with `greedy_wrong`, also a fourth run `r1-d`: the `r1-a` values on a pair file built with `pairs --greedy-wrong-only` (the design's "greedy-wrong only" against "all" comparison). Send each run's `train_log.jsonl`.
- [ ] **Step 2: Implementer.** For each config: last `eval_pref_acc` and `eval_loss` against the first `eval_loss` (step 0 is sft-q1c). Pick the highest `eval_pref_acc` among the configs whose `eval_loss` rose by at most 5% (`r1-d` is measured on the full held-out file too, so the four compare); none qualifies → stop and report.
- [ ] **Step 3: HUMAN.** `merge` the picked config; note the HF commit sha of `TripleTrouble/lifemate-sft-q1d`. In parallel, `interpolate` (α 0.7 and 0.85), noting both shas.
- [ ] **Step 4: Format check for WiSE-FT** (molab_run, Split `train`, subset `dev`, ASR `gold`, extra infer args `--limit 50 --agent-kw revision=<sha>`, agent presets `sft-q1c-wise70-…` and `-wise85-…` plus `sft-q1c-refs-fixes-all-clock` as reference): an α whose share of samples with `stop == "invalid"` or a schema rejection is above the reference's by more than 2 samples is dropped.

### Task 10: Gate (**HUMAN (molab)**, then implementer)

- [ ] **Step 1: HUMAN.** `notebooks/molab_run.py`, branch `feat/onpolicy-pref`, Split `train`, ASR preset `pho-large-ft-b`, HF model owner `TripleTrouble`:

| Run | Agent preset | Extra infer args | Subset |
|---|---|---|---|
| B1 | `sft-q1c-refs-fixes-all-clock` | `--agent-kw revision=8cb4336e529f0a4b2d6f10cdcfc63d15c45b651d` | holdout |
| B2 | same | same | dev |
| Q1 | `sft-q1d-refs-fixes-all-clock` | `--agent-kw revision=<q1d sha>` | holdout |
| Q2 | same | same | dev |
| W1, W2 | the WiSE-FT preset(s) kept by Task 9 Step 4 | `--agent-kw revision=<sha>` | holdout, dev |

B1/B2 re-run the baseline on the current code (the 12 Nov G3/G4 ran before the clock review fixes and the warmup change).
- [ ] **Step 2: Implementer.** `lifemate.evaluate` each run; `lifemate.compare B1 Q1`, `B2 Q2` (and the W runs); `lifemate.loss_report` on Q1 against B1's failure table: the `other tools` and `ask/act flip` rows must shrink for the change to be the intended one.
- [ ] **Step 3: Decide.** A model passes with holdout ≥ +1.0 and CI above 0, and dev ≥ -0.3. Both q1d and a WiSE-FT model pass → the plan's Task 11 runs q1d first (the main track), then the better WiSE-FT. Neither passes → stop; the image stays at 63.44; write the rows and the reason.
- [ ] **Step 4:** `RESULTS.md` rows on `docs/results-q1d` (from `origin/main`, own PR), in the format of the 12 Nov rows.

### Task 11: Public run and ship (only after a pass in Task 10)

- [ ] **Step 1: HUMAN (molab).** `molab_run.py`, Split `public_test`, ASR `pho-large-ft-all-beam5`, the passing preset, extra infer args `--time-field --agent-kw revision=<sha>`. Validate the submission (`zaic_tools --validate`), `over_time_limit` 0, submit; the score goes in its `RESULTS.md` row.
- [ ] **Step 2: Decide.** Ship when the leaderboard is ≥ 63.44; below, the image stays.
- [ ] **Step 3: Ship PR** on `feat/ship-q1d` from `origin/main` (after the code PR is merged): `configs/shipped.json` `agent_preset` and `agent_kw.revision` of the passing model, `records_commit` = the commit of `feat/onpolicy-pref` that produced the pairs and the training; copy the pair file into `docker/training_code/pairs/`; `docker/training_code/README.md` gains the reproduction steps (records with seed 7, `sample_steps` → the shipped pair file instead of re-sampling, `train --pairs`, `merge`) and the WiSE-FT command if that model ships; `docs/shipped-model.md` and `README.md` describe the new model. `pytest -q`; CI builds and smoke-tests the image.
- [ ] **Step 4: HUMAN.** Merge, tag `docker-*`, Test Solution (only offered until 15 Nov 23:59), submit link + MD5 before 16 Nov 23:59. Never commit the Drive link.
- [ ] **Step 5: Round 2** only if Step 2 shipped and more than 24 hours remain before 16 Nov 23:59: Tasks 8 to 11 again with `model` = q1d at its sha and name `r2`.

### Later (not planned in detail): trajectory-level pairs

Only after Task 11, and only with more than a day left before 16 Nov 23:59: the design's section "Later, if time remains". It gets its own plan then; nothing in Tasks 1 to 11 depends on it.
