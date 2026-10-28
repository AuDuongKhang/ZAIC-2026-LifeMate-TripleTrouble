# Experiments Batch 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared experiment infrastructure (fixed splits, paired comparison, ASR cache, number-normalized WER), then run A1 (dynamic few-shot) and B1 (off-the-shelf ASR comparison).

**Architecture:**
- `lifemate.splits` writes `configs/splits.json` once (holdout apps, a dev set and the training pool). `infer --subset` runs on a split part.
- `lifemate.compare` does paired bootstrap comparison of two runs.
- `lifemate.transcribe` fills `asr_cache/<split>/<name>.jsonl`, which the `cached` ASR backend replays, and `lifemate.asr_report` scores it.
- Few-shot examples are retrieved from the training pool with BM25 plus an app filter, and rendered into the system prompt.

All of this is pure Python, so it can be tested locally. Model runs happen on molab.

**Tech Stack:** Python ≥ 3.10 stdlib (no new runtime dependencies), pytest; transformers on molab.

**Spec:** `docs/superpowers/specs/2026-10-27-agent-roadmap-design.md`, section "Experiments" (with "Evaluation" for the holdout rules).

## Global Constraints

- No new entry in `requirements.txt` may depend on torch (`tests/test_requirements.py`). Experimental ASR packages (qwen-asr, chunkformer) are installed by hand on molab with `--no-deps`, never through requirements.
- Solution code never reads `<split>_environment/`. Few-shot reads only `train/train.jsonl` (labels and results of our own training data).
- **Leakage rule:** few-shot retrieval and any training never use dev or holdout samples. Dev and holdout numbers are the only ones reported for experiments.
- Defaults stay at the best measured setup: `examples=0`, `prompt_version="v1"`, `check_refs=False`, `dtype="float16"`. New behaviour is opt-in through `--agent-kw` / `--asr-kw`.
- Determinism: anything random (`splits`, bootstrap) takes an explicit seed, default 0.
- All code, comments and commits in English. Branch `feat/experiments-infra` cut from `main` after the docs PR (this plan + spec) merges. PR into `main`.

## Review Focus

1. **Holdout leakage through history.** A sample whose history uses a holdout app is a holdout sample too, even if its gold calls do not use it. The app set of a sample is its gold calls ∪ history calls. Tested in Task 1.
2. **Cache misses.** A `cached` ASR run with an audio file missing from the cache must fail loudly naming the file, not silently return "". Tested in Task 4.
3. **The few-shot pool must never contain the current sample**, even when it runs on the training pool itself (for example when debugging). Tested in Task 6.
4. **Comparing runs with different ids:** compare only the intersection and print how many were dropped. Tested in Task 2.
5. **Numbers inside words** ("S+", "4G", "MP3") must not be rewritten by number normalization; only standalone digit tokens are. Tested in Task 3.

---

### Task 1: Fixed splits and `infer --subset`

**Files:**
- Create: `src/lifemate/splits.py`, `configs/splits.json` (generated, committed)
- Modify: `src/lifemate/infer.py` (add `--subset`)
- Test: `tests/test_splits.py`

**Interfaces:**
- Produces:
  - `sample_apps(sample: dict) -> set[str]`: the app prefixes of the gold calls (if labelled) and of all history `tool_calls`. `sys` is excluded.
  - `make_splits(samples: list[dict], seed: int = 0, n_holdout_apps: int = 8, dev_size: int = 300) -> dict`, which returns `{"seed", "holdout_apps": [...], "holdout": [ids], "dev": [ids], "pool": [ids]}`.
  - `load_splits(path: Path = REPO_ROOT / "configs/splits.json") -> dict`.
  - CLI `python -m lifemate.splits --seed 0`, which writes the JSON sorted and indented.
- Rules:
  - Holdout apps are drawn with `random.Random(seed)` from apps used by 15–120 train samples. The upper bound keeps big apps such as contacts, sms and calendar in training.
  - The draw is retried until the holdout share of train is 10–20% and the apps span ≥ 4 catalog `group`s.
  - Dev takes 300 non-holdout samples, stratified proportionally by `evaluate.label_kind` × `evaluate.history_bucket`.
  - The pool is everything else.
- `infer --subset dev|holdout|pool` keeps only those ids, applied before `--limit`. It needs `--split train`.

- [ ] **Step 1: Write the failing tests** on the real train file (skip without data, like `test_pipeline.py`):

```python
def test_splits_are_disjoint_and_cover_train(train_samples):
    s = make_splits(train_samples, seed=0)
    ids = {x["id"] for x in train_samples}
    assert set(s["holdout"]) | set(s["dev"]) | set(s["pool"]) == ids
    assert not (set(s["holdout"]) & set(s["dev"])) and not (set(s["dev"]) & set(s["pool"]))
    assert len(s["dev"]) == 300 and 0.10 <= len(s["holdout"]) / len(ids) <= 0.20

def test_history_app_makes_a_sample_holdout(train_samples):
    s = make_splits(train_samples, seed=0)
    held = set(s["holdout_apps"])
    for x in train_samples:
        if sample_apps(x) & held:
            assert x["id"] in s["holdout"]

def test_splits_are_deterministic(train_samples):
    assert make_splits(train_samples, seed=0) == make_splits(train_samples, seed=0)

def test_committed_splits_match_generator(train_samples):
    assert load_splits() == make_splits(train_samples, seed=0)
```

- [ ] **Step 2:** `pytest tests/test_splits.py -v` → FAIL (ImportError).
- [ ] **Step 3:** Implement `splits.py`. Generate `configs/splits.json` with `python -m lifemate.splits --seed 0`. Add `--subset` to `infer.py`.
- [ ] **Step 4:** `pytest -v` → all pass.
- [ ] **Step 5:** Commit `Add fixed train splits (holdout apps, dev 300, pool) and infer --subset`.

### Task 2: Paired run comparison

**Files:**
- Create: `src/lifemate/compare.py`
- Test: `tests/test_compare.py`

**Interfaces:**
- Consumes: `per_sample.jsonl` from `evaluate` (fields `id`, `score`).
- Produces:
  - `compare(a: dict[str, float], b: dict[str, float], n_boot: int = 2000, seed: int = 0) -> dict`, which returns `{"n", "dropped", "mean_a", "mean_b", "diff", "ci_low", "ci_high", "better", "worse", "same"}`. Scores are on the 0–100 scale. `better` counts samples where b > a.
  - CLI `python -m lifemate.compare runs/<a> runs/<b>`, which prints one line plus `WIN`, `LOSS` or `NO CLEAR DIFFERENCE` depending on whether the CI excludes 0.

- [ ] **Step 1: Write the failing tests**

```python
def test_identical_runs_have_zero_interval():
    a = {f"s{i}": i % 2 for i in range(50)}
    r = compare(a, a)
    assert r["diff"] == 0 and r["ci_low"] == 0 == r["ci_high"] and r["same"] == 50

def test_clear_improvement_is_a_win():
    a = {f"s{i}": 0.0 for i in range(100)}; b = {f"s{i}": 1.0 if i < 40 else 0.0 for i in range(100)}
    r = compare(a, b)
    assert r["diff"] == 40.0 and r["ci_low"] > 0 and r["better"] == 40

def test_only_shared_ids_are_compared():
    r = compare({"x": 1.0, "y": 0.0}, {"y": 1.0, "z": 1.0})
    assert r["n"] == 1 and r["dropped"] == 2
```

- [ ] **Step 2:** `pytest tests/test_compare.py -v` → FAIL.
- [ ] **Step 3:** Implement it with a paired bootstrap over the per-sample differences, using `random.Random(seed)`.
- [ ] **Step 4:** `pytest tests/test_compare.py -v` → PASS.
- [ ] **Step 5:** Commit `Add paired bootstrap comparison of two runs`.

### Task 3: Number-normalized WER and CER

**Files:**
- Modify: `src/lifemate/asr/wer.py`
- Test: `tests/test_wer.py` (extend)

**Interfaces:**
- Produces:
  - `number_words(n: int) -> str`: Vietnamese reading of 0 ≤ n < 10¹². Rules:
    - "mười" for 10–19;
    - "mốt" for a units 1 after a tens ≥ 20;
    - "lăm" for a units 5 after any tens;
    - "lẻ" for a zero tens before a nonzero units in hundreds;
    - "nghìn", "triệu" and "tỷ" groups.
  - `normalize(text)` now also:
    - rewrites standalone digit tokens (`^\d+$` after punctuation stripping) with `number_words`;
    - maps the variants "ngàn" → "nghìn" and "linh" → "lẻ".
  - `cer(ref: str, hyp: str) -> float`: character Levenshtein over the normalized words joined by single spaces.
  - `mean_wer` keeps its signature.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("n,words", [(5, "năm"), (10, "mười"), (15, "mười lăm"), (21, "hai mươi mốt"), (25, "hai mươi lăm"),
    (105, "một trăm lẻ năm"), (1000, "một nghìn"), (2800000, "hai triệu tám trăm nghìn"), (0, "không")])
def test_number_words(n, words):
    assert number_words(n) == words

def test_digits_match_spelled_numbers():
    assert wer("Đặt báo thức 5 giờ", "đặt báo thức năm giờ") == 0.0
    assert wer("nạp 50 ngàn", "nạp năm mươi nghìn") == 0.0

def test_digits_inside_words_are_kept():
    assert normalize("kênh S+ Sport 4G") == ["kênh", "s", "sport", "4g"]

def test_cer():
    assert cer("đèn", "đen") == 1 / 3
```

- [ ] **Step 2:** `pytest tests/test_wer.py -v` → the new tests fail.
- [ ] **Step 3:** Implement it.
- [ ] **Step 4:** `pytest -v` → all pass. The earlier WER tests still hold.
- [ ] **Step 5:** Commit `Normalize numbers in WER and add CER`.

### Task 4: ASR cache: transcribe once, replay everywhere

**Files:**
- Create: `src/lifemate/transcribe.py`, `src/lifemate/asr_report.py`, `src/lifemate/asr/cached.py`
- Modify: `src/lifemate/asr/__init__.py` (register `"cached"`), `src/lifemate/asr/whisper.py` (`dtype` kwarg), `.gitignore` (`asr_cache/`)
- Test: `tests/test_asr_cache.py`

**Interfaces:**
- Consumes: `load_splits` (Task 1), `wer`/`cer` (Task 3), `build_asr` (existing).
- Produces:
  - **Cache file:** `asr_cache/<split>/<name>.jsonl`, one line per clip: `{"audio": "<relative path as in the sample>", "text": str, "seconds": float}`. The default cache root is `REPO_ROOT / "asr_cache"`, overridable with `LIFEMATE_ASR_CACHE`.
  - **Transcribe CLI:** `python -m lifemate.transcribe --split S --asr NAME [--asr-kw K=V ...] --name CACHE [--subset dev|holdout|pool|eval] [--limit N]`. It transcribes the final clip and every history user clip of each selected sample. `eval` means dev ∪ holdout. It writes the cache and prints the mean and p95 seconds. A clip already in the cache is skipped, so an interrupted molab run can resume.
  - **`class CachedASR(ASR)`** with `name = "cached"`, `__init__(name: str, split: str | None = None)`. `transcribe` looks up `sample`'s audio path and raises `KeyError(f"{name}: no cached transcript for {rel}")` on a miss. The split is taken from the path when it is omitted.
  - **Report CLI:** `python -m lifemate.asr_report --split train CACHE [CACHE ...]` prints one table row per cache:
    - n clips;
    - WER and CER (normalized), against `audio_text` of each clip;
    - p50 and p95 seconds.

    It also writes `asr_cache/<split>/report.md`.
  - **Whisper precision:** `WhisperASR.__init__` gains `dtype: str = "float16"`, which accepts `float32`.

- [ ] **Step 1: Write the failing tests** with a fake ASR registered only inside the test (`monkeypatch.setitem(BACKENDS, "fake", FakeASR)`) and a temp cache root:

```python
def test_cached_replays_transcribed_text(tmp_cache, train_samples):
    write_cache(tmp_cache / "train" / "c.jsonl", [{"audio": train_samples[0]["audio"], "text": "xin chào", "seconds": 0.5}])
    asr = CachedASR("c", split="train")
    assert asr.transcribe(Path("data/train") / train_samples[0]["audio"], train_samples[0]) == "xin chào"

def test_cache_miss_fails_loudly(tmp_cache, train_samples):
    write_cache(tmp_cache / "train" / "c.jsonl", [])
    with pytest.raises(KeyError, match=train_samples[0]["audio"].split("/")[-1]):
        CachedASR("c", split="train").transcribe(Path("data/train") / train_samples[0]["audio"], train_samples[0])

def test_transcribe_covers_history_clips_and_resumes(tmp_cache, monkeypatch):
    # run transcribe on 3 samples with a FakeASR counting calls; run again; second run makes 0 calls
    ...
    assert n_lines == n_final_clips + n_history_user_clips and fake.calls_second_run == 0

def test_report_scores_against_audio_text(tmp_cache, train_samples):
    # cache = gold audio_text for 2 clips -> WER 0.0
    ...
```

  (The `...` bodies follow the comments: build the 3 samples from `train_samples[:3]` and run `transcribe.main([...])` twice. `tmp_cache` is a fixture that sets `LIFEMATE_ASR_CACHE` to `tmp_path`.)
- [ ] **Step 2:** `pytest tests/test_asr_cache.py -v` → FAIL.
- [ ] **Step 3:** Implement the three modules, the `dtype` kwarg and the `.gitignore` line, then register `cached`.
- [ ] **Step 4:** `pytest -v` → all pass, `test_registry` included (no torch import).
- [ ] **Step 5:** Commit `Add ASR transcript cache, cached backend and ASR report`.

### Task 5: Progress output and HF token on molab

**Files:**
- Modify: `src/lifemate/infer.py` (progress line every 10 samples with running mean seconds), `notebooks/molab_run.py` (pass `HF_TOKEN` from the UI field to the infer/transcribe subprocess env; add a free-text "command" mode that runs `python -m lifemate.<module> ...` for `transcribe`/`asr_report`/`compare`)
- Test: none new (runner output). Verify with `pytest -v`, and `marimo check notebooks/molab_run.py` if available.

- [ ] **Step 1:** Implement both changes. The notebook change must keep the existing fields and defaults.
- [ ] **Step 2:** `pytest -v` → all pass.
- [ ] **Step 3:** Commit `Show progress every 10 samples; pass HF token to model downloads on molab`. Tell the team the notebook must be re-uploaded to molab.

### Task 6: Dynamic few-shot examples (A1)

**Files:**
- Create: `src/lifemate/agents/fewshot.py`
- Modify: `src/lifemate/agents/react.py` (`examples: int = 0`), `src/lifemate/agents/prompting.py` (`build_messages(..., examples_text: str = "")` appended to the system prompt)
- Test: `tests/test_fewshot.py`, `tests/test_react_agent.py` (extend)

**Interfaces:**
- Consumes: `load_splits` (Task 1), `sample_apps` (Task 1).
- Produces:
  - `class ExampleBank`: `__init__(samples: list[dict], pool_ids: set[str])`. It indexes the pool samples' final `audio_text`, gold calls and results.
  - `ExampleBank.select(request: str, installed_apps: set[str], k: int, exclude_id: str | None) -> list[dict]`:
    - candidates are pool samples whose gold-call apps are all in `installed_apps`, plus out-of-scope pool samples;
    - candidates are ranked by BM25 (k1=1.5, b=0.75) of `request` against their `audio_text`, tokenized with `asr.wer.normalize`;
    - ties break by id;
    - `exclude_id` is never returned.
  - `render_examples(examples: list[dict], max_result_chars: int = 300) -> str`: an "Examples of correct behaviour" block. Per example:
    - `User: <audio_text>`;
    - one line per gold call, in the native `<tool_call>{json}</tool_call>` form;
    - `-> <result JSON truncated to max_result_chars>`;
    - `(no tool call)` for out-of-scope examples.
  - `ExampleBank.from_data(data_root: Path) -> ExampleBank`, which loads `train/train.jsonl` and the pool from `configs/splits.json`.
  - **`ReActAgent(examples=k)`:** builds the bank once in `__init__` when k > 0. Per sample it passes `render_examples(bank.select(task.request, set(task.context["installed_apps"]), k, exclude_id=task.id))`. `info["examples"] = k`.

- [ ] **Step 1: Write the failing tests**

```python
def test_select_never_returns_the_sample_itself(train_samples):
    bank = ExampleBank(train_samples, pool_ids={s["id"] for s in train_samples})
    s = train_samples[0]
    picked = bank.select(s["audio_text"], set(s["context"]["installed_apps"]), k=3, exclude_id=s["id"])
    assert s["id"] not in [p["id"] for p in picked] and len(picked) == 3

def test_select_only_uses_pool(train_samples):
    pool = {s["id"] for s in train_samples[100:]}
    bank = ExampleBank(train_samples, pool_ids=pool)
    picked = bank.select("chuyển tiền cho mẹ", set(train_samples[0]["context"]["installed_apps"]), k=4, exclude_id=None)
    assert all(p["id"] in pool for p in picked)

def test_select_requires_example_apps_installed(train_samples):
    bank = ExampleBank(train_samples, pool_ids={s["id"] for s in train_samples})
    picked = bank.select("bật đèn phòng khách", {"smart_home"}, k=2, exclude_id=None)
    assert all(sample_apps(p) <= {"smart_home"} for p in picked)

def test_render_examples_shows_calls_and_truncated_results():
    ex = [{"id": "x", "audio_text": "tắt đèn", "label": {"tool_calls": [{"name": "smart_home.light_set", "arguments": {"device_id": "l1", "power": "off"}}],
                                                         "tool_results": [{"ok": True, "pad": "z" * 1000}]}}]
    text = render_examples(ex, max_result_chars=50)
    assert "User: tắt đèn" in text and '<tool_call>{"name": "smart_home.light_set"' in text and "z" * 60 not in text
```

  Plus in `test_react_agent.py`: with a stub bank (`agent.bank = StubBank("EXAMPLE BLOCK")`, examples=2), the system message given to the LLM contains `"EXAMPLE BLOCK"`. With the default `examples=0`, the system message is unchanged.
- [ ] **Step 2:** `pytest tests/test_fewshot.py tests/test_react_agent.py -v` → FAIL.
- [ ] **Step 3:** Implement `fewshot.py`, the `examples_text` parameter and the agent wiring.
- [ ] **Step 4:** `pytest -v` → all pass.
- [ ] **Step 5:** Check the token budget with the Qwen3 tokenizer, as on 27 Oct, now with `examples=4` on every train sample: max prompt tokens < 16,384 − 384. Ledger the max. If it is over, lower `max_result_chars` and ledger the change.
- [ ] **Step 6:** Commit `Add dynamic few-shot examples from the training pool`.

### Task 7: A1 runs on molab (Lead)

Each run: Branch `feat/experiments-infra`, split `train`, agent `react`, asr `gold`, extra `--subset dev` plus the setting below. Download the full run folder into `runs/`.

- [ ] **Step 1:** Run `--agent-kw examples=0` (the dev baseline). Expected: 300 samples, 0 errors.
- [ ] **Step 2:** Run `--agent-kw examples=2`, then `--agent-kw examples=4`.
- [ ] **Step 3:** `python -m lifemate.compare runs/<k0> runs/<k2>` and `... runs/<k0> runs/<k4>`. Record both lines and the three `RESULTS.md` rows. Note max seconds; it must stay < 15.
- [ ] **Step 4:** If a k wins, repeat it with `--subset holdout` against k=0 on holdout. That checks the gain transfers to apps without examples in the pool.

### Task 8: B1 runs on molab (Partner)

- [ ] **Step 1:** Transcribe the eval clips for each PhoWhisper setting, by passing `--asr whisper --subset eval` plus the arguments below to `python -m lifemate.transcribe` in the notebook's command mode:

  | Cache name (`--name`) | ASR arguments |
  |---|---|
  | `pho-medium-fp16` | `--asr-kw model=vinai/PhoWhisper-medium` |
  | `pho-large-fp16` | `--asr-kw model=vinai/PhoWhisper-large` |
  | `pho-large-fp32` | `--asr-kw model=vinai/PhoWhisper-large --asr-kw dtype=float32` |

  Expected: one cache file each, covering every final and history clip of dev ∪ holdout.
- [ ] **Step 2:** `python -m lifemate.asr_report --split train pho-medium-fp16 pho-large-fp16 pho-large-fp32`. Commit `asr_cache/train/report.md` content into `RESULTS.md` under a new "ASR" table, not the cache files themselves.
- [ ] **Step 3:** Run end to end with the best cache: `--agent react --asr cached --asr-kw name=<best> --subset dev`, then `compare` against the Task 7 k=0 gold run. That measures the score lost to ASR on dev.

### Task 9: More ASR backends for B1 (Partner)

**Files:**
- Create: `src/lifemate/asr/qwen3asr.py` (`name = "qwen3-asr"`), `src/lifemate/asr/chunkformer.py` (`name = "chunkformer"`)
- Modify: `src/lifemate/asr/__init__.py`
- Test: `tests/test_registry.py` (both registered, no torch import at registry load)

**Interfaces:**
- Produces: two `ASR` subclasses with `__init__(model: str = <hub id>, device: str = "cuda")`.
  - `qwen3-asr` defaults to `Qwen/Qwen3-ASR-1.7B`.
  - `chunkformer` defaults to `khanhld/chunkformer-ctc-large-vie`.
  - Their packages are imported inside `__init__`, with an error message giving the exact `uv pip install --no-deps <pkg>` command.
  - Decoding is greedy.

- [ ] **Step 1:** Read each model card's usage section. Before writing code, write down in the PR the licence, release date and parameter count, and drop any candidate whose licence does not allow use in the competition.
- [ ] **Step 2:** Extend `test_registry.py`, then run it and watch it fail.
- [ ] **Step 3:** Implement both backends, following the card usage and keeping decoding greedy.
- [ ] **Step 4:** `pytest -v` → all pass.
- [ ] **Step 5:** On molab, install with `--no-deps` and check `torch.__file__` is still the system one. Then run Task 8 Steps 1–2 with caches `qwen3-asr-1.7b` and `chunkformer-large`.
- [ ] **Step 6:** Commit `Add Qwen3-ASR and ChunkFormer backends`, with the B1 table updated in `RESULTS.md`.
