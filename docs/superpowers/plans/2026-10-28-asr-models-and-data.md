# ASR Models and Fine-Tuning Data Plan (B1 extension + B2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Check every ASR candidate and public dataset the team found, add the cheap ones to B1, and build the data and
training script for B2 (fine-tune the B1 winner on in-domain + regional Vietnamese speech).

**Depends on:** `2026-10-27-experiments-batch1.md` Tasks 1, 3 and 4 (splits, normalized WER/CER, ASR cache). Tasks 1–2
below can start before them; Tasks 3–5 cannot.

**Spec:** `docs/superpowers/specs/2026-10-27-agent-roadmap-design.md`, section "Experiments", B1 and B2.

## Why

- Train 100, same agent: gold transcripts 16.58, PhoWhisper-medium 8.17. ASR halves the score.
- WER is spread from < 0.1 to 0.85 by speaker, and PhoWhisper medium and large fail on the same clips (strong
  regional accents). No Hugging Face model is trained specifically for regional accents; the best ones are general
  Vietnamese models trained on North/Central/South data. So the gain has to come from fine-tuning on accented and
  in-domain audio.

## Global Constraints

- Everything in batch 1's Global Constraints applies (no torch in `requirements.txt`, experimental packages installed
  by hand with `--no-deps`, never train on dev or holdout clips).
- Parameter budget: ASR ≤ ~1.9B next to a ~4B LLM, whole system < 6B. Count the real checkpoint.
- Every model and dataset must be released before 26 Oct 2026 with a licence that allows use in the competition.
  Record the evidence (link + quote of the licence line) in the audit table. "Unclear" means not used until the
  organisers confirm.
- Never transcribe, pseudo-label or train on public_test or private_test audio.
- The Docker package ships training code **and data**. Public datasets of hundreds of hours may not fit, so ship the
  exact clip-id manifests plus a download script, and ask the organisers if that is acceptable (Task 1).

---

### Task 1: Licence and release audit (Partner, no GPU)

**Files:**
- Create: `docs/asr_audit.md`

- [ ] **Step 1:** Fill one row per candidate: hub id, parameter count (from the checkpoint, not the name), licence,
  release date, required package, and whether it can be used. Expected rows:

  | Model | Size | Known so far |
  |---|---|---|
  | vinai/PhoWhisper-large | 1.55B | baseline; WER 0.283 on train 100 (before number normalization) |
  | Qwen/Qwen3-ASR-1.7B, Qwen/Qwen3-ASR-0.6B | 1.7B / 0.6B | Apache 2.0, Jan 2026; `qwen-asr`; 0.6B WER 9.21% FLEURS-vi (community number) |
  | khanhld/chunkformer-ctc-large-vie | ~110M | WER 4.18 VIVOS, 6.66 CV-vi; licence and training data unclear |
  | nvidia parakeet-ctc-0.6b-vi | 0.6B | 4,000+ h, code-switching; NVIDIA licence; HF repo and date unconfirmed; NeMo |
  | PhoASR-whisper-small (Qualcomm, EACL 2026) | 244M | 3,000 h, punctuation; card says weak on Central accents |
  | VieGem-ASR, whisper-small-vi, wav2vec2-base-vi-250h | small | fallbacks only |

- [ ] **Step 2:** Same for datasets: hours, regions, licence, size on disk and HF id.

  | Dataset | Hours | Known so far |
  |---|---|---|
  | Our train (pool clips only) | ≤ 6.8 h, 5,036 clips in total | in-domain: same vocabulary (app names, English words such as "S+ Sport") |
  | ViMD (EMNLP 2024) | 102.56 h | 63 provinces, region + province labels; licence "for research" |
  | VietSpeech (NhutP) | 1,100+ h | North, Central, South |
  | Bud500 | ~500 h | North, Central, South |

- [ ] **Step 3:** Post two questions to the organisers and record the answers in the audit file:
  1. Does a "for research" licence (ViMD) allow use in the challenge?
  2. Can the Docker package ship download scripts + clip manifests instead of hundreds of hours of public audio?
- [ ] **Step 4:** Commit `Add ASR model and dataset audit`.

### Task 2: Whisper-family candidates through the existing backend (Partner)

PhoASR-whisper-small and whisper-small-vi are Whisper checkpoints, so they need no new code: they run through
`--asr whisper --asr-kw model=<hub id>`. This only adds rows to batch 1 Task 8.

- [ ] **Step 1:** After batch 1 Task 4, transcribe the eval clips with the caches `phoasr-small-fp16` and
  `whisper-small-vi-fp16` (only the ones that passed Task 1).
- [ ] **Step 2:** Add them to the `asr_report` call and to the ASR table in `RESULTS.md`.
- [ ] **Step 3:** If parakeet passed Task 1, add `src/lifemate/asr/parakeet.py` (`name = "parakeet"`) following batch 1
  Task 9. Install NeMo with `--no-deps` on molab and check `torch.__file__` is still the system one. Drop it if NeMo
  cannot be installed without replacing torch.

**B1 decision (end of this task):** pick at most two models for B2, by dev + holdout WER/CER first and p95 seconds
second. Write the choice and the numbers into the PR.

### Task 3: Fine-tuning manifests (Partner)

**Files:**
- Create: `src/lifemate/asr_data.py`, `configs/asr_mix.json`
- Test: `tests/test_asr_data.py`

**Interfaces:**
- Manifest line: `{"audio": "<path or hf://dataset/split/idx>", "text": str, "source": "train|vimd|vietspeech|bud500|tts", "region": "north|central|south|unknown", "seconds": float}`.
- `build_train_manifest(samples, pool_ids) -> list[dict]`: every final and history user clip of **pool** samples only,
  text from `audio_text`.
- `build_public_manifest(name, hours, seed=0) -> list[dict]`: a seeded, region-balanced subset (equal hours per
  region where labels exist). Drops clips over 30 s and empty transcripts.
- Transcript style must match `audio_text`: check how train writes numbers, times, money and English words, then
  convert public transcripts to that style. If they cannot be converted, normalize both sides with batch 1's
  `normalize` and train on normalized text.
- `configs/asr_mix.json` holds the mix, for example `{"train": {"repeat": 4}, "vimd": {"hours": 40}, "vietspeech": {"hours": 30}, "bud500": {"hours": 20}}`.
  Keep public audio around 10× the in-domain hours, so one epoch fits a 12 h molab session.

- [ ] **Step 1: Write the failing tests**

```python
def test_train_manifest_uses_pool_only(train_samples):
    s = load_splits()
    m = build_train_manifest(train_samples, set(s["pool"]))
    held = set(s["dev"]) | set(s["holdout"])
    assert m and not any(any(h in r["audio"] for h in held) for r in m)

def test_train_manifest_includes_history_clips(train_samples):
    s = load_splits()
    m = build_train_manifest(train_samples, set(s["pool"]))
    assert any("_u1.wav" in r["audio"] for r in m)

def test_public_manifest_is_deterministic_and_capped():
    a = build_public_manifest("fake", hours=1, seed=0); b = build_public_manifest("fake", hours=1, seed=0)
    assert a == b and sum(r["seconds"] for r in a) <= 3600 and all(r["seconds"] <= 30 for r in a)
```

  (`fake` is a tiny in-test dataset registered with `monkeypatch`, so the test needs no download.)
- [ ] **Step 2:** `pytest tests/test_asr_data.py -v` → FAIL.
- [ ] **Step 3:** Implement it. Public audio is read through `datasets` on molab only, imported inside the function.
- [ ] **Step 4:** `pytest -v` → all pass.
- [ ] **Step 5:** Commit `Add ASR fine-tuning manifests and data mix`.

### Task 4: Fine-tuning script (Partner)

**Files:**
- Create: `scripts/train_asr.py`

- [ ] **Step 1:** Seq2Seq fine-tuning for the Whisper family (PhoWhisper, PhoASR) with HF `Seq2SeqTrainer`:
  - bf16, fixed seed, SpecAugment on, LR around 1e-5 for large (sweep only if time allows);
  - freeze the encoder for the first run if memory or time is short;
  - checkpoint every ~30 min to a private HF model repo, resume with `resume_from_checkpoint` (same rules as Tier 2);
  - evaluate dev + holdout WER/CER (batch 1 `wer`/`cer`) at each checkpoint, and keep the best one;
  - plots of train loss, learning rate and dev WER/CER, also per region where known.
- [ ] **Step 2:** If Qwen3-ASR wins B1, follow its official fine-tuning recipe instead and keep the same outputs.
- [ ] **Step 3:** Smoke test on 50 clips for 20 steps on molab: loss goes down and resume works.
- [ ] **Step 4:** Commit `Add ASR fine-tuning script`.

### Task 5: B2 runs and gate (Partner)

- [ ] **Step 1:** Run A: train clips only (`repeat` 4). Run B: the full `asr_mix.json`. Same seed and steps budget.
- [ ] **Step 2:** For each run, transcribe the eval clips into a cache (`<model>-ft-a`, `<model>-ft-b`) and add them to
  `asr_report`.
- [ ] **Step 3:** End to end: `--agent react --asr cached --asr-kw name=<best ft> --subset dev`, then `compare` against
  the best B1 cache and against the gold run.
- [ ] **Step 4:** Ship the fine-tuned model only if the CI excludes 0 (WIN). Record the rows in `RESULTS.md` and
  push the weights to the private HF model repo.

### Task 6 (optional): TTS data for in-domain vocabulary

Only if the per-clip errors after B2 are still mostly names, app names, numbers or English words:
- generate request text from the **train** tool catalog (app and device names, contact names, addresses, amounts,
  times), never from public_test;
- voice it with an open Vietnamese TTS in several voices and regions (allowed during development only);
- add it to `asr_mix.json` as `tts` at a capped share, and rerun Task 5 Run B.
