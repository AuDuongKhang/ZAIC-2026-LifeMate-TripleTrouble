# Results log

One row per run. `python -m lifemate.evaluate runs/<run>` (and the molab notebook) prints the row ready to paste.
Train scores are local (`lifemate.scoring`); public_test scores come from the leaderboard, so fill in "score" by hand.
Timing is per sample on the machine that ran it; the scoring A30 is slower than molab's RTX Pro 6000.

| date | commit | split | n | agent | asr | score | F1 no hash | act | ask | empty | mean s | max s | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-27 05:50 | 5a7ce91 | train | 100 | react | gold | 16.58 | 27.78 | 12.5 | 11.11 | 83.33 | 3.043 | 8.578 | Tier 0 baseline: Qwen3-4B-Instruct-2507 (HF, bf16), first 100 train samples, molab |
| 2026-10-27 06:08 | cb9d9cf | train | 100 | react | whisper | 8.17 | 15.98 | 3.86 | 0.0 | 83.33 | 3.259 | 12.055 | WER 0.2982; PhoWhisper-medium, same agent as the gold row |
| 2026-10-27 06:20 | 0e4a696 | train | 100 | empty | whisper | 6.0 | 6.0 | 0.0 | 0.0 | 100.0 | 0.768 | 2.02 | WER 0.2831; ASR-only check, PhoWhisper-large (medium: 0.2982); part of the WER is digits vs spelled-out numbers |
| 2026-10-27 06:43 | 354cbb0 | train | 100 | react | gold | 14.88 | 28.73 | 13.27 | 0.0 | 66.67 | 3.24 | 8.022 | prompt v2 + check_refs; vs v1: 5 samples better, 6 worse (asks lost: model called by phone_number instead of asking) |
| 2026-10-27 08:11 | 5baab9a | public_test | 634 | react | whisper | 18.95 | - | - | - | - | 3.262 | 8.613 | prompt v1; PhoWhisper-large; first leaderboard submission |
| 2026-10-29 08:04 | 4ac0018 | train | 50 | react | gold | 8.27 | 24.06 | 9.74 | 3.33 | 0.0 | 3.677 | 8.198 | prompt v1; first 50 of dev (--subset dev --limit 50), same ids as the cached row below |
| 2026-10-29 07:32 | 4ac0018 | train | 50 | react | cached | 4.33 | 11.53 | 5.56 | 0.0 | 0.0 | 2.749 | 8.931 | prompt v1; cache pho-large-fp16, first 50 of dev (--subset dev --limit 50); WER 0.2313; vs gold row above: -3.93 [95% CI -8.47, -0.67], better 0 / worse 4 / same 46; out-of-scope sample lost only because the agent made read calls |

## ASR

`python -m lifemate.asr_report --split train <cache> ...` over the eval clips (dev + holdout, final requests and history
user turns), number-normalized WER/CER against the train transcripts. Seconds are per clip on molab (RTX Pro 6000).

| date | commit | cache | model | clips | WER | CER | p50 s | p95 s | notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-29 | 4ac0018 | pho-medium-fp16 | vinai/PhoWhisper-medium, fp16 | 2108 | 0.2803 | 0.1564 | 0.205 | 0.487 | |
| 2026-10-29 | 4ac0018 | pho-large-fp16 | vinai/PhoWhisper-large, fp16 | 2108 | 0.2627 | 0.1449 | 0.261 | 0.616 | B1 best so far |
| 2026-10-29 | 4ac0018 | pho-large-fp32 | vinai/PhoWhisper-large, fp32 | 2108 | 0.2627 | 0.1449 | 0.321 | 0.674 | same WER/CER as fp16 but slower: keep fp16 |
