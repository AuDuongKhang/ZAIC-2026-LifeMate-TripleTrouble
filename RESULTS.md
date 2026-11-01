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
| 2026-10-29 07:03 | 3eaa35f | train | 300 | react | gold | 21.02 | 34.21 | 19.04 | 4.7 | 68.0 | 2.624 | 8.306 | prompt v1 , react-v1 gold, subset dev |
| 2026-10-29 07:31 | 3eaa35f | train | 300 | react | gold | 21.38 | 35.53 | 21.35 | 4.17 | 52.0 | 2.571 | 8.032 | prompt v2 , react ref v2, subset dev |
| 2026-10-29 08:33 | 3eaa35f | train | 300 | react | gold | 38.35 | 53.86 | 36.51 | 33.45 | 64.0 | 2.793 | 10.205 | prompt v1, react fewshot k4, subset dev |
| 2026-10-29 08:49 | 3eaa35f | train | 300 | react | gold | 34.98 | 49.72 | 33.93 | 28.56 | 56.0 | 2.605 | 10.512 | prompt v1,react fewshot k2, subset dev |
| 2026-10-29 09:13 | 3eaa35f | train | 297 | react | gold | 29.69 | 42.81 | 31.56 | 6.31 | 85.71 | 2.332 | 8.141 | prompt v1, react fewshot k4, subset holdout |
| 2026-10-29 09:26 | 3eaa35f | train | 297 | react | gold | 27.65 | 39.34 | 29.49 | 6.76 | 71.43 | 2.337 | 8.971 | prompt v1, react-v1 gold, subset holdout |
| 2026-10-29 10:33 | 3eaa35f | public_test | 634 | react | whisper | 21.46 | - | - | - | - | 3.416 | 15.258 | prompt v1, react fewshot k4, public test |

## ASR

`python -m lifemate.asr_report --split train <cache> ...` over the eval clips (dev + holdout, final requests and history
user turns), number-normalized WER/CER against the train transcripts. Seconds are per clip on molab (RTX Pro 6000).

| date | commit | cache | model | clips | WER | CER | p50 s | p95 s | notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-29 | 4ac0018 | pho-medium-fp16 | vinai/PhoWhisper-medium, fp16 | 2108 | 0.2803 | 0.1564 | 0.205 | 0.487 | |
| 2026-10-29 | 4ac0018 | pho-large-fp16 | vinai/PhoWhisper-large, fp16 | 2108 | 0.2627 | 0.1449 | 0.261 | 0.616 | B1 best so far |
| 2026-10-29 | 4ac0018 | pho-large-fp32 | vinai/PhoWhisper-large, fp32 | 2108 | 0.2627 | 0.1449 | 0.321 | 0.674 | same WER/CER as fp16 but slower: keep fp16 |
| 2026-10-29 | 4ac0018 | phoasr-small-fp16 | Qualcomm-AI-Research/PhoASR-whisper-small, fp16 | 2108 | 0.2654 | 0.1536 | 0.110 | 0.250 | 0.24B: WER within 0.003 of PhoWhisper-large at 1/6 the size and ~2.5x faster; CER worse |
| 2026-10-31 | 1c15b73 | phoasr-small-ft-a | PhoASR-whisper-small fine-tuned on the 2928 pool clips (3.98 h), `train_asr --epochs 4 --lr 1e-5 --eval-steps 100`, best checkpoint | 2108 | 0.1349 | 0.0802 | 0.122 | 0.264 | B2 run A: WER halved vs PhoWhisper-large (0.2627). No speaker labels, so pool and eval may share speakers: confirm on public_test |
