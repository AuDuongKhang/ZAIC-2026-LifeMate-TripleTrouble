# Shipped model: sft-final-h + pho-large-ft-all beam 5

The system in `configs/shipped.json`, best on the public leaderboard: **57.5** (7 Nov 2026, run commit `0c7152b`, mean 3.84 s
and max 11.93 s per sample on molab, 634/634 samples). The previous best, sft-final with the same ASR, scored 56.01, so the
difference (+1.49) comes from the agent alone.

## Agent: `TripleTrouble/lifemate-sft-final-h`

Pinned revision `e3a659cb166907460f564c31a01e613786a835b6` (merge commit "merged .../checkpoints/sft-final-h2/last", 6 Nov
13:22 UTC). The repo also holds checkpoint folders of other runs (`checkpoints/sft-final-h`, `checkpoints/sft-v1`); only
`checkpoints/sft-final-h2` produced the shipped model. `run_config.json` and `train_log.jsonl` are stored next to the weights.

It is the sft-final recipe (all 1,545 train samples) plus few-shot examples hidden for half of the samples:

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen3-4B-Instruct-2507` |
| Train records | `gold-all-m1-h50-asr-pho-large-fp16`: 4,629 records (gold replay, 1 function-masked copy, 1 ASR-noisy copy per sample) |
| ASR-noisy copy | user turns from the `pho-large-fp16` cache (off-the-shelf `vinai/PhoWhisper-large`, not a fine-tuned model) |
| Hidden apps | `--hide-gold-apps 0.5`: 769 samples (with all their copies) rendered with few-shot examples that use none of the sample's gold apps |
| Perturb, date hints, synthetic data | none |
| LoRA | r 16, alpha 32, dropout 0.05 |
| Optimiser | lr 1e-4, warmup 3%, micro batch 1, grad accum 16, 2 epochs = 579 steps, max length 12,288, seed 0 |
| Time | 3 h 7 min on molab (RTX Pro 6000); final train loss about 0.05 |
| Eval records | none (all train is trained on, so there is no clean local score) |

Served with preset `sft-final-h`: prompt v1, 4 few-shot examples by BM25 from the train bank, `check_refs`.

### Code that rendered the records

The records were built on branch `feat/fewshot-unseen` (`--hide-gold-apps`), which was **not merged**: `main` gained
`--leave-app-out P` instead. The two are close but not identical:

- `--hide-gold-apps 0.5` (used here) picks about half of the **samples**, and all copies of a picked sample hide their apps.
- `--leave-app-out 0.5` on `main` draws each **record** separately (name suffix `-lao50`).

To rebuild this exact model, train from `feat/fewshot-unseen` with the notebook exported from that branch: *Train records* =
all train, *masked copies* = 1, *hide gold apps* = 0.5, *ASR-noisy copy* = `pho-large-fp16`, run name `sft-final-h2`. On
`main`, the nearest recipe is the same fields with *leave-app-out* = 0.5; it has not been trained or scored.

Neither option has a clean holdout gate: the public score is the only evidence for hiding apps.

## ASR: `TripleTrouble/pho-large-ft-all`, beam 5

Pinned revision `73bb56a73e1e1470bbda0fad99353b6f1ed6c338`. `vinai/PhoWhisper-large` fine-tuned on all train clips
(`train_asr --subset all`), decoded with `num_beams` 5 (preset `pho-large-ft-all-beam5`). On public it raised sft-final
from 53.58 (pho-large-ft-b, greedy) to 56.01.

## Total size

PhoWhisper-large 1.55 B + Qwen3-4B 4.02 B = 5.57 B parameters, under the 6 B limit; the LLM is under 5 B.

## Not shipped

Models trained after sft-final-h failed the holdout gate (`RESULTS.md`): sft-kd1, sft-q1, beam search 3, sft-r1 and
sft-v6d (pool + recovery records: holdout 38.58 vs 41.37).
