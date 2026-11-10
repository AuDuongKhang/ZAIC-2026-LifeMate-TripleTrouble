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

### How it was trained

Both ASR models were trained with `notebooks/molab_asr.py` (which runs `python -m lifemate.train_asr`) on molab. No
training log was kept, so the recipe below is rebuilt from the repo history. Each line says where it comes from.

| Setting | `pho-large-ft-b` (pool, used for gates) | `pho-large-ft-all` (shipped) | Source |
|---|---|---|---|
| Base model | `vinai/PhoWhisper-large` | same | notebook default; HF commit titles |
| Clips | 2,928 pool clips (final requests and history user turns, 3.98 h) | all train clips: pool + dev + holdout, about 5,036 | `train_asr --subset pool` / `--subset all`; plan `2026-11-02-final-sprint.md` P5 ("same hyper-parameters as pho-large-ft-b, `--subset all`") |
| Epochs, lr, batch, grad accum | 3, 1e-5, 8, 2 (effective batch 16) | same | notebook defaults on 1 Nov (`1321406`, run name default `pho-large-ft-b`); the ft-a recipe in `RESULTS.md` |
| Warmup, SpecAugment | 100 steps, `mask_time_prob` 0.05 | same | `train_asr` defaults; `config.json` of both repos (`apply_spec_augment` true, `mask_time_prob` 0.05) |
| Eval and checkpoint | every 100 steps on a fixed 400-clip dev + holdout subset (seed 0); **checkpoint-500** kept (400-clip WER 0.1057) | no eval; the model after the last step | HF commit `04c0bdb` "checkpoint-500 of pho-large-ft-b"; `selection_args("all")` |
| Seed, precision | 0, bf16 | same | `train_asr` defaults |
| Time | about 1 h | about 1.5 h (code `ac2862c` at 04:50 UTC, upload 06:41 UTC on 4 Nov) | commit and upload times |
| Upload | weights + processor only | same, plus the base tokenizer files (`73bb56a`) | HF commits |

To reproduce `pho-large-ft-all`:

```bash
python -m lifemate.train_asr --model vinai/PhoWhisper-large --out asr_runs/pho-large-ft-all \
    --epochs 3 --lr 1e-5 --batch 8 --grad-accum 2 --eval-steps 100 --subset all
# then upload asr_runs/pho-large-ft-all/final (weights + processor) to the model repo
```

The epochs, lr, batch and grad accum of the two runs were not logged. They are the notebook defaults of that day and the
recipe the plan names. A run that changed them would have needed someone to edit the fields by hand. The ASR
augmentation flags added later (PR #67) are off by default and were not used for either model.

## Total size

PhoWhisper-large 1.55 B + Qwen3-4B 4.02 B = 5.57 B parameters, under the 6 B limit; the LLM is under 5 B.

## Not shipped

Models trained after sft-final-h failed the holdout gate (`RESULTS.md`): sft-kd1, sft-q1, beam search 3, sft-r1 and
sft-v6d (pool + recovery records: holdout 38.58 vs 41.37).
