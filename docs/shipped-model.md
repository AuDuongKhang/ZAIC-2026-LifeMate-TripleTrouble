# Shipped model: sft-q1c + four agent rules + pho-large-ft-all beam 5

The system in `configs/shipped.json` (agent preset `sft-q1c-refs-fixes-all`), best on the public leaderboard: **63.23**
(11 Nov 2026, run commit `d200053`, mean 5.11 s and max 14.71 s per sample on molab, 634/634 samples, 0 over the time
limit). The same rules on sft-q1b scored 62.83, and sft-q1b without them 61.05. The holdout/dev gate runs
(`RESULTS.md`, 11 Nov): sft-q1c vs sft-q1b with the same rules except `empty_retry` and `weekday_fix`, holdout +2.18
[-1.52, +5.89], dev -1.78 [-4.92, +1.48], pooled +0.19; `empty_retry` alone on sft-q1b, holdout +0.95 [+0.01, +1.99], dev
+0.80 [+0.13, +1.68]. The public difference between sft-q1c and sft-q1b (+0.40) is within noise.

## Agent: `TripleTrouble/lifemate-sft-q1c`, preset `sft-q1c-refs-fixes-all`

Pinned revision `8cb4336e529f0a4b2d6f10cdcfc63d15c45b651d` (merge commit "merged .../checkpoints/sft-q1c/last", 11 Nov
09:09 UTC; the gate and public runs ran after it, unpinned, on that latest commit). `run_config.json` and
`train_log.jsonl` are next to the weights in `checkpoints/sft-q1c/last/`. 4.21 B parameters (bf16).

Agent options: prompt v1, 4 few-shot examples by BM25 from the train bank, `check_refs`, `id_fix`, `spoken_email`,
`empty_retry`, `weekday_fix` (`configs/presets.json`, `src/lifemate/agents/`).

It is the sft-q1b recipe (Qwen3.5-4B student on the experiment pool) with the gold apps left out of the few-shot
examples for half of the records:

| Setting | Value |
|---|---|
| Base model | `TripleTrouble/qwen35-4b-text` @ `300ccc1`: text-only weights of `Qwen/Qwen3.5-4B` @ `851bf6e`, chat template `configs/chat_templates/qwen_json_nothink.jinja` (`lifemate.sft.text_only`) |
| Train records | `gold-pool-m1-asr-pho-large-fp16-lao50`: pool samples (train minus dev and holdout), gold replay + 1 function-masked copy + 1 ASR-noisy copy per sample |
| ASR-noisy copy | user turns from the `pho-large-fp16` cache (off-the-shelf `vinai/PhoWhisper-large`) |
| Leave-app-out | `--leave-app-out 0.5`: each record drawn separately, half get few-shot examples without their gold apps |
| Eval records | `gold-dev` (final eval loss 0.0350) |
| LoRA | r 16, alpha 32, dropout 0.05, targets `q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,gate_proj,up_proj,down_proj` |
| Optimiser | lr 1e-4, warmup 3%, micro batch 1, grad accum 16, 2 epochs = 356 steps, max length 12,288, seed 0 |
| Time | 2 h 34 min on molab (RTX Pro 6000), with the `fla` kernels of PR #82 (sft-q1b took 10 h without them) |
| Code | repo commit `4f37ecbb` (`records_commit`): `main` when the notebook cloned it; its `src/` equals `595e080`, #82 changed only the Dockerfile and requirements |

Commands: `docker/training_code/README.md`, steps 0 and 2–4.

## Previous shipped agent: `TripleTrouble/lifemate-sft-final-h` (57.5)

Pinned revision `e3a659cb166907460f564c31a01e613786a835b6` (merge commit "merged .../checkpoints/sft-final-h2/last", 6 Nov
13:22 UTC). The repo also holds checkpoint folders of other runs (`checkpoints/sft-final-h`, `checkpoints/sft-v1`); only
`checkpoints/sft-final-h2` produced that model. `run_config.json` and `train_log.jsonl` are stored next to the weights.

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

## Docker image

Built by `.github/workflows/docker-image.yml` (`scripts/docker_build.sh`), layout as in the organiser Docker guide:
`/code` holds `predict.sh`, `start_jupyter.sh`, `predict_notebook.ipynb`, `README.md`, `requirements.txt` and
`training_code/` (README, this code, and the code at `records_commit` that trained the models); the weights are in the
HF cache under `/models`; input is the split folder under `/data`, output `/result/submission.jsonl` (and
`/result/jupyter_submission.jsonl` from the notebook), one line per sample with `time` in ms.

| | |
|---|---|
| Base image | `pytorch/pytorch:2.15.0-cuda13.2-cudnn9-runtime` (torch 2.15.0+cu132, kernels sm_75 to sm_120: A30 and RTX 5090) |
| Libraries | `docker/constraints.txt` (transformers 5.21.0, huggingface_hub 2.3.0, tokenizers 0.23.3) |
| Size | image 20.78 GB, `zac2026_TripleTrouble.tar.gz` 14.44 GiB |
| CPU smoke test | every build: models load offline, `predict.sh` and the notebook give the same 634 lines on public_test, `zaic_tools --validate` 0 errors |

Portal *Test Solution* runs:

| Date | Commit | MD5 | Result |
|---|---|---|---|
| 2026-11-10 | `28e209e` (tag `docker-test4`) | `51fe14c8721c93178f03ad42af613e7b` | "Test passed. Your solution runs correctly." |

The Drive links are not kept in the repo (the image carries `train.jsonl`, licensed data); `rclone cat
gdrive:zaic2026/<folder>/info.txt` gives the link of each upload. Still to check on GPU: score and time per sample
with the image's library versions (molab run of `sft-final-h` on public_test from this branch).

## Total size

PhoWhisper-large 1.54 B + Qwen3.5-4B text-only 4.21 B = 5.75 B parameters (safetensors metadata of the pinned
revisions), under the 6 B limit; the LLM is under 5 B.

## Image libraries

The image installs `flash-linear-attention` / `fla-core` 0.6.0 (PR #82) for the gated delta net layers of Qwen3.5. The
public runs of 11 Nov ran on `d200053`, before #82, so without these kernels: the image decodes the same greedy text
through different kernels, which may change a few outputs. A molab run of this branch on public_test (runbook C1 with
the pinned revisions) gives the score and time with the image's libraries.

## Not shipped

Models trained after sft-final-h failed the holdout gate (`RESULTS.md`): sft-kd1, sft-q1, beam search 3, sft-r1 and
sft-v6d (pool + recovery records: holdout 38.58 vs 41.37).
