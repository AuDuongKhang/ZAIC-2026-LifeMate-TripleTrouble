# LifeMate, team TripleTrouble: solution and training code

The image runs the system in `/code/configs/shipped.json`: ASR `pho-large-ft-all` (beam 5) feeding a ReAct agent on
`lifemate-sft-q1c` (preset `sft-q1c-refs-fixes-all`). Public leaderboard: 63.23 (11 Nov 2026). The full source is in `/code/src/lifemate`, the
model card with every training setting in `/code/docs/shipped-model.md`, and experiment logs in `/code/RESULTS.md`.

## Idea

1. **ASR.** Every user audio clip (the final request and the history user turns) is transcribed by PhoWhisper-large
   fine-tuned on the train clips, beam search 5.
2. **Agent.** A ReAct loop on a LoRA-fine-tuned Qwen3.5-4B (text-only weights). The prompt holds the tool schemas of the
   sample's installed apps, the history (with its earlier tool calls and results), the user profile and current time,
   and 4 few-shot examples picked by BM25 from the train requests. The model writes one `<tool_call>` per step; the
   runner checks it with `env.validate` (free, not scored), sends it with `env.call`, and appends the result. It
   stops when the model answers without a call (done, or nothing to do) or calls `sys.ask_missing_info`.
   `check_refs` rejects calls whose IDs were never seen in the history or earlier results. Four schema-driven
   rules (no app or tool name decides anything): `id_fix` passes an entity's id where a write call names it,
   `spoken_email` respells a misheard spoken email address, `weekday_fix` puts the calendar date of the request's only
   "thứ X (tuần sau)" phrase into the call, and `empty_retry` re-sends a read that found nothing without its optional,
   non-date filters.
3. **One `ToolEnv` per sample**, created by the runner; app data is only read through tool calls.

| Part | Model | Parameters | Notes |
|---|---|---|---|
| ASR | [`TripleTrouble/pho-large-ft-all`](https://huggingface.co/TripleTrouble/pho-large-ft-all) @ `73bb56a` | 1.55 B | from `vinai/PhoWhisper-large`, fp16, beam 5 |
| LLM | [`TripleTrouble/lifemate-sft-q1c`](https://huggingface.co/TripleTrouble/lifemate-sft-q1c) @ `8cb4336` | 4.21 B | from `Qwen/Qwen3.5-4B` (text-only weights, `TripleTrouble/qwen35-4b-text`), merged LoRA, bf16, greedy, context ≤ 16,384 tokens |

Total 5.75 B parameters. Only these fine-tuned weights are in the image; the base models are downloaded from Hugging
Face when training.

## Data

Only the organisers' train split (`train.jsonl`, `train_tools_catalog.json`, `audio/`, `train_environment/`) is used.
No external or paid data, no LLM API. The SFT records are rendered from it:

- **Gold replay**: each train sample is run through the same ReAct prompt as inference with its gold tool calls and
  results replayed (`lifemate.sft.records`, `lifemate.sft.replay`), so training text equals inference text.
- **Function-masked copy** (1 per sample): apps, tools and parameters renamed and tools shuffled
  (`lifemate.sft.mask`), so the model reads the schemas instead of memorising app names.
- **ASR-noisy copy** (1 per sample): user turns replaced by off-the-shelf PhoWhisper-large transcripts
  (`lifemate.transcribe`, cache `pho-large-fp16`).
- **Gold apps left out** of the few-shot examples for half of the records, as for an app never seen in training.

The LLM is trained on the experiment pool (train minus the fixed dev and holdout subsets used for local scoring).

## What is in this folder

| Path | Content |
|---|---|
| `code_at_4f37ecb/` | **The code that trained the shipped LLM** (repo commit `4f37ecbb`, 11 Nov 2026): `src/lifemate` (`transcribe`, SFT `sft.text_only` / `sft.records` / `sft.train` / `sft.merge`, ASR fine-tune `train_asr`), `configs/`, `notebooks/` (the molab notebooks that ran it), `requirements*.txt`. Use it to reproduce the models below. The ASR model was trained on 4 Nov with the `train_asr` options of step 1, which this code still has. |
| `src/`, `configs/`, `notebooks/`, `scripts/`, `pyproject.toml`, `requirements*.txt` | The code of this image (same as `/code/src`): inference, plus the training modules as they are now. |

## Data

Only the organisers' train split is used: put it at `data/train/` (`train.jsonl`, `train_tools_catalog.json`,
`audio/`, `train_environment/`), next to `src/`, or point `LIFEMATE_DATA_ROOT` to the folder that holds `train/`. No
external or paid data and no LLM API. Everything else is generated from it by the code (steps 2 and 3 below), so no
other data file has to be downloaded:

- **Gold replay**: each train sample is run through the same ReAct prompt as inference with its gold tool calls and
  results replayed (`lifemate.sft.records`, `lifemate.sft.replay`), so training text equals inference text.
- **Function-masked copy** (1 per sample): apps, tools and parameters renamed and tools shuffled
  (`lifemate.sft.mask`), so the model reads the schemas instead of memorising app names.
- **ASR-noisy copy** (1 per sample): user turns replaced by off-the-shelf PhoWhisper-large transcripts
  (`lifemate.transcribe`, cache `pho-large-fp16`).
- **Gold apps left out** (`--leave-app-out 0.5`): each record is drawn separately, and half of them get few-shot
  examples that use none of the record's gold apps, as for an app never seen in training.

The LLM is trained on the experiment pool (`--subset pool`: train minus the fixed dev and holdout subsets of
`lifemate.splits`), with the dev subset as eval loss.

## Models

| | Download |
|---|---|
| Base ASR | https://huggingface.co/vinai/PhoWhisper-large |
| Base LLM | https://huggingface.co/Qwen/Qwen3.5-4B @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (step 0 keeps its text-only weights) |
| Fine-tuned ASR (output of step 1) | in this image under `/models` (HF cache of `TripleTrouble/pho-large-ft-all` @ `73bb56a`) |
| Fine-tuned LLM (output of step 4) | in this image under `/models` (HF cache of `TripleTrouble/lifemate-sft-q1c` @ `8cb4336`) |

## Reproduce

Run from `code_at_4f37ecb/` with Python ≥ 3.10 and one large GPU (a 96 GB RTX Pro 6000 on molab was used):
`pip install -r requirements.txt` and `pip install --no-deps -r requirements-train.txt`. Seeds are fixed (0) in every
step (`lifemate.utils.seed_everything`, `--seed 0`), decoding is greedy (beam 5 for the ASR).

```bash
cd code_at_4f37ecb && export PYTHONPATH=src
# 0. Text-only base LLM: Qwen3.5-4B without its vision tower, with the repo's chat template (4.21 B parameters)
python -m lifemate.sft.text_only --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
    --chat-template configs/chat_templates/qwen_json_nothink.jinja --out merged/qwen35-4b-text

# 1. ASR fine-tune on all train clips (about 1.5 h on molab); the model is <out>/final
python -m lifemate.train_asr --model vinai/PhoWhisper-large --out asr_runs/pho-large-ft-all \
    --epochs 3 --lr 1e-5 --batch 8 --grad-accum 2 --eval-steps 100 --subset all

# 2. ASR-noisy text for the records (off-the-shelf PhoWhisper-large, the pool's clips)
python -m lifemate.transcribe --split train --asr whisper --asr-kw model=vinai/PhoWhisper-large --name pho-large-fp16 \
    --subset pool

# 3. SFT records (train: pool, eval loss: dev)
python -m lifemate.sft.records --name gold-pool-m1-asr-pho-large-fp16-lao50 --subset pool --examples 4 --tokenizer none \
    --mask 1 --leave-app-out 0.5 --text gold --text cache:pho-large-fp16
python -m lifemate.sft.records --name gold-dev --subset dev --examples 4 --tokenizer none

# 4. LoRA SFT on the text-only base (r 16, alpha 32, dropout 0.05, lr 1e-4, warmup 3%, micro batch 1, grad accum 16,
#    2 epochs = 356 steps, max length 12,288, seed 0; about 2.6 h on molab), then merge the adapter into the base
python -m lifemate.sft.train --records gold-pool-m1-asr-pho-large-fp16-lao50 --eval-records gold-dev \
    --out checkpoints/sft-q1c --model merged/qwen35-4b-text \
    --lora-targets q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,gate_proj,up_proj,down_proj
python -m lifemate.sft.merge --adapter checkpoints/sft-q1c/last --out merged/sft-q1c
```

The team ran step 0 once and used its upload, `TripleTrouble/qwen35-4b-text` @ `300ccc1`, as `--model` in step 4; the
local folder `merged/qwen35-4b-text` holds the same weights. The run's `run_config.json` and `train_log.jsonl` (356
steps, final eval loss 0.0350) are next to the weights in `TripleTrouble/lifemate-sft-q1c` (`checkpoints/sft-q1c/last/`).

`notebooks/molab_asr.py` (step 1) and `notebooks/molab_train.py` (steps 2–4; step 0 ran through the command field of
`notebooks/molab_run.py`) in `code_at_4f37ecb/` ran these steps on molab with these values. The ASR settings were not logged; they are the notebook defaults of 1–4 Nov
(`/code/docs/shipped-model.md` gives the sources).

## Inference

`bash /code/predict.sh` (input: the split folder under `/data`; output: `/result/submission.jsonl`) or
`/code/predict_notebook.ipynb` (output: `/result/jupyter_submission.jsonl`). Both run `configs/shipped.json` offline
with the models above.
