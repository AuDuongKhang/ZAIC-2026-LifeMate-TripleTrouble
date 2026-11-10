# LifeMate, team TripleTrouble: solution and training code

The image runs the system in `/code/configs/shipped.json`: ASR `pho-large-ft-all` (beam 5) feeding a ReAct agent on
`lifemate-sft-final-h`. Public leaderboard: 57.5 (7 Nov 2026). The full source is in `/code/src/lifemate`, the
model card with every training setting in `/code/docs/shipped-model.md`, and experiment logs in `/code/RESULTS.md`.

## Idea

1. **ASR.** Every user audio clip (the final request and the history user turns) is transcribed by PhoWhisper-large
   fine-tuned on the train clips, beam search 5.
2. **Agent.** A ReAct loop on a LoRA-fine-tuned Qwen3-4B-Instruct-2507. The prompt holds the tool schemas of the
   sample's installed apps, the history (with its earlier tool calls and results), the user profile and current time,
   and 4 few-shot examples picked by BM25 from the train requests. The model writes one `<tool_call>` per step; the
   runner checks it with `env.validate` (free, not scored), sends it with `env.call`, and appends the result. It
   stops when the model answers without a call (done, or nothing to do) or calls `sys.ask_missing_info`.
   `check_refs` rejects calls whose IDs were never seen in the history or earlier results.
3. **One `ToolEnv` per sample**, created by the runner; app data is only read through tool calls.

| Part | Model | Parameters | Notes |
|---|---|---|---|
| ASR | [`TripleTrouble/pho-large-ft-all`](https://huggingface.co/TripleTrouble/pho-large-ft-all) @ `73bb56a` | 1.55 B | from `vinai/PhoWhisper-large`, fp16, beam 5 |
| LLM | [`TripleTrouble/lifemate-sft-final-h`](https://huggingface.co/TripleTrouble/lifemate-sft-final-h) @ `e3a659c` | 4.02 B | from `Qwen/Qwen3-4B-Instruct-2507`, merged LoRA, bf16, greedy, context ≤ 16,384 tokens |

Total 5.57 B parameters. Only these fine-tuned weights are in the image; the base models are downloaded from Hugging
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
- **Hidden gold apps** for half of the samples: their few-shot examples use none of the sample's gold apps, as for
  an app never seen in training.

## Reproduce

Python ≥ 3.10, one large GPU for the LLM SFT (a 96 GB RTX Pro 6000 on molab was used), `pip install -r requirements.txt`
and `pip install --no-deps -r requirements-train.txt`, the train data under `data/train/`. Seeds are fixed (0) in
every step (`lifemate.utils.seed_everything`, `--seed 0`).

```bash
# 1. ASR fine-tune on all train clips (about 1.5 h), then upload <out>/final
python -m lifemate.train_asr --model vinai/PhoWhisper-large --out asr_runs/pho-large-ft-all \
    --epochs 3 --lr 1e-5 --batch 8 --grad-accum 2 --eval-steps 100 --subset all

# 2. ASR-noisy text for the records (off-the-shelf PhoWhisper-large)
python -m lifemate.transcribe --split train --asr whisper --asr-kw model=vinai/PhoWhisper-large \
    --name pho-large-fp16                                  # no --subset: every train clip

# 3. SFT records (branch feat/fewshot-unseen, which has --hide-gold-apps): 4,629 records
python -m lifemate.sft.records --name gold-all-m1-h50-asr-pho-large-fp16 --subset all \
    --text gold --text cache:pho-large-fp16 --mask 1 --hide-gold-apps 0.5

# 4. LoRA SFT (r 16, alpha 32, dropout 0.05, lr 1e-4, warmup 3%, micro batch 1, grad accum 16, 2 epochs = 579 steps,
#    max length 12,288; about 3 h), then merge the adapter into the base model
python -m lifemate.sft.train --records gold-all-m1-h50-asr-pho-large-fp16 --out checkpoints/sft-final-h2
python -m lifemate.sft.merge --adapter checkpoints/sft-final-h2/last --out merged/sft-final-h
```

Inference: `bash /code/predict.sh` (input under `/data`, output `/result/submission.jsonl`) or
`/code/predict_notebook.ipynb` (output `/result/jupyter_submission.jsonl`).
