"""Fine-tune a Whisper-family ASR (PhoWhisper, PhoASR-whisper) on the train pool clips (B2, run A).

    python -m lifemate.train_asr --model Qualcomm-AI-Research/PhoASR-whisper-small --out asr_runs/phoasr-small-ft-a
    python -m lifemate.train_asr ... --hub-repo <user>/phoasr-small-ft-a            # checkpoints to a private HF repo
    python -m lifemate.train_asr ... --hub-repo <user>/phoasr-small-ft-a --resume   # new session: continue from there

Trains on every pool clip (final requests and history user turns), never on dev or holdout. Every --eval-steps it
transcribes a fixed subset of dev + holdout clips greedily and logs WER/CER (lifemate.asr.wer, as asr_report), saves a
checkpoint and keeps the best one by WER. The best model and its processor end up in <out>/final, which the whisper
backend loads directly:

    python -m lifemate.transcribe --split train --asr whisper --asr-kw model=asr_runs/phoasr-small-ft-a/final \
        --name phoasr-small-ft-a --subset eval

Plots of train loss, learning rate and eval WER/CER go to <out>/training.png, the raw log to <out>/log_history.json.
Needs accelerate, installed by hand with --no-deps (it depends on torch, so it stays out of requirements.txt).
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .asr.base import SAMPLE_RATE, load_audio
from .asr.wer import cer, wer
from .asr_data import build_eval_manifest, build_train_manifest
from .data import audio_path, default_data_root, load_samples
from .splits import load_splits
from .utils import seed_everything


class ClipDataset:
    def __init__(self, rows: list[dict], data_root: Path, processor):
        self.rows, self.data_root, self.processor = rows, data_root, processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        wav = load_audio(audio_path(self.data_root, "train", row["audio"]))
        feats = self.processor.feature_extractor(wav, sampling_rate=SAMPLE_RATE).input_features[0]
        return {"input_features": feats, "labels": self.processor.tokenizer(row["text"]).input_ids}


@dataclass
class Collator:
    processor: object
    decoder_start_token_id: int

    def __call__(self, items: list[dict]) -> dict:
        batch = self.processor.feature_extractor.pad([{"input_features": x["input_features"]} for x in items],
                                                     return_tensors="pt")
        labels = self.processor.tokenizer.pad([{"input_ids": x["labels"]} for x in items], return_tensors="pt")
        ids = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
        if (ids[:, 0] == self.decoder_start_token_id).all():  # the model prepends it again
            ids = ids[:, 1:]
        batch["labels"] = ids
        return batch


def plot_history(history: list[dict], out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: no training.png")
        return
    series = {k: [(h["step"], h[k]) for h in history if k in h] for k in ("loss", "learning_rate", "eval_wer", "eval_cer")}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, keys, title in zip(axes, (["loss"], ["learning_rate"], ["eval_wer", "eval_cer"]),
                               ("train loss", "learning rate", "eval WER / CER")):
        for k in keys:
            if series[k]:
                ax.plot(*zip(*series[k]), marker="o" if k.startswith("eval") else None, label=k)
        ax.set_title(title)
        ax.set_xlabel("step")
        if ax.lines:  # a short run may log no train loss yet
            ax.legend()
    fig.tight_layout()
    fig.savefig(out / "training.png", dpi=120)
    print(f"-> {out / 'training.png'}")


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qualcomm-AI-Research/PhoASR-whisper-small")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=default_data_root())
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--eval-steps", type=int, default=200, help="evaluate + checkpoint every N optimizer steps")
    ap.add_argument("--eval-clips", type=int, default=400, help="fixed random subset of dev + holdout clips")
    ap.add_argument("--mask-time-prob", type=float, default=0.05, help="SpecAugment; 0 turns it off")
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--grad-checkpointing", action="store_true", help="for large models on small GPUs")
    ap.add_argument("--limit", type=int, default=None, help="first N train clips only (smoke test)")
    ap.add_argument("--hub-repo", default=None, help="private HF model repo for checkpoints (token: HF_TOKEN)")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint (local, else --hub-repo)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    try:
        import accelerate  # noqa: F401  (Trainer needs it; kept out of requirements.txt because it depends on torch)
    except ImportError as e:
        raise ImportError("train_asr needs accelerate: uv pip install --python <python> --no-deps 'accelerate>=1.1.0'") from e
    import torch
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration,
                              WhisperProcessor)

    seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    gen = model.generation_config
    if getattr(gen, "lang_to_id", None):  # same rule as the whisper backend: force Vietnamese when the checkpoint can
        processor.tokenizer.set_prefix_tokens(language="vietnamese", task="transcribe")
        gen.language, gen.task = "vi", "transcribe"
    gen.forced_decoder_ids = None
    gen.num_beams, gen.do_sample = 1, False
    model.config.apply_spec_augment = args.mask_time_prob > 0
    model.config.mask_time_prob = args.mask_time_prob
    if args.freeze_encoder:
        model.freeze_encoder()
    if args.grad_checkpointing:
        model.config.use_cache = False

    samples = load_samples(args.data_root, "train")
    train_rows = build_train_manifest(samples, set(load_splits()["pool"]))[: args.limit]
    eval_rows = build_eval_manifest(samples, limit=args.eval_clips, seed=args.seed)
    print(f"train clips {len(train_rows)}, eval clips {len(eval_rows)}", flush=True)
    (args.out / "manifest_train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train_rows),
                                                   encoding="utf-8")

    tok = processor.tokenizer

    def compute_metrics(pred) -> dict:
        ids = pred.predictions
        ids[ids == -100] = tok.pad_token_id
        labels = pred.label_ids
        labels[labels == -100] = tok.pad_token_id
        hyps = tok.batch_decode(ids, skip_special_tokens=True)
        refs = tok.batch_decode(labels, skip_special_tokens=True)
        return {"wer": sum(map(wer, refs, hyps)) / len(refs), "cer": sum(map(cer, refs, hyps)) / len(refs)}

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    targs = Seq2SeqTrainingArguments(
        output_dir=str(args.out), seed=args.seed, report_to="none",
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs, bf16=bf16, fp16=torch.cuda.is_available() and not bf16,
        gradient_checkpointing=args.grad_checkpointing,
        eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="steps", save_steps=args.eval_steps,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False,
        predict_with_generate=True, generation_max_length=225, generation_num_beams=1,
        logging_steps=25, dataloader_num_workers=2, remove_unused_columns=False, label_names=["labels"],
        push_to_hub=bool(args.hub_repo), hub_model_id=args.hub_repo, hub_private_repo=True,
        hub_strategy="checkpoint", hub_token=os.environ.get("HF_TOKEN"),
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=ClipDataset(train_rows, args.data_root, processor),
        eval_dataset=ClipDataset(eval_rows, args.data_root, processor),
        data_collator=Collator(processor, model.config.decoder_start_token_id),
        compute_metrics=compute_metrics, processing_class=processor.feature_extractor,
    )

    resume = None
    if args.resume:
        local = sorted(args.out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if local:
            resume = str(local[-1])
        elif args.hub_repo:
            from huggingface_hub import snapshot_download
            snapshot_download(args.hub_repo, allow_patterns=["last-checkpoint/*"], local_dir=args.out,
                              token=os.environ.get("HF_TOKEN"))
            resume = str(args.out / "last-checkpoint")
        print(f"resuming from {resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume)

    final = args.out / "final"
    trainer.save_model(str(final))  # the best checkpoint by eval WER (load_best_model_at_end)
    processor.save_pretrained(str(final))
    history = trainer.state.log_history
    (args.out / "log_history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    plot_history(history, args.out)
    best = trainer.state.best_metric
    print(f"-> {final}  best eval WER {best:.4f} on {len(eval_rows)} clips" if best is not None else f"-> {final}")
    if args.hub_repo:
        trainer.push_to_hub(commit_message="final (best eval WER)")
    return final


if __name__ == "__main__":
    main()
