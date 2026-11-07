"""LoRA SFT of the agent LLM on records from lifemate.sft.records, plain PyTorch + peft (no Trainer).

    python -m lifemate.sft.records --name gold-pool --subset pool
    python -m lifemate.sft.records --name gold-dev --subset dev
    python -m lifemate.sft.train --records gold-pool --eval-records gold-dev --out checkpoints/sft-v1 \
        --hf-repo <owner>/lifemate-sft-v1                          # checkpoints to a private HF repo (HF_TOKEN, write)
    python -m lifemate.sft.train ... --resume                     # new session: continue from the last checkpoint
    python -m lifemate.sft.merge --adapter checkpoints/sft-v1/last --out merged/sft-v1 --hf-repo <owner>/lifemate-sft-v1

The loss is the mean cross-entropy over the labelled (assistant) tokens of each micro-batch; logits are computed only
at those positions, so a 12k-token record does not need a full-vocabulary logit tensor. A checkpoint (adapter,
optimizer, scheduler, RNG state and data position) is written every --save-minutes and at the end to <out>/last and
uploaded to <hf-repo>/checkpoints/<run>/last. --resume continues from it with the same data order and LR schedule;
with --hf-repo the HF copy is used (a local copy only when it is newer, e.g. after a failed upload).
<out>/train_log.jsonl gets one line per optimizer step; loss.png, lr.png and grad_norm.png are redrawn at each save.
--max-steps only stops early (smoke tests): the LR schedule is always that of the full run.

Needs torch (preinstalled on molab) and requirements-train.txt (peft, accelerate), installed with --no-deps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import signal
import tempfile
import time
from pathlib import Path

from ..utils import seed_everything
from .plots import plot_log, read_log
from .records import RECORDS_DIR
from .state import TrainState, batches, total_steps, truncate_log
from .tokenize import MAX_LEN, encode_record

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
# Arguments a resumed run must share with its checkpoint (the rest, such as --max-steps or --hf-repo, may change).
RUN_KEYS = ("records", "model", "epochs", "lr", "warmup", "micro_bs", "grad_accum", "lora_r", "lora_alpha",
            "lora_dropout", "max_len", "seed")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_encoded(name: str, tokenizer, max_len: int, records_dir: Path = RECORDS_DIR) -> list[dict]:
    path = records_dir / f"{name}.jsonl"
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    encoded = [e for e in (encode_record(r, tokenizer, max_len) for r in records) if e is not None]
    tokens = sum(len(e["input_ids"]) for e in encoded)
    print(f"{path.name}: {len(encoded)} records, {len(records) - len(encoded)} skipped as longer than {max_len} "
          f"tokens, {tokens} tokens", flush=True)
    return encoded


def checkpoint_step(folder: Path) -> int | None:
    state = folder / "state.json"
    return TrainState.from_json(state.read_text(encoding="utf-8")).step if state.exists() else None


def labelled_count(enc: dict) -> int:
    return sum(label != -100 for label in enc["labels"][1:])


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True, help="train_records/<NAME>.jsonl")
    ap.add_argument("--eval-records", default=None, help="records for the eval loss (dev), never trained on")
    ap.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    ap.add_argument("--out", type=Path, required=True, help="checkpoints/<run>; the run name is its last part")
    ap.add_argument("--hf-repo", default=None, help="private HF model repo for checkpoints (token: HF_TOKEN)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last, else from --hf-repo")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (0 = full run)")
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.03, help="share of the steps")
    ap.add_argument("--micro-bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--eval-every", type=int, default=50, help="optimizer steps between eval losses")
    ap.add_argument("--save-minutes", type=float, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    from ..agents.llm import load_causal_lm

    run, out = args.out.name, args.out
    out.mkdir(parents=True, exist_ok=True)
    ckpt, log_path = out / "last", out / "train_log.jsonl"
    api = None
    if args.hf_repo:
        from huggingface_hub import HfApi, snapshot_download

        api = HfApi()
        api.create_repo(args.hf_repo, private=True, exist_ok=True)
        if not args.resume and any(f.startswith(f"checkpoints/{run}/") for f in api.list_repo_files(args.hf_repo)):
            raise SystemExit(f"{args.hf_repo} already holds checkpoints/{run}; pass --resume or use another run name")
    if args.resume and api:  # the HF copy is the reference; a local one wins only when newer (a failed upload)
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_download(args.hf_repo, allow_patterns=[f"checkpoints/{run}/last/*"], local_dir=tmp)
            hub = Path(tmp) / "checkpoints" / run / "last"
            hub_step, local_step = checkpoint_step(hub), checkpoint_step(ckpt)
            if hub_step is not None and (local_step is None or hub_step >= local_step):
                shutil.rmtree(ckpt, ignore_errors=True)
                shutil.copytree(hub, ckpt)
                print(f"resume source: {args.hf_repo} checkpoints/{run}/last (step {hub_step})"
                      + (f", replacing the local copy at step {local_step}" if local_step is not None else ""), flush=True)
            elif local_step is not None:
                print(f"resume source: local {ckpt} (step {local_step}); {args.hf_repo} has "
                      + ("no checkpoint" if hub_step is None else f"an older one (step {hub_step})"), flush=True)
    elif args.resume:
        print(f"resume source: local {ckpt} (no --hf-repo)", flush=True)
    records_sha = hashlib.sha256((args.records_dir / f"{args.records}.jsonl").read_bytes()).hexdigest()[:16]
    if args.resume:
        if not ckpt.exists():
            raise SystemExit(f"--resume: no checkpoint at {ckpt}" + (f" nor in {args.hf_repo}" if api else ""))
        saved = json.loads((ckpt / "run_config.json").read_text(encoding="utf-8"))
        if saved.get("records_sha") != records_sha:  # another file would shift the data order
            raise SystemExit(f"--resume: train_records/{args.records}.jsonl differs from the one the checkpoint was "
                             f"trained on (sha {records_sha} vs {saved.get('records_sha')}); rebuild it with the same "
                             "code and options")
        changed = [f"--{k.replace('_', '-')} {saved.get(k)} -> {getattr(args, k)}" for k in RUN_KEYS
                   if saved.get(k) != getattr(args, k)]
        if changed:  # the schedule, the surviving records or the model would silently change mid-run
            raise SystemExit(f"--resume: training arguments differ from the checkpoint's: {'; '.join(changed)}")
        if (ckpt / "train_log.jsonl").exists() and len(read_log(log_path)) < len(read_log(ckpt / "train_log.jsonl")):
            shutil.copy(ckpt / "train_log.jsonl", log_path)
        state = TrainState.from_json((ckpt / "state.json").read_text(encoding="utf-8"))
        truncate_log(log_path, state.step)  # steps after the checkpoint run again
        print(f"resuming {run} at step {state.step} (epoch {state.epoch}, position {state.pos})", flush=True)
    else:
        if ckpt.exists():
            raise SystemExit(f"{out} already holds a run; pass --resume or use another --out")
        log_path.unlink(missing_ok=True)  # steps of a run stopped before its first checkpoint
        state = TrainState(epoch=0, pos=0, step=0, seed=args.seed)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (out / "run_config.json").write_text(json.dumps({**config, "records_sha": records_sha}, indent=1), encoding="utf-8")

    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train = load_encoded(args.records, tokenizer, args.max_len, args.records_dir)
    evals = load_encoded(args.eval_records, tokenizer, args.max_len, args.records_dir) if args.eval_records else []

    device = "cuda" if torch.cuda.is_available() else "cpu"  # cpu: local smoke tests with a tiny model only
    model = load_causal_lm(args.model, torch.bfloat16).to(device)  # refuses a checkpoint whose weights would stay random
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if args.resume:
        model = PeftModel.from_pretrained(model, str(ckpt), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                                 target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    n, mb_per_epoch = len(train), math.ceil(len(train) / args.micro_bs)
    total = total_steps(n, args.micro_bs, args.epochs, args.grad_accum)
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(args.warmup * total), total)
    if args.resume:
        optimizer.load_state_dict(torch.load(ckpt / "optimizer.pt", weights_only=False))
        scheduler.load_state_dict(torch.load(ckpt / "scheduler.pt", weights_only=False))
        rng = torch.load(ckpt / "rng.pt", weights_only=False)
        random.setstate(rng["python"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"]:
            torch.cuda.set_rng_state_all(rng["cuda"])
    stop_at = min(total, args.max_steps) if args.max_steps else total
    print(f"{n} train records, {total} optimizer steps in the full run, stopping at step {stop_at}", flush=True)

    def record_loss(enc: dict):
        """(sum of cross-entropy over the labelled tokens, their count), logits only at those positions."""
        ids = torch.tensor([enc["input_ids"]], device=device)
        labels = torch.tensor(enc["labels"], device=device)
        pos = (labels[1:] != -100).nonzero(as_tuple=True)[0]
        with torch.autocast(device, dtype=torch.bfloat16):
            logits = model(input_ids=ids, logits_to_keep=pos).logits[0]
        return F.cross_entropy(logits.float(), labels[pos + 1], reduction="sum"), len(pos)

    def eval_loss() -> float:
        model.eval()
        total_loss, count = 0.0, 0
        with torch.no_grad():
            for enc in evals:
                loss, k = record_loss(enc)
                total_loss, count = total_loss + loss.item(), count + k
        model.train()
        return total_loss / max(count, 1)

    def save() -> None:
        tmp = out / "last.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        model.save_pretrained(str(tmp))  # the adapter only
        torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
        torch.save(scheduler.state_dict(), tmp / "scheduler.pt")
        torch.save({"python": random.getstate(), "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if device == "cuda" else []}, tmp / "rng.pt")
        (tmp / "state.json").write_text(state.to_json(), encoding="utf-8")
        shutil.copy(out / "run_config.json", tmp / "run_config.json")
        if log_path.exists():
            shutil.copy(log_path, tmp / "train_log.jsonl")
        plot_log(log_path, out)
        plot_log(log_path, tmp)
        shutil.rmtree(ckpt, ignore_errors=True)
        tmp.rename(ckpt)
        print(f"checkpoint step {state.step} -> {ckpt}", flush=True)
        if api:
            try:
                api.upload_folder(repo_id=args.hf_repo, folder_path=str(ckpt), path_in_repo=f"checkpoints/{run}/last",
                                  commit_message=f"{run}: step {state.step}")
            except Exception as e:  # a failed upload must not kill the run; the next save retries
                print(f"upload to {args.hf_repo} failed: {type(e).__name__}: {e}", flush=True)

    stop_file = out / "STOP"  # written by the notebook's Stop button: save at the next step, then exit
    stop_file.unlink(missing_ok=True)
    stop = {"signal": False}
    try:
        signal.signal(signal.SIGTERM, lambda *_: stop.update(signal=True))
    except ValueError:  # not the main thread
        pass
    prev = read_log(log_path)
    offset = prev[-1]["seconds"] if prev else 0.0
    t0 = last_save = step_start = time.time()
    start_step = state.step
    step_loss, step_count, step_tokens = 0.0, 0, 0
    model.train()
    if state.step < stop_at:
        for batch in batches(n, state, args.micro_bs, args.epochs):
            count = sum(labelled_count(train[i]) for i in batch)
            for i in batch:
                loss, k = record_loss(train[i])
                (loss / max(count, 1) / args.grad_accum).backward()
                step_loss, step_count, step_tokens = step_loss + loss.item(), step_count + k, step_tokens + len(train[i]["input_ids"])
            done = state.epoch * mb_per_epoch + math.ceil(state.pos / args.micro_bs)
            if done % args.grad_accum and done < args.epochs * mb_per_epoch:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state.step += 1
            now = time.time()
            row = {"step": state.step, "epoch": round(done / mb_per_epoch, 4), "loss": round(step_loss / max(step_count, 1), 5),
                   "lr": lr, "grad_norm": round(grad_norm, 5), "tokens": step_tokens,
                   "seconds": round(offset + now - t0, 1)}
            if evals and (state.step % args.eval_every == 0 or state.step == stop_at):
                row["eval_loss"] = round(eval_loss(), 5)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            secs = now - step_start
            eta = (stop_at - state.step) * (now - t0) / (state.step - start_step)
            print(f"step {state.step}/{stop_at} loss {row['loss']:.4f}" + (f" eval {row['eval_loss']:.4f}" if "eval_loss" in row else "")
                  + f" lr {lr:.2e} gnorm {grad_norm:.3f} {step_tokens / secs:.0f} tok/s, eta {eta / 60:.0f} min", flush=True)
            step_loss, step_count, step_tokens, step_start = 0.0, 0, 0, time.time()
            if state.step >= stop_at:
                break
            if stop["signal"] or stop_file.exists():
                print(f"stop requested: saving at step {state.step}", flush=True)
                break
            if time.time() - last_save >= args.save_minutes * 60:
                save()
                last_save = time.time()
    save()
    stop_file.unlink(missing_ok=True)
    print(f"-> {ckpt} (step {state.step}/{total})", flush=True)
    return ckpt


if __name__ == "__main__":
    main()
