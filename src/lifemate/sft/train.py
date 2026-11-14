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

--decision-weight W (default 1) gives the tokens that decide a turn (its first token, tool names, argument keys) W times the
weight of the others in the training loss, which is the weighted sum over the weight sum. The logged `loss` and the eval
loss stay plain cross-entropy over the same tokens, so curves compare with runs without the option.

--kd-teachers A,B adds logit distillation (lifemate.sft.kd). Each teacher runs once over the train records before the
student is built; the soft targets go to <out>/kd_targets.pt and their summary to <out>/kd_report.json. --kd-report-only
stops there, which shows whether a teacher says anything beyond the hard labels before a training slot is spent. The
logged `loss` and `eval_loss` stay plain cross-entropy, so curves compare with runs without distillation; `kd` is the
extra term.

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
from .kd import kd_sum, merge_targets, soft_targets, target_report
from .state import TrainState, batches, total_steps, truncate_log
from .pref import encode_pair, rpo_loss, step_logps
from .tokenize import MAX_LEN, encode_record

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
# Arguments a resumed run must share with its checkpoint (the rest, such as --max-steps or --hf-repo, may change).
RUN_KEYS = ("records", "model", "epochs", "lr", "warmup", "micro_bs", "grad_accum", "lora_r", "lora_alpha",
            "lora_dropout", "max_len", "seed", "decision_weight", "lora_targets", "kd_teachers", "kd_alpha", "kd_temp",
            "kd_top_k", "pairs", "beta", "sft_weight", "sft_share", "revision")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_encoded(name: str, tokenizer, max_len: int, records_dir: Path = RECORDS_DIR,
                 decision_weight: float = 1.0) -> list[dict]:
    path = records_dir / f"{name}.jsonl"
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    encoded = [e for e in (encode_record(r, tokenizer, max_len, decision_weight) for r in records) if e is not None]
    tokens = sum(len(e["input_ids"]) for e in encoded)
    print(f"{path.name}: {len(encoded)} records, {len(records) - len(encoded)} skipped as longer than {max_len} "
          f"tokens, {tokens} tokens", flush=True)
    return encoded


def checkpoint_step(folder: Path) -> int | None:
    state = folder / "state.json"
    return TrainState.from_json(state.read_text(encoding="utf-8")).step if state.exists() else None


def labelled_count(enc: dict) -> int:
    return sum(label != -100 for label in enc["labels"][1:])


def labelled_weight(enc: dict) -> float:
    """What the training loss divides by: the weights of the labelled tokens (their count without weights)."""
    if "weights" not in enc:
        return labelled_count(enc)
    return sum(w for label, w in zip(enc["labels"][1:], enc["weights"][1:]) if label != -100)


def teacher_targets(teachers: list[str], train: list[dict], tokenizer, temp: float, top_k: int) -> list[tuple]:
    """One (ids, probs) per train record, on the CPU: every teacher run once in eval mode, then freed."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = tokenizer.get_vocab()
    for name in teachers:
        if AutoTokenizer.from_pretrained(name).get_vocab() != vocab:
            raise SystemExit(f"teacher {name} has another vocabulary than the student: logits cannot be compared")
    per_teacher = []
    for name in teachers:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device).eval()
        rows, t0 = [], time.time()
        with torch.no_grad():  # not inference_mode: its tensors cannot enter the student's backward pass
            for i, enc in enumerate(train):
                ids = torch.tensor([enc["input_ids"]], device=device)
                pos = (torch.tensor(enc["labels"], device=device)[1:] != -100).nonzero(as_tuple=True)[0]
                with torch.autocast(device, dtype=torch.bfloat16):
                    logits = model(input_ids=ids, logits_to_keep=pos).logits[0]
                rows.append(tuple(t.cpu() for t in soft_targets(logits, temp, top_k)))
                if (i + 1) % 200 == 0:
                    print(f"teacher {name}: {i + 1}/{len(train)} records, {time.time() - t0:.0f} s", flush=True)
        per_teacher.append(rows)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return [merge_targets([rows[i] for rows in per_teacher], top_k) for i in range(len(train))]


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
    ap.add_argument("--lora-targets", default=",".join(LORA_TARGETS), help="comma-separated module names for LoRA")
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--eval-every", type=int, default=50, help="optimizer steps between eval losses")
    ap.add_argument("--save-minutes", type=float, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decision-weight", type=float, default=1.0,
                    help="loss weight of the tokens that decide a turn (its first token, tool names, argument keys); "
                         "1 = every token counts the same. The logged loss and the eval loss stay unweighted")
    ap.add_argument("--kd-teachers", default="", help="comma-separated teacher models with the student's vocabulary")
    ap.add_argument("--kd-alpha", type=float, default=0.5, help="share of the KL term in the loss")
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--kd-top-k", type=int, default=20, help="teacher probabilities kept per labelled token")
    ap.add_argument("--kd-report-only", action="store_true", help="write kd_report.json and stop before training")
    ap.add_argument("--revision", default=None, help="commit sha of --model (and of its tokenizer)")
    ap.add_argument("--pairs", default=None, help="preference mode: <records-dir>/pairs/<NAME>.jsonl (lifemate.sft.pairs)")
    ap.add_argument("--eval-pairs", default=None, help="held-out pairs for eval_pref_acc, never trained on")
    ap.add_argument("--beta", type=float, default=0.1, help="--pairs: DPO beta")
    ap.add_argument("--sft-weight", type=float, default=1.0, help="--pairs: weight of the NLL of the chosen step")
    ap.add_argument("--sft-share", type=float, default=0.2, help="--pairs: share of plain --records items in the mix")
    args = ap.parse_args(argv)
    if args.kd_report_only and not args.kd_teachers:
        ap.error("--kd-report-only needs --kd-teachers")
    if args.pairs and (args.kd_teachers or args.decision_weight != 1.0):
        ap.error("--pairs cannot be combined with --kd-teachers or --decision-weight")
    if args.pairs and not 0 <= args.sft_share < 1:
        ap.error("--sft-share is a share in [0, 1)")

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

    def pairs_path(name: str) -> Path:
        return args.records_dir / "pairs" / f"{name}.jsonl"

    def file_sha(path: Path | None) -> str | None:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path is not None else None

    pairs_sha = file_sha(pairs_path(args.pairs)) if args.pairs else None
    eval_pairs_sha = file_sha(pairs_path(args.eval_pairs)) if args.eval_pairs else None
    if args.resume:
        if not ckpt.exists():
            raise SystemExit(f"--resume: no checkpoint at {ckpt}" + (f" nor in {args.hf_repo}" if api else ""))
        saved = json.loads((ckpt / "run_config.json").read_text(encoding="utf-8"))
        if saved.get("pairs_sha") != pairs_sha:
            raise SystemExit(f"--resume: pairs/{args.pairs}.jsonl differs from the one the checkpoint was trained on")
        if saved.get("records_sha") != records_sha:  # another file would shift the data order
            raise SystemExit(f"--resume: train_records/{args.records}.jsonl differs from the one the checkpoint was "
                             f"trained on (sha {records_sha} vs {saved.get('records_sha')}); rebuild it with the same "
                             "code and options")
        changed = [f"--{k.replace('_', '-')} {saved.get(k, ap.get_default(k))} -> {getattr(args, k)}" for k in RUN_KEYS
                   if saved.get(k, ap.get_default(k)) != getattr(args, k)]
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
    (out / "run_config.json").write_text(json.dumps({**config, "records_sha": records_sha, "pairs_sha": pairs_sha}, indent=1), encoding="utf-8")

    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    # --pairs reads the records itself: it needs their sample ids to keep held-out pair samples out of the SFT share
    train = [] if args.pairs else load_encoded(args.records, tokenizer, args.max_len, args.records_dir, args.decision_weight)
    eval_pairs: list[dict] = []
    if args.pairs:  # preference mode: the pairs, then a seeded share of plain records (behaviour away from decisions)
        def load_pairs(name: str) -> list[dict]:
            with open(pairs_path(name), encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            encoded = [e for e in (encode_pair(r, tokenizer, args.max_len) for r in rows) if e is not None]
            print(f"pairs/{name}.jsonl: {len(encoded)} pairs, {len(rows) - len(encoded)} skipped as longer than "
                  f"{args.max_len} tokens", flush=True)
            return encoded

        pairs = load_pairs(args.pairs)
        held_ids: set[str] = set()
        if args.eval_pairs:  # a held-out sample trained on through its gold record would inflate eval_pref_acc
            with open(pairs_path(args.eval_pairs), encoding="utf-8") as f:
                held_ids = {json.loads(line)["id"] for line in f if line.strip()}
        with open(args.records_dir / f"{args.records}.jsonl", encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]
        kept = [r for r in raw if r["id"] not in held_ids]
        pool = [e for e in (encode_record(r, tokenizer, args.max_len) for r in kept) if e is not None]
        n_sft = min(len(pool), round(args.sft_share / (1 - args.sft_share) * len(pairs)))
        picked = sorted(random.Random(args.seed).sample(range(len(pool)), n_sft))
        train = pairs + [pool[i] for i in picked]
        eval_pairs = load_pairs(args.eval_pairs) if args.eval_pairs else []
        print(f"preference mix: {len(pairs)} pairs + {n_sft} records ({len(raw) - len(kept)} records of "
              f"{len(held_ids)} held-out samples left out)", flush=True)
        mix_sha = hashlib.sha256(json.dumps([sorted(held_ids), picked]).encode()).hexdigest()[:16]
        if args.resume:  # the same records must come back, or the data order changes mid-run
            saved_mix = json.loads((ckpt / "run_config.json").read_text(encoding="utf-8")).get("mix_sha")
            if saved_mix != mix_sha:
                raise SystemExit(f"--resume: the preference mix differs from the checkpoint's (mix_sha {mix_sha} vs "
                                 f"{saved_mix}); a checkpoint made before held-out samples were left out resumes only "
                                 "with the code it was trained with")
        run_config = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
        (out / "run_config.json").write_text(json.dumps({**run_config, "mix_sha": mix_sha}, indent=1), encoding="utf-8")
    evals = load_encoded(args.eval_records, tokenizer, args.max_len, args.records_dir) if args.eval_records else []
    targets = None
    if args.kd_teachers:
        cache = out / "kd_targets.pt"
        key = {"records_sha": records_sha, "teachers": args.kd_teachers, "temp": args.kd_temp, "top_k": args.kd_top_k,
               "max_len": args.max_len}
        saved_targets = torch.load(cache, weights_only=False) if cache.exists() else None
        if saved_targets is not None and saved_targets["key"] == key:
            targets = saved_targets["targets"]
        else:
            targets = teacher_targets(args.kd_teachers.split(","), train, tokenizer, args.kd_temp, args.kd_top_k)
            torch.save({"key": key, "targets": targets}, cache)
        gold = [torch.tensor(e["labels"][1:])[torch.tensor(e["labels"][1:]) != -100] for e in train]
        report = dict(target_report(targets, gold), teachers=args.kd_teachers, temp=args.kd_temp, top_k=args.kd_top_k)
        (out / "kd_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"soft targets: {report}", flush=True)
        if args.kd_report_only:
            return out / "kd_report.json"
        seed_everything(args.seed)  # the student starts as it would without teachers

    device = "cuda" if torch.cuda.is_available() else "cpu"  # cpu: local smoke tests with a tiny model only
    model = load_causal_lm(args.model, torch.bfloat16, args.revision).to(device)  # refuses a checkpoint whose weights would stay random
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if args.resume:
        model = PeftModel.from_pretrained(model, str(ckpt), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                                 target_modules=args.lora_targets.split(","), task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    if args.pairs:  # the reference is the same network with the adapter disabled: computed once, cached on disk
        cache = out / "ref_logps.pt"
        key = {"pairs_sha": pairs_sha, "eval_pairs_sha": eval_pairs_sha, "model": args.model, "revision": args.revision,
               "max_len": args.max_len}
        saved_ref = torch.load(cache, weights_only=False) if cache.exists() else None
        if saved_ref is None or saved_ref["key"] != key:
            model.eval()
            with torch.no_grad(), model.disable_adapter():
                saved_ref = {"key": key, **{name: [(step_logps(model, e["prefix"], e["chosen"], device)[0].item(),
                                                    step_logps(model, e["prefix"], e["rejected"], device)[0].item())
                                                   for e in rows]
                                            for name, rows in (("train", [e for e in train if "prefix" in e]),
                                                               ("eval", eval_pairs))}}
            model.train()
            torch.save(saved_ref, cache)
        for e, ref in zip([e for e in train if "prefix" in e], saved_ref["train"]):
            e["ref"] = ref
        for e, ref in zip(eval_pairs, saved_ref["eval"]):
            e["ref"] = ref
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

    def record_logits(enc: dict, weighted: bool = False):
        """(logits, gold ids, weights) at the labelled tokens only. weights is None unless weighted=True and the record
        has them (--decision-weight): training minimises the weighted sum, everything logged stays plain."""
        ids = torch.tensor([enc["input_ids"]], device=device)
        labels = torch.tensor(enc["labels"], device=device)
        pos = (labels[1:] != -100).nonzero(as_tuple=True)[0]
        with torch.autocast(device, dtype=torch.bfloat16):
            logits = model(input_ids=ids, logits_to_keep=pos).logits[0]
        w = torch.tensor(enc["weights"], device=device)[pos + 1] if weighted and "weights" in enc else None
        return logits.float(), labels[pos + 1], w

    def record_loss(enc: dict):
        """(sum of cross-entropy over the labelled tokens, their count): the plain loss, used for eval."""
        logits, gold_ids, _ = record_logits(enc)
        return F.cross_entropy(logits, gold_ids, reduction="sum"), len(gold_ids)

    def eval_loss() -> float:
        model.eval()
        total_loss, count = 0.0, 0
        with torch.no_grad():
            for enc in evals:
                loss, k = record_loss(enc)
                total_loss, count = total_loss + loss.item(), count + k
        model.train()
        return total_loss / max(count, 1)

    def pair_margin(e: dict):
        """(margin, mean NLL of the chosen step, policy log-probs) of one encoded pair against its reference."""
        pc, nll = step_logps(model, e["prefix"], e["chosen"], device)
        pr, _ = step_logps(model, e["prefix"], e["rejected"], device)
        rc, rr = (torch.tensor(v, device=device) for v in e["ref"])
        return rpo_loss(pc, pr, rc, rr, nll, args.beta, args.sft_weight), nll

    def eval_pref_acc() -> float:
        """Share of the held-out pairs whose margin (policy vs reference) prefers the gold step."""
        model.eval()
        with torch.no_grad():
            wins = sum(pair_margin(e)[0][1].item() > 0 for e in eval_pairs)
        model.train()
        return wins / max(len(eval_pairs), 1)

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
    step_loss, step_kd, step_count, step_tokens = 0.0, 0.0, 0, 0
    pref = {"pairs": 0, "dpo": 0.0, "nll": 0.0, "margin": 0.0, "wins": 0}
    model.train()
    if state.step < stop_at:
        for batch in batches(n, state, args.micro_bs, args.epochs):
            if args.pairs:  # one loss per item, averaged over the accumulated micro-batches
                for i in batch:
                    item = train[i]
                    if "prefix" in item:
                        (loss, margin), nll = pair_margin(item)
                        pref["pairs"] += 1
                        pref["dpo"] += loss.item() - args.sft_weight * nll.item()
                        pref["nll"] += nll.item()
                        pref["margin"] += margin.item()
                        pref["wins"] += margin.item() > 0
                        tokens = len(item["prefix"]) + len(item["chosen"]) + len(item["rejected"])
                    else:
                        logits, gold_ids, _ = record_logits(item)
                        loss = F.cross_entropy(logits, gold_ids)
                        tokens = len(item["input_ids"])
                    (loss / len(batch) / args.grad_accum).backward()
                    step_loss, step_count, step_tokens = step_loss + loss.item(), step_count + 1, step_tokens + tokens
            count = 0 if args.pairs else sum(labelled_weight(train[i]) for i in batch)
            for i in ([] if args.pairs else batch):
                logits, gold_ids, w = record_logits(train[i], weighted=True)
                if w is None:
                    ce = ce_weighted = F.cross_entropy(logits, gold_ids, reduction="sum")
                else:  # --decision-weight: the weighted sum trains, the plain sum is what gets logged
                    each = F.cross_entropy(logits, gold_ids, reduction="none")
                    ce, ce_weighted = each.sum(), (each * w).sum()
                total_loss = ce_weighted
                if targets is not None:  # --kd-teachers: the KL term is summed per token and divided like the CE
                    top_ids, top_probs = (t.to(device) for t in targets[i])
                    kd = kd_sum(logits, top_ids, top_probs, args.kd_temp)
                    total_loss = (1 - args.kd_alpha) * ce_weighted + args.kd_alpha * kd
                    step_kd += kd.item()
                (total_loss / max(count, 1) / args.grad_accum).backward()
                step_loss, step_count, step_tokens = step_loss + ce.item(), step_count + len(gold_ids), step_tokens + len(train[i]["input_ids"])
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
            if targets is not None:
                row["kd"] = round(step_kd / max(step_count, 1), 5)
            if pref["pairs"]:
                k = pref["pairs"]
                row.update(pairs=k, dpo=round(pref["dpo"] / k, 5), nll=round(pref["nll"] / k, 5),
                           margin=round(pref["margin"] / k, 5), pref_acc=round(pref["wins"] / k, 4))
                pref = {"pairs": 0, "dpo": 0.0, "nll": 0.0, "margin": 0.0, "wins": 0}
            if (evals or eval_pairs) and (state.step % args.eval_every == 0 or state.step == stop_at):
                if evals:
                    row["eval_loss"] = round(eval_loss(), 5)
                if eval_pairs:
                    row["eval_pref_acc"] = round(eval_pref_acc(), 4)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            secs = now - step_start
            eta = (stop_at - state.step) * (now - t0) / (state.step - start_step)
            print(f"step {state.step}/{stop_at} loss {row['loss']:.4f}" + (f" eval {row['eval_loss']:.4f}" if "eval_loss" in row else "")
                  + f" lr {lr:.2e} gnorm {grad_norm:.3f} {step_tokens / secs:.0f} tok/s, eta {eta / 60:.0f} min", flush=True)
            step_loss, step_kd, step_count, step_tokens, step_start = 0.0, 0.0, 0, 0, time.time()
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
