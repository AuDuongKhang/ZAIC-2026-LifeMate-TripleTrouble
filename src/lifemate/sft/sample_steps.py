"""Sample the agent model at every agent turn of masked pool records, for preference pairs (lifemate.sft.pairs).

    python -m lifemate.sft.sample_steps --records pref-r1-gold --records pref-r1-asr --name r1-t0.8 \
        --model TripleTrouble/lifemate-sft-q1c --revision <sha> --k 4 --temperature 0.8 --top-p 0.95 --hf-repo <dataset>

Development only: temperature sampling makes training data and is never used at inference; agents/, infer.py and
docker_entry.py never import this module. Only function-masked records (source "masked", written by lifemate.sft.records
--mask) are sampled, so the model sees the tools under names it never trained on; every record must belong to the pool.

A decision point is an assistant turn after the last user message. Its prefix (the record's messages before it, with
the generation prompt) gets one greedy completion and K sampled ones, seeded by point_seed, and one row in
<records-dir>/steps/<name>.jsonl. The file is append-only: a rerun skips the points it already holds (after pulling the
HF copy with --hf-repo), so a dead session resumes where it stopped. A prefix longer than --max-len is skipped and
counted, never cut.
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

from ..splits import load_splits
from .records import RECORDS_DIR
from .thoughts import _pull_hf_copy, _trim_torn, _upload
from .tokenize import MAX_LEN, render


def decision_points(record: dict) -> list[int]:
    """Indices of the assistant messages after the last user message (the turns the agent generates)."""
    messages = record["messages"]
    last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
    return [i for i in range(last_user + 1, len(messages)) if messages[i]["role"] == "assistant"]


def point_seed(seed: int, records: str, record: int, turn: int) -> int:
    return zlib.crc32(f"{seed}:{records}:{record}:{turn}".encode())


def load_model(model: str, revision: str):
    """(model, tokenizer) on the GPU, the model at its pinned revision."""
    import torch
    from transformers import AutoTokenizer

    from ..agents.llm import load_causal_lm

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    lm = load_causal_lm(model, torch.bfloat16, revision).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return lm, tokenizer


def sample_texts(model, tokenizer, messages: list[dict], tools: list[dict], k: int, temperature: float, top_p: float,
                 seed: int, max_new_tokens: int = 384) -> tuple[str, list[str]]:
    """(greedy completion, k sampled completions) of the prompt the agent would send (HFChatLLM._encode), cut as
    HFChatLLM.generate cuts its output."""
    import torch

    from ..agents.llm import finish_text, stop_token_ids

    inputs = tokenizer.apply_chat_template(messages, tools=tools or None, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True).to(model.device)
    stop_strings = {t for t in (tokenizer.eos_token, tokenizer.pad_token, "<|im_end|>", "<|endoftext|>") if t}
    eos = stop_token_ids(tokenizer, model.generation_config)
    prompt_len = inputs["input_ids"].shape[1]
    open_think = tokenizer.decode(inputs["input_ids"][0, -8:], skip_special_tokens=False).endswith("<think>\n")

    def texts(out) -> list[str]:
        return [finish_text(tokenizer.decode(row[prompt_len:], skip_special_tokens=False), stop_strings, open_think)
                for row in out]

    with torch.inference_mode():
        greedy = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                                top_k=None, eos_token_id=eos)
        torch.manual_seed(seed)
        sampled = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                                 top_p=top_p, top_k=0, num_return_sequences=k, eos_token_id=eos)
    return texts(greedy)[0], texts(sampled)


def check_rows(rows: list[dict], records: dict[tuple[str, int], dict], args) -> None:
    """Stop before a resumed run reuses rows that another model, revision, sampling setting or records file made:
    a skipped point must be exactly the point this run would sample."""
    for r in rows:
        made = (r["model"], r["revision"], r["temperature"], r["top_p"], len(r["samples"]))
        wanted = (args.model, args.revision, args.temperature, args.top_p, args.k)
        if made != wanted or r["seed"] != point_seed(args.seed, r["records"], r["record"], r["turn"]):
            raise SystemExit(f"steps file rows were made with other settings {made} than this run {wanted}: use another "
                             "--name, or the same settings")
        rec = records.get((r["records"], r["record"]))
        if rec is not None and rec["id"] != r["id"]:
            raise SystemExit(f"{r['records']} record {r['record']} is {rec['id']} now but the steps file sampled {r['id']}: "
                             "the records file was rebuilt; use another --name")


def _jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", action="append", required=True, help="train_records/<NAME>.jsonl (masked records used)")
    ap.add_argument("--name", required=True, help="train_records/steps/<NAME>.jsonl")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True, help="model commit sha, written in every row")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="records of the first N distinct sample ids only")
    ap.add_argument("--max-len", type=int, default=MAX_LEN, help="prefixes longer than this (tokens) are skipped")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--hf-repo", default=None, help="private HF dataset holding steps/<name>.jsonl")
    ap.add_argument("--upload-every", type=int, default=200, help="points between uploads")
    ap.add_argument("--progress-every", type=int, default=10, help="points between progress lines")
    ap.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    args = ap.parse_args(argv)

    pool = set(load_splits()["pool"])
    todo: list[tuple[str, int, dict]] = []
    for name in args.records:
        for i, rec in enumerate(_jsonl(args.records_dir / f"{name}.jsonl")):
            if rec.get("source") != "masked":
                continue
            if rec["id"] not in pool:
                raise SystemExit(f"{name}: record {i} ({rec['id']}) is not a pool sample; only the pool may be sampled")
            todo.append((name, i, rec))
    if args.limit is not None:
        keep = list(dict.fromkeys(rec["id"] for _, _, rec in todo))[: args.limit]
        todo = [t for t in todo if t[2]["id"] in set(keep)]

    out = args.records_dir / "steps" / f"{args.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.hf_repo and not out.exists():
        _pull_hf_copy(args.hf_repo, [out], folder="steps")
    _trim_torn(out)
    existing = _jsonl(out) if out.exists() else []
    check_rows(existing, {(name, i): rec for name, i, rec in todo}, args)
    done = {(r["records"], r["record"], r["turn"]) for r in existing}

    points = [(name, i, turn) for name, i, rec in todo for turn in decision_points(rec)]
    already = sum(p in done for p in points)
    print(f"{len(points)} points to sample ({already} already done) from {len(todo)} records -> {out}", flush=True)
    model, tokenizer = load_model(args.model, args.revision)
    written = skipped = since_upload = 0
    current, t0 = already, time.time()

    def report() -> None:  # one line the notebook's progress bar reads
        new = current - already
        secs = (time.time() - t0) / new if new else 0.0
        print(f"progress {current}/{len(points)} points ({100 * current / max(len(points), 1):.0f}%), "
              f"{secs:.1f} s/point, eta {secs * (len(points) - current) / 60:.0f} min", flush=True)

    for name, i, rec in todo:
        for turn in decision_points(rec):
            if (name, i, turn) in done:
                continue
            current += 1
            report_due = (current - already) % args.progress_every == 0 or current == len(points)
            prefix = rec["messages"][:turn]
            if len(render(tokenizer, prefix, rec["tools"], add_generation_prompt=True)) > args.max_len:
                skipped += 1
                if report_due:
                    report()
                continue
            seed = point_seed(args.seed, name, i, turn)
            greedy, samples = sample_texts(model, tokenizer, prefix, rec["tools"], args.k, args.temperature, args.top_p,
                                           seed, args.max_new_tokens)
            row = {"records": name, "record": i, "id": rec["id"], "turn": turn, "greedy": greedy, "samples": samples,
                   "temperature": args.temperature, "top_p": args.top_p, "seed": seed, "model": args.model,
                   "revision": args.revision}
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            since_upload += 1
            if args.hf_repo and since_upload >= args.upload_every:
                _upload(args.hf_repo, [out], folder="steps")
                since_upload = 0
            if report_due:
                report()
    if args.hf_repo and since_upload:
        _upload(args.hf_repo, [out], folder="steps")
    print(f"{written} points sampled, {len(done)} already in {out}; {skipped} points over --max-len skipped", flush=True)
    return out


if __name__ == "__main__":
    main()
