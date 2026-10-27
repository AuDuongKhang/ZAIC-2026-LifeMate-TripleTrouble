"""Run an agent on a split and write runs/<run>/{submission.jsonl, trace.jsonl, meta.json}.

    python -m lifemate.infer --split train --agent oracle --asr gold --limit 50
    python -m lifemate.infer --split public_test --agent <name> --asr <name>
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

from .agents import AGENTS, build_agent
from .asr import BACKENDS, build_asr
from .data import REPO_ROOT, audio_path, default_data_root, import_zaic_tools, installed_tool_names, load_catalog, load_samples
from .task import Task
from .utils import git_info, parse_kw, seed_everything

TIME_LIMIT_S = 30.0  # per sample on the scoring A30, ASR included


def build_task(sample: dict, asr, catalog: dict[str, dict], data_root: Path, split: str) -> Task:
    history = []
    for turn in sample["history"]:
        turn = {k: v for k, v in turn.items() if k != "audio_text"}  # agents only see ASR output
        if turn["role"] == "user" and turn.get("audio"):
            turn["text"] = asr.transcribe(audio_path(data_root, split, turn["audio"]), sample)
        history.append(turn)
    return Task(
        id=sample["id"],
        request=asr.transcribe(audio_path(data_root, split, sample["audio"]), sample),
        history=history,
        context=sample["context"],
        tools=[catalog[name] for name in installed_tool_names(sample) if name in catalog],
        label=sample.get("label"),
    )


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--data-root", type=Path, default=default_data_root())
    ap.add_argument("--agent", default="empty", choices=sorted(AGENTS))
    ap.add_argument("--asr", default="gold", choices=sorted(BACKENDS))
    ap.add_argument("--limit", type=int, default=None, help="first N samples only")
    ap.add_argument("--ids", nargs="*", default=None, help="only these sample ids")
    ap.add_argument("--out", type=Path, default=None, help="run folder (default runs/<time>_<split>_<agent>_<asr>)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agent-kw", action="append", metavar="KEY=VALUE", help="agent constructor option (repeatable)")
    ap.add_argument("--asr-kw", action="append", metavar="KEY=VALUE", help="ASR constructor option (repeatable)")
    args = ap.parse_args(argv)

    zaic_tools = import_zaic_tools(args.data_root, args.split)
    samples = load_samples(args.data_root, args.split)
    if args.asr == "gold" and samples and "audio_text" not in samples[0]:
        ap.error(f"--asr gold needs transcripts, which only train has; use a real ASR or --asr none on {args.split}")
    if args.ids:
        wanted = set(args.ids)
        samples = [s for s in samples if s["id"] in wanted]
    if args.limit:
        samples = samples[: args.limit]
    catalog = load_catalog(args.data_root, args.split)

    t_load = time.perf_counter()
    asr_kw, agent_kw = parse_kw(args.asr_kw), parse_kw(args.agent_kw)
    asr = build_asr(args.asr, **asr_kw)
    agent = build_agent(args.agent, **agent_kw)
    load_s = time.perf_counter() - t_load
    seed_everything(args.seed)

    out = args.out or REPO_ROOT / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}_{args.split}_{args.agent}_{args.asr}"
    out.mkdir(parents=True, exist_ok=True)
    times, errors = [], 0
    with open(out / "submission.jsonl", "w", encoding="utf-8") as fsub, open(out / "trace.jsonl", "w", encoding="utf-8") as ftr:
        for i, sample in enumerate(samples, 1):
            t0 = time.perf_counter()
            env = zaic_tools.ToolEnv(args.split, sample["id"], root=args.data_root)  # the sample's one and only env
            task, error = None, None
            try:
                task = build_task(sample, asr, catalog, args.data_root, args.split)
                agent.run(task, env)
            except Exception:
                error = traceback.format_exc()
                errors += 1
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            line = {"id": sample["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()}
            fsub.write(json.dumps(line, ensure_ascii=False) + "\n")
            ftr.write(json.dumps({"id": sample["id"], "request": task.request if task else None, "calls": env.log,
                                  "seconds": round(elapsed, 3), "error": error}, ensure_ascii=False) + "\n")
            if error or elapsed > TIME_LIMIT_S or i % 50 == 0 or i == len(samples):
                flag = " ERROR" if error else (" OVER 30s" if elapsed > TIME_LIMIT_S else "")
                print(f"[{i}/{len(samples)}] {sample['id']} {elapsed:.2f}s{flag}", flush=True)

    meta = {
        "split": args.split, "agent": args.agent, "asr": args.asr, "agent_kw": agent_kw, "asr_kw": asr_kw,
        "seed": args.seed, "limit": args.limit,
        "n": len(samples), "errors": errors, "load_seconds": round(load_s, 2),
        "mean_seconds": round(sum(times) / max(len(times), 1), 3), "max_seconds": round(max(times, default=0), 3),
        "over_time_limit": sum(t > TIME_LIMIT_S for t in times),
        "git": git_info(), "finished": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {out}  ({meta['n']} samples, {errors} errors, mean {meta['mean_seconds']}s, max {meta['max_seconds']}s)")
    return out


if __name__ == "__main__":
    main()
