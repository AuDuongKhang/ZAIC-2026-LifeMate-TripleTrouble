"""Run an agent on a split and write runs/<run>/{submission.jsonl, trace.jsonl, meta.json}.

    python -m lifemate.infer --split train --agent oracle --asr gold --limit 50
    python -m lifemate.infer --split public_test --agent <name> --asr <name>
"""
from __future__ import annotations

import argparse
import inspect
import json
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType

from .agents import AGENTS, build_agent
from .asr import BACKENDS, build_asr
from .data import REPO_ROOT, audio_path, default_data_root, import_zaic_tools, installed_tool_names, load_catalog, load_samples
from .presets import choose
from .splits import SUBSETS, load_splits
from .task import Task
from .utils import git_info, seed_everything

TIME_LIMIT_S = 30.0  # per sample on the scoring A30, ASR included


def build_task(sample: dict, asr, catalog: dict[str, dict], data_root: Path, split: str,
               started_at: float | None = None) -> Task:
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
        started_at=started_at,
    )


def with_split(agent_cls: type, agent_kw: dict, split: str, data_root: Path) -> dict:
    """agent_kw plus split= and data_root= for agents that take them (the few-shot bank depends on both), unless set."""
    params = inspect.signature(agent_cls).parameters
    extra = {"split": split, "data_root": data_root}
    return {**{k: v for k, v in extra.items() if k in params}, **agent_kw}


@dataclass
class Pipeline:
    """What every sample needs: the split's zaic_tools and catalog, and the loaded ASR and agent."""
    split: str
    data_root: Path
    zaic_tools: ModuleType
    catalog: dict[str, dict]
    asr: object
    agent: object


def load_pipeline(split: str, data_root: Path, agent_name: str, agent_kw: dict, asr_name: str,
                  asr_kw: dict) -> Pipeline:
    """Import the split's zaic_tools, load its catalog and build the ASR, then the agent. Does not seed."""
    zaic_tools = import_zaic_tools(data_root, split)
    catalog = load_catalog(data_root, split)
    agent_kw = with_split(AGENTS[agent_name], agent_kw, split, data_root)
    asr = build_asr(asr_name, **asr_kw)
    agent = build_agent(agent_name, **agent_kw)
    return Pipeline(split, data_root, zaic_tools, catalog, asr, agent)


def run_sample(p: Pipeline, sample: dict) -> tuple[dict, dict]:
    """Run one sample in its own ToolEnv: (submission line, trace record). Never raises: an error goes to the trace.

    `time` is the sample's milliseconds from before its ToolEnv and audio to the end of the agent, as the organiser
    guide measures it."""
    t0 = time.perf_counter()
    env = p.zaic_tools.ToolEnv(p.split, sample["id"], root=p.data_root)  # the sample's one and only env
    task, error = None, None
    p.agent.last_trace = None
    try:
        task = build_task(sample, p.asr, p.catalog, p.data_root, p.split, started_at=t0)
        p.agent.run(task, env)
    except Exception:
        error = traceback.format_exc()
    elapsed = time.perf_counter() - t0
    line = {"id": sample["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash(),
            "time": int(elapsed * 1000)}
    trace = {"id": sample["id"], "request": task.request if task else None, "calls": env.log,
             "agent": p.agent.last_trace, "seconds": round(elapsed, 3), "error": error}
    return line, trace


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--data-root", type=Path, default=default_data_root())
    ap.add_argument("--agent", default=None, choices=sorted(AGENTS), help="default: empty")
    ap.add_argument("--asr", default=None, choices=sorted(BACKENDS), help="default: gold")
    ap.add_argument("--agent-preset", default=None, metavar="NAME", help="named agent setup from configs/presets.json")
    ap.add_argument("--asr-preset", default=None, metavar="NAME",
                    help="named ASR setup from configs/presets.json, or cached:<cache>")
    ap.add_argument("--limit", type=int, default=None, help="first N samples only")
    ap.add_argument("--ids", nargs="*", default=None, help="only these sample ids")
    ap.add_argument("--subset", choices=SUBSETS, default=None, help="train only: a part of configs/splits.json")
    ap.add_argument("--out", type=Path, default=None, help="run folder (default runs/<time>_<split>_<agent>_<asr>)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-field", action="store_true",
                    help="keep each sample's `time` (ms) in submission.jsonl, as the Docker guide asks; off by default "
                         "so that two identical runs give byte-identical files")
    ap.add_argument("--agent-kw", action="append", metavar="KEY=VALUE", help="agent constructor option (repeatable)")
    ap.add_argument("--asr-kw", action="append", metavar="KEY=VALUE", help="ASR constructor option (repeatable)")
    args = ap.parse_args(argv)
    try:
        args.agent_name, agent_kw = choose("agent", args.agent, args.agent_preset, args.agent_kw, "empty")
        args.asr_name, asr_kw = choose("asr", args.asr, args.asr_preset, args.asr_kw, "gold")
    except (ValueError, KeyError) as e:
        ap.error(str(e.args[0]))

    samples = load_samples(args.data_root, args.split)
    if args.asr_name == "gold" and samples and "audio_text" not in samples[0]:
        ap.error(f"--asr gold needs transcripts, which only train has; use a real ASR or --asr none on {args.split}")
    if args.subset:
        if args.split != "train":
            ap.error("--subset needs --split train")
        wanted = set(load_splits()[args.subset])
        samples = [s for s in samples if s["id"] in wanted]
    if args.ids:
        wanted = set(args.ids)
        samples = [s for s in samples if s["id"] in wanted]
    if args.limit:
        samples = samples[: args.limit]
    t_load = time.perf_counter()
    p = load_pipeline(args.split, args.data_root, args.agent_name, agent_kw, args.asr_name, asr_kw)
    agent = p.agent
    load_s = time.perf_counter() - t_load
    # the run record keeps split= but not data_root= (a machine path, needed only to build the agent)
    agent_kw = {k: v for k, v in with_split(AGENTS[args.agent_name], agent_kw, args.split, args.data_root).items()
                if k != "data_root"}
    seed_everything(args.seed)

    out = args.out or REPO_ROOT / "runs" / (
        f"{datetime.now():%Y%m%d-%H%M%S}_{args.split}_{(args.agent_preset or args.agent_name).replace(':', '-')}"
        f"_{(args.asr_preset or args.asr_name).replace(':', '-')}")
    out.mkdir(parents=True, exist_ok=True)
    times, errors = [], 0
    with open(out / "submission.jsonl", "w", encoding="utf-8") as fsub, open(out / "trace.jsonl", "w", encoding="utf-8") as ftr:
        for i, sample in enumerate(samples, 1):
            line, trace = run_sample(p, sample)
            elapsed, error = trace["seconds"], trace["error"]
            errors += bool(error)
            times.append(elapsed)
            if not args.time_field:
                line.pop("time")
            fsub.write(json.dumps(line, ensure_ascii=False) + "\n")
            ftr.write(json.dumps(trace, ensure_ascii=False) + "\n")
            if error or elapsed > TIME_LIMIT_S or i % 10 == 0 or i == len(samples):
                flag = " ERROR" if error else (" OVER 30s" if elapsed > TIME_LIMIT_S else "")
                print(f"[{i}/{len(samples)}] {sample['id']} {elapsed:.2f}s (mean {sum(times) / len(times):.2f}s){flag}",
                      flush=True)

    meta = {
        "split": args.split, "agent": args.agent_name, "asr": args.asr_name,
        "agent_preset": args.agent_preset, "asr_preset": args.asr_preset, "agent_kw": agent_kw, "asr_kw": asr_kw,
        "agent_info": getattr(agent, "info", {}),
        "seed": args.seed, "limit": args.limit, "subset": args.subset,
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
