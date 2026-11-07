"""Synthetic SFT scenarios: what a scenario writer is shown (probe), and the proof that its calls really run (verify).

A synthetic sample points at a real train environment (`env_id`), so the history's applied calls and the app data
are the same ones the scenario was written against. This module only checks and packages scenarios; it never writes
them, and nothing here calls an API or any paid service (team rule). Scenarios come from a free source, then pass
`accept`, and `lifemate.sft.records --extra-samples` renders the survivors.

Command line, no network:

    python -m lifemate.sft.synth pack   --name p1 --offset 0 --limit 40 --per-prompt 4   # -> p1/prompts/prompt_00.txt ...
    # paste each prompt into a free chat model, save its whole answer as p1/scenarios/prompt_00.json (fences are fine)
    python -m lifemate.sft.synth ingest --name p1                          # p1/scenarios/* -> p1.jsonl + report

or let a local open model write the answers (development only, on a GPU; never at inference):

    python -m lifemate.sft.synth answer --name p1 --teacher Qwen/Qwen3.5-27B --revision <sha>

An answer maps a base sample id to [{"request", "kind": act|ask|empty, "calls": [{"name", "arguments"}]}]. The prompts
hold train data (history, app contents) but not the base sample's own request or gold calls.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ..agents.refs import scalar_strings, unseen_refs
from ..data import REPO_ROOT, SYS_TOOL, default_data_root, import_zaic_tools, installed_tool_names, load_catalog, load_samples


def holdout_apps() -> set[str]:
    """The apps of the holdout split: the gate would no longer test unseen apps if generated data touched them."""
    from ..splits import load_splits

    return set(load_splits()["holdout_apps"])


def uses_holdout_app(calls: list[dict], holdout: set[str] | None = None) -> bool:
    """Any call to a tool of a holdout app, or an ask about one."""
    holdout = holdout_apps() if holdout is None else holdout
    for call in calls:
        name = call["arguments"].get("tool_name", "") if call["name"] == SYS_TOOL else call["name"]
        if name.split(".")[0] in holdout:
            return True
    return False


def env_id(sample: dict) -> str:
    """The train environment a sample runs in: its own id, or the real sample a synthetic one was built on."""
    return sample.get("env_id", sample["id"])


def _new_env(zaic_tools, sample: dict, data_root: Path):
    return zaic_tools.ToolEnv("train", env_id(sample), root=data_root)


def probe(sample: dict, catalog: dict[str, dict], zaic_tools, data_root: Path) -> list[dict]:
    """Every installed read tool that needs no argument, called with {} in a fresh env: context for the teacher
    (what the user's apps hold), never part of a synthetic label."""
    env = _new_env(zaic_tools, sample, data_root)
    out, holdout = [], holdout_apps()
    for name in installed_tool_names(sample):
        spec = catalog.get(name)
        if spec is None or spec.get("kind") != "read" or spec["parameters"].get("required") or name.split(".")[0] in holdout:
            continue
        result = env.call(name, {})
        if result.get("ok"):  # a one-of read that needs an argument is rejected by the schema: not a probe
            out.append({"name": name, "arguments": {}, "result": result})
    return out


def verify(sample: dict, calls: list[dict], kind: str, catalog: dict[str, dict], zaic_tools,
           data_root: Path) -> str | None:
    """None if the calls are acceptable as a scenario of this kind (act, ask, empty), else the reason."""
    if kind == "empty":
        return None if not calls else "calls on empty"
    if not calls:
        return "no calls"
    if uses_holdout_app(calls):
        return "holdout app"
    if (kind == "ask" and calls[-1]["name"] != SYS_TOOL) or (kind == "act" and any(c["name"] == SYS_TOOL for c in calls)):
        return "kind mismatch"
    installed = set(installed_tool_names(sample))
    env = _new_env(zaic_tools, sample, data_root)
    seen = scalar_strings([turn.get("tool_results") for turn in sample["history"]], set())
    for call in calls:
        name, args = call["name"], call["arguments"]
        if name not in installed:
            return f"not installed: {name}"
        check = env.validate(name, args)
        if not check.get("ok"):
            return f"schema: {check.get('error')}"
        unseen = unseen_refs(catalog.get(name), args, seen)
        if unseen:
            return f"unseen id: {unseen[0]}"
        result = env.call(name, args)
        scalar_strings(result, seen)
        if result.get("ok") and result.get("done") is False:
            return f"no effect: {name}"
    if any(c["name"] == SYS_TOOL for c in calls[:-1]):
        return "ask not last"
    keys = [(c["name"], repr(sorted(c["arguments"].items()))) for c in calls]
    if len(set(keys)) != len(keys):
        return "repeat"
    return None


def synthetic_sample(base: dict, request: str, calls: list[dict], n: int) -> dict:
    """A new sample on base's environment. The history stays: the environment state already includes it."""
    return {"id": f"synth_{base['id']}_{n}", "env_id": base["id"], "audio": base["audio"], "audio_text": request,
            "context": base["context"], "history": base["history"], "label": {"tool_calls": calls},
            "source": "synth"}


def accept(base: dict, scenarios: list[dict], catalog: dict, zaic_tools, data_root: Path):
    """(accepted synthetic samples of this base, rejection reasons, kinds of the accepted) for scenarios given as
    {"request", "kind", "calls": [{"name", "arguments"}]}."""
    accepted, reasons, kinds = [], Counter(), Counter()
    for scenario in scenarios:
        reason = verify(base, scenario["calls"], scenario["kind"], catalog, zaic_tools, data_root)
        if reason:
            reasons[reason.split(":")[0]] += 1
            continue
        accepted.append(synthetic_sample(base, scenario["request"], scenario["calls"], len(accepted)))
        kinds[scenario["kind"]] += 1
    return accepted, reasons, kinds


SYNTH_DIR = REPO_ROOT / "train_records" / "synth"


def _brief(base: dict, catalog: dict, zaic_tools, data_root: Path) -> dict:
    """Everything a scenario writer needs, compact: profile, history, what the apps hold, tool signatures."""
    tools, holdout = {}, holdout_apps()
    for name in installed_tool_names(base):
        spec = catalog.get(name)
        if spec is None or name.split(".")[0] in holdout:  # the writer never sees a holdout app
            continue
        params = spec["parameters"]
        tools[name] = {
            "kind": spec.get("kind"), "required": params.get("required", []), "one_of": params.get("x-one-of"),
            "params": {k: {kk: v[kk] for kk in ("type", "enum", "x-ref", "x-match", "minimum", "maximum", "format")
                           if kk in v} for k, v in params.get("properties", {}).items()}}
    ctx = base["context"]
    return {"id": base["id"], "now": ctx["current_time"], "profile": ctx.get("user_profile"),
            "history": [{k: v for k, v in t.items() if k not in ("audio",)} for t in base["history"]],
            "apps_hold": {p["name"]: p["result"] for p in probe(base, catalog, zaic_tools, data_root)},
            "tools": tools}


INSTRUCTIONS = """You write training scenarios for a Vietnamese voice assistant that controls phone apps through tool calls.
Below are several BRIEFS, one per user. For each brief write {n} new scenarios: something that user could say to the
assistant right now, in spoken Vietnamese, with the exact tool calls a perfect assistant would make.

Rules for the calls:
- Every call is scored, so make only the calls the request needs. Read tools only when their result is needed to fill a
  later argument or to decide a condition. Leave optional arguments out unless the user states them.
- An id (contact_id, event_id, ...) must come from "apps_hold", from the history's tool results, or from a result of a
  read call earlier in the same scenario (then use a plausible read call first and assume it returns what "apps_hold"
  shows). Never invent one.
- Use only values the tool's params allow (type, enum, minimum, maximum, format). Times are ISO 8601 with the timezone of
  "now". Work out relative dates ("mai", "thứ sáu tuần sau") from "now".
- kind "ask": a required value is missing, or a read would return several candidates that fit. The last call is
  sys.ask_missing_info with arguments {{"tool_name": "<app.tool>", "params": ["<required param still unknown>"]}}. No call
  comes after it.
- kind "empty": no installed tool can do what the user asks, or the user cancels their own request, or only chats.
  "calls" is [].
- kind "act": everything needed is known; the calls carry out the request.
- For each brief write at least one pair: one pair with the same intent, made of two scenarios that differ in one detail.
- The kind "act" one: a detail in the request picks exactly one record, so the calls carry out the request.
- The kind "ask" one: several records fit or a required value is missing, so the last call asks.
- Read calls pass only the arguments the user stated: a filter the user did not say is left out.
- Never put a placeholder such as "Đã xong" in an argument. Free-text arguments (title, content, name) are short phrases
  copying names and numbers word for word from the request or from tool results.
- Use only tools listed in that brief's "tools".

Rules for the request: what a person says aloud; numbers and e-mail addresses as speech, regional words, fillers,
polite particles; vary length and style. Mix per brief: about 60% act (1-4 calls), 25% ask, 15% empty.

Answer with ONLY one JSON object, no commentary, mapping each brief "id" to its list of scenarios:
{{"<id>": [{{"request": "...", "kind": "act", "calls": [{{"name": "alarm.create", "arguments": {{"time": "..."}}}}]}}]}}

BRIEFS:
"""


def parse_answer(text: str) -> dict[str, list[dict]]:
    """The first JSON object in a chat answer (markdown fences and commentary around it are ignored); {} if none."""
    start = text.find("{")
    if start < 0:
        return {}
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return {}
    out = {}
    for base_id, scenarios in obj.items() if isinstance(obj, dict) else []:
        good = []
        for sc in scenarios if isinstance(scenarios, list) else []:
            try:
                calls = [{"name": c["name"], "arguments": c.get("arguments", {})} for c in sc["calls"]]
                if (isinstance(sc["request"], str) and sc["request"].strip() and sc["kind"] in ("act", "ask", "empty")
                        and all(isinstance(c["name"], str) and isinstance(c["arguments"], dict) for c in calls)):
                    good.append({"request": sc["request"].strip(), "kind": sc["kind"], "calls": calls})
            except (KeyError, TypeError, AttributeError):
                continue
        out[str(base_id)] = good
    return out


ANSWER_TOKENS = 2048


def answer(run: Path, llm=None, teacher: str | None = None, revision: str | None = None) -> int:
    """Write run/scenarios/prompt_NN.json for every packed prompt that has no answer yet; returns how many it wrote.
    `llm.generate(messages) -> text` is the local teacher (built from `teacher` when not given). Each answer is saved as
    soon as it is written, so an interrupted run resumes where it stopped. The teacher module is imported here only."""
    prompts = sorted((run / "prompts").glob("prompt_*.txt"))
    todo = [(p, run / "scenarios" / (p.stem + ".json")) for p in prompts]
    todo = [(p, out) for p, out in todo if not (out.exists() and out.stat().st_size)]
    if todo and llm is None:
        from .thoughts import Teacher

        llm = Teacher(teacher, revision, max_new_tokens=ANSWER_TOKENS)
    (run / "scenarios").mkdir(parents=True, exist_ok=True)
    for i, (prompt, out) in enumerate(todo):
        out.write_text(llm.generate([{"role": "user", "content": prompt.read_text(encoding="utf-8")}]), encoding="utf-8")
        print(f"answered {out.name} ({i + 1}/{len(todo)})", flush=True)
    print(f"-> {run / 'scenarios'}: {len(todo)} answered, {len(prompts) - len(todo)} already there", flush=True)
    return len(todo)


def main(argv: list[str] | None = None, llm=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("pack", "ingest", "answer"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--subset", choices=("pool", "all"), default="pool")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--per-prompt", type=int, default=4, help="briefs per prompt file")
    ap.add_argument("--scenarios", type=int, default=3, help="scenarios asked per brief")
    ap.add_argument("--hf-repo", default=None, help="ingest: private HF dataset to upload <name>.jsonl and its report to (synth/)")
    ap.add_argument("--teacher", help="answer: HF id of the local model that writes the scenarios")
    ap.add_argument("--revision", help="answer: pinned commit of --teacher")
    ap.add_argument("--data-root", type=Path, default=default_data_root())
    ap.add_argument("--out-dir", type=Path, default=SYNTH_DIR)
    args = ap.parse_args(argv)
    if args.mode == "answer":
        if llm is None and not (args.teacher and args.revision):
            ap.error("answer needs --teacher and --revision (a pinned commit)")
        answer(args.out_dir / args.name, llm, args.teacher, args.revision)
        return
    from ..splits import load_splits

    samples = load_samples(args.data_root, "train")
    wanted = {s["id"] for s in samples} if args.subset == "all" else set(load_splits()["pool"])
    bases = [s for s in samples if s["id"] in wanted]
    zaic_tools = import_zaic_tools(args.data_root, "train")
    catalog = load_catalog(args.data_root, "train")
    run = args.out_dir / args.name
    if args.mode == "pack":
        (run / "prompts").mkdir(parents=True, exist_ok=True)
        (run / "scenarios").mkdir(exist_ok=True)
        chosen = bases[args.offset: args.offset + args.limit]
        for i in range(0, len(chosen), args.per_prompt):
            briefs = [_brief(b, catalog, zaic_tools, args.data_root) for b in chosen[i: i + args.per_prompt]]
            text = INSTRUCTIONS.format(n=args.scenarios) + "\n".join(json.dumps(b, ensure_ascii=False) for b in briefs)
            (run / "prompts" / f"prompt_{i // args.per_prompt:02d}.txt").write_text(text, encoding="utf-8")
        print(f"-> {run / 'prompts'}: {-(-len(chosen) // args.per_prompt)} prompts for {len(chosen)} samples", flush=True)
        return
    by_id = {b["id"]: b for b in bases}
    accepted, reasons, kinds, total = [], Counter(), Counter(), 0
    for path in sorted(p for p in (run / "scenarios").iterdir() if p.is_file()):
        for base_id, scenarios in parse_answer(path.read_text(encoding="utf-8")).items():
            if base_id not in by_id:
                reasons["unknown base"] += 1
                continue
            total += len(scenarios)
            got, why, kind = accept(by_id[base_id], scenarios, catalog, zaic_tools, args.data_root)
            accepted += [{**s, "id": f"{s['id']}_{path.stem}"} for s in got]
            reasons.update(why)
            kinds.update(kind)
    (args.out_dir / f"{args.name}.jsonl").write_text(
        "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in accepted), encoding="utf-8")
    report = {"scenarios": total, "accepted": len(accepted), "rejected_by_reason": dict(reasons),
              "accepted_by_kind": dict(kinds)}
    (args.out_dir / f"{args.name}.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                           encoding="utf-8")
    print(f"-> {args.out_dir / (args.name + '.jsonl')}: {report}", flush=True)
    if args.hf_repo:  # the training notebook reads synthetic samples from the dataset, from another session
        from huggingface_hub import HfApi

        for path in (args.out_dir / f"{args.name}.jsonl", args.out_dir / f"{args.name}.report.json"):
            HfApi().upload_file(path_or_fileobj=str(path), path_in_repo=f"synth/{path.name}", repo_id=args.hf_repo,
                                repo_type="dataset")
        print(f"uploaded synth/{args.name}.jsonl to {args.hf_repo}", flush=True)


if __name__ == "__main__":
    main()
