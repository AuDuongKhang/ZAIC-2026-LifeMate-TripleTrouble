"""Preference pairs from the model's own wrong decisions (design docs/superpowers/specs/2026-11-12-onpolicy-preference-design.md).

Development only: never imported by agents/, infer.py or docker_entry.py.

A step's decision is what the agent does next: write to a tool, read a tool, ask (`sys.ask_missing_info`), answer
without a call, or other (unparsable output, a tool the record does not show). A sampled step makes a pair with the gold
step when the two decisions differ in a way gold is the single right answer for (pair_type), or when both write to the
same tool and a non-fuzzy argument has another value (args_differ, type `write:args`). Reads against other reads,
read arguments and fuzzy free text never make a pair: gold is not consistent there.
"""
from __future__ import annotations

import argparse
import json
import random
import zlib
from collections import Counter
from pathlib import Path

from ..data import SYS_TOOL, default_data_root, fuzzy_params, load_catalog
from .records import RECORDS_DIR

Decision = tuple[str, "str | None"]


def tool_kinds(catalog: dict[str, dict]) -> dict[str, str]:
    """{tool name: "read" | "write"} for every catalog tool except sys.ask_missing_info."""
    return {name: spec["kind"] for name, spec in catalog.items() if name != SYS_TOOL}


def _first_call(message: dict) -> dict | None:
    calls = message.get("tool_calls") or []
    return calls[0]["function"] if calls else None


def decision(message: dict, name_map: dict[str, str], kinds: dict[str, str]) -> Decision:
    """The decision of an assistant message (its first tool call), with masked names mapped back by name_map."""
    call = _first_call(message)
    if call is None:
        if "<tool_call>" in (message.get("content") or ""):  # a call that did not parse
            return ("other", None)
        return ("answer", None)
    name = name_map.get(call["name"], call["name"])
    if name == SYS_TOOL:
        target = (call.get("arguments") or {}).get("tool_name")
        return ("ask", name_map.get(target, target) if isinstance(target, str) else None)
    kind = kinds.get(name)
    if kind not in ("read", "write"):
        return ("other", None)
    return (kind, name)


def pair_type(gold: Decision, model: Decision) -> str | None:
    """The pair type for a model decision against the gold one, or None when they do not make a pair."""
    g, m = gold[0], model[0]
    if m == "other" or g == "other":
        return None
    if g == "write":
        if m == "write":
            return "write:other_write" if model[1] != gold[1] else None
        return f"write:{m}"
    if g == "ask":
        return None if m == "ask" else f"ask:{m}"
    if g == "answer":
        # Any call where gold stops is scored as extra (an out-of-scope sample scores only when empty), reads included.
        return None if m == "answer" else f"answer:{m}"
    if g == "read":
        return None if m == "read" else f"read:{m}"
    return None


def args_differ(gold: dict, model: dict, name_map: dict[str, str], fuzzy: dict[str, set[str]]) -> bool:
    """True when both messages call the same tool and a non-fuzzy argument present on both sides has another value
    (compared with != as scoring.call_matches does). The caller checks that gold is a write."""
    g, m = _first_call(gold), _first_call(model)
    if g is None or m is None:
        return False
    tool = name_map.get(g["name"], g["name"])
    if name_map.get(m["name"], m["name"]) != tool:
        return False
    model_args = {name_map.get(k, k): v for k, v in (m.get("arguments") or {}).items()}
    loose = fuzzy.get(tool, set())
    return any(k in model_args and k not in loose and model_args[k] != v
               for k, v in (g.get("arguments") or {}).items())


def _assistant(text: str) -> dict:
    from .replay import assistant_message  # replay imports the agent; only the pair building needs it

    return assistant_message(text)


# Pair types that are conventions, not decisions (pilot of 13 Nov, 30 pairs read by hand): an extra read before a write
# (a fare or a balance) is another valid order of work and only costs precision.
DROPPED_TYPES = frozenset({"write:read"})


def choose_rejected(gold_message: dict, candidates: list[str], name_map: dict[str, str], kinds: dict[str, str],
                    fuzzy: dict[str, set[str]], later: set | frozenset = frozenset()) -> tuple[str, str, bool] | None:
    """(type, rejected text, greedy_wrong) for the most frequent wrong decision among candidates ([greedy] + samples),
    or None when no candidate makes a pair. Ties go to the earliest candidate; write:args candidates form one group.
    later holds the decisions of the gold steps after this one: a candidate doing one of them first is another valid
    order, not a mistake (stopping is never such a step). Types in DROPPED_TYPES never make a pair."""
    gold = decision(gold_message, name_map, kinds)
    swaps = set(later) - {("answer", None)}
    groups: dict[tuple, list[int]] = {}
    types: dict[tuple, str] = {}
    for i, text in enumerate(candidates):
        message = _assistant(text)
        model = decision(message, name_map, kinds)
        kind = pair_type(gold, model)
        key: tuple = (kind, model)
        if kind is None and gold[0] == "write" and model == gold and args_differ(gold_message, message, name_map, fuzzy):
            kind, key = "write:args", ("write:args",)
        if kind is None or kind in DROPPED_TYPES or (kind != "write:args" and model in swaps):
            continue
        groups.setdefault(key, []).append(i)
        types[key] = kind
    if not groups:
        return None
    best = max(groups, key=lambda k: (len(groups[k]), -groups[k][0]))
    greedy_wrong = any(0 in members for members in groups.values())
    return types[best], candidates[groups[best][0]], greedy_wrong


def is_heldout(sample_id: str) -> bool:
    """10% of the samples, by id, whose pairs are held out of training (cheap selection before the gate)."""
    return zlib.crc32(sample_id.encode()) % 10 == 0


def _review(pairs: list[dict]) -> str:
    lines = ["# Pairs to read by hand", "",
             "Mark each pair `decision` (gold is the single right answer) or `convention` (another step could be right).", ""]
    for p in pairs:
        user = next(m["content"] for m in reversed(p["messages"]) if m["role"] == "user")
        lines += [f"## {p['id']} turn {p['turn']} — {p['type']}", "", f"- request: {user}",
                  f"- gold: `{json.dumps(p['chosen'].get('tool_calls') or p['chosen'].get('content'), ensure_ascii=False)}`",
                  f"- model: `{json.dumps(p['rejected'].get('tool_calls') or p['rejected'].get('content'), ensure_ascii=False)}`",
                  "- verdict: ", ""]
    return "\n".join(lines)


def _jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Build preference pairs from a steps file (lifemate.sft.sample_steps).")
    ap.add_argument("--records", action="append", required=True, help="train_records/<NAME>.jsonl the steps came from")
    ap.add_argument("--steps", required=True, help="train_records/steps/<NAME>.jsonl")
    ap.add_argument("--out", required=True, help="train_records/pairs/<NAME>.jsonl (+ -heldout.jsonl, -review.md)")
    ap.add_argument("--greedy-wrong-only", action="store_true", help="keep only pairs where the greedy step was wrong")
    ap.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    ap.add_argument("--data-root", type=Path, default=default_data_root())
    args = ap.parse_args(argv)

    catalog = load_catalog(args.data_root, "train")
    kinds, fuzzy = tool_kinds(catalog), fuzzy_params(catalog)
    records = {name: _jsonl(args.records_dir / f"{name}.jsonl") for name in args.records}
    train, held = [], []
    points = with_pair = greedy_wrong = 0
    by_type: Counter = Counter()
    for row in _jsonl(args.records_dir / "steps" / f"{args.steps}.jsonl"):
        try:
            record = records[row["records"]][row["record"]]
        except (KeyError, IndexError):
            raise ValueError(f"steps row {row['records']}:{row['record']} has no record") from None
        points += 1
        name_map = record.get("name_map", {})
        gold = record["messages"][row["turn"]]
        later = {decision(m, name_map, kinds) for m in record["messages"][row["turn"] + 1:] if m["role"] == "assistant"}
        chosen = choose_rejected(gold, [row["greedy"], *row["samples"]], name_map, kinds, fuzzy, later)
        if chosen is None:
            continue
        kind, text, wrong = chosen
        with_pair += 1
        greedy_wrong += wrong
        by_type[kind] += 1
        if args.greedy_wrong_only and not wrong:
            continue
        pair = {"id": record["id"], "record": row["record"], "turn": row["turn"], "type": kind, "greedy_wrong": wrong,
                "messages": record["messages"][: row["turn"]], "tools": record["tools"], "chosen": gold,
                "rejected": _assistant(text)}
        (held if is_heldout(record["id"]) else train).append(pair)

    out = args.records_dir / "pairs"
    out.mkdir(parents=True, exist_ok=True)
    for path, rows in ((out / f"{args.out}.jsonl", train), (out / f"{args.out}-heldout.jsonl", held)):
        path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in rows), encoding="utf-8")
    picked = random.Random(0).sample(train + held, min(30, len(train) + len(held)))
    (out / f"{args.out}-review.md").write_text(_review(picked), encoding="utf-8")
    report = {"points": points, "with_pair": with_pair, "rate": with_pair / points if points else 0.0,
              "greedy_wrong": greedy_wrong, "by_type": dict(by_type), "train": len(train), "heldout": len(held)}
    print(f"{points} decision points, {with_pair} with a pair ({report['rate']:.1%}); greedy wrong {greedy_wrong}; "
          f"by type {dict(by_type)}; {len(train)} train pairs, {len(held)} held out -> {out / args.out}.jsonl", flush=True)
    return report


if __name__ == "__main__":
    main()
