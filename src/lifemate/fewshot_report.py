"""How often the few-shot examples hold a tool analogous to a gold tool, for a train subset.

    python -m lifemate.fewshot_report --subset holdout --analog-weight 2

For every sample of the subset with at least one gold tool, the examples the bank picks (k, gold request text, the
installed tools of the sample) are checked against the gold tools: an example tool is analogous to a gold tool when both
have the same name after the app, or their tool_tokens (agents/fewshot.py) have Jaccard >= 0.5. Reported for all such
samples and for those calling a tool of an app the bank has no example of, which is where selection by tool structure
can change anything. It reads the labels for this report only; it needs no model, no ASR and no GPU.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .agents.fewshot import ExampleBank, _jaccard, gold_tool_names, tool_tokens
from .data import default_data_root, installed_tool_names, load_catalog, load_samples
from .splits import load_splits

JACCARD = 0.5


def analogous(a: dict, b: dict) -> bool:
    """Same name after the app, or the same tool shape."""
    return a["name"].split(".", 1)[-1] == b["name"].split(".", 1)[-1] or _jaccard(tool_tokens(a), tool_tokens(b)) >= JACCARD


def report(samples: list[dict], bank: ExampleBank, catalog: dict[str, dict], analog_weight: float, k: int = 4) -> dict:
    """{"n", "hits", "share", "unseen_n", "unseen_hits", "unseen_share"} over the samples that have a gold tool."""
    n = hits = unseen_n = unseen_hits = 0
    for s in samples:
        gold = [{**catalog[t], "name": t} for t in gold_tool_names(s) if t in catalog]
        if not gold:
            continue
        installed = [{**catalog[t], "name": t} for t in installed_tool_names(s) if t in catalog]
        picked = bank.select(s["audio_text"], set(s["context"]["installed_apps"]), k, exclude_id=s["id"],
                             tools=installed, analog_weight=analog_weight)
        shown = [{**catalog[t], "name": t} for ex in picked for t in gold_tool_names(ex) if t in catalog]
        hit = any(analogous(e, g) for e in shown for g in gold)
        unseen = any(g["name"].split(".", 1)[0] not in bank.bank_apps for g in gold)
        n, hits = n + 1, hits + hit
        unseen_n, unseen_hits = unseen_n + unseen, unseen_hits + (unseen and hit)
    pct = lambda a, b: round(100 * a / b, 2) if b else 0.0  # noqa: E731
    return {"n": n, "hits": hits, "share": pct(hits, n), "unseen_n": unseen_n, "unseen_hits": unseen_hits,
            "unseen_share": pct(unseen_hits, unseen_n)}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default="holdout", choices=["dev", "holdout", "pool"])
    ap.add_argument("--analog-weight", type=float, default=0.0)
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--data-root", type=Path, default=default_data_root())
    args = ap.parse_args(argv)
    ids = set(load_splits()[args.subset])
    samples = [s for s in load_samples(args.data_root, "train") if s["id"] in ids]
    out = report(samples, ExampleBank.from_data(args.data_root, "train"), load_catalog(args.data_root, "train"),
                 args.analog_weight, args.examples)
    print(f"{args.subset} analog_weight {args.analog_weight}: {out['hits']}/{out['n']} samples ({out['share']}%) have an "
          f"analogous tool in their examples; calling a tool of an app absent from the bank: "
          f"{out['unseen_hits']}/{out['unseen_n']} ({out['unseen_share']}%)", flush=True)
    return out


if __name__ == "__main__":
    main()
