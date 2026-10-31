"""Paired comparison of two scored runs on their shared sample ids, with a bootstrap 95% interval of the difference.

    python -m lifemate.compare runs/<a> runs/<b>       # both need per_sample.jsonl (run lifemate.evaluate first)

A change counts as a win only when the interval excludes 0.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def compare(a: dict[str, float], b: dict[str, float], n_boot: int = 2000, seed: int = 0) -> dict:
    """Scores are 0-100. diff = mean(b) - mean(a); `better` counts samples where b > a."""
    ids = sorted(a.keys() & b.keys())
    if not ids:
        raise ValueError("the runs share no sample ids")
    d = [b[i] - a[i] for i in ids]
    n = len(d)
    rng = random.Random(seed)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    r2 = lambda x: round(x, 2)  # noqa: E731
    return {"n": n, "dropped": len(a.keys() ^ b.keys()),
            "mean_a": r2(sum(a[i] for i in ids) / n), "mean_b": r2(sum(b[i] for i in ids) / n), "diff": r2(sum(d) / n),
            "ci_low": r2(means[int(0.025 * n_boot)]), "ci_high": r2(means[int(0.975 * n_boot) - 1]),
            "better": sum(x > 0 for x in d), "worse": sum(x < 0 for x in d), "same": sum(x == 0 for x in d)}


def verdict(r: dict) -> str:
    if r["ci_low"] > 0:
        return "WIN"
    if r["ci_high"] < 0:
        return "LOSS"
    return "NO CLEAR DIFFERENCE"


def load_scores(run: Path) -> dict[str, float]:
    path = run / "per_sample.jsonl" if run.is_dir() else run
    if not path.exists():
        raise SystemExit(f"{path} not found; run `python -m lifemate.evaluate {run}` first")
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return {r["id"]: 100 * r["score"] for r in rows}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path, help="baseline run folder (or its per_sample.jsonl)")
    ap.add_argument("b", type=Path, help="new run folder")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    r = compare(load_scores(args.a), load_scores(args.b), n_boot=args.n_boot, seed=args.seed)
    print(f"n={r['n']} (dropped {r['dropped']})  a {r['mean_a']}  b {r['mean_b']}  diff {r['diff']:+} "
          f"[95% CI {r['ci_low']:+}, {r['ci_high']:+}]  better {r['better']} / worse {r['worse']} / same {r['same']}  "
          f"{verdict(r)}")
    return r


if __name__ == "__main__":
    main()
