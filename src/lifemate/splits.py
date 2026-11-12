"""Fixed train splits: holdout apps (stand-in for unseen private-test apps), a 300-sample dev set and the training pool.

    python -m lifemate.splits --seed 0        # writes configs/splits.json (committed; regenerate only on purpose)

Every experiment reports dev and holdout numbers; few-shot retrieval and training draw only from the pool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from .data import REPO_ROOT, default_data_root, load_samples
from .evaluate import history_bucket, label_kind

SPLITS_PATH = REPO_ROOT / "configs" / "splits.json"
SUBSETS = ("dev", "holdout", "pool")
APP_USE_RANGE = (15, 120)    # candidate holdout apps: used by this many train samples (keeps contacts, calendar, ... in training)
HOLDOUT_SHARE = (0.10, 0.20)
MIN_GROUPS = 4
MAX_DRAWS = 10_000


def sample_apps(sample: dict) -> set[str]:
    """App prefixes of the gold calls (if labelled) and of every history tool call, without sys."""
    calls = list((sample.get("label") or {}).get("tool_calls", []))
    calls += [c for turn in sample["history"] for c in turn.get("tool_calls") or []]
    return {c["name"].split(".", 1)[0] for c in calls} - {"sys"}


def load_app_groups(data_root: Path | None = None) -> dict[str, str]:
    path = (data_root or default_data_root()) / "train" / "train_tools_catalog.json"
    with open(path, encoding="utf-8") as f:
        return {app["id"]: app["group"] for app in json.load(f)["apps"]}


def _draw_holdout_apps(samples: list[dict], rng: random.Random, n_apps: int, groups: dict[str, str]) -> list[str]:
    users = defaultdict(int)
    for s in samples:
        for app in sample_apps(s):
            users[app] += 1
    candidates = sorted(a for a, n in users.items() if APP_USE_RANGE[0] <= n <= APP_USE_RANGE[1])
    for _ in range(MAX_DRAWS):
        apps = set(rng.sample(candidates, n_apps))
        share = sum(bool(sample_apps(s) & apps) for s in samples) / len(samples)
        if HOLDOUT_SHARE[0] <= share <= HOLDOUT_SHARE[1] and len({groups.get(a) for a in apps}) >= MIN_GROUPS:
            return sorted(apps)
    raise RuntimeError(f"no holdout draw of {n_apps} apps met the share and group rules in {MAX_DRAWS} tries")


def _stratified(samples: list[dict], rng: random.Random, size: int) -> list[str]:
    """`size` ids, allocated to label_kind x history_bucket strata by largest remainder, drawn at random inside each."""
    strata = defaultdict(list)
    for s in samples:
        strata[(label_kind(s["label"]["tool_calls"]), history_bucket(len(s["history"])))].append(s["id"])
    keys = sorted(strata)
    exact = {k: size * len(strata[k]) / len(samples) for k in keys}
    quota = {k: int(exact[k]) for k in keys}
    for k in sorted(keys, key=lambda k: (-(exact[k] - quota[k]), k))[: size - sum(quota.values())]:
        quota[k] += 1
    return [i for k in keys for i in rng.sample(sorted(strata[k]), quota[k])]


def make_splits(samples: list[dict], seed: int = 0, n_holdout_apps: int = 8, dev_size: int = 300,
                groups: dict[str, str] | None = None) -> dict:
    rng = random.Random(seed)
    holdout_apps = _draw_holdout_apps(samples, rng, n_holdout_apps, groups or load_app_groups())
    held = set(holdout_apps)
    holdout = sorted(s["id"] for s in samples if sample_apps(s) & held)
    rest = [s for s in samples if not sample_apps(s) & held]
    dev = set(_stratified(rest, rng, dev_size))
    return {"seed": seed, "holdout_apps": holdout_apps, "holdout": holdout, "dev": sorted(dev),
            "pool": sorted(s["id"] for s in rest if s["id"] not in dev)}


def load_splits(path: Path = SPLITS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pool_fold(sample_id: str, folds: int) -> int:
    """Cross-fit fold of a pool sample, stable across runs and machines (sha1 of the id, not Python's hash)."""
    return int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest(), 16) % folds


def fold_ids(pool, folds: int, fold: int) -> set[str]:
    """Pool ids in `fold`. Clips of one sample (final request and history turns) always share a fold."""
    if not 0 <= fold < folds:
        raise ValueError(f"fold {fold} out of range for {folds} folds")
    return {i for i in pool if pool_fold(i, folds) == fold}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", type=Path, default=default_data_root())
    ap.add_argument("--out", type=Path, default=SPLITS_PATH)
    args = ap.parse_args(argv)
    splits = make_splits(load_samples(args.data_root, "train"), seed=args.seed, groups=load_app_groups(args.data_root))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(splits, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"-> {args.out}: holdout apps {splits['holdout_apps']}; "
          f"holdout {len(splits['holdout'])}, dev {len(splits['dev'])}, pool {len(splits['pool'])}")


if __name__ == "__main__":
    main()
