"""Docker entry point: run the shipped system on the split mounted at /data, fully offline.

    docker run --rm --gpus all --network none -v <split folder>:/data:ro -v <out>:/app/result lifemate:<tag>

`/data` is the split folder itself (audio/, <split>.jsonl, <split>_environment/, zaic_tools/,
<split>_tools_catalog.json); the split name comes from the catalog file. The shipped presets, extra kwargs and
model revisions live in configs/shipped.json, which the image build also reads (`fetch_models`), so the weights
baked into the image are exactly the ones the entry loads. The few-shot bank needs train/train.jsonl and the train
catalog, which the image carries in /app/data/train.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
from pathlib import Path

from .data import REPO_ROOT

CONFIG_PATH = REPO_ROOT / "configs" / "shipped.json"
TRAIN_FILES = ("train.jsonl", "train_tools_catalog.json")
CATALOG_SUFFIX = "_tools_catalog.json"


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_split(data_dir: Path) -> str:
    found = sorted(p.name for p in Path(data_dir).glob(f"*{CATALOG_SUFFIX}"))
    if len(found) != 1:
        raise SystemExit(f"{data_dir}: expected one *{CATALOG_SUFFIX}, found {found or 'none'}")
    return found[0][: -len(CATALOG_SUFFIX)]


def _model_and_revision(kind: str, config: dict) -> tuple[str | None, str | None]:
    """The HF repo a preset loads (its `model` kwarg, else the backend's default) and the pinned revision."""
    from .agents import AGENTS
    from .asr import BACKENDS
    from .presets import resolve

    backend, kw = resolve(kind, config[f"{kind}_preset"], {"HF_MODEL_OWNER": config["hf_model_owner"]})
    kw.update(config.get(f"{kind}_kw", {}))
    if "model" not in kw:
        param = inspect.signature((AGENTS if kind == "agent" else BACKENDS)[backend]).parameters.get("model")
        if param is None or param.default is inspect.Parameter.empty:
            return None, None
        kw["model"] = param.default
    return kw["model"], kw.get("revision")


def fetch_models(config: dict) -> None:
    """Download every model the shipped presets load, at the pinned revisions, into HF_HOME (image build only)."""
    import huggingface_hub

    for kind in ("agent", "asr"):
        model, revision = _model_and_revision(kind, config)
        if model:
            huggingface_hub.snapshot_download(repo_id=model, revision=revision)


def _kw_flags(flag: str, kw: dict) -> list[str]:
    # infer parses KEY=VALUE values as JSON, so encode them: a revision like "1e5" must stay a string
    return [x for k, v in kw.items() for x in (flag, f"{k}={json.dumps(v)}")]


def _lines(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("/data"), help="the mounted split folder")
    ap.add_argument("--out", type=Path, default=Path("result/submission.jsonl"))
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--work", type=Path, default=Path("/tmp/lifemate"))
    ap.add_argument("--train-dir", type=Path, default=REPO_ROOT / "data" / "train",
                    help="holds train.jsonl and the train catalog (few-shot bank)")
    args = ap.parse_args(argv)
    config = load_config(args.config)
    split = find_split(args.data)

    os.environ["HF_HUB_OFFLINE"] = "1"  # before anything imports transformers or huggingface_hub
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_MODEL_OWNER"] = config["hf_model_owner"]

    root = args.work / "data"
    root.mkdir(parents=True, exist_ok=True)
    link = root / split
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(args.data.resolve(), link, target_is_directory=True)
    (root / "train").mkdir(exist_ok=True)
    for name in TRAIN_FILES:
        shutil.copy2(args.train_dir / name, root / "train" / name)

    from . import infer

    run = infer.main(["--split", split, "--data-root", str(root), "--out", str(args.work / "run"),
                      "--agent-preset", config["agent_preset"], "--asr-preset", config["asr_preset"],
                      *_kw_flags("--agent-kw", config.get("agent_kw", {})),
                      *_kw_flags("--asr-kw", config.get("asr_kw", {}))])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run / "submission.jsonl", args.out)
    if (run / "meta.json").exists():
        shutil.copy2(run / "meta.json", args.out.parent / "meta.json")  # timing, presets, commit
    expected, written = _lines(args.data / f"{split}.jsonl"), _lines(args.out)
    if written != expected:
        raise SystemExit(f"{args.out}: {written} lines for {expected} samples")
    print(f"-> {args.out}: {written} samples ({split})", flush=True)
    return args.out


if __name__ == "__main__":
    main()
