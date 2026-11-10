"""Docker entry point (`/code/predict.sh`): run the shipped system on the split mounted under /data, fully offline.

    docker run --rm --gpus all --network none -v <folder holding private_test/>:/data:ro -v <out>:/result <image>
    python -m lifemate.docker_entry --check-models     # every shipped model is in the HF cache (no network)

`/data` holds the split folder (`/data/private_test/` with audio/, <split>.jsonl, zaic_tools/,
<split>_tools_catalog.json), as the organiser guide mounts it; `/data` being the split folder itself works too. The
split name comes from the catalog file, and the result goes to /result/submission.jsonl. The shipped presets, extra
kwargs and model revisions live in configs/shipped.json, which the image build also reads (`fetch_models`), so the
weights baked into the image are exactly the ones the entry loads; LIFEMATE_SHIPPED points to another config (the
CPU smoke test uses configs/smoke.json). The few-shot bank needs train/train.jsonl and the train catalog, which the
image carries in /code/data/train.
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
SKIP_FILES = ["checkpoints/*", "*.pt"]  # other runs' adapters and optimizer states in the model repos


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_split(data_dir: Path) -> str:
    found = sorted(p.name for p in Path(data_dir).glob(f"*{CATALOG_SUFFIX}"))
    if len(found) != 1:
        raise SystemExit(f"{data_dir}: expected one *{CATALOG_SUFFIX}, found {found or 'none'}")
    return found[0][: -len(CATALOG_SUFFIX)]


def locate_split(data: Path) -> tuple[Path, str]:
    """(split folder, split name): `data` itself when it holds a catalog, else its one child `<name>/` holding
    `<name>_tools_catalog.json`."""
    data = Path(data)
    if any(data.glob(f"*{CATALOG_SUFFIX}")):
        return data, find_split(data)
    found = sorted(c for c in data.iterdir() if c.is_dir() and (c / f"{c.name}{CATALOG_SUFFIX}").is_file())
    if len(found) != 1:
        raise SystemExit(f"{data}: expected one split folder with its *{CATALOG_SUFFIX}, "
                         f"found {[c.name for c in found] or 'none'}")
    return found[0], found[0].name


def prepare(data: Path, work: Path, train_dir: Path) -> tuple[Path, str]:
    """A writable data root under `work`: a link to the mounted (read-only) split folder plus a copy of the train
    few-shot bank. Returns (data root, split)."""
    split_dir, split = locate_split(data)
    root = Path(work) / "data"
    root.mkdir(parents=True, exist_ok=True)
    link = root / split
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(split_dir.resolve(), link, target_is_directory=True)
    (root / "train").mkdir(exist_ok=True)
    for name in TRAIN_FILES:
        shutil.copy2(Path(train_dir) / name, root / "train" / name)
    return root, split


def set_offline(config: dict) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"  # before anything imports transformers or huggingface_hub
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_MODEL_OWNER"] = config["hf_model_owner"]


def shipped_backends(config: dict) -> tuple[str, dict, str, dict]:
    """(agent backend, agent kwargs, ASR backend, ASR kwargs): the config's presets plus its extra kwargs."""
    from .presets import resolve

    env = {"HF_MODEL_OWNER": config["hf_model_owner"]}
    agent, agent_kw = resolve("agent", config["agent_preset"], env)
    asr, asr_kw = resolve("asr", config["asr_preset"], env)
    return agent, {**agent_kw, **config.get("agent_kw", {})}, asr, {**asr_kw, **config.get("asr_kw", {})}


def load_shipped(config: dict, split: str, data_root: Path):
    """The infer.Pipeline of the configured system, models loaded, HF offline."""
    set_offline(config)
    from . import infer

    agent, agent_kw, asr, asr_kw = shipped_backends(config)
    return infer.load_pipeline(split, data_root, agent, agent_kw, asr, asr_kw)


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
            huggingface_hub.snapshot_download(repo_id=model, revision=revision, ignore_patterns=SKIP_FILES)


def check_models(config: dict, load: bool = True) -> None:
    """Exit non-zero unless every shipped model is in the HF cache (meant to run with the network off). With `load`,
    also read each model's config and tokenizer (agent) or processor (ASR) by repo id and revision, as inference
    does, so a library or offline-cache problem shows up without a GPU."""
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    for kind in ("agent", "asr"):
        model, revision = _model_and_revision(kind, config)
        if not model:
            continue
        try:
            path = huggingface_hub.snapshot_download(repo_id=model, revision=revision, ignore_patterns=SKIP_FILES,
                                                     local_files_only=True)
        except LocalEntryNotFoundError as e:
            raise SystemExit(f"{kind} model {model}@{revision} is not in the cache: {e}")
        print(f"{kind}: {model}@{revision} -> {path}", flush=True)
        # snapshot_download(local_files_only=True) also returns a snapshot folder that holds only part of the repo
        if not (Path(path) / "config.json").is_file() or not any(Path(path).glob("*.safetensors")):
            raise SystemExit(f"{kind} model {model}@{revision}: incomplete snapshot in {path} "
                             "(no config.json or *.safetensors)")
        if load:
            from transformers import AutoConfig, AutoProcessor, AutoTokenizer

            cfg = AutoConfig.from_pretrained(model, revision=revision)
            (AutoTokenizer if kind == "agent" else AutoProcessor).from_pretrained(model, revision=revision)
            print(f"  loaded config ({cfg.model_type}) and {'tokenizer' if kind == 'agent' else 'processor'}",
                  flush=True)


def _kw_flags(flag: str, kw: dict) -> list[str]:
    # infer parses KEY=VALUE values as JSON, so encode them: a revision like "1e5" must stay a string
    return [x for k, v in kw.items() for x in (flag, f"{k}={json.dumps(v)}")]


def _lines(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main(argv: list[str] | None = None) -> Path | None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("/data"), help="the mounted folder (holds the split folder)")
    ap.add_argument("--out", type=Path, default=Path("/result/submission.jsonl"))
    ap.add_argument("--config", type=Path, default=Path(os.environ.get("LIFEMATE_SHIPPED", CONFIG_PATH)))
    ap.add_argument("--work", type=Path, default=Path("/tmp/lifemate"))
    ap.add_argument("--train-dir", type=Path, default=REPO_ROOT / "data" / "train",
                    help="holds train.jsonl and the train catalog (few-shot bank)")
    ap.add_argument("--check-models", action="store_true", help="only check that the shipped models are cached")
    args = ap.parse_args(argv)
    config = load_config(args.config)
    set_offline(config)
    if args.check_models:
        check_models(config)
        return None
    root, split = prepare(args.data, args.work, args.train_dir)

    from . import infer

    run = infer.main(["--split", split, "--data-root", str(root), "--out", str(args.work / "run"), "--time-field",
                      "--agent-preset", config["agent_preset"], "--asr-preset", config["asr_preset"],
                      *_kw_flags("--agent-kw", config.get("agent_kw", {})),
                      *_kw_flags("--asr-kw", config.get("asr_kw", {}))])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run / "submission.jsonl", args.out)
    meta = {}
    if (run / "meta.json").exists():
        shutil.copy2(run / "meta.json", args.out.parent / "meta.json")  # timing, presets, commit
        meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    expected, written = _lines(root / split / f"{split}.jsonl"), _lines(args.out)
    if written != expected:
        raise SystemExit(f"{args.out}: {written} lines for {expected} samples")
    if meta:
        n, mean = meta.get("n", written), meta.get("mean_seconds", 0)
        print(f"load model: {meta.get('load_seconds')} s", flush=True)
        print(f"predict: {n} samples, {n * mean:.1f} s, mean {mean * 1000:.0f} ms/sample, "
              f"max {meta.get('max_seconds', 0) * 1000:.0f} ms, over 30 s: {meta.get('over_time_limit')}", flush=True)
    print(f"-> {args.out}: {written} samples ({split})", flush=True)
    return args.out


if __name__ == "__main__":
    main()
