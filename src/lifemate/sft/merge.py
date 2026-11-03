"""Merge a LoRA adapter from lifemate.sft.train into bf16 base weights, so inference stays on HFChatLLM without peft.

    python -m lifemate.sft.merge --adapter checkpoints/sft-v1/last --out merged/sft-v1 --hf-repo <owner>/lifemate-sft-v1
    python -m lifemate.infer --split train --subset dev --agent react --agent-kw model=<owner>/lifemate-sft-v1 ...

With --adapter-repo, the adapter of checkpoints/<run>/last in that HF repo (run = the folder above "last") is used
unless the local --adapter folder holds a complete and newer one; a folder left incomplete by a dead session is
replaced by the HF copy. Without --adapter-repo the local folder must be complete. The merged model and its tokenizer
go to --out and, with --hf-repo, to the root of that private repo (token: HF_TOKEN, write).
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from .state import ADAPTER_FILES, MERGE_FILES, checkpoint_step


def resolve_adapter(adapter: Path, repo: str | None) -> None:
    """Leave a complete adapter in the folder adapter: the HF copy in repo unless the local one is complete and newer."""
    local_step = checkpoint_step(adapter, MERGE_FILES)
    local_ok = all((adapter / name).exists() for name in ADAPTER_FILES)  # e.g. an adapter made without state.json
    if repo:
        from huggingface_hub import snapshot_download

        sub = f"checkpoints/{adapter.parent.name}/{adapter.name}"
        with tempfile.TemporaryDirectory() as tmp:  # the adapter, its state and logs; not the optimizer state
            snapshot_download(repo, allow_patterns=[f"{sub}/*"], ignore_patterns=["*.pt"], local_dir=tmp)
            hub = Path(tmp) / sub
            hub_step = checkpoint_step(hub, MERGE_FILES)
            if hub_step is not None and (local_step is None or hub_step >= local_step):
                shutil.rmtree(adapter, ignore_errors=True)
                shutil.copytree(hub, adapter)
                print(f"adapter source: {repo} {sub} (step {hub_step})"
                      + ("" if local_ok else f", the local {adapter} was missing or incomplete"), flush=True)
                return
    if not local_ok:
        raise SystemExit(f"no complete adapter at {adapter}" + (f" nor in {repo} {sub}" if repo else
                                                                "; pass --adapter-repo to download it"))
    print(f"adapter source: local {adapter}" + (f" (step {local_step})" if local_step is not None else ""), flush=True)


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, required=True, help="checkpoints/<run>/last")
    ap.add_argument("--adapter-repo", default=None, help="HF repo holding checkpoints/<run>/last (used unless the local one is complete and newer)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hf-repo", default=None, help="private HF model repo for the merged model")
    ap.add_argument("--model", default=None, help="base model (default: the adapter's base_model_name_or_path)")
    args = ap.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolve_adapter(args.adapter, args.adapter_repo)
    base = args.model or json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))[
        "base_model_name_or_path"]
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(args.adapter)).merge_and_unload()
    model = model.to(torch.bfloat16)  # the adapter is fp32; keep the merged weights in the base dtype
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out), safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(str(args.out))
    for name in ("train_log.jsonl", "run_config.json", "loss.png"):  # how the model was made, next to it
        if (args.adapter / name).exists():
            shutil.copy(args.adapter / name, args.out / name)
    print(f"merged {args.adapter} into {base} -> {args.out}", flush=True)
    if args.hf_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.hf_repo, private=True, exist_ok=True)
        api.upload_folder(repo_id=args.hf_repo, folder_path=str(args.out), commit_message=f"merged {args.adapter}")
        print(f"-> {args.hf_repo}", flush=True)
    return args.out


if __name__ == "__main__":
    main()
