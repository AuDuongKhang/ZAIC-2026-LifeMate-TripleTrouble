"""Merge a LoRA adapter from lifemate.sft.train into bf16 base weights, so inference stays on HFChatLLM without peft.

    python -m lifemate.sft.merge --adapter checkpoints/sft-v1/last --out merged/sft-v1 --hf-repo <owner>/lifemate-sft-v1
    python -m lifemate.infer --split train --subset dev --agent react --agent-kw model=<owner>/lifemate-sft-v1 ...

Without a local --adapter folder, --adapter-repo downloads checkpoints/<run>/last from that HF repo (run = the
folder above "last"). The merged model and its tokenizer go to --out and, with --hf-repo, to the root of that private
repo (token: HF_TOKEN, write).
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, required=True, help="checkpoints/<run>/last")
    ap.add_argument("--adapter-repo", default=None, help="HF repo holding checkpoints/<run>/last, if not local")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hf-repo", default=None, help="private HF model repo for the merged model")
    ap.add_argument("--model", default=None, help="base model (default: the adapter's base_model_name_or_path)")
    ap.add_argument("--revision", default=None, help="commit sha of the base model (and of its tokenizer)")
    args = ap.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    from ..agents.llm import load_causal_lm

    if not args.adapter.exists():
        if not args.adapter_repo:
            raise SystemExit(f"no adapter at {args.adapter}; pass --adapter-repo to download it")
        from huggingface_hub import snapshot_download

        sub = f"checkpoints/{args.adapter.parent.name}/{args.adapter.name}"
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_download(args.adapter_repo, allow_patterns=[f"{sub}/*"], local_dir=tmp)
            if not (Path(tmp) / sub).exists():
                raise SystemExit(f"{args.adapter_repo} has no {sub}")
            shutil.copytree(Path(tmp) / sub, args.adapter)
    base = args.model or json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))[
        "base_model_name_or_path"]
    model = load_causal_lm(base, torch.bfloat16, args.revision)
    model = PeftModel.from_pretrained(model, str(args.adapter)).merge_and_unload()
    model = model.to(torch.bfloat16)  # the adapter is fp32; keep the merged weights in the base dtype
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out), safe_serialization=True)
    AutoTokenizer.from_pretrained(base, revision=args.revision).save_pretrained(str(args.out))
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
