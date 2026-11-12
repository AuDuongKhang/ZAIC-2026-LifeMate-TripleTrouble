"""WiSE-FT: interpolate the weights of a fine-tuned model with its base, theta = alpha * a + (1 - alpha) * b.

    python -m lifemate.sft.interpolate --a TripleTrouble/lifemate-sft-q1c --a-revision <sha> \
        --b TripleTrouble/qwen35-4b-text --b-revision <sha> --alpha 0.85 --out merged/sft-q1c-wise85 \
        --hf-repo TripleTrouble/lifemate-sft-q1c-wise85

Both models must have the same parameter names and shapes (a fine-tune and its base do). The result keeps a's config,
tokenizer and chat template, is saved in bf16 and, with --hf-repo, uploaded to that private repo (token: HF_TOKEN).
"""
from __future__ import annotations

import argparse
from pathlib import Path


def interpolate_state(a: dict, b: dict, alpha: float) -> dict:
    """alpha * a + (1 - alpha) * b for floating tensors, a's value for the others. ValueError naming the first key
    whose presence or shape differs."""
    for key in sorted(set(a) ^ set(b)):
        raise ValueError(f"parameter {key} is in only one of the two models")
    out = {}
    for key, va in a.items():
        vb = b[key]
        if va.shape != vb.shape:
            raise ValueError(f"parameter {key}: shape {tuple(va.shape)} vs {tuple(vb.shape)}")
        out[key] = (alpha * va.float() + (1 - alpha) * vb.float()).to(va.dtype) if va.is_floating_point() else va
    return out


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="the fine-tuned model (config and tokenizer are taken from it)")
    ap.add_argument("--a-revision", default=None)
    ap.add_argument("--b", required=True, help="the base model")
    ap.add_argument("--b-revision", default=None)
    ap.add_argument("--alpha", type=float, required=True, help="weight of --a, in [0, 1]")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hf-repo", default=None)
    args = ap.parse_args(argv)
    if not 0 <= args.alpha <= 1:
        ap.error("--alpha is a weight in [0, 1]")

    import torch
    from transformers import AutoTokenizer

    from ..agents.llm import load_causal_lm

    model = load_causal_lm(args.a, torch.bfloat16, args.a_revision)
    base = load_causal_lm(args.b, torch.bfloat16, args.b_revision)
    model.load_state_dict(interpolate_state(model.state_dict(), base.state_dict(), args.alpha))
    del base
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out), safe_serialization=True)
    AutoTokenizer.from_pretrained(args.a, revision=args.a_revision).save_pretrained(str(args.out))
    print(f"{args.alpha} * {args.a} + {1 - args.alpha:.2f} * {args.b} -> {args.out}", flush=True)
    if args.hf_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.hf_repo, private=True, exist_ok=True)
        api.upload_folder(repo_id=args.hf_repo, folder_path=str(args.out),
                          commit_message=f"WiSE-FT alpha {args.alpha}: {args.a}@{args.a_revision} + {args.b}@{args.b_revision}")
        print(f"-> {args.hf_repo}", flush=True)
    return args.out


if __name__ == "__main__":
    main()
