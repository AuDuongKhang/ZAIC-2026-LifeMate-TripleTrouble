"""Convert a multimodal Qwen3.5 checkpoint into a text-only causal LM that the rest of the pipeline loads unchanged.

    python -m lifemate.sft.text_only --model Qwen/Qwen3.5-4B --revision <sha> \
        --chat-template configs/chat_templates/qwen_json_nothink.jinja --out merged/qwen35-4b-text \
        --hf-repo TripleTrouble/qwen35-4b-text

Qwen3.5 ships as a conditional-generation model: the text weights are `model.language_model.*`, next to `model.visual.*`
(and an MTP head the HF class does not load). AutoModelForCausalLM on that checkpoint builds a text model with other
parameter names, does not raise, and leaves every weight random. This loads the checkpoint with
AutoModelForImageTextToText, copies `model.language_model.*` and `lm_head` into the matching causal-LM class (vision and
MTP are dropped, which also takes them out of the parameter count the competition limits), stores the chat template in
the tokenizer and prints the parameter count. Development only: nothing at inference imports it.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

LANGUAGE_PREFIX = "model.language_model."


def convert_model(model):
    """A causal LM of the text part of `model` (a *ForConditionalGeneration), with its weights, in its dtype."""
    from transformers import AutoModelForCausalLM

    text_config = copy.deepcopy(model.config.text_config)
    # The flag lives on the composite config; the text config does not inherit it, and an untied copy would carry a second
    # vocab x hidden matrix (about 0.6B parameters for Qwen3.5-4B) that the competition counts.
    text_config.tie_word_embeddings = bool(getattr(model.config, "tie_word_embeddings", False)
                                           or getattr(model.config.text_config, "tie_word_embeddings", False))
    dtype = next(model.parameters()).dtype
    target = AutoModelForCausalLM.from_config(text_config, dtype=dtype)
    state = {}
    for key, value in model.state_dict().items():
        if key.startswith(LANGUAGE_PREFIX):
            state["model." + key[len(LANGUAGE_PREFIX):]] = value
        elif key == "lm_head.weight":
            state[key] = value
    missing, unexpected = target.load_state_dict(state, strict=False)
    target.tie_weights()
    # The end-of-turn tokens live in the generation config; the default one built from the text config stops nowhere.
    target.generation_config = copy.deepcopy(model.generation_config)
    tied = {"lm_head.weight"} if getattr(text_config, "tie_word_embeddings", False) else set()
    if set(missing) - tied or unexpected:
        raise RuntimeError(f"text-only conversion: missing {sorted(set(missing) - tied)[:5]}, unexpected {unexpected[:5]}")
    return target


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF id of the multimodal checkpoint")
    ap.add_argument("--revision", default=None, help="commit of --model (pin it)")
    ap.add_argument("--chat-template", type=Path, required=True, help="Jinja file stored in the tokenizer")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hf-repo", default=None, help="private HF model repo to upload to (HF_TOKEN with write access)")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from ..agents.llm import load_causal_lm

    source = AutoModelForImageTextToText.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16)
    text = convert_model(source)
    total = sum(p.numel() for p in source.parameters())
    del source
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tokenizer.chat_template = args.chat_template.read_text(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    text.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    params = sum(p.numel() for p in text.parameters())
    del text
    load_causal_lm(str(args.out), torch.bfloat16)  # the saved folder has to load with no missing weight
    print(f"{args.model}: {total / 1e9:.3f}B parameters -> {params / 1e9:.3f}B text-only, saved to {args.out}", flush=True)
    if args.hf_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.hf_repo, private=True, exist_ok=True)
        api.upload_folder(repo_id=args.hf_repo, folder_path=str(args.out))
        print(f"uploaded to {args.hf_repo} (private)", flush=True)
    return args.out


if __name__ == "__main__":
    main()
