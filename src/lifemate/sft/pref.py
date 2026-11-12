"""Preference pairs as token ids, the log-probability of a step, and the DPO + NLL ("RPO") loss.

Development only (training); never imported by agents/, infer.py or docker_entry.py.
A pair (lifemate.sft.pairs) is a conversation prefix and two candidate next assistant turns, chosen (gold) and rejected
(the model's own wrong step). Both are rendered exactly as encode_record renders a labelled turn: the prefix with the
generation prompt, then the turn's extra tokens.
"""
from __future__ import annotations

from .tokenize import MAX_LEN, render


def encode_pair(pair: dict, tokenizer, max_len: int = MAX_LEN) -> dict | None:
    """{"prefix", "chosen", "rejected"} token ids; None when prefix + the longer step exceeds max_len (skipped, never
    cut). ValueError when the chat template does not render the prefix as a prefix of a step."""
    messages, tools = pair["messages"], pair["tools"]
    prefix = render(tokenizer, messages, tools, add_generation_prompt=True)
    steps = {}
    for side in ("chosen", "rejected"):
        full = render(tokenizer, messages + [pair[side]], tools)
        if full[: len(prefix)] != prefix:
            raise ValueError(f"{pair.get('id')}: chat template does not render the prefix of the {side} step as a prefix")
        steps[side] = full[len(prefix):]
    if len(prefix) + max(len(steps["chosen"]), len(steps["rejected"])) > max_len:
        return None
    return {"prefix": prefix, **steps}


def step_logps(model, prefix: list[int], step: list[int], device):
    """(sum of the log-probabilities of the step tokens, mean NLL of the step tokens) under model; logits are computed
    only at the positions that predict the step."""
    import torch

    ids = torch.tensor([prefix + step], device=device)
    pos = torch.arange(len(prefix) - 1, len(prefix) + len(step) - 1, device=device)
    with torch.autocast(device if isinstance(device, str) else device.type, dtype=torch.bfloat16,
                        enabled=(device if isinstance(device, str) else device.type) == "cuda"):
        logits = model(input_ids=ids, logits_to_keep=pos).logits[0].float()
    token_logps = logits.log_softmax(-1).gather(1, torch.tensor(step, device=device)[:, None])[:, 0]
    total = token_logps.sum()
    return total, -total / len(step)


def rpo_loss(pc, pr, rc, rr, nll_c, beta: float, sft_weight: float):
    """(loss, margin): margin = beta * ((pc - rc) - (pr - rr)), loss = -logsigmoid(margin) + sft_weight * nll_c.
    pc/pr: policy log-probs of the chosen/rejected step; rc/rr: the reference's; nll_c: mean NLL of the chosen step."""
    import torch.nn.functional as F

    margin = beta * ((pc - rc) - (pr - rr))
    return -F.logsigmoid(margin) + sft_weight * nll_c, margin
