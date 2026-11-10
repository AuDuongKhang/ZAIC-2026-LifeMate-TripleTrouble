"""Logit distillation for SFT: soft targets from one or more teacher models, kept as the top-k probabilities per
labelled token plus one tail bin, and the KL term of the student's loss.

    total = (1 - alpha) * cross-entropy + alpha * temp^2 * KL(teacher || student)

Teachers see the student's exact token ids, so they must share its vocabulary (train.py refuses another one). Several
teachers are averaged; a token outside one teacher's top-k counts as probability 0 for it. torch is imported inside each
function, so the module loads without it. Development only: nothing at inference imports this.
"""
from __future__ import annotations

EPS = 1e-12


def soft_targets(logits, temp: float, top_k: int):
    """(ids, probs), both (positions, top_k): the largest probabilities of softmax(logits / temp)."""
    import torch

    top = torch.softmax(logits.float() / temp, -1).topk(top_k, dim=-1)
    return top.indices, top.values


def merge_targets(targets: list[tuple], top_k: int):
    """Mean over the teachers of one record's (ids, probs), cut back to the top_k largest."""
    import torch

    if len(targets) == 1:
        return targets[0]
    size = int(max(ids.max() for ids, _ in targets)) + 1
    total = torch.zeros(targets[0][0].shape[0], size, dtype=torch.float32)
    for ids, probs in targets:
        total.scatter_add_(1, ids, probs)
    top = (total / len(targets)).topk(top_k, dim=-1)
    return top.indices, top.values


def kd_sum(student_logits, ids, probs, temp: float):
    """temp^2 * KL(teacher || student) summed over the positions, on the top-k tokens plus one tail bin."""
    import torch

    log_q = torch.log_softmax(student_logits.float() / temp, -1).gather(-1, ids)
    tail_p = (1 - probs.sum(-1)).clamp_min(0)
    tail_q = (1 - log_q.exp().sum(-1)).clamp_min(EPS)
    kl = (probs * (probs.clamp_min(EPS).log() - log_q)).sum(-1) + tail_p * (tail_p.clamp_min(EPS).log() - tail_q.log())
    return kl.sum() * temp ** 2


def target_report(targets: list[tuple], gold: list) -> dict:
    """How much the soft targets say beyond the hard labels. All shares are per labelled position."""
    import torch

    ids = torch.cat([t[0] for t in targets])
    probs = torch.cat([t[1] for t in targets])
    labels = torch.cat(list(gold)).to(ids.device)
    tail = (1 - probs.sum(-1)).clamp_min(0)
    entropy = -(probs * probs.clamp_min(EPS).log()).sum(-1) - tail * tail.clamp_min(EPS).log()
    return {"positions": int(ids.shape[0]), "mean_entropy": round(entropy.mean().item(), 5),
            "share_top1_below_0.9": round((probs[:, 0] < 0.9).float().mean().item(), 5),
            "share_top1_not_gold": round((ids[:, 0] != labels).float().mean().item(), 5)}
