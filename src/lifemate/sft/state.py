"""Where training is in the data, kept in the checkpoint so a resumed run continues the same order: no example is
repeated or skipped. Pure Python, so it is tested without torch."""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


def epoch_order(n: int, epoch: int, seed: int) -> list[int]:
    order = list(range(n))
    random.Random(seed * 1000 + epoch).shuffle(order)
    return order


@dataclass
class TrainState:
    epoch: int  # current epoch
    pos: int    # index into that epoch's order of the first micro-batch not yet yielded
    step: int   # optimizer steps done
    seed: int

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "TrainState":
        return cls(**json.loads(text))


def batches(n: int, state: TrainState, micro_bs: int, epochs: int) -> Iterator[list[int]]:
    """Micro-batches of example indices from (state.epoch, state.pos) to the end of the last epoch. state advances
    before each yield, so a checkpoint saved after the optimizer step that used a batch resumes after it."""
    while state.epoch < epochs:
        if state.pos >= n:
            state.epoch, state.pos = state.epoch + 1, 0
            continue
        batch = epoch_order(n, state.epoch, state.seed)[state.pos: state.pos + micro_bs]
        state.pos += len(batch)
        yield batch


def total_steps(n: int, micro_bs: int, epochs: int, grad_accum: int) -> int:
    """Optimizer steps of a full run; a last partial accumulation still makes a step."""
    return math.ceil(epochs * math.ceil(n / micro_bs) / grad_accum)


def truncate_log(path: Path, step: int) -> None:
    """Drop log lines written after the checkpoint being resumed (those steps run again)."""
    if not path.exists():
        return
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    path.write_text("".join(line + "\n" for line in lines if json.loads(line)["step"] <= step), encoding="utf-8")
