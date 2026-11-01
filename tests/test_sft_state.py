import json

import pytest

from lifemate.sft.plots import plot_log
from lifemate.sft.state import TrainState, batches, epoch_order, total_steps, truncate_log


def test_resume_continues_the_same_order():
    full = list(batches(10, TrainState(epoch=0, pos=0, step=0, seed=0), micro_bs=1, epochs=2))
    assert len(full) == 20
    st = TrainState(epoch=0, pos=0, step=0, seed=0)
    first = []
    for b in batches(10, st, micro_bs=1, epochs=2):
        first.append(b)
        if len(first) == 13:  # stop inside epoch 1
            break
    resumed = TrainState.from_json(st.to_json())
    assert first + list(batches(10, resumed, micro_bs=1, epochs=2)) == full


def test_resume_at_an_epoch_boundary():
    full = list(batches(6, TrainState(epoch=0, pos=0, step=0, seed=1), micro_bs=2, epochs=2))
    st = TrainState(epoch=0, pos=0, step=0, seed=1)
    it = batches(6, st, micro_bs=2, epochs=2)
    first = [next(it) for _ in range(3)]  # exactly one epoch
    assert first + list(batches(6, TrainState.from_json(st.to_json()), micro_bs=2, epochs=2)) == full


def test_every_example_once_per_epoch():
    out = [i for b in batches(7, TrainState(epoch=0, pos=0, step=0, seed=0), micro_bs=3, epochs=1) for i in b]
    assert sorted(out) == list(range(7))


def test_epochs_use_different_orders():
    assert epoch_order(50, 0, 0) != epoch_order(50, 1, 0) and sorted(epoch_order(50, 1, 0)) == list(range(50))
    assert epoch_order(50, 0, 0) == epoch_order(50, 0, 0) and epoch_order(50, 0, 0) != epoch_order(50, 0, 1)


def test_total_steps_counts_the_last_partial_step():
    assert total_steps(n=948, micro_bs=1, epochs=2, grad_accum=16) == 119
    assert total_steps(n=32, micro_bs=1, epochs=1, grad_accum=16) == 2


def test_truncate_log_drops_steps_after_the_checkpoint(tmp_path):
    log = tmp_path / "train_log.jsonl"
    log.write_text("".join(json.dumps({"step": s, "loss": 1.0}) + "\n" for s in range(1, 8)), encoding="utf-8")
    truncate_log(log, step=4)
    assert [json.loads(line)["step"] for line in log.read_text(encoding="utf-8").splitlines()] == [1, 2, 3, 4]


def test_plot_log_writes_pngs(tmp_path):
    pytest.importorskip("matplotlib")
    log = tmp_path / "train_log.jsonl"
    rows = [{"step": s, "epoch": s / 5, "loss": 2.0 / s, "lr": 1e-4, "grad_norm": 0.5, "tokens": 100, "seconds": s}
            for s in range(1, 6)]
    rows[-1]["eval_loss"] = 0.7
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    plot_log(log, tmp_path)
    assert all((tmp_path / f).exists() for f in ("loss.png", "lr.png", "grad_norm.png"))
