"""Logit distillation: soft targets, their loss and the report. Pure tensors, no model."""
import pytest

torch = pytest.importorskip("torch")

from lifemate.sft.kd import kd_sum, merge_targets, soft_targets, target_report  # noqa: E402


def _logits():
    torch.manual_seed(0)
    return torch.randn(5, 50)


def test_soft_targets_keep_the_largest_probabilities():
    logits = _logits()
    ids, probs = soft_targets(logits, temp=2.0, top_k=4)
    full = torch.softmax(logits / 2.0, -1)
    assert ids.shape == probs.shape == (5, 4)
    assert torch.allclose(probs, full.gather(-1, ids))
    assert torch.equal(ids[:, 0], full.argmax(-1))


def test_kd_is_zero_when_student_equals_teacher():
    logits = _logits()
    ids, probs = soft_targets(logits, temp=2.0, top_k=4)
    assert kd_sum(logits, ids, probs, temp=2.0).item() == pytest.approx(0.0, abs=1e-4)


def test_kd_is_positive_when_they_differ_and_scales_with_temperature_squared():
    teacher, student = _logits(), _logits().flip(0)
    ids, probs = soft_targets(teacher, temp=2.0, top_k=4)
    loss = kd_sum(student, ids, probs, temp=2.0)
    assert loss.item() > 0
    log_q = torch.log_softmax(student / 2.0, -1).gather(-1, ids)
    tail_p, tail_q = 1 - probs.sum(-1), 1 - log_q.exp().sum(-1)
    kl = (probs * (probs.log() - log_q)).sum(-1) + tail_p * (tail_p.log() - tail_q.log())
    assert loss.item() == pytest.approx((kl.sum() * 4).item(), rel=1e-4)


def test_kd_gradient_moves_the_student_towards_the_teacher():
    teacher = _logits()
    ids, probs = soft_targets(teacher, temp=2.0, top_k=4)
    student = torch.zeros_like(teacher, requires_grad=True)
    optimizer = torch.optim.SGD([student], lr=0.5)
    before = kd_sum(student, ids, probs, temp=2.0).item()
    for _ in range(50):
        optimizer.zero_grad()
        kd_sum(student, ids, probs, temp=2.0).backward()
        optimizer.step()
    assert kd_sum(student, ids, probs, temp=2.0).item() < before / 2


def test_kd_stays_finite_on_a_confident_student_and_teacher():
    logits = torch.zeros(1, 50)
    logits[0, 3] = 200.0
    ids, probs = soft_targets(logits, temp=1.0, top_k=4)
    student = logits.clone().requires_grad_()
    loss = kd_sum(student, ids, probs, temp=1.0)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(student.grad).all()


def test_merge_averages_teachers_and_counts_a_missing_token_as_zero():
    a = (torch.tensor([[1, 2]]), torch.tensor([[0.6, 0.4]]))
    b = (torch.tensor([[2, 3]]), torch.tensor([[0.8, 0.2]]))
    ids, probs = merge_targets([a, b], top_k=2)
    assert ids.tolist() == [[2, 1]]
    assert probs[0].tolist() == pytest.approx([0.6, 0.3])


def test_merge_of_one_teacher_is_unchanged():
    ids, probs = soft_targets(_logits(), temp=1.0, top_k=4)
    merged_ids, merged_probs = merge_targets([(ids, probs)], top_k=4)
    assert torch.equal(merged_ids, ids) and torch.allclose(merged_probs, probs)


def test_report_counts_soft_positions_and_disagreement_with_gold():
    ids = torch.tensor([[7, 1], [8, 2], [9, 3]])
    probs = torch.tensor([[0.99, 0.01], [0.6, 0.3], [0.95, 0.05]])
    gold = torch.tensor([7, 2, 9])
    r = target_report([(ids, probs)], [gold])
    assert r["positions"] == 3
    assert r["share_top1_below_0.9"] == pytest.approx(1 / 3, abs=1e-4)  # the report rounds to 5 places
    assert r["share_top1_not_gold"] == pytest.approx(1 / 3, abs=1e-4)
    assert r["mean_entropy"] > 0
