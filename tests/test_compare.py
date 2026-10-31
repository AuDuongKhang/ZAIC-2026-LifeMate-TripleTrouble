import json

from lifemate.compare import compare, load_scores, verdict


def test_identical_runs_have_zero_interval():
    a = {f"s{i}": float(i % 2) for i in range(50)}
    r = compare(a, a)
    assert r["diff"] == 0 and r["ci_low"] == 0 == r["ci_high"] and r["same"] == 50
    assert verdict(r) == "NO CLEAR DIFFERENCE"


def test_clear_improvement_is_a_win():
    a = {f"s{i}": 0.0 for i in range(100)}
    b = {f"s{i}": 100.0 if i < 40 else 0.0 for i in range(100)}
    r = compare(a, b)
    assert r["diff"] == 40.0 and r["ci_low"] > 0 and r["better"] == 40 and verdict(r) == "WIN"
    assert verdict(compare(b, a)) == "LOSS"


def test_only_shared_ids_are_compared():
    r = compare({"x": 100.0, "y": 0.0}, {"y": 100.0, "z": 100.0})
    assert r["n"] == 1 and r["dropped"] == 2


def test_bootstrap_is_deterministic():
    a = {f"s{i}": float(i % 3) for i in range(30)}
    b = {f"s{i}": float(i % 5) for i in range(30)}
    assert compare(a, b, seed=0) == compare(a, b, seed=0)


def test_load_scores_reads_per_sample_on_0_100(tmp_path):
    (tmp_path / "per_sample.jsonl").write_text(json.dumps({"id": "a", "score": 0.5}) + "\n", encoding="utf-8")
    assert load_scores(tmp_path) == {"a": 50.0}
