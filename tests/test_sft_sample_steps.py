"""The step sampler with the model replaced by fakes (CPU, no download); one test runs the real sampling on a tiny model."""
import ast
import json
import zlib
from pathlib import Path

import pytest

from lifemate.sft import sample_steps
from lifemate.splits import load_splits

SRC = Path(__file__).parent.parent / "src" / "lifemate"


class FakeTokenizer:
    def apply_chat_template(self, messages, tools=None, tokenize=True, add_generation_prompt=False):
        return list(range(sum(len(str(m.get("content") or "")) + 10 for m in messages)))


def _record(sample_id, source="masked"):
    call = {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "orbit.make", "arguments": {"stamp": "2026-11-06T20:30:00+07:00"}}}]}
    return {"id": sample_id, "source": source, "name_map": {"orbit.make": "alarm.create"}, "tools": [],
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "đặt báo thức"}, call,
                         {"role": "tool", "content": "{}"}, {"role": "assistant", "content": "Đã xong."}]}


def _write(tmp_path, records, name="rec"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                                            encoding="utf-8")


def _fake(monkeypatch, fail_after=None):
    calls = {"n": 0}

    def fake_sample(model, tokenizer, messages, tools, k, temperature, top_p, seed, max_new_tokens=384):
        calls["n"] += 1
        if fail_after is not None and calls["n"] > fail_after:
            raise RuntimeError("session died")
        return f"greedy-{len(messages)}", [f"s{seed % 97}-{j}" for j in range(k)]

    monkeypatch.setattr(sample_steps, "load_model", lambda model, revision: (None, FakeTokenizer()))
    monkeypatch.setattr(sample_steps, "sample_texts", fake_sample)


def _run(tmp_path, *extra):
    return sample_steps.main(["--records", "rec", "--name", "st", "--model", "m", "--revision", "r", "--k", "2",
                              "--records-dir", str(tmp_path), *extra])


def _rows(tmp_path):
    return [json.loads(line) for line in (tmp_path / "steps" / "st.jsonl").read_text(encoding="utf-8").splitlines()]


POOL = load_splits()["pool"]


def test_decision_points_are_the_agent_turns():
    assert sample_steps.decision_points(_record("x")) == [2, 4]


def test_point_seed_is_stable():
    assert sample_steps.point_seed(0, "a", 3, 5) == zlib.crc32(b"0:a:3:5")


def test_refuses_records_outside_the_pool(tmp_path, monkeypatch):
    _fake(monkeypatch)
    _write(tmp_path, [_record(POOL[0]), _record(load_splits()["holdout"][0])])
    with pytest.raises(SystemExit):
        _run(tmp_path)
    assert not (tmp_path / "steps" / "st.jsonl").exists()


def test_writes_one_row_per_point_of_the_masked_records(tmp_path, monkeypatch):
    _fake(monkeypatch)
    _write(tmp_path, [_record(POOL[0], source="gold"), _record(POOL[0]), _record(POOL[1])])
    _run(tmp_path)
    rows = _rows(tmp_path)
    assert [(r["record"], r["turn"]) for r in rows] == [(1, 2), (1, 4), (2, 2), (2, 4)]
    first = rows[0]
    assert first["id"] == POOL[0] and first["records"] == "rec" and len(first["samples"]) == 2
    assert first["seed"] == sample_steps.point_seed(0, "rec", 1, 2) and first["model"] == "m" and first["revision"] == "r"


def test_limit_keeps_the_first_sample_ids(tmp_path, monkeypatch):
    _fake(monkeypatch)
    _write(tmp_path, [_record(POOL[0]), _record(POOL[1]), _record(POOL[0])])
    _run(tmp_path, "--limit", "1")
    assert {r["record"] for r in _rows(tmp_path)} == {0, 2}


def test_skips_and_counts_long_prefixes(tmp_path, monkeypatch, capsys):
    _fake(monkeypatch)
    _write(tmp_path, [_record(POOL[0])])
    _run(tmp_path, "--max-len", "20")
    assert not (tmp_path / "steps" / "st.jsonl").exists() or _rows(tmp_path) == []
    assert "2 points over --max-len skipped" in capsys.readouterr().out


def test_resume_gives_the_rows_of_an_uninterrupted_run(tmp_path, monkeypatch):
    records = [_record(POOL[0]), _record(POOL[1])]
    clean = tmp_path / "clean"
    _write(clean, records)
    _fake(monkeypatch)
    _run(clean)
    broken = tmp_path / "broken"
    _write(broken, records)
    _fake(monkeypatch, fail_after=2)
    with pytest.raises(RuntimeError):
        _run(broken)
    _fake(monkeypatch)
    _run(broken)
    assert _rows(broken) == _rows(clean)


def test_inference_never_imports_the_preference_modules():
    files = [*(SRC / "agents").glob("*.py"), SRC / "infer.py", SRC / "docker_entry.py"]
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
                assert not any(n.split(".")[-1] in ("sample_steps", "pairs", "pref") for n in names), path


def test_sample_texts_on_a_tiny_model():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", local_files_only=True)
    except OSError:
        pytest.skip("tokenizer not in the local HF cache")
    cfg = transformers.Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=1, head_dim=16)
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(cfg).eval()
    messages = [{"role": "user", "content": "xin chào"}]
    a = sample_steps.sample_texts(model, tok, messages, [], 3, 1.0, 0.95, seed=5, max_new_tokens=8)
    b = sample_steps.sample_texts(model, tok, messages, [], 3, 1.0, 0.95, seed=5, max_new_tokens=8)
    assert a == b and isinstance(a[0], str) and len(a[1]) == 3


def test_prints_the_total_and_progress_lines(tmp_path, monkeypatch, capsys):
    _fake(monkeypatch)
    _write(tmp_path, [_record(POOL[0]), _record(POOL[1])])
    _run(tmp_path, "--progress-every", "1")
    out = capsys.readouterr().out
    assert "4 points to sample (0 already done)" in out
    assert "progress 1/4 points" in out and "progress 4/4 points" in out


def test_progress_counts_points_already_done_after_a_resume(tmp_path, monkeypatch, capsys):
    _write(tmp_path, [_record(POOL[0]), _record(POOL[1])])
    _fake(monkeypatch, fail_after=2)
    with pytest.raises(RuntimeError):
        _run(tmp_path, "--progress-every", "1")
    capsys.readouterr()
    _fake(monkeypatch)
    _run(tmp_path, "--progress-every", "1")
    out = capsys.readouterr().out
    assert "4 points to sample (2 already done)" in out and "progress 3/4 points" in out


@pytest.mark.parametrize("change", [["--revision", "r2"], ["--temperature", "1.0"], ["--k", "3"], ["--model", "other"]])
def test_resume_refuses_rows_made_with_other_settings(tmp_path, monkeypatch, change):
    _fake(monkeypatch)
    _write(tmp_path, [_record(POOL[0])])
    _run(tmp_path)
    before = _rows(tmp_path)
    with pytest.raises(SystemExit, match="other settings"):
        _run(tmp_path, *change)
    assert _rows(tmp_path) == before


def test_resume_refuses_rows_whose_record_changed(tmp_path, monkeypatch):
    _fake(monkeypatch)
    _write(tmp_path, [_record(POOL[0])])
    _run(tmp_path)
    _write(tmp_path, [_record(POOL[1])])  # rebuilt records: index 0 is now another sample
    with pytest.raises(SystemExit, match="record"):
        _run(tmp_path)
