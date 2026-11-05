"""The sentence writer CLI: output format, resume, HF copy and upload. A fake LLM stands in for the teacher."""
import json
import random
import shutil

import pytest

from lifemate.evaluate import label_kind
from lifemate.sft import thoughts
from lifemate.sft.thoughts import load_thoughts, step_calls
from lifemate.splits import load_splits

SENTENCE = "Cần làm việc này trước."


class FixedLLM:
    """Always the same valid sentence; the first `fail_first` generate calls return an empty text instead."""

    def __init__(self, fail_first=0):
        self.calls, self.fail_first, self.seen = 0, fail_first, []

    def generate(self, messages):
        self.calls += 1
        self.seen.append(messages)
        return "" if self.calls <= self.fail_first else SENTENCE


def _run(tmp_path, llm, *extra, name="t", subset="pool", limit=3, revision="r"):
    argv = ["--name", name, "--subset", subset, "--revision", revision, "--out-dir", str(tmp_path)]
    if limit:
        argv += ["--limit", str(limit)]
    return thoughts.main(argv + list(extra), llm=llm)


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _subset(train_samples, name):
    ids = set(load_splits()[name])
    return [s for s in train_samples if s["id"] in ids]


def test_writes_one_line_per_sample_in_the_spec_format(train_samples, tmp_path):
    out = _run(tmp_path, FixedLLM())
    rows = _rows(out)
    assert out == tmp_path / "t.jsonl"
    assert all(set(r) == {"id", "teacher", "revision", "steps"} for r in rows) and len(rows) == 3
    assert all(r["teacher"] == "Qwen/Qwen3.5-27B" and r["revision"] == "r" for r in rows)
    assert all(s["thought"] == SENTENCE for r in rows for s in r["steps"])


def test_step_counts_match_the_design(train_samples):
    def steps(name):
        return sum(len(step_calls(s)) for s in _subset(train_samples, name))
    assert steps("pool") == 2271 and steps("dev") == 725 and steps("holdout") == 656


def test_limit_draws_a_mix_of_kinds(train_samples, tmp_path):
    rows = _rows(_run(tmp_path, FixedLLM(), limit=100))
    by_id = {s["id"]: s for s in train_samples}
    assert {label_kind(by_id[r["id"]]["label"]["tool_calls"]) for r in rows} == {"act", "ask", "empty"}


def test_resume_skips_written_and_failed_samples(train_samples, tmp_path):
    first = FixedLLM(fail_first=2)  # the first sample fails both attempts, the other two are written
    out = _run(tmp_path, first)
    assert len(_rows(out)) == 2 and len(_rows(tmp_path / "t.failed.jsonl")) == 1
    before = out.read_text(encoding="utf-8")
    second = FixedLLM()
    _run(tmp_path, second)
    assert second.calls == 0 and out.read_text(encoding="utf-8") == before


def test_resume_trims_a_torn_last_line(train_samples, tmp_path):
    out = _run(tmp_path, FixedLLM())
    rows = _rows(out)
    torn_id = rows[-1]["id"]
    kept = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[:-1])
    out.write_text(kept + '{"id": "' + torn_id + '", "tea', encoding="utf-8")
    llm = FixedLLM()
    _run(tmp_path, llm)
    again = _rows(out)  # every line parses
    assert [r["id"] for r in again] == [r["id"] for r in rows] and llm.calls == len(rows[-1]["steps"])
    assert [r["id"] for r in again].count(torn_id) == 1


def test_load_thoughts_ignores_a_torn_last_line(tmp_path):
    path = tmp_path / "th.jsonl"
    path.write_text('{"id": "a", "steps": []}\n{"id": "b", "ste', encoding="utf-8")
    assert list(load_thoughts(path)) == ["a"]


def test_resume_refuses_another_teacher_or_revision(train_samples, tmp_path):
    out = _run(tmp_path, FixedLLM())
    before = out.read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="revision"):
        _run(tmp_path, FixedLLM(), revision="other")
    with pytest.raises(SystemExit, match="teacher"):
        _run(tmp_path, FixedLLM(), "--teacher", "someone/else")
    assert out.read_text(encoding="utf-8") == before


class FakeApi:
    uploads = []

    def upload_file(self, **kw):
        type(self).uploads.append(kw)


@pytest.fixture
def fake_api(monkeypatch):
    import huggingface_hub

    FakeApi.uploads = []
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    return FakeApi


def _main_uploads(api):
    return [u for u in api.uploads if u["path_in_repo"] == "thoughts/t.jsonl"]


def test_a_new_session_continues_from_the_hf_copy(train_samples, tmp_path, monkeypatch, fake_api):
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError

    done = random.Random(0).sample(_subset(train_samples, "pool"), 3)[0]  # the first sample the run would write
    asked = []

    def fake_download(repo_id, filename, repo_type=None, local_dir=None, **kw):
        asked.append((repo_id, filename, repo_type))
        if filename.endswith(".failed.jsonl"):
            raise EntryNotFoundError("no such file")
        path = tmp_path / "hf" / "t.jsonl"
        path.parent.mkdir(exist_ok=True)
        row = {"id": done["id"], "teacher": "Qwen/Qwen3.5-27B", "revision": "r",
               "steps": [{"call": c["name"] if c else None, "thought": "Từ HF."} for c in step_calls(done)]}
        path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    llm = FixedLLM()
    out = _run(tmp_path / "out", llm, "--hf-repo", "me/data")
    assert ("me/data", "thoughts/t.jsonl", "dataset") in asked
    assert not any(done["audio_text"] in m["content"] for msgs in llm.seen for m in msgs)
    assert [r["id"] for r in _rows(out)].count(done["id"]) == 1 and len(_rows(out)) == 3
    assert _rows(out)[0]["steps"][0]["thought"] == "Từ HF."


def test_a_new_run_starts_empty_when_hf_has_no_file(train_samples, tmp_path, monkeypatch, fake_api):
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError

    def missing(*a, **kw):
        raise EntryNotFoundError("no such file")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    assert len(_rows(_run(tmp_path, FixedLLM(), "--hf-repo", "me/data"))) == 3


def test_another_hf_error_stops_the_run(train_samples, tmp_path, monkeypatch, fake_api):
    import huggingface_hub

    def broken(*a, **kw):
        raise OSError("network down")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", broken)
    with pytest.raises(OSError, match="network down"):
        _run(tmp_path, FixedLLM(), "--hf-repo", "me/data")
    assert fake_api.uploads == [] and not (tmp_path / "t.jsonl").exists()


def test_being_offline_is_not_a_missing_file(train_samples, tmp_path, monkeypatch, fake_api):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    def offline(*a, **kw):
        raise LocalEntryNotFoundError("cannot reach the hub")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    with pytest.raises(LocalEntryNotFoundError):
        _run(tmp_path, FixedLLM(), "--hf-repo", "me/data")


def test_uploads_every_n_samples_and_at_the_end(train_samples, tmp_path, monkeypatch, fake_api):
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: (_ for _ in ()).throw(EntryNotFoundError("x")))
    seen_rows = []
    real_upload = fake_api.upload_file

    def recording(self, **kw):
        seen_rows.append(len(_rows(tmp_path / "t.jsonl")))
        real_upload(self, **kw)

    monkeypatch.setattr(fake_api, "upload_file", recording)
    _run(tmp_path, FixedLLM(), "--hf-repo", "me/data", "--upload-every", "2")
    assert seen_rows == [2, 3]
    main = _main_uploads(fake_api)
    assert len(main) == 2 and main[0]["repo_id"] == "me/data" and main[0]["repo_type"] == "dataset"


def test_a_failed_upload_does_not_stop_the_run(train_samples, tmp_path, monkeypatch, fake_api, capsys):
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: (_ for _ in ()).throw(EntryNotFoundError("x")))

    def boom(self, **kw):
        raise RuntimeError("quota")

    monkeypatch.setattr(fake_api, "upload_file", boom)
    out = _run(tmp_path, FixedLLM(), "--hf-repo", "me/data", "--upload-every", "1")
    assert len(_rows(out)) == 3 and "upload" in capsys.readouterr().out


def test_there_is_no_split_option():
    with pytest.raises(SystemExit):
        thoughts.main(["--name", "x", "--subset", "pool", "--revision", "r", "--split", "public_test"])


def test_the_completion_is_cut_to_the_sentence():
    assert thoughts.clean_completion("<think>\n\n</think>\n\nCâu một.<|im_end|>", {"<|im_end|>"}) == "Câu một."
    assert thoughts.clean_completion("lý do dài</think>Câu hai.", set()) == "Câu hai."


def test_prints_the_pilot_numbers(train_samples, tmp_path, capsys):
    _run(tmp_path, FixedLLM())
    out = capsys.readouterr().out
    assert "pass rate" in out and "hours for 3652 steps" in out and "kinds" in out


def test_a_multimodal_checkpoint_is_loaded_with_the_class_it_was_saved_from():
    # Qwen3.5-27B ships as Qwen3_5ForConditionalGeneration (weights under model.language_model.*): the causal-LM auto
    # class would build a model with different parameter names instead of raising ValueError
    assert thoughts.teacher_loader(["Qwen3_5ForConditionalGeneration"]) == "AutoModelForImageTextToText"
    assert thoughts.teacher_loader(["Qwen3ForCausalLM"]) == "AutoModelForCausalLM"
    assert thoughts.teacher_loader(None) == "AutoModelForCausalLM"
