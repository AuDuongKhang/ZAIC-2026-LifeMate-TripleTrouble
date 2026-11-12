"""ASR cache: transcribe once (final + history clips, resumable), replay with the cached backend, score with asr_report."""
import json
from pathlib import Path

import pytest

from lifemate import asr_report, transcribe
from lifemate.asr import BACKENDS
from lifemate.asr.base import ASR
from lifemate.asr.cached import CachedASR, cache_path, read_cache, sample_clips
from lifemate.data import default_data_root, load_samples

pytestmark = pytest.mark.skipif(not (default_data_root() / "train").is_dir(), reason="data/train not downloaded")


class FakeASR(ASR):
    name = "fake"
    calls = 0

    def transcribe(self, audio, sample):
        FakeASR.calls += 1
        return f"text of {audio.name}"


@pytest.fixture(scope="module")
def train_samples():
    return load_samples(default_data_root(), "train")


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("LIFEMATE_ASR_CACHE", str(tmp_path))
    return tmp_path


def write_cache(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_cached_replays_transcribed_text(tmp_cache, train_samples):
    s = train_samples[0]
    write_cache(tmp_cache / "train" / "c.jsonl", [{"audio": s["audio"], "text": "xin chào", "seconds": 0.5}])
    assert CachedASR("c", split="train").transcribe(Path("data/train") / s["audio"], s) == "xin chào"
    assert CachedASR("c").transcribe(Path("data/train") / s["audio"], s) == "xin chào"  # split taken from the path


def test_build_asr_passes_cache_name(tmp_cache):
    from lifemate.asr import build_asr

    assert build_asr("cached", name="c").cache_name == "c"


def test_cache_miss_fails_loudly(tmp_cache, train_samples):
    s = train_samples[0]
    write_cache(tmp_cache / "train" / "c.jsonl", [])
    with pytest.raises(KeyError, match=Path(s["audio"]).name):
        CachedASR("c", split="train").transcribe(Path("data/train") / s["audio"], s)


def test_transcribe_covers_history_clips_and_resumes(tmp_cache, train_samples, monkeypatch):
    monkeypatch.setitem(BACKENDS, "fake", FakeASR)
    picked = [s for s in train_samples if s["history"]][:2] + [s for s in train_samples if not s["history"]][:1]
    argv = ["--split", "train", "--asr", "fake", "--name", "f"]
    monkeypatch.setattr(transcribe, "load_samples", lambda root, split: picked)
    FakeASR.calls = 0
    transcribe.main(argv)
    first = FakeASR.calls
    transcribe.main(argv)
    rows = read_cache(cache_path("train", "f"))
    assert first == len(rows) == sum(len(sample_clips(s)) for s in picked) > len(picked)
    assert FakeASR.calls == first  # second run transcribed nothing


def test_report_scores_against_audio_text(tmp_cache, train_samples):
    s = next(x for x in train_samples if x["history"])
    user_turn = next(t for t in s["history"] if t["role"] == "user" and t.get("audio"))
    write_cache(cache_path("train", "g"), [{"audio": s["audio"], "text": s["audio_text"], "seconds": 1.0},
                                           {"audio": user_turn["audio"], "text": user_turn["audio_text"], "seconds": 3.0}])
    [row] = asr_report.main(["--split", "train", "g"])
    assert row["clips"] == 2 and row["wer"] == 0.0 and row["cer"] == 0.0
    assert (tmp_cache / "train" / "report.md").exists()


def test_transcribe_fold_writes_only_that_pool_fold(tmp_path, monkeypatch):
    from lifemate.data import default_data_root
    if not (default_data_root() / "train").is_dir():
        pytest.skip("data/train not downloaded")
    from lifemate import transcribe
    from lifemate.asr.cached import read_cache, sample_clips
    from lifemate.data import load_samples
    from lifemate.splits import fold_ids, load_splits

    monkeypatch.setenv("LIFEMATE_ASR_CACHE", str(tmp_path))
    out = transcribe.main(["--split", "train", "--asr", "none", "--name", "xfit", "--subset", "pool",
                           "--folds", "2", "--fold", "1"])
    fold = fold_ids(load_splits()["pool"], 2, 1)
    want = {c for s in load_samples(default_data_root(), "train") if s["id"] in fold for c in sample_clips(s)}
    assert set(read_cache(out)) == want
    with pytest.raises(SystemExit):
        transcribe.main(["--split", "train", "--asr", "none", "--name", "x", "--subset", "dev", "--fold", "0"])
