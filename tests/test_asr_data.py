"""ASR fine-tuning manifests: train on pool clips only, history clips included, eval subset fixed."""
import pytest

from lifemate.asr_data import build_eval_manifest, build_train_manifest
from lifemate.data import default_data_root, load_samples
from lifemate.splits import load_splits

pytestmark = pytest.mark.skipif(not (default_data_root() / "train").is_dir(), reason="data/train not downloaded")


@pytest.fixture(scope="module")
def train_samples():
    return load_samples(default_data_root(), "train")


def test_train_manifest_uses_pool_only(train_samples):
    s = load_splits()
    m = build_train_manifest(train_samples, set(s["pool"]))
    held = set(s["dev"]) | set(s["holdout"])
    assert m and not any(r["sample"] in held for r in m)
    assert all(r["text"].strip() for r in m)


def test_train_manifest_includes_history_clips(train_samples):
    m = build_train_manifest(train_samples, set(load_splits()["pool"]))
    assert any(r["audio"].endswith("_u1.wav") for r in m)


def test_eval_manifest_is_disjoint_from_train_and_deterministic(train_samples):
    train = {r["audio"] for r in build_train_manifest(train_samples, set(load_splits()["pool"]))}
    full = build_eval_manifest(train_samples)
    assert full and not train & {r["audio"] for r in full}
    a, b = build_eval_manifest(train_samples, limit=50), build_eval_manifest(train_samples, limit=50)
    assert a == b and len(a) == 50
