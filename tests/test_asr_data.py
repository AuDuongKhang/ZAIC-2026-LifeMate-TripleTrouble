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


def test_final_retrain_uses_every_clip_and_selects_nothing(train_samples):
    from lifemate.train_asr import selection_args, training_ids

    assert training_ids("all", train_samples) == {x["id"] for x in train_samples}
    assert training_ids("pool", train_samples) == set(load_splits()["pool"])
    assert len(build_train_manifest(train_samples, training_ids("all", train_samples))) == 5036
    final = selection_args("all", 100)
    assert final["eval_strategy"] == "no" and final["load_best_model_at_end"] is False and "metric_for_best_model" not in final
    pool = selection_args("pool", 100)
    assert pool["eval_steps"] == 100 and pool["load_best_model_at_end"] is True and pool["metric_for_best_model"] == "wer"


def test_crossfit_folds_split_the_pool_and_never_train_on_the_left_out_fold(train_samples):
    from lifemate.splits import fold_ids, pool_fold
    from lifemate.train_asr import training_ids

    pool = set(load_splits()["pool"])
    f0, f1 = fold_ids(pool, 2, 0), fold_ids(pool, 2, 1)
    assert f0 and f1 and not f0 & f1 and f0 | f1 == pool
    assert pool_fold("train_0001", 2) == pool_fold("train_0001", 2)  # sha1, stable across processes
    for k, left_out in ((0, f0), (1, f1)):
        ids = training_ids("pool", train_samples, 2, k)
        assert ids == pool - left_out
        trained = {r["audio"] for r in build_train_manifest(train_samples, ids)}
        held = {r["audio"] for r in build_train_manifest(train_samples, left_out)}
        assert held and not trained & held  # history clips follow their sample's fold
    with pytest.raises(ValueError):
        training_ids("all", train_samples, 2, 0)
    with pytest.raises(ValueError):
        fold_ids(pool, 2, 2)
