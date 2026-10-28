"""Fixed train splits: disjoint, deterministic, history apps count for holdout, committed file is up to date."""
import pytest

from lifemate.data import default_data_root, load_samples
from lifemate.splits import load_splits, make_splits, sample_apps

pytestmark = pytest.mark.skipif(not (default_data_root() / "train").is_dir(), reason="data/train not downloaded")


@pytest.fixture(scope="module")
def train_samples():
    return load_samples(default_data_root(), "train")


@pytest.fixture(scope="module")
def splits(train_samples):
    return make_splits(train_samples, seed=0)


def test_splits_are_disjoint_and_cover_train(train_samples, splits):
    s = splits
    ids = {x["id"] for x in train_samples}
    assert set(s["holdout"]) | set(s["dev"]) | set(s["pool"]) == ids
    assert not (set(s["holdout"]) & set(s["dev"])) and not (set(s["dev"]) & set(s["pool"]))
    assert not (set(s["holdout"]) & set(s["pool"]))
    assert len(s["dev"]) == 300 and 0.10 <= len(s["holdout"]) / len(ids) <= 0.20


def test_history_app_makes_a_sample_holdout(train_samples, splits):
    held = set(splits["holdout_apps"])
    for x in train_samples:
        if sample_apps(x) & held:
            assert x["id"] in splits["holdout"]


def test_sample_apps_includes_history_and_skips_sys():
    sample = {"label": {"tool_calls": [{"name": "sys.ask_missing_info", "arguments": {}}]},
              "history": [{"role": "user", "text": "x"},
                          {"role": "assistant", "tool_calls": [{"name": "alarm.list", "arguments": {}}]}]}
    assert sample_apps(sample) == {"alarm"}


def test_splits_are_deterministic(train_samples, splits):
    assert make_splits(train_samples, seed=0) == splits


def test_committed_splits_match_generator(splits):
    assert load_splits() == splits
