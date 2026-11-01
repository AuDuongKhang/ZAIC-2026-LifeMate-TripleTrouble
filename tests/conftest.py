import pytest

from lifemate.data import default_data_root, load_samples


@pytest.fixture(scope="session")
def train_samples():
    if not (default_data_root() / "train").is_dir():
        pytest.skip("data/train not downloaded")
    return load_samples(default_data_root(), "train")
