import functools

import pytest

from lifemate.data import default_data_root, load_samples


@pytest.fixture(scope="session")
def train_samples():
    if not (default_data_root() / "train").is_dir():
        pytest.skip("data/train not downloaded")
    return load_samples(default_data_root(), "train")


@functools.lru_cache(maxsize=None)
def _train_parts(examples: bool):
    from lifemate.agents.fewshot import ExampleBank
    from lifemate.data import import_zaic_tools, load_catalog

    root = default_data_root()
    bank = ExampleBank.from_data(root, "train") if examples else None
    return import_zaic_tools(root, "train"), load_catalog(root, "train"), bank


def replay_sample(sample: dict, examples: int = 0) -> dict | None:
    """SFT record of one train sample: gold-ASR task, a fresh train ToolEnv, replayed through the agent."""
    from lifemate.asr.base import GoldTranscriptASR
    from lifemate.infer import build_task
    from lifemate.sft.replay import replay

    zaic_tools, catalog, bank = _train_parts(bool(examples))
    root = default_data_root()
    task = build_task(sample, GoldTranscriptASR(), catalog, root, "train")
    return replay(sample, task, zaic_tools.ToolEnv("train", sample["id"], root=root), examples=examples, bank=bank)
