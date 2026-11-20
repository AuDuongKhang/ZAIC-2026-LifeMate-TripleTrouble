"""The runner notebook offers the private test once it is released (context/SUBMISSION_GUIDANCE.md, round 2)."""
from pathlib import Path

NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "molab_run.py"


def test_the_runner_offers_the_private_test_split():
    assert 'mo.ui.dropdown(["train", "public_test", "private_test"]' in NOTEBOOK.read_text(encoding="utf-8")
