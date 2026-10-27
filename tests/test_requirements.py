"""molab (and the Docker base image) ship torch + torchvision built together. A requirement that depends on torch
makes the installer put a second torch into the venv, which breaks the preinstalled torchvision at import time."""
import re

from lifemate.data import REPO_ROOT

PULLS_TORCH = {"torch", "torchvision", "torchaudio", "accelerate", "peft", "trl", "bitsandbytes", "vllm"}


def test_runtime_requirements_do_not_pull_torch():
    names = set()
    for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            names.add(re.split(r"[<>=!~\[ ;]", line, maxsplit=1)[0].lower())
    assert not names & PULLS_TORCH, names & PULLS_TORCH
