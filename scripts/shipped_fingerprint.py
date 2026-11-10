"""Print what the Docker image ships at a git ref: configs/shipped.json plus the two presets it names, as one JSON
line. Two refs with the same line build an image that runs the same models with the same settings, whatever else
changed in configs/presets.json (new experiment presets, for example). Stdlib only.

    python scripts/shipped_fingerprint.py <git ref>
"""
from __future__ import annotations

import json
import subprocess
import sys


def at(ref: str, path: str) -> dict:
    return json.loads(subprocess.run(["git", "show", f"{ref}:{path}"], check=True, capture_output=True,
                                     text=True, encoding="utf-8").stdout)


def fingerprint(ref: str) -> str:
    shipped = at(ref, "configs/shipped.json")
    presets = at(ref, "configs/presets.json")
    used = {"agent": presets["agent"].get(shipped["agent_preset"]), "asr": presets["asr"].get(shipped["asr_preset"])}
    return json.dumps({"shipped": shipped, "presets": used}, sort_keys=True)


if __name__ == "__main__":
    print(fingerprint(sys.argv[1]))
