"""Reasoning sentences for SFT: one short Vietnamese sentence per gold step, written by a local open model on molab
(see docs/superpowers/specs/2026-11-05-reasoning-lines-design.md). Development tooling only: nothing in the inference
path (agents/, infer.py, docker_entry.py) may import this module."""
from __future__ import annotations

import json
from pathlib import Path


def load_thoughts(path: Path) -> dict[str, dict]:
    """Rows of a thoughts file by sample id. A torn last line (a session that died mid-write) is ignored; a broken
    line anywhere else raises."""
    rows: dict[str, dict] = {}
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break
            raise
        rows[row["id"]] = row
    return rows
