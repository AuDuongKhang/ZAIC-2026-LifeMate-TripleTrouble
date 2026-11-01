"""Named agent / ASR configurations from configs/presets.json, so one name means the same setup in the CLI, the molab
notebook and Docker, and meta.json records it.

    python -m lifemate.presets --list       # {"agent": [names], "asr": [names]} as JSON, for the notebook

A string value may hold ${VAR}, filled from the environment when the preset is resolved, so private HF repo owners
are never committed (for example "${HF_MODEL_OWNER}/lifemate-sft-v1").
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Mapping

from .data import REPO_ROOT
from .utils import parse_kw

PRESETS_PATH = REPO_ROOT / "configs" / "presets.json"
CACHED_PREFIX = "cached:"
_VAR = re.compile(r"\$\{(\w+)\}")


def load_presets(path: Path = PRESETS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fill(value, environ: Mapping[str, str], preset: str):
    if isinstance(value, str):
        def sub(m: re.Match) -> str:
            if m.group(1) not in environ:
                raise ValueError(f"preset {preset!r} needs the environment variable {m.group(1)}")
            return environ[m.group(1)]
        return _VAR.sub(sub, value)
    if isinstance(value, dict):
        return {k: _fill(v, environ, preset) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, environ, preset) for v in value]
    return value


def resolve(kind: str, name: str, environ: Mapping[str, str] = os.environ, path: Path = PRESETS_PATH) -> tuple[str, dict]:
    """(backend name, kwargs) of a preset; kind is "agent" or "asr". `cached:<cache>` is an ASR preset without an entry."""
    if kind == "asr" and name.startswith(CACHED_PREFIX):
        return "cached", {"name": name[len(CACHED_PREFIX):]}
    known = load_presets(path)[kind]
    if name not in known:
        raise KeyError(f"unknown {kind} preset {name!r}; known: {sorted(known)}")
    entry = known[name]
    return entry[kind], _fill(entry.get(f"{kind}_kw", {}), environ, name)


def choose(kind: str, backend: str | None, preset: str | None, kw_items: list[str] | None,
           default: str) -> tuple[str, dict]:
    """The backend and kwargs for one --agent/--asr, --<kind>-preset and --<kind>-kw combination. Explicit
    --<kind>-kw items override single keys of the preset."""
    if backend and preset:
        raise ValueError(f"give --{kind} or --{kind}-preset, not both")
    if preset:
        name, kw = resolve(kind, preset)
    else:
        name, kw = backend or default, {}
    return name, {**kw, **parse_kw(kw_items)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", required=True)
    ap.parse_args(argv)
    presets = load_presets()
    print(json.dumps({"agent": sorted(presets["agent"]), "asr": sorted(presets["asr"])}))


if __name__ == "__main__":
    main()
