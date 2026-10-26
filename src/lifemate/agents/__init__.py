"""Agents, picked by name with --agent. Add one: subclass Agent and register it in AGENTS."""
from __future__ import annotations

from .base import Agent, EmptyAgent, OracleAgent

AGENTS: dict[str, type[Agent]] = {
    "empty": EmptyAgent,
    "oracle": OracleAgent,
}


def build_agent(name: str, **kwargs) -> Agent:
    if name not in AGENTS:
        raise ValueError(f"unknown agent {name!r}; choose from {sorted(AGENTS)}")
    return AGENTS[name](**kwargs)
