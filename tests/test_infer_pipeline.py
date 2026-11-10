import time
import types
from pathlib import Path

import pytest

from lifemate import infer


class _Env:
    def __init__(self, split, sample_id, root):
        self.log = [{"name": "alarm.list"}]

    def get_tool_calls(self):
        return [{"name": "alarm.list", "arguments": {}}]

    def get_state_hash(self):
        return "h0"


class _Asr:
    def transcribe(self, audio, sample):
        return "đặt báo thức"


class _Agent:
    last_trace = None

    def __init__(self, sleep=0.0, fail=False):
        self.sleep, self.fail = sleep, fail

    def run(self, task, env):
        time.sleep(self.sleep)
        if self.fail:
            raise RuntimeError("boom")


def _pipeline(agent):
    tools = types.SimpleNamespace(ToolEnv=_Env)
    return infer.Pipeline(split="private_test", data_root=Path("/nowhere"), zaic_tools=tools, catalog={},
                          asr=_Asr(), agent=agent)


SAMPLE = {"id": "p1", "audio": "audio/p1/p1.wav", "history": [], "context": {"installed_apps": {}}}


def test_run_sample_line_has_time_in_ms():
    line, trace = infer.run_sample(_pipeline(_Agent(sleep=0.05)), SAMPLE)
    assert set(line) == {"id", "tool_calls", "state_hash", "time"}
    assert line["id"] == "p1" and line["state_hash"] == "h0"
    assert isinstance(line["time"], int) and line["time"] >= 50
    assert trace["error"] is None and trace["request"] == "đặt báo thức"


def test_run_sample_keeps_going_when_the_agent_raises():
    line, trace = infer.run_sample(_pipeline(_Agent(fail=True)), SAMPLE)
    assert line["tool_calls"] == [{"name": "alarm.list", "arguments": {}}] and line["state_hash"] == "h0"
    assert isinstance(line["time"], int)
    assert "boom" in trace["error"]


def test_seed_everything_sets_cudnn_flags():
    torch = pytest.importorskip("torch")
    from lifemate.utils import seed_everything

    torch.backends.cudnn.benchmark = True
    seed_everything(0)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
