import dataclasses
import json

from lifemate.agents.react import ReActAgent
from lifemate.task import Task

TASK = Task(id="t", request="xoá báo thức sớm nhất", history=[], tools=[],
            context={"current_time": "2026-11-13T12:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh"})


def CALL(name, args):
    return f'<tool_call>{{"name": "{name}", "arguments": {json.dumps(args)}}}</tool_call>'


class FakeLLM:
    context_limit = 16384
    max_new_tokens = 0

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.seen = []  # messages passed to each generate call
        self.max_times = []

    def generate(self, messages, tools, max_time=None):
        self.seen.append([dict(m) for m in messages])
        self.max_times.append(max_time)
        return self.outputs.pop(0) if self.outputs else "Xong."

    def count_tokens(self, messages, tools):
        return 1


class FakeEnv:
    def __init__(self, bad=()):
        self.bad, self.validated, self.called = set(bad), [], []

    def validate(self, name, args):
        self.validated.append(name)
        return {"ok": False, "error": "missing contact_id"} if name in self.bad else {"ok": True}

    def call(self, name, args):
        self.called.append((name, args))
        return {"ok": True}


def test_plain_answer_makes_no_calls():
    env = FakeEnv()
    ReActAgent(llm=FakeLLM(["Không làm được."])).run(TASK, env)
    assert env.called == []


def test_calls_are_executed_and_results_fed_back():
    env, llm = FakeEnv(), FakeLLM([CALL("alarm.list", {}), CALL("alarm.delete", {"alarm_id": "a1"}), "Đã xoá."])
    ReActAgent(llm=llm).run(TASK, env)
    assert env.called == [("alarm.list", {}), ("alarm.delete", {"alarm_id": "a1"})]
    assert [m["role"] for m in llm.seen[1][-2:]] == ["assistant", "tool"]


def test_invalid_call_is_never_executed():
    env, llm = FakeEnv(bad={"sms.send"}), FakeLLM([CALL("sms.send", {}), "Xong."])
    ReActAgent(llm=llm).run(TASK, env)
    assert env.called == [] and env.validated == ["sms.send"]
    assert "missing contact_id" in llm.seen[1][-1]["content"]


def test_stops_after_ask_missing_info():
    env = FakeEnv()
    ReActAgent(llm=FakeLLM([CALL("sys.ask_missing_info", {"tool_name": "phone.call", "params": ["contact_id"]}),
                            CALL("phone.call", {"contact_id": "c1"})])).run(TASK, env)
    assert [c[0] for c in env.called] == ["sys.ask_missing_info"]


def test_ask_missing_info_ends_the_batch_too():
    env = FakeEnv()
    both = CALL("sys.ask_missing_info", {"tool_name": "phone.call", "params": ["contact_id"]}) + CALL("sms.send", {})
    ReActAgent(llm=FakeLLM([both])).run(TASK, env)
    assert [c[0] for c in env.called] == ["sys.ask_missing_info"]


def test_identical_call_is_not_repeated():
    env = FakeEnv()
    ReActAgent(llm=FakeLLM([CALL("alarm.list", {})] * 10), max_steps=8).run(TASK, env)
    assert [c[0] for c in env.called] == ["alarm.list"]


def test_max_steps_bounds_generations():
    llm = FakeLLM([CALL("alarm.get", {"i": i}) for i in range(20)])
    ReActAgent(llm=llm, max_steps=3).run(TASK, FakeEnv())
    assert len(llm.seen) == 3


def test_time_budget_stops_but_keeps_calls():
    ticks = iter([0.0, 1.0, 25.0, 26.0, 27.0])
    env = FakeEnv()
    ReActAgent(llm=FakeLLM([CALL("alarm.list", {}), CALL("alarm.delete", {"alarm_id": "a"})]),
               time_budget_s=20, clock=lambda: next(ticks, 99.0)).run(TASK, env)
    assert [c[0] for c in env.called] == ["alarm.list"]


def test_broken_json_retried_then_gives_up():
    env, llm = FakeEnv(), FakeLLM(['<tool_call>{bad</tool_call>'] * 5)
    ReActAgent(llm=llm, max_invalid=2).run(TASK, env)
    assert env.called == [] and len(llm.seen) == 3


def test_time_budget_counts_from_sample_start():
    # ASR already used 19 s of a 20 s budget: no time left for a generation step (min_step_s=3).
    llm = FakeLLM([CALL("alarm.list", {})])
    ReActAgent(llm=llm, time_budget_s=20, min_step_s=3, clock=lambda: 19.0).run(dataclasses.replace(TASK, started_at=0.0), FakeEnv())
    assert llm.seen == []


def test_generation_capped_by_remaining_time():
    llm = FakeLLM(["Xong."])
    ReActAgent(llm=llm, time_budget_s=20, clock=lambda: 5.0).run(dataclasses.replace(TASK, started_at=0.0), FakeEnv())
    assert llm.max_times == [15.0]
