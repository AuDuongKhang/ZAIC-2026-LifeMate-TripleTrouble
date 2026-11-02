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
    def __init__(self, bad=(), results=None):
        self.bad, self.validated, self.called = set(bad), [], []
        self.results = results or {}

    def validate(self, name, args):
        self.validated.append(name)
        return {"ok": False, "error": "missing contact_id"} if name in self.bad else {"ok": True}

    def call(self, name, args):
        self.called.append((name, args))
        return self.results.get(name, {"ok": True})


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


PAY = {"name": "bill_payment.pay", "description": "Pay a bill", "parameters": {"type": "object", "properties": {
    "bill_id": {"type": "string", "x-ref": "bill_payment.bills"}, "source": {"type": "string"}}, "required": ["bill_id"]}}
LIST = {"name": "bill_payment.list_bills", "description": "List bills", "parameters": {"type": "object", "properties": {}}}
BILL_TASK = dataclasses.replace(TASK, tools=[PAY, LIST])


def test_invented_ref_id_is_not_sent_until_looked_up():
    env = FakeEnv(results={"bill_payment.list_bills": {"ok": True, "bills": [{"id": "bill_04", "amount": 300000}]}})
    llm = FakeLLM([CALL("bill_payment.pay", {"bill_id": "bill_12345"}), CALL("bill_payment.list_bills", {}),
                   CALL("bill_payment.pay", {"bill_id": "bill_04"}), "Xong."])
    ReActAgent(llm=llm, check_refs=True).run(BILL_TASK, env)
    assert [c for c in env.called] == [("bill_payment.list_bills", {}), ("bill_payment.pay", {"bill_id": "bill_04"})]
    assert "bill_12345" in llm.seen[1][-1]["content"]


def test_ref_id_from_history_results_is_accepted():
    task = dataclasses.replace(BILL_TASK, history=[
        {"role": "user", "text": "hoá đơn nào?"},
        {"role": "assistant", "text": "Có 1.", "tool_calls": [{"name": "bill_payment.list_bills", "arguments": {}}],
         "tool_results": [{"ok": True, "bills": [{"id": "bill_04"}]}]}])
    env = FakeEnv()
    ReActAgent(llm=FakeLLM([CALL("bill_payment.pay", {"bill_id": "bill_04"}), "Xong."]), check_refs=True).run(task, env)
    assert env.called == [("bill_payment.pay", {"bill_id": "bill_04"})]


def test_ref_check_is_off_by_default():
    env = FakeEnv()
    ReActAgent(llm=FakeLLM([CALL("bill_payment.pay", {"bill_id": "bill_12345"}), "Xong."])).run(BILL_TASK, env)
    assert env.called == [("bill_payment.pay", {"bill_id": "bill_12345"})]


def test_agent_info_records_prompt_and_options():
    agent = ReActAgent(llm=FakeLLM([]), prompt_version="v1", check_refs=False)
    assert agent.info["prompt"] == "v1" and agent.info["check_refs"] is False


def test_trace_records_rejections_and_stop_reason():
    llm = FakeLLM([CALL("alarm.delete", {}), "Xong."])
    agent = ReActAgent(llm=llm)
    agent.run(TASK, FakeEnv(bad={"alarm.delete"}))
    assert agent.last_trace["steps"][0]["rejected"] == ["alarm.delete: missing contact_id"]
    assert agent.last_trace["stop"] == "no_call"


def test_ask_stops_with_asked():
    agent = ReActAgent(llm=FakeLLM([CALL("sys.ask_missing_info", {"tool_name": "sms.send", "params": ["contact_id"]})]))
    agent.run(TASK, FakeEnv())
    assert agent.last_trace["stop"] == "asked"


def test_trace_keeps_model_output_and_resets_between_runs():
    agent = ReActAgent(llm=FakeLLM([CALL("alarm.list", {}), "Xong.", "Không làm được."]))
    agent.run(TASK, FakeEnv())
    assert [s["output"][:11] for s in agent.last_trace["steps"]] == ["<tool_call>", "Xong."]
    agent.run(TASK, FakeEnv())
    assert [s["output"] for s in agent.last_trace["steps"]] == ["Không làm được."]


def test_stop_reasons_for_repeats_budget_and_steps():
    repeat = ReActAgent(llm=FakeLLM([CALL("alarm.list", {})] * 5), max_invalid=1)
    repeat.run(TASK, FakeEnv())
    assert repeat.last_trace["stop"] == "invalid"
    assert repeat.last_trace["steps"][1]["rejected"] == ["alarm.list: already called"]
    loop = ReActAgent(llm=FakeLLM([CALL("alarm.list", {"n": i}) for i in range(5)]), max_steps=3)
    loop.run(TASK, FakeEnv())
    assert loop.last_trace["stop"] == "max_steps"
    late = ReActAgent(llm=FakeLLM(["x"]), clock=lambda: 100.0)
    late.run(dataclasses.replace(TASK, started_at=0.0), FakeEnv())
    assert late.last_trace["stop"] == "time" and late.last_trace["steps"] == []


def test_invalid_json_is_listed_as_rejected():
    agent = ReActAgent(llm=FakeLLM(['<tool_call>{"name": </tool_call>', "Xong."]))
    agent.run(TASK, FakeEnv())
    assert agent.last_trace["steps"][0]["rejected"][0].startswith("invalid JSON")


class StubBank:
    def __init__(self, text):
        self.text, self.asked = text, []

    def __len__(self):
        return 1

    def select(self, request, installed_apps, k, exclude_id):
        self.asked.append((request, installed_apps, k, exclude_id))
        return [{"id": "e", "audio_text": self.text, "label": {"tool_calls": [], "tool_results": []}}]


def test_examples_are_added_to_the_system_message():
    bank, llm = StubBank("EXAMPLE BLOCK"), FakeLLM(["Xong."])
    task = dataclasses.replace(TASK, context={**TASK.context, "installed_apps": {"alarm": {}}})
    ReActAgent(llm=llm, examples=2, bank=bank).run(task, FakeEnv())
    assert "EXAMPLE BLOCK" in llm.seen[0][0]["content"]
    assert bank.asked == [("xoá báo thức sớm nhất", {"alarm"}, 2, "t")]


def test_default_has_no_examples_and_builds_no_bank():
    llm = FakeLLM(["Xong."])
    agent = ReActAgent(llm=llm)
    agent.run(TASK, FakeEnv())
    assert agent.bank is None and "Examples" not in llm.seen[0][0]["content"]
    assert agent.info["examples"] == 0


def test_a_crashing_step_does_not_report_max_steps():
    class Boom(FakeLLM):
        def generate(self, messages, tools, max_time=None):
            raise RuntimeError("cuda oom")

    agent = ReActAgent(llm=Boom([]))
    try:
        agent.run(TASK, FakeEnv())
    except RuntimeError:
        pass
    assert agent.last_trace["stop"] is None


def test_agent_passes_the_model_revision_to_the_llm(monkeypatch):
    import lifemate.agents.react as react

    made = []
    monkeypatch.setattr(react, "HFChatLLM", lambda model, **kw: made.append((model, kw)) or FakeLLM([]))
    agent = ReActAgent(model="team/lifemate-sft-v2", revision="abc123")
    assert made == [("team/lifemate-sft-v2", {"revision": "abc123"})]
    assert agent.info["model"] == "team/lifemate-sft-v2" and agent.info["revision"] == "abc123"


def test_hf_llm_loads_tokenizer_and_weights_at_the_revision(monkeypatch):
    import sys
    import types

    from lifemate.agents.llm import HFChatLLM

    calls = []

    class Loaded:
        eos_token, pad_token = "<|im_end|>", None

        def to(self, device):
            return self

        def eval(self):
            pass

    def loader(kind):
        return types.SimpleNamespace(from_pretrained=lambda name, **kw: calls.append((kind, name, kw)) or Loaded())

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=loader("tok"), AutoModelForCausalLM=loader("model")))
    HFChatLLM("team/m", revision="abc123")
    assert [(k, n, kw.get("revision")) for k, n, kw in calls] == [("tok", "team/m", "abc123"),
                                                                   ("model", "team/m", "abc123")]
