import dataclasses
import json

import pytest

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


def test_date_hints_reach_the_system_prompt_and_info():
    llm = FakeLLM(["Xong."])
    agent = ReActAgent(llm=llm, date_hints=True)
    agent.run(TASK, FakeEnv())
    assert agent.info["date_hints"] is True
    assert "Calendar (weeks start on Monday):" in llm.seen[0][0]["content"]
    assert ReActAgent(llm=FakeLLM([])).info["date_hints"] is False


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
        self.text, self.asked, self.offsets = text, [], []

    def __len__(self):
        return 1

    def select(self, request, installed_apps, k, exclude_id, offset=0):
        self.asked.append((request, installed_apps, k, exclude_id))
        self.offsets.append(offset)
        return [{"id": "e", "audio_text": f"{self.text} @{offset}", "label": {"tool_calls": [], "tool_results": []}}]


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
        generation_config = types.SimpleNamespace(eos_token_id=None)

        def get_vocab(self):
            return {"<|im_end|>": 3}

        def to(self, device):
            return self

        def eval(self):
            pass

    def loader(kind):
        def load(name, **kw):
            calls.append((kind, name, kw))
            return (Loaded(), {"missing_keys": []}) if kw.get("output_loading_info") else Loaded()
        return types.SimpleNamespace(from_pretrained=load)

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=loader("tok"), AutoModelForCausalLM=loader("model")))
    llm = HFChatLLM("team/m", revision="abc123")
    assert llm.eos_token_id == [3]
    assert [(k, n, kw.get("revision")) for k, n, kw in calls] == [("tok", "team/m", "abc123"),
                                                                   ("model", "team/m", "abc123")]


EMPTY_FLIGHTS = {"flights.search": {"ok": True, "flights": []}}
HINT = ("\nNo results. Try once more with other values: another date (up to 7 days away), fewer optional filters or a "
        "shorter query. If that also finds nothing, reply in one short sentence without calling any tool.")
SEARCH = CALL("flights.search", {"to": "Hue"})


def test_empty_result_gets_a_hint():
    llm = FakeLLM([SEARCH, "Không có chuyến bay."])
    ReActAgent(llm=llm, empty_hint=True).run(TASK, FakeEnv(results=EMPTY_FLIGHTS))
    last = llm.seen[1][-1]
    assert last["role"] == "tool" and last["content"] == json.dumps({"ok": True, "flights": []}) + HINT


def test_empty_hint_changes_only_the_message():
    outputs = [SEARCH, CALL("flights.search", {"to": "Hue", "date": "2026-11-14"}), SEARCH, "Xong."]
    runs = []
    for flag in (False, True):
        env = FakeEnv(results=EMPTY_FLIGHTS)
        ReActAgent(llm=FakeLLM(outputs), empty_hint=flag).run(TASK, env)
        runs.append(env.called)
    assert runs[0] == runs[1] and len(runs[0]) == 2


def test_repeat_of_an_empty_call_gets_the_empty_text():
    llm = FakeLLM([SEARCH, SEARCH, "Xong."])
    ReActAgent(llm=llm, empty_hint=True).run(TASK, FakeEnv(results=EMPTY_FLIGHTS))
    assert llm.seen[2][-1]["content"] == "Already called and it returned nothing. Change the arguments as suggested, or stop."
    other = FakeLLM([CALL("alarm.list", {})] * 2 + ["Xong."])  # a repeat of a non-empty call keeps today's text
    ReActAgent(llm=other, empty_hint=True).run(TASK, FakeEnv())
    assert other.seen[2][-1]["content"] == "Already called; reuse the earlier result."


def test_empty_hint_off_by_default():
    llm = FakeLLM([SEARCH, SEARCH, "Xong."])
    agent = ReActAgent(llm=llm)
    agent.run(TASK, FakeEnv(results=EMPTY_FLIGHTS))
    assert llm.seen[1][-1]["content"] == json.dumps({"ok": True, "flights": []})
    assert llm.seen[2][-1]["content"] == "Already called; reuse the earlier result."
    assert agent.info["empty_hint"] is False


def test_is_empty_result():
    from lifemate.agents.react import is_empty_result

    assert is_empty_result([]) and is_empty_result({"ok": True, "flights": []})
    assert is_empty_result({"ok": True, "a": [], "b": []})
    assert not is_empty_result({"ok": True}) and not is_empty_result({"ok": True, "done": False})
    assert not is_empty_result({"flights": [], "hotels": [{"id": 1}]}) and not is_empty_result([{"id": 1}])
    assert not is_empty_result("") and not is_empty_result(None)


REFUSE = "Không làm được."
END_SPEC = {"name": "parking.end_session", "description": "Kết thúc phiên gửi xe",
            "parameters": {"type": "object", "properties": {}}}
PARKING_TASK = dataclasses.replace(TASK, request="kết thúc phiên gửi xe", tools=[END_SPEC])
CHECK = ("Check again: the installed tool(s) parking.end_session may handle this request. If one does, call it. "
         "If the request is out of scope or the user cancelled it, reply in one short sentence without calling any tool.")


def test_refusal_check_fires_once():
    llm, env = FakeLLM([REFUSE, REFUSE]), FakeEnv()
    agent = ReActAgent(llm=llm, refusal_check=True)
    agent.run(PARKING_TASK, env)
    assert len(llm.seen) == 2 and env.called == [] and agent.last_trace["stop"] == "no_call"
    assert llm.seen[1][-2:] == [{"role": "assistant", "content": REFUSE, "tool_calls": []},
                                {"role": "user", "content": CHECK}]
    assert agent.last_trace["refusal_check"] == ["parking.end_session"]


def test_refusal_check_lets_the_model_call():
    llm, env = FakeLLM([REFUSE, CALL("parking.end_session", {}), "Xong."]), FakeEnv()
    ReActAgent(llm=llm, refusal_check=True).run(PARKING_TASK, env)
    assert env.called == [("parking.end_session", {})] and len(llm.seen) == 3


def test_no_refusal_check_after_a_call():
    llm = FakeLLM([CALL("parking.end_session", {}), "Xong."])
    agent = ReActAgent(llm=llm, refusal_check=True)
    agent.run(PARKING_TASK, FakeEnv())
    assert len(llm.seen) == 2 and not any(m["content"] == CHECK for m in llm.seen[1] if m["role"] == "user")
    assert "refusal_check" not in agent.last_trace


def test_no_refusal_check_without_a_matching_tool():
    llm = FakeLLM([REFUSE])
    agent = ReActAgent(llm=llm, refusal_check=True)
    agent.run(dataclasses.replace(PARKING_TASK, request="bật đèn phòng khách"), FakeEnv())
    assert len(llm.seen) == 1 and "refusal_check" not in agent.last_trace


def test_refusal_min_hits_is_passed_on():
    llm = FakeLLM([REFUSE])
    ReActAgent(llm=llm, refusal_check=True, refusal_min_hits=5).run(PARKING_TASK, FakeEnv())
    assert len(llm.seen) == 1


def test_refusal_check_off_by_default():
    llm = FakeLLM([REFUSE, REFUSE])
    agent = ReActAgent(llm=llm)
    agent.run(PARKING_TASK, FakeEnv())
    assert len(llm.seen) == 1 and agent.info["refusal_check"] is False and agent.info["refusal_min_hits"] == 1
    assert "refusal_check" not in agent.last_trace


SMS_SPEC = {"name": "sms.send", "description": "Gửi tin nhắn",
            "parameters": {"type": "object", "properties": {
                "phone_number": {"type": "string"},
                "content": {"type": "string", "x-match": "fuzzy"}}}}
SMS_TASK = dataclasses.replace(TASK, request="nhắn cho mẹ là tối nay về ăn cơm", tools=[SMS_SPEC])
LEAK = "Not sent: content repeats your closing reply. Write the content the user asked for."


def test_closing_text_as_content_is_rejected():
    env = FakeEnv()
    llm = FakeLLM([CALL("sms.send", {"phone_number": "0901", "content": "Đã xong."}), "Xong."])
    agent = ReActAgent(llm=llm, leak_guard=True)
    agent.run(SMS_TASK, env)
    assert env.called == [] and llm.seen[1][-1] == {"role": "tool", "content": LEAK}
    assert agent.last_trace["steps"][0]["rejected"] == ["sms.send: content repeats your closing reply"]
    # folded equality: case, diacritics and punctuation do not matter; the refusal text counts too
    env = FakeEnv()
    llm = FakeLLM([CALL("sms.send", {"content": "da xong"}), CALL("sms.send", {"content": "Xin lỗi, tôi không làm được việc này"}),
                   "Xong."])
    ReActAgent(llm=llm, leak_guard=True).run(SMS_TASK, env)
    assert env.called == []


def test_real_content_is_sent():
    env = FakeEnv()
    args = {"phone_number": "0901", "content": "Tối nay về ăn cơm"}
    ReActAgent(llm=FakeLLM([CALL("sms.send", args), "Xong."]), leak_guard=True).run(SMS_TASK, env)
    assert env.called == [("sms.send", args)]
    # a non-fuzzy param that happens to equal the closing text is left alone
    env = FakeEnv()
    args = {"phone_number": "Đã xong.", "content": "Tối nay về ăn cơm"}
    ReActAgent(llm=FakeLLM([CALL("sms.send", args), "Xong."]), leak_guard=True).run(SMS_TASK, env)
    assert env.called == [("sms.send", args)]


def test_leak_guard_off_by_default():
    env, llm = FakeEnv(), FakeLLM([CALL("sms.send", {"content": "Đã xong."}), "Xong."])
    agent = ReActAgent(llm=llm)
    agent.run(SMS_TASK, env)
    assert env.called == [("sms.send", {"content": "Đã xong."})] and agent.info["leak_guard"] is False
    assert ReActAgent(llm=FakeLLM([]), leak_guard=True).info["leak_guard"] is True


DELETE_SPEC = {"name": "alarm.delete", "kind": "write", "description": "Xoá báo thức",
               "parameters": {"type": "object", "properties": {"alarm_id": {"type": "string"}}}}
LIST_SPEC = {"name": "alarm.list", "kind": "read", "description": "Liệt kê báo thức",
             "parameters": {"type": "object", "properties": {}}}
NOTE_SPEC = {"name": "notes.create", "kind": "write", "description": "Tạo ghi chú",
             "parameters": {"type": "object", "properties": {"folder_id": {"type": "string"},
                                                             "title": {"type": "string", "x-match": "fuzzy"}}}}
VOTE_TASK = dataclasses.replace(TASK, tools=[DELETE_SPEC, LIST_SPEC, NOTE_SPEC],
                                context={**TASK.context, "installed_apps": {"alarm": {}, "notes": {}}})
ASK_DELETE = {"tool_name": "alarm.delete", "params": ["alarm_id"]}


def DEL(alarm_id):
    return CALL("alarm.delete", {"alarm_id": alarm_id})


def vote_run(outputs, task=VOTE_TASK, **kw):
    llm, env, bank = FakeLLM(outputs), FakeEnv(), StubBank("EX")
    agent = ReActAgent(llm=llm, examples=2, bank=bank, **{"vote_prompts": 3, **kw})
    agent.run(task, env)
    return agent, llm, env, bank


def test_vote_majority_replaces_the_pick():
    agent, llm, env, bank = vote_run([DEL("a1"), DEL("a2"), DEL("a2"), "Xong."])
    assert env.called[-1] == ("alarm.delete", {"alarm_id": "a2"}) and len(env.called) == 1
    vote = agent.last_trace["vote"]
    assert vote["winner"] == 1 and vote["applied"] is True and vote["step"] == 0 and len(vote["keys"]) == 3
    assert vote["keys"][1] == vote["keys"][2] != vote["keys"][0]
    # the variants see the same conversation; only the system message (the examples) changes
    assert bank.offsets == [0, 2, 4]
    assert llm.seen[1][1:] == llm.seen[0][1:] and "EX @2" in llm.seen[1][0]["content"]
    assert "EX @4" in llm.seen[2][0]["content"] and "EX @0" in llm.seen[0][0]["content"]
    assert agent.last_trace["steps"][0]["output"] == DEL("a2")
    assert llm.seen[3][-2]["tool_calls"][0]["function"]["arguments"] == {"alarm_id": "a2"}
    assert llm.seen[3][0] == llm.seen[0][0]  # later steps keep the original prompt


def test_vote_variants_keep_the_date_table():
    agent, llm, env, _ = vote_run([DEL("a1"), DEL("a1"), DEL("a1"), "Xong."], date_hints=True)
    assert all("Calendar (weeks start on Monday)" in llm.seen[i][0]["content"] for i in range(3))


def test_vote_tie_keeps_the_first_prompt():
    agent, llm, env, _ = vote_run([DEL("a1"), DEL("a2"), DEL("a3"), "Xong."])
    assert env.called == [("alarm.delete", {"alarm_id": "a1"})] and agent.last_trace["vote"]["winner"] == 0


def test_vote_ignores_free_text_differences():
    from lifemate.agents.react import vote_key

    outputs = [CALL("notes.create", {"folder_id": "f1", "title": t}) for t in ("Mua sữa", "mua sua", "Mua sữa tươi")]
    agent, llm, env, _ = vote_run(outputs + ["Xong."])
    assert env.called == [("notes.create", {"folder_id": "f1", "title": "Mua sữa"})]
    vote = agent.last_trace["vote"]
    assert len(set(vote["keys"])) == 1 and vote["winner"] == 0
    assert vote_key({"name": "notes.create", "arguments": {"title": "a", "folder_id": "f1"}}, NOTE_SPEC) == \
        vote_key({"name": "notes.create", "arguments": {"folder_id": "f1", "title": "b"}}, NOTE_SPEC)
    ask = {"name": "sys.ask_missing_info", "arguments": {"tool_name": "sms.send", "params": ["b", "a"]}}
    assert vote_key(ask, None) == vote_key({**ask, "arguments": {"tool_name": "sms.send", "params": ["a", "b"]}}, None)


def test_ask_and_act_are_different_keys():
    ask = CALL("sys.ask_missing_info", ASK_DELETE)
    agent, llm, env, _ = vote_run([DEL("a1"), ask, ask])
    assert env.called == [("sys.ask_missing_info", ASK_DELETE)] and agent.last_trace["stop"] == "asked"
    assert agent.last_trace["vote"]["winner"] == 1


def test_reads_are_not_voted():
    agent, llm, env, _ = vote_run([CALL("alarm.list", {}), "Xong."])
    assert len(llm.seen) == 2 and env.called == [("alarm.list", {})] and agent.last_trace["vote"] is None


def test_vote_runs_once_per_sample():
    agent, llm, env, _ = vote_run([DEL("a1"), DEL("a1"), DEL("a1"), DEL("a2"), "Xong."])
    assert len(llm.seen) == 3 + 1 + 1  # K proposals for the first write, one for the second, then the closing text
    assert env.called == [("alarm.delete", {"alarm_id": "a1"}), ("alarm.delete", {"alarm_id": "a2"})]
    assert agent.last_trace["vote"]["step"] == 0


def test_vote_record_only_sends_the_original():
    agent, llm, env, _ = vote_run([DEL("a1"), DEL("a2"), DEL("a2"), "Xong."], vote_apply=False)
    assert env.called == [("alarm.delete", {"alarm_id": "a1"})]
    assert agent.last_trace["vote"]["winner"] == 1 and agent.last_trace["vote"]["applied"] is False
    assert agent.last_trace["steps"][0]["output"] == DEL("a1")


def test_invalid_proposals_never_win():
    agent, llm, env, _ = vote_run([DEL("a1"), "<tool_call>{bad</tool_call>", "<tool_call>{bad</tool_call>", "Xong."])
    assert env.called == [("alarm.delete", {"alarm_id": "a1"})] and agent.last_trace["vote"]["winner"] == 0
    assert agent.last_trace["vote"]["keys"][1] == '["invalid"]'


def test_vote_skipped_when_time_is_short():
    # 8 s left after the first proposal: enough for a step (min_step_s=3), not for 3 proposals (9 s)
    agent, llm, env, _ = vote_run([DEL("a1"), "Xong."], task=dataclasses.replace(VOTE_TASK, started_at=0.0),
                                  time_budget_s=20, min_step_s=3, clock=lambda: 12.0)
    assert env.called == [("alarm.delete", {"alarm_id": "a1"})] and len(llm.seen) == 2
    assert agent.last_trace["vote"] is None


def test_vote_off_by_default():
    runs = []
    for kw in ({}, {"vote_prompts": 0, "vote_apply": False}):
        llm, env, bank = FakeLLM([DEL("a1"), DEL("a2"), DEL("a2"), "Xong."]), FakeEnv(), StubBank("EX")
        agent = ReActAgent(llm=llm, examples=2, bank=bank, **kw)
        agent.run(VOTE_TASK, env)
        runs.append((json.dumps(agent.last_trace), json.dumps(llm.seen), env.called, bank.offsets))
    assert runs[0] == runs[1] and runs[0][3] == [0]
    trace, _, called, _ = runs[0]
    assert "vote" not in json.loads(trace) and called[0] == ("alarm.delete", {"alarm_id": "a1"})
    agent = ReActAgent(llm=FakeLLM([]))
    assert agent.info["vote_prompts"] == 0 and agent.info["vote_apply"] is True


def test_plain_answer_majority_does_not_win():
    agent, llm, env, _ = vote_run([DEL("a1"), "Không làm được.", "Không làm được.", "Xong."])
    assert env.called == [("alarm.delete", {"alarm_id": "a1"})] and agent.last_trace["steps"][0]["output"] == DEL("a1")
    vote = agent.last_trace["vote"]
    assert vote["winner"] == 0 and vote["keys"][1] == vote["keys"][2] == "[]"


def test_read_only_majority_does_not_win():
    read = CALL("alarm.list", {})
    agent, llm, env, _ = vote_run([DEL("a1"), read, read, "Xong."])
    assert env.called == [("alarm.delete", {"alarm_id": "a1"})] and len(llm.seen) == 4
    vote = agent.last_trace["vote"]
    assert vote["winner"] == 0 and vote["step"] == 0 and vote["keys"][1] == vote["keys"][2] != vote["keys"][0]


def test_vote_needs_examples():
    with pytest.raises(ValueError, match="vote_prompts"):
        ReActAgent(llm=FakeLLM([]), vote_prompts=3)


# --- date_fix ----------------------------------------------------------------------------------------------------------

DATE_TASK = Task(id="d", request="họp chiều mai", history=[], tools=[],
                 context={"current_time": "2026-10-01T09:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh"})


def test_date_fix_corrects_the_date_before_validation():
    env = FakeEnv()
    agent = ReActAgent(llm=FakeLLM([CALL("calendar.create_event", {"start": "2026-09-30T15:00:00+07:00"}), "Xong."]),
                       date_fix=True)
    agent.run(DATE_TASK, env)
    assert env.called == [("calendar.create_event", {"start": "2026-10-02T15:00:00+07:00"})]
    assert agent.info["date_fix"] is True


def test_date_fix_is_off_by_default():
    env = FakeEnv()
    agent = ReActAgent(llm=FakeLLM([CALL("calendar.create_event", {"start": "2026-09-30T15:00:00+07:00"}), "Xong."]))
    agent.run(DATE_TASK, env)
    assert env.called == [("calendar.create_event", {"start": "2026-09-30T15:00:00+07:00"})]
    assert agent.info["date_fix"] is False


def test_date_fix_shows_the_model_the_corrected_call():
    env, llm = FakeEnv(), FakeLLM([CALL("calendar.create_event", {"start": "2026-09-30T15:00:00+07:00"}), "Xong."])
    ReActAgent(llm=llm, date_fix=True).run(DATE_TASK, env)
    shown = llm.seen[1][-2]["tool_calls"][0]["function"]["arguments"]
    assert shown == {"start": "2026-10-02T15:00:00+07:00"}  # what the model reads back is what was sent


def test_date_fix_leaves_a_turn_with_two_dated_calls_alone():
    env = FakeEnv()
    two = CALL("calendar.create_event", {"start": "2026-09-30T15:00:00+07:00"}) + CALL("tasks.add", {"due": "2026-09-30T18:00:00+07:00"})
    ReActAgent(llm=FakeLLM([two, "Xong."]), date_fix=True).run(DATE_TASK, env)
    assert [a for _, a in env.called] == [{"start": "2026-09-30T15:00:00+07:00"}, {"due": "2026-09-30T18:00:00+07:00"}]


# --- analog_weight -----------------------------------------------------------------------------------------------------

class AnalogBank(StubBank):
    def select(self, request, installed_apps, k, exclude_id, offset=0, tools=None, analog_weight=0.0):
        self.asked.append((request, installed_apps, k, exclude_id, [t["name"] for t in tools or []], analog_weight))
        return super().select(request, installed_apps, k, exclude_id, offset)


def test_analog_weight_is_passed_to_the_bank_with_the_installed_tools():
    bank = AnalogBank("EX")
    task = dataclasses.replace(TASK, tools=[{"name": "alarm.list"}], context={**TASK.context, "installed_apps": {"alarm": {}}})
    agent = ReActAgent(llm=FakeLLM(["Xong."]), examples=2, bank=bank, analog_weight=2.0)
    agent.run(task, FakeEnv())
    assert bank.asked[0][-2:] == (["alarm.list"], 2.0) and agent.info["analog_weight"] == 2.0


def test_analog_weight_off_keeps_the_bank_call_as_before():
    bank = StubBank("EX")  # a bank without the new keywords still works
    agent = ReActAgent(llm=FakeLLM(["Xong."]), examples=2, bank=bank)
    agent.run(dataclasses.replace(TASK, context={**TASK.context, "installed_apps": {"alarm": {}}}), FakeEnv())
    assert agent.info["analog_weight"] == 0.0 and len(bank.asked) == 1


# --- beam search and token budget --------------------------------------------------------------------------------------

def test_num_beams_and_max_new_tokens_reach_the_llm_and_info(monkeypatch):
    import lifemate.agents.react as react
    seen = {}

    class Fake(FakeLLM):
        def __init__(self, model, revision=None, max_new_tokens=384, num_beams=1):
            super().__init__([])
            seen.update(model=model, max_new_tokens=max_new_tokens, num_beams=num_beams)
            self.max_new_tokens = max_new_tokens

    monkeypatch.setattr(react, "HFChatLLM", Fake)
    agent = ReActAgent(model="m", max_new_tokens=1024, num_beams=3)
    assert seen == {"model": "m", "max_new_tokens": 1024, "num_beams": 3}
    assert agent.info["max_new_tokens"] == 1024 and agent.info["num_beams"] == 3
    assert ReActAgent(llm=FakeLLM([])).info["num_beams"] == 1


# --- leave_gold_apps_out -----------------------------------------------------------------------------------------------

def test_leave_gold_apps_out_passes_the_label_apps_to_the_bank():
    seen = {}

    class Bank:
        def __len__(self):
            return 1

        def select(self, *args, **kw):
            seen.update(kw)
            return []

    label = {"tool_calls": [{"name": "alarm.delete", "arguments": {"alarm_id": "a1"}}]}
    task = dataclasses.replace(TASK, label=label, context={**TASK.context, "installed_apps": {"alarm": {"tools": []}}})
    agent = ReActAgent(llm=FakeLLM(["Xong."]), examples=2, bank=Bank(), leave_gold_apps_out=True)
    agent.run(task, FakeEnv())
    assert seen["exclude_apps"] == {"alarm"} and agent.info["leave_gold_apps_out"] is True


def test_leave_gold_apps_out_refuses_a_task_without_a_label():
    task = dataclasses.replace(TASK, context={**TASK.context, "installed_apps": {}})
    agent = ReActAgent(llm=FakeLLM(["Xong."]), examples=2, bank=StubBank("EX"), leave_gold_apps_out=True)
    with pytest.raises(ValueError, match="label"):
        agent.run(task, FakeEnv())


def test_leave_gold_apps_out_off_never_passes_the_keyword():
    bank = StubBank("EX")  # its select() has no exclude_apps parameter
    agent = ReActAgent(llm=FakeLLM(["Xong."]), examples=2, bank=bank)
    agent.run(dataclasses.replace(TASK, context={**TASK.context, "installed_apps": {"alarm": {}}}), FakeEnv())
    assert agent.info["leave_gold_apps_out"] is False and len(bank.asked) == 1


def test_leave_gold_apps_out_reaches_the_prompt_vote_selects_too():
    seen = []

    class Bank(StubBank):
        def select(self, request, installed_apps, k, exclude_id, offset=0, exclude_apps=frozenset()):
            seen.append(exclude_apps)
            return super().select(request, installed_apps, k, exclude_id, offset)

    label = {"tool_calls": [{"name": "alarm.delete", "arguments": {"alarm_id": "a1"}}]}
    task = dataclasses.replace(VOTE_TASK, label=label)
    agent = ReActAgent(llm=FakeLLM([DEL("a1"), DEL("a1"), DEL("a1"), "Xong."]), examples=2, bank=Bank("EX"),
                       vote_prompts=3, leave_gold_apps_out=True)
    agent.run(task, FakeEnv())
    assert len(seen) == 3 and all(s == {"alarm"} for s in seen)
