from lifemate.agents.id_fix import IDS, entity_names, fix_call_ids, fix_calls_ids

REDEEM = {"name": "loyalty.redeem", "kind": "write",
          "parameters": {"properties": {"program": {"type": "string"}, "reward_id": {"type": "string"}}}}
NOTE = {"name": "notes.create", "kind": "write",
        "parameters": {"properties": {"title": {"type": "string", "x-match": "fuzzy"}}}}
POINTS = {"name": "loyalty.get_points", "kind": "read", "parameters": {"properties": {"program": {"type": "string"}}}}
RESULT = {"ok": True, "programs": [{"id": "prog_815", "name": "Cao Nguyên Rewards",
                                    "rewards": [{"id": "rw_02_1", "name": "Ly phin"}]}]}


def test_entity_names_maps_folded_names_titles_labels_to_ids():
    assert entity_names(RESULT, {}) == {"cao nguyên rewards": {"prog_815"}, "ly phin": {"rw_02_1"},
                                        IDS: {"prog_815", "rw_02_1"}}
    assert entity_names([{"id": "a1", "title": "Họp nhóm", "label": "Nhà"}], {}) == {"họp nhóm": {"a1"}, "nhà": {"a1"},
                                                                                    IDS: {"a1"}}


def test_id_of_one_entity_that_names_another_stays():
    names = entity_names([{"id": "c1", "name": "x"}, {"id": "c2", "name": "c1"}], {})
    call = {"name": "loyalty.redeem", "arguments": {"program": "c1", "reward_id": "r"}}
    assert fix_call_ids(call, REDEEM, names) is call


def test_blank_argument_is_left_alone():
    call = {"name": "loyalty.redeem", "arguments": {"program": "  ", "reward_id": "r"}}
    assert fix_call_ids(call, REDEEM, entity_names(RESULT, {})) is call


def test_name_of_a_seen_entity_becomes_its_id():
    call = {"name": "loyalty.redeem", "arguments": {"program": " cao nguyên REWARDS", "reward_id": "rw_02_1"}}
    fixed = fix_call_ids(call, REDEEM, entity_names(RESULT, {}))
    assert fixed["arguments"] == {"program": "prog_815", "reward_id": "rw_02_1"}
    assert call["arguments"]["program"] == " cao nguyên REWARDS"


def test_seen_id_free_text_read_and_non_strings_are_left_alone():
    names = entity_names(RESULT, {})
    for spec, call in [(REDEEM, {"name": "loyalty.redeem", "arguments": {"program": "prog_815", "reward_id": "rw_02_1"}}),
                       (NOTE, {"name": "notes.create", "arguments": {"title": "Cao Nguyên Rewards"}}),
                       (POINTS, {"name": "loyalty.get_points", "arguments": {"program": "Cao Nguyên Rewards"}}),
                       (REDEEM, {"name": "loyalty.redeem", "arguments": {"program": 5, "reward_id": "rw_02_1"}}),
                       (None, {"name": "x.unknown", "arguments": {"program": "Cao Nguyên Rewards"}})]:
        assert fix_call_ids(call, spec, names) is call


def test_an_id_that_is_also_a_name_stays():
    names = entity_names([{"id": "Phòng khách", "name": "Phòng khách"}], {})
    call = {"name": "loyalty.redeem", "arguments": {"program": "Phòng khách", "reward_id": "r"}}
    assert fix_call_ids(call, REDEEM, names) is call


def test_a_name_shared_by_two_entities_is_left_alone():
    names = entity_names([{"id": "p1", "name": "MinhPoint"}, {"id": "p2", "name": "minhpoint"}], {})
    call = {"name": "loyalty.redeem", "arguments": {"program": "MinhPoint", "reward_id": "r"}}
    assert fix_call_ids(call, REDEEM, names) is call


def test_fix_calls_ids_returns_the_same_list_when_nothing_changes():
    calls = [{"name": "loyalty.redeem", "arguments": {"program": "prog_815", "reward_id": "rw_02_1"}}]
    assert fix_calls_ids(calls, {"loyalty.redeem": REDEEM}, entity_names(RESULT, {})) is calls
    by_name = [{"name": "loyalty.redeem", "arguments": {"program": "Cao Nguyên Rewards", "reward_id": "rw_02_1"}}]
    fixed = fix_calls_ids(by_name, {"loyalty.redeem": REDEEM}, entity_names(RESULT, {}))
    assert fixed is not by_name and fixed[0]["arguments"]["program"] == "prog_815"
