from lifemate.agents.empty_retry import relaxed_call

SEARCH = {"name": "contacts.search", "kind": "read",
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": []}}
FLIGHTS = {"name": "flights.search", "kind": "read", "parameters": {"type": "object", "required": ["origin", "destination", "date"],
           "properties": {"origin": {}, "destination": {}, "date": {}, "passengers": {"type": "integer"}}}}
EVENTS = {"name": "calendar.list_events", "kind": "read", "parameters": {"type": "object", "required": [],
          "properties": {"date_from": {}, "date_to": {}, "query": {}}}}
PLACE = {"name": "maps.search_place", "kind": "read",
         "parameters": {"type": "object", "properties": {"query": {}, "near": {}}, "required": ["query"]}}
SEND = {"name": "sms.send", "kind": "write", "parameters": {"type": "object", "properties": {"note": {}}, "required": []}}


def test_optional_filter_is_dropped():
    assert relaxed_call({"name": "contacts.search", "arguments": {"query": "Sếp Nguyễn"}}, SEARCH) == {
        "name": "contacts.search", "arguments": {}}


def test_required_arguments_stay():
    assert relaxed_call({"name": "maps.search_place", "arguments": {"query": "hiệu thuốc", "near": "Hàng Bài"}}, PLACE) == {
        "name": "maps.search_place", "arguments": {"query": "hiệu thuốc"}}


def test_optional_number_is_dropped():
    call = {"name": "flights.search", "arguments": {"origin": "HAN", "destination": "DAD", "date": "2026-10-03", "passengers": 2}}
    assert relaxed_call(call, FLIGHTS)["arguments"] == {"origin": "HAN", "destination": "DAD", "date": "2026-10-03"}


def test_nothing_to_drop_is_none():
    assert relaxed_call({"name": "flights.search", "arguments": {"origin": "HAN", "destination": "DAD", "date": "2026-10-03"}}, FLIGHTS) is None
    assert relaxed_call({"name": "contacts.search", "arguments": {}}, SEARCH) is None


def test_a_date_filter_is_kept():
    day = {"date_from": "2026-11-22", "date_to": "2026-11-22"}
    assert relaxed_call({"name": "calendar.list_events", "arguments": day}, EVENTS) is None
    assert relaxed_call({"name": "calendar.list_events", "arguments": {**day, "query": "họp"}}, EVENTS)["arguments"] == day


def test_only_read_tools():
    assert relaxed_call({"name": "sms.send", "arguments": {"note": "x"}}, SEND) is None
    assert relaxed_call({"name": "contacts.search", "arguments": {"query": "x"}}, None) is None


def test_the_call_is_not_mutated():
    call = {"name": "contacts.search", "arguments": {"query": "Lan"}}
    relaxed_call(call, SEARCH)
    assert call["arguments"] == {"query": "Lan"}
