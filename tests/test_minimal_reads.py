from lifemate.agents.minimal_reads import drop_unstated_optionals, minimal_reads_rewrite
from lifemate.sim import REWRITES

APPLIANCES = {"name": "appliances.list", "kind": "read", "parameters": {"properties": {
    "type": {"type": "string", "enum": ["washer", "dishwasher"]}}, "required": []}}
CONTACTS = {"name": "contacts.search", "kind": "read", "parameters": {"properties": {
    "query": {"type": "string"}}, "required": []}}
TRAINS = {"name": "trains.search", "kind": "read", "parameters": {"properties": {
    "to": {"type": "string"}, "seats": {"type": "integer"}, "kind": {"type": "string"}}, "required": ["to"]}}
CALL = {"name": "phone.call", "kind": "write", "parameters": {"properties": {
    "contact_id": {"type": "string"}}, "required": []}}


def test_unstated_filter_is_dropped():
    call = {"name": "appliances.list", "arguments": {"type": "dishwasher"}}
    assert drop_unstated_optionals(call, APPLIANCES, "máy rửa chén còn bao lâu xong") == {
        "name": "appliances.list", "arguments": {}}
    assert call["arguments"] == {"type": "dishwasher"}  # never mutates


def test_stated_query_is_kept():
    call = {"name": "contacts.search", "arguments": {"query": "Sếp Dũng"}}
    assert drop_unstated_optionals(call, CONTACTS, "nhắn sếp dũng") == call


def test_folded_match_keeps_the_value():
    call = {"name": "trains.search", "arguments": {"to": "Hà Nội", "kind": "Đà Nẵng"}}
    assert drop_unstated_optionals(call, TRAINS, "vé đi da nang") == call


def test_writes_are_untouched():
    call = {"name": "phone.call", "arguments": {"contact_id": "c9"}}
    assert drop_unstated_optionals(call, CALL, "gọi cho mẹ") == call


def test_required_args_are_kept():
    call = {"name": "trains.search", "arguments": {"to": "Huế", "kind": "ghế mềm", "seats": 2}}
    out = drop_unstated_optionals(call, TRAINS, "đặt vé tàu")
    assert out["arguments"] == {"to": "Huế", "seats": 2}  # required and non-string kept, unstated optional dropped


def test_no_spec_is_kept():
    call = {"name": "x.y", "arguments": {"q": "abc"}}
    assert drop_unstated_optionals(call, None, "") == call


def test_rewrite_changes_only_reads_and_reports_none_when_unchanged():
    catalog = {"appliances.list": APPLIANCES, "phone.call": CALL}
    trace = {"id": "t", "request": "máy rửa chén còn bao lâu xong", "calls": [
        {"name": "appliances.list", "arguments": {"type": "dishwasher"}, "result": {}},
        {"name": "phone.call", "arguments": {"contact_id": "c9"}, "result": {}}]}
    assert minimal_reads_rewrite()(trace, catalog) == [
        {"name": "appliances.list", "arguments": {}}, {"name": "phone.call", "arguments": {"contact_id": "c9"}}]
    assert minimal_reads_rewrite()(dict(trace, request="máy dishwasher"), catalog) is None


def test_registered_in_sim():
    assert REWRITES["minimal_reads"] is minimal_reads_rewrite
