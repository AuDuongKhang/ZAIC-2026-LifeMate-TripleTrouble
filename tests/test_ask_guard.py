from lifemate.agents.ask_guard import ask_instead, candidate_lists, fold, match_score

RESERVE = {"name": "bike_rental.reserve", "kind": "write", "parameters": {"type": "object", "required": ["bike_id"],
           "properties": {"bike_id": {"type": "string", "x-ref": "bike_rental.bikes"}}}}
BIKES = [{"id": "b1", "station": "Trạm Biển Đông", "available": True},
         {"id": "b2", "station": "Trạm Cầu Rồng", "available": True}]


def _candidates(items):
    out = {}
    candidate_lists({"ok": True, "bikes": items}, out)
    return out


def test_fold_strips_diacritics():
    assert fold("Đường Lò Đúc") == ["duong", "lo", "duc"]


def test_match_score_is_the_share_of_the_items_words_in_the_request():
    assert match_score("giữ xe ở trạm Biển Đông", BIKES[0]) == 1.0
    assert abs(match_score("giữ xe ở trạm Biển Đông", BIKES[1]) - 1 / 3) < 1e-9


def test_no_ask_when_the_request_names_the_station():
    call = {"name": "bike_rental.reserve", "arguments": {"bike_id": "b1"}}
    assert ask_instead("giữ xe ở trạm Biển Đông", call, RESERVE, _candidates(BIKES)) is None


def test_ask_when_nothing_in_the_request_picks_one():
    call = {"name": "bike_rental.reserve", "arguments": {"bike_id": "b1"}}
    assert ask_instead("giữ giùm một chiếc xe", call, RESERVE, _candidates(BIKES)) == {
        "name": "sys.ask_missing_info", "arguments": {"tool_name": "bike_rental.reserve", "params": ["bike_id"]}}


def test_ask_when_the_pick_is_not_clearly_better_than_another():
    bikes = BIKES + [{"id": "b3", "station": "Trạm Biển Đông", "available": True}]
    call = {"name": "bike_rental.reserve", "arguments": {"bike_id": "b1"}}
    assert ask_instead("giữ xe ở trạm Biển Đông", call, RESERVE, _candidates(bikes)) is not None


def test_no_ask_for_a_single_candidate():
    call = {"name": "bike_rental.reserve", "arguments": {"bike_id": "b1"}}
    assert ask_instead("giữ giùm một chiếc xe", call, RESERVE, _candidates(BIKES[:1])) is None


def test_no_ask_for_read_calls():
    read = dict(RESERVE, kind="read")
    call = {"name": "bike_rental.reserve", "arguments": {"bike_id": "b1"}}
    assert ask_instead("giữ giùm một chiếc xe", call, read, _candidates(BIKES)) is None


def test_no_ask_for_an_id_that_came_from_no_list():
    call = {"name": "bike_rental.reserve", "arguments": {"bike_id": "b9"}}
    assert ask_instead("giữ giùm một chiếc xe", call, RESERVE, _candidates(BIKES)) is None


def test_candidate_lists_find_nested_lists_of_items():
    out = {}
    candidate_lists({"ok": True, "data": {"bikes": BIKES}, "one": [{"id": "x"}]}, out)
    assert set(out) == {"b1", "b2"} and out["b1"] == BIKES
