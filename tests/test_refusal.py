from lifemate.agents.refusal import matching_tools

END = {"name": "parking.end_session", "description": "Kết thúc phiên gửi xe", "parameters": {"type": "object", "properties": {}}}
START = {"name": "parking.start_session", "description": "Bắt đầu phiên gửi xe", "parameters": {"type": "object", "properties": {}}}
ALARM = {"name": "alarm.create", "description": "Đặt báo thức mới", "parameters": {"type": "object", "properties": {}}}
ASK = {"name": "sys.ask_missing_info", "description": "Hỏi lại người dùng kết thúc phiên", "parameters": {"type": "object", "properties": {}}}


def test_parking_request_matches_end_session():
    assert matching_tools("kết thúc phiên gửi xe ở toà nhà", [END, ALARM]) == ["parking.end_session"]


def test_unrelated_request_matches_nothing():
    assert matching_tools("bật đèn phòng khách", [END, START, ALARM]) == []


def test_best_first_ties_by_name_and_at_most_k():
    tools = [START, END, ALARM]
    # END: 4 shared bigrams, START: 2 ("phien gui", "gui xe"), so END comes first.
    assert matching_tools("kết thúc phiên gửi xe", tools, k=2) == ["parking.end_session", "parking.start_session"]
    assert matching_tools("kết thúc phiên gửi xe", tools, k=1) == ["parking.end_session"]
    assert matching_tools("phiên gửi xe", [START, END]) == ["parking.end_session", "parking.start_session"]  # tie: by name


def test_min_hits_raises_the_bar():
    assert matching_tools("phiên gửi", [END], min_hits=2) == []
    assert matching_tools("phiên gửi xe", [END], min_hits=2) == ["parking.end_session"]


def test_tool_name_parts_count_and_sys_tools_are_excluded():
    assert matching_tools("end session", [END]) == ["parking.end_session"]
    assert matching_tools("kết thúc phiên", [ASK]) == []


def test_a_single_word_request_has_no_bigram():
    assert matching_tools("kết", [END]) == []
