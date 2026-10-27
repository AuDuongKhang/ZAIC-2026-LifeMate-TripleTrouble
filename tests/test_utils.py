from lifemate.utils import parse_kw


def test_parse_kw_types():
    assert parse_kw(["model=Qwen/Qwen3-4B-Instruct-2507", "max_steps=6", "time_budget_s=12.5", "flag=true"]) == {
        "model": "Qwen/Qwen3-4B-Instruct-2507", "max_steps": 6, "time_budget_s": 12.5, "flag": True}


def test_parse_kw_keeps_equals_in_value():
    assert parse_kw(["prompt=a=b"]) == {"prompt": "a=b"}


def test_parse_kw_empty():
    assert parse_kw(None) == {}
