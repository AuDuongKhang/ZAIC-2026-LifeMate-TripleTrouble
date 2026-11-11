from lifemate.agents.result_fixes import fix_checking_account, fix_result_calls, result_fixes_rewrite


def test_fix_checking_account_uses_checking_when_balance_covers_total():
    prior = [{"name": "banking.get_balance", "arguments": {},
              "result": {"accounts": [{"id": "acc_chk", "name": "Tài khoản thanh toán", "balance": 5000000},
                                      {"id": "acc_sav", "name": "Tài khoản tiết kiệm", "balance": 20000000}]}}]
    batch = [{"name": "banking.transfer", "arguments": {"from_account_id": "acc_sav", "to": "A", "amount": 2000000}},
             {"name": "banking.transfer", "arguments": {"from_account_id": "acc_sav", "to": "B", "amount": 1500000}}]
    req = "Chuyển 2 triệu cho A và 1.5 triệu cho B, lấy từ tài khoản thanh toán, nếu không đủ thì lấy tiết kiệm"
    fixed = fix_result_calls(batch, prior, req)
    assert [c["arguments"]["from_account_id"] for c in fixed] == ["acc_chk", "acc_chk"]


def test_fix_checking_account_keeps_savings_when_checking_is_insufficient():
    prior = [{"name": "banking.get_balance", "arguments": {},
              "result": {"accounts": [{"id": "acc_chk", "name": "Tài khoản thanh toán", "balance": 1000000},
                                      {"id": "acc_sav", "name": "Tài khoản tiết kiệm", "balance": 20000000}]}}]
    call = {"name": "banking.transfer", "arguments": {"from_account_id": "acc_sav", "to": "A", "amount": 2000000}}
    req = "Chuyển 2 triệu cho A, ưu tiên tài khoản thanh toán, nếu không đủ thì lấy tiết kiệm"
    assert fix_checking_account(call, [call], prior, req) is call


def test_result_fixes_rewrite_returns_none_when_unchanged():
    rw = result_fixes_rewrite()
    trace = {"id": "t1", "request": "bật đèn", "calls": [{"name": "smart_home.light_set", "arguments": {"power": "on"}}]}
    assert rw(trace, {}) is None
