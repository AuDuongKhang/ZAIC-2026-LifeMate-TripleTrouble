"""Correct write-call arguments that misread an earlier read result in the same conversation.

Two systematic LLM copy errors after a read call:
1. After a unit conversion returning `{"value": V, "unit": U, "from_value": FV}` (e.g. `calculator.convert_unit`),
   the model calls `shopping_list.add` with `quantity=V` (or `FV`) instead of `quantity=1` and leaves `"<V> <U>"` out
   of `name`.
2. When the request asks to transfer from the checking account first and only fall back to savings if checking is
   insufficient ("tài khoản thanh toán ... không đủ ... tiết kiệm"), and `banking.get_balance` shows the checking
   balance covers the total transfer amount of the turn, the model still passes the savings account id in
   `from_account_id`.

Shared by the simulator (`lifemate.sim`, rewrite "result_fixes") and the agent (`result_fixes=True`).
"""
from __future__ import annotations

from .ask_guard import fold


def _format_num(val: int | float) -> str:
    return str(int(val)) if isinstance(val, float) and val.is_integer() else str(val)


def fix_converted_item(call: dict, prior_calls: list[dict]) -> dict:
    """Fix `shopping_list.add` when `quantity` copied the converted value instead of `1`."""
    if call.get("name") != "shopping_list.add":
        return call
    args = call.get("arguments") or {}
    qty, name = args.get("quantity"), args.get("name")
    if not isinstance(qty, (int, float)) or not isinstance(name, str):
        return call
    conv = None
    for prev in reversed(prior_calls):
        res = prev.get("result")
        if isinstance(res, dict) and "value" in res and "unit" in res and "from_value" in res:
            conv = res
            break
    if conv is None:
        return call
    val, unit, fval = conv.get("value"), conv.get("unit"), conv.get("from_value")
    if not isinstance(val, (int, float)) or not isinstance(unit, str) or val == 1:
        return call
    matches_val = abs(qty - val) < 1e-6 or (isinstance(val, float) and qty == round(val))
    matches_fval = isinstance(fval, (int, float)) and (abs(qty - fval) < 1e-6 or qty == round(fval))
    if not (matches_val or matches_fval):
        return call
    v_str = _format_num(val)
    new_name = name if v_str in name else f"{name.rstrip()} {v_str} {unit}"
    if qty == 1 and new_name == name:
        return call
    return dict(call, arguments={**args, "name": new_name, "quantity": 1})


def _prefers_checking(request: str) -> bool:
    folded = " ".join(fold(request))
    return "thanh toan" in folded and "khong du" in folded and "tiet kiem" in folded


def fix_checking_account(call: dict, batch_calls: list[dict], prior_calls: list[dict], request: str) -> dict:
    """Use the checking account id in `banking.transfer` when its balance covers the requested transfer(s)."""
    if call.get("name") != "banking.transfer" or not _prefers_checking(request):
        return call
    accounts = None
    for prev in reversed(prior_calls):
        res = prev.get("result")
        if prev.get("name") == "banking.get_balance" and isinstance(res, dict) and isinstance(res.get("accounts"), list):
            accounts = res["accounts"]
            break
    if not accounts:
        return call
    checking = next(
        (a for a in accounts if isinstance(a, dict)
         and ("thanh toan" in " ".join(fold(str(a.get("name", "")))) or a.get("type") == "checking")),
        None,
    )
    if not checking or not isinstance(checking.get("balance"), (int, float)) or not checking.get("id"):
        return call
    total = sum(
        c.get("arguments", {}).get("amount", 0)
        for c in batch_calls
        if c.get("name") == "banking.transfer" and isinstance(c.get("arguments", {}).get("amount"), (int, float))
    )
    args = call.get("arguments") or {}
    if total <= 0 or checking["balance"] < total or args.get("from_account_id") == checking["id"]:
        return call
    return dict(call, arguments={**args, "from_account_id": checking["id"]})


def fix_result_call(call: dict, batch_calls: list[dict], prior_calls: list[dict], request: str) -> dict:
    """Apply result-based argument fixes to a single `call`."""
    fixed = fix_converted_item(call, prior_calls)
    return fix_checking_account(fixed, batch_calls, prior_calls, request)


def fix_result_calls(calls: list[dict], prior_calls: list[dict], request: str) -> list[dict]:
    """Apply `fix_result_call` across `calls`; returns `calls` itself when nothing changes."""
    fixed = [fix_result_call(c, calls, prior_calls, request) for c in calls]
    return calls if all(f is c for f, c in zip(fixed, calls)) else fixed


def result_fixes_rewrite():
    """Simulator rewrite (see lifemate.sim): apply `fix_result_call` across each trace's calls."""
    def rewrite(trace: dict, catalog: dict[str, dict]) -> list[dict] | None:
        prior: list[dict] = []
        raw_calls = trace.get("calls") or []
        out, changed = [], False
        for c in raw_calls:
            new = fix_result_call(c, raw_calls, prior, trace.get("request") or "")
            changed |= new["arguments"] != c["arguments"]
            out.append({"name": new["name"], "arguments": new["arguments"]})
            prior.append(c)
        return out if changed else None
    return rewrite
