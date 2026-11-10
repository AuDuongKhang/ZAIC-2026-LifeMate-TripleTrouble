import pytest

from lifemate.agents.spoken import respell_emails


@pytest.mark.parametrize("heard, said", [
    ("gửi Family Account Example chấm tết báo là", "gửi Family a còng example chấm test báo là"),
    ("gửi Family Item Example chấm test, báo", "gửi Family a còng Example chấm test, báo"),
    ("địa chỉ vendor acong example.step nhé", "địa chỉ vendor a còng example chấm test nhé"),
    ("gửi nhóm bảy mươi ba a còng example chấm test báo", "gửi nhóm bảy mươi ba a còng example chấm test báo"),
])
def test_misheard_email_parts_get_the_spoken_form(heard, said):
    assert respell_emails(heard) == said


@pytest.mark.parametrize("text", ["Cho anh xem một example về cách nấu phở", "Kiểm tra account ngân hàng",
                                  "thư nháp gửi Lego icon example.test, bằng câu đã dịch",
                                  "Đặt báo thức sáu giờ", ""])
def test_text_without_an_email_is_unchanged(text):
    assert respell_emails(text) == text
