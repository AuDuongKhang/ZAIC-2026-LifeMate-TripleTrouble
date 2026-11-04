from lifemate.asr.context import context_prompt
from lifemate.asr_prompt_check import check, echo

SAMPLE = {
    "id": "s", "audio": "a.wav", "audio_text": "SECRETFINAL",
    "label": {"tool_calls": [{"name": "x", "arguments": {"content": "SECRETLABEL"}}], "tool_results": ["SECRETLABEL"]},
    "context": {"current_time": "2026-11-13T12:00:00+07:00", "user_profile": {"name": "Anh Khoa", "city": "TP.HCM"}},
    "history": [
        {"role": "user", "audio": "u1.wav", "audio_text": "SECRETTURN", "text": "SECRETTEXT"},
        {"role": "assistant", "text": "Mai có họp sớm.", "tool_calls": [{"name": "c", "arguments": {"q": "SECRETARG"}}],
         "tool_results": [{"ok": True, "id": "alm_01", "status": "confirmed", "time": "2026-11-14T06:15:00+07:00",
                           "phone": "09870829665", "email": "a@b.test", "contacts": [{"id": "c_1", "name": "Minh"}],
                           "label": "họp sớm"}]},
    ],
}


def test_prompt_never_reads_transcripts_labels_or_user_turns():
    assert "SECRET" not in context_prompt(SAMPLE)


def test_prompt_keeps_names_and_phrases_and_drops_ids_values_and_contacts_details():
    words = context_prompt(SAMPLE).split(", ")
    assert {"Anh Khoa", "Minh", "họp sớm", "Mai có họp sớm."} <= set(words)
    assert not {"alm_01", "c_1", "confirmed", "09870829665", "a@b.test", "2026-11-14T06:15:00+07:00"} & set(words)


def test_prompt_keeps_an_address_that_starts_with_a_number():
    sample = {"context": {"user_profile": {"home_address": "12 Nguyễn Huệ, Quận 1"}}, "history": []}
    assert context_prompt(sample) == "12 Nguyễn Huệ, Quận 1"


def test_repeated_phrase_keeps_its_latest_place():
    sample = {"context": {}, "history": [{"role": "assistant", "text": "Lan"}, {"role": "assistant", "text": "Minh"},
                                         {"role": "assistant", "text": "Lan"}]}
    assert context_prompt(sample) == "Minh, Lan"


def test_empty_sample_gives_no_prompt():
    assert context_prompt({"context": {}, "history": []}) == ""


def test_echo_ignores_what_the_speaker_said():
    prompt = "gọi cho anh Minh bên kế toán"
    assert echo("gọi cho anh Minh bên kế toán", "gọi cho anh Minh bên kế toán", prompt) == set()
    assert echo("đặt báo thức gọi cho anh Minh", "đặt báo thức", prompt) == {("gọi", "cho", "anh", "minh")}


def test_check_counts_echoes_and_final_request_wer():
    sample = {"audio": "a.wav", "audio_text": "đặt báo thức", "context": {}, "history": [
        {"role": "user", "audio": "u.wav", "audio_text": "chào em"},
        {"role": "assistant", "text": "gọi cho anh Minh nhé"}]}
    rows = {"a.wav": {"text": "đặt báo thức gọi cho anh Minh"}, "u.wav": {"text": "chào em"}}
    result = check(rows, [sample])
    assert result["clips"] == 2 and result["echo_clips"] == 1 and result["echoes"][0]["audio"] == "a.wav"
    assert result["final_wer"] > 0 and abs(result["wer"] - result["final_wer"] / 2) < 0.001
