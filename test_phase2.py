"""
Phase 2 tests. No voice, no GPU, no models.   Run:  python -m pytest test_phase2.py -v

The 'caller' is a list of typed turns, so the same dialogue code that runs in
the voice pipeline is exercised end to end against a temporary seeded DB.
"""
import re
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "db"))
import seed_students  # noqa: E402

from dialogue import (DIGIT_WORDS, MAX_FAILURES, DialogueState, WifiDiagnosisDialogue,  # noqa: E402
                      extract_digits, normalize, parse_new_device, parse_yes_no,
                      speak_digits)
from session import SessionIdentity, establish_identity_stub  # noqa: E402
from student_tools import (get_notification_history, get_verification_status,  # noqa: E402
                           get_wifi_status, lookup_student)
from wifi_diagnosis import PROVISIONED_STATES, WifiState  # noqa: E402

COMPLAINT = "my wifi is not working"


# ----------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def db(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("db") / "students.sqlite")
    manifest = seed_students.seed(path, include_edge=True)
    return path, manifest


def make_dialogue(db_path, tmp_path, intent=True):
    return WifiDiagnosisDialogue(intent_fn=lambda _t: intent, db_path=db_path,
                                 log_dir=str(tmp_path / "logs"))


def say_digits(enrol):
    return " ".join(DIGIT_WORDS[d] for d in enrol)


def run_walk(dlg, enrol, device_answer=None):
    """Drive one full call. Returns list of agent utterances."""
    out = []

    def turn(text):
        r = dlg.handle(text)
        assert r.handled
        if r.text:
            out.append(r.text)
        return r

    turn(COMPLAINT)
    turn(say_digits(enrol))
    turn("yes")
    if dlg.state is DialogueState.ASK_DEVICE:
        assert device_answer is not None, "device question asked but no answer supplied"
        turn(device_answer)
    return out


# --------------------------------------------------------------- parsing
@pytest.mark.parametrize("text,expected", [
    ("1 0 0 0 0 1", "100001"),
    ("100001", "100001"),
    ("100,001", "100001"),
    ("1-0-0-0-0-1", "100001"),
    ("one zero zero zero zero one", "100001"),
    ("one oh oh oh oh one", "100001"),
    ("my number is one zero zero zero zero one", "100001"),
    ("double zero one two three four", "001234"),
    ("triple five one two three", "555123"),
    ("hello there", ""),
    ("", ""),
])
def test_extract_digits(text, expected):
    assert extract_digits(text) == expected


def test_speak_digits():
    assert speak_digits("100001") == "one, zero, zero, zero, zero, one"


@pytest.mark.parametrize("text,expected", [
    ("yes", True), ("Yeah.", True), ("that's right", True), ("correct", True),
    ("no", False), ("nope", False), ("that's wrong", False),
    ("no that is not right", None),     # contradictory
    ("banana", None), ("", None),
])
def test_yes_no(text, expected):
    assert parse_yes_no(normalize(text)) is expected


@pytest.mark.parametrize("text,expected", [
    ("yes", True), ("no", False),
    ("I have a new phone", True), ("I got a different laptop", True),
    ("no, same phone", False), ("it's the same device", False),
    ("I haven't changed it", False),
    ("hmm", None),
])
def test_new_device(text, expected):
    assert parse_new_device(normalize(text)) is expected


# ----------------------------------------------------------------- tools
def test_tools_require_session_identity(db):
    path, _ = db
    for tool in (lookup_student, get_wifi_status, get_verification_status, get_notification_history):
        with pytest.raises(TypeError):
            tool("100001", path)              # raw string must be rejected
        with pytest.raises(TypeError):
            tool({"enrolment": "100001"}, path)


def test_identity_is_immutable():
    ident = SessionIdentity("100001", "x")
    with pytest.raises(Exception):
        ident.enrolment = "999999"


def test_auth_stub_unknown_enrolment(db):
    path, _ = db
    assert establish_identity_stub("999999", path) is None
    assert establish_identity_stub("100001", path).enrolment == "100001"


def test_lookup_returns_minimal_fields_only(db):
    path, _ = db
    ident = establish_identity_stub("100006", path)
    student = lookup_student(ident, path)
    assert set(student) == {"enrolment", "first_name", "has_hall", "mobile_verified"}
    assert "mobile" not in student and "email" not in student


def test_verification_normalisation_with_malformed_entries(db):
    path, _ = db
    v = get_verification_status(establish_identity_stub("200007", path), path)
    assert v["provost_approved"] is True            # valid entry survived the garbage
    assert v["required_approver"] == "provost" and v["required_approved"] is True


def test_verification_null_is_pending_not_crash(db):
    path, _ = db
    v = get_verification_status(establish_identity_stub("200008", path), path)
    assert v["required_approved"] is False and v["rejected"] is False


def test_missing_and_null_services(db):
    path, _ = db
    for enrol in ("200003", "200004"):
        w = get_wifi_status(establish_identity_stub(enrol, path), path)
        assert w["has_wifi_entry"] is False and w["applied"] is False


def test_wifi_found_regardless_of_position(db):
    path, _ = db
    w = get_wifi_status(establish_identity_stub("100002", path), path)  # variant 1: WiFi listed last
    assert w["has_wifi_entry"] is True


def test_rejection_details_come_through(db):
    path, manifest = db
    v = get_verification_status(establish_identity_stub("100011", path), path)
    assert v["rejected"] and v["rejected_by_role"] == "provost"
    assert v["rejection_comment"] == manifest["100011"]["comment"]


# ------------------------------------- THE GATE: all 7 states, every record
STANDARD = list(seed_students.SEED_MANIFEST.items())
EDGE = [(d["enroll"], m) for d, m in seed_students.build_edge_cases()]


@pytest.mark.parametrize("enrol,expected", STANDARD + EDGE, ids=[e for e, _ in STANDARD + EDGE])
def test_every_record_reaches_the_right_diagnosis(db, tmp_path, enrol, expected):
    path, _ = db
    dlg = make_dialogue(path, tmp_path)
    ans = None
    if expected["new_device"] is not None:
        ans = "yes I have a new phone" if expected["new_device"] else "no"
    run_walk(dlg, enrol, ans)

    d = dlg.last_diagnosis
    assert d is not None, "dialogue ended without a diagnosis"
    assert d.state.value == expected["state"]

    # no unnecessary questions: provisioned accounts get exactly one, others none
    assert dlg.device_question_asked == (d.state in PROVISIONED_STATES)

    if expected.get("approver") and expected["state"] == "verification_pending":
        assert d.detail["pending_approver"] == expected["approver"]
    if expected.get("approver") and expected["state"] == "rejected":
        assert d.detail["rejected_by"] == expected["approver"]
    assert dlg.state is DialogueState.IDLE         # session cleaned up


def test_rejection_comment_is_read_back(db, tmp_path):
    path, manifest = db
    dlg = make_dialogue(path, tmp_path)
    out = run_walk(dlg, "100011")
    assert manifest["100011"]["comment"].rstrip(".") in out[-1]


def test_rejection_without_comment_does_not_read_garbage(db, tmp_path):
    path, _ = db
    dlg = make_dialogue(path, tmp_path)
    out = run_walk(dlg, "200002")
    assert "No reason was recorded" in out[-1]


def test_state_actions(db, tmp_path):
    path, _ = db
    expected = {"100016": "expedite_ticket_stub", "100021": "mac_update_stub",
                "100031": "technical_followup_stub", "100001": "none"}
    for enrol, action in expected.items():
        dlg = make_dialogue(path, tmp_path)
        ans = "yes" if enrol == "100021" else "no"
        run_walk(dlg, enrol, ans)
        assert dlg.last_diagnosis.action == action, enrol


# ------------------------------------------------- spoken-response hygiene
def all_responses(db, tmp_path):
    path, _ = db
    responses = []
    for enrol, exp in STANDARD + EDGE:
        dlg = make_dialogue(path, tmp_path)
        ans = None
        if exp["new_device"] is not None:
            ans = "yes" if exp["new_device"] else "no"
        responses.append((enrol, run_walk(dlg, enrol, ans)[-1]))
    return responses


def test_responses_are_phone_friendly(db, tmp_path):
    for enrol, text in all_responses(db, tmp_path):
        assert not re.search(r"[0-9*#_\[\]<>{}|]", text), f"{enrol}: {text!r}"
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]
        assert len(sentences) <= 4, f"{enrol}: too many sentences"
        assert len(text.split()) <= 70, f"{enrol}: too long for a phone call"


# ----------------------------------------------------- dialogue behaviour
def test_not_our_turn_passes_through(db, tmp_path):
    path, _ = db
    dlg = make_dialogue(path, tmp_path, intent=False)
    r = dlg.handle("how long does approval take")
    assert r.handled is False and dlg.state is DialogueState.IDLE


def test_number_in_first_utterance_skips_the_ask(db, tmp_path):
    path, _ = db
    dlg = make_dialogue(path, tmp_path)
    r = dlg.handle("my wifi is not working my number is one zero zero zero zero six")
    assert dlg.state is DialogueState.CONFIRM_ENROLMENT
    assert "one, zero, zero, zero, zero, six" in r.text


def test_wrong_readback_then_correct(db, tmp_path):
    path, _ = db
    dlg = make_dialogue(path, tmp_path)
    dlg.handle(COMPLAINT)
    dlg.handle("one zero zero zero zero seven")     # misheard
    dlg.handle("no")
    assert dlg.state is DialogueState.ASK_ENROLMENT
    dlg.handle("one zero zero zero zero six")
    dlg.handle("yes")
    assert dlg.last_diagnosis.state is WifiState.VERIFICATION_PENDING


def test_caller_restates_number_instead_of_saying_no(db, tmp_path):
    path, _ = db
    dlg = make_dialogue(path, tmp_path)
    dlg.handle(COMPLAINT)
    dlg.handle("one zero zero zero zero seven")
    r = dlg.handle("no it is one zero zero zero zero six")
    assert "one, zero, zero, zero, zero, six" in r.text
    assert dlg.state is DialogueState.CONFIRM_ENROLMENT


def test_unknown_enrolment_escalates_after_max_failures(db, tmp_path):
    path, _ = db
    dlg = make_dialogue(path, tmp_path)
    dlg.handle(COMPLAINT)
    last = None
    for _ in range(MAX_FAILURES):
        dlg.handle("nine nine nine nine nine nine")
        last = dlg.handle("yes")
    assert "note this down" in last.text
    assert dlg.state is DialogueState.IDLE
    log = (tmp_path / "logs" / "stub_actions.log").read_text()
    assert "ticket_stub" in log


def test_no_digits_repeatedly_escalates(db, tmp_path):
    path, _ = db
    dlg = make_dialogue(path, tmp_path)
    dlg.handle(COMPLAINT)
    r = None
    for _ in range(MAX_FAILURES):
        r = dlg.handle("I don't remember it")
    assert "note this down" in r.text and dlg.state is DialogueState.IDLE


def test_whisper_noise_is_ignored_midflow(db, tmp_path):
    path, _ = db
    dlg = make_dialogue(path, tmp_path)
    dlg.handle(COMPLAINT)
    for noise in ("Thank you.", "Thanks for watching!", "you", ""):
        r = dlg.handle(noise)
        assert r.handled and r.text is None
    assert dlg.failures == 0 and dlg.state is DialogueState.ASK_ENROLMENT


def test_cancel_mid_flow(db, tmp_path):
    path, _ = db
    dlg = make_dialogue(path, tmp_path)
    dlg.handle(COMPLAINT)
    r = dlg.handle("never mind")
    assert r.handled and dlg.state is DialogueState.IDLE


def test_unclear_device_answer_reprompts_then_escalates(db, tmp_path):
    path, _ = db
    dlg = make_dialogue(path, tmp_path)
    dlg.handle(COMPLAINT)
    dlg.handle("one zero zero zero three one")
    dlg.handle("yes")
    assert dlg.state is DialogueState.ASK_DEVICE
    r = dlg.handle("hmm maybe")                       # failure 1: re-prompt
    assert "yes or no" in r.text
    for _ in range(MAX_FAILURES - 1):                 # failures 2..MAX: escalate on the last
        r = dlg.handle("hmm maybe")
    assert "note this down" in r.text and dlg.state is DialogueState.IDLE


def test_two_calls_in_a_row_do_not_leak_state(db, tmp_path):
    path, _ = db
    dlg = make_dialogue(path, tmp_path)
    run_walk(dlg, "100001")
    first = dlg.last_diagnosis.state
    run_walk(dlg, "100016")
    assert first is WifiState.NEVER_APPLIED
    assert dlg.last_diagnosis.state is WifiState.PROVISIONING_BACKLOG
    assert dlg.identity is None


def test_caller_speech_cannot_choose_the_record(db, tmp_path):
    """The identity comes from the confirmed digits only; claims about whose account it is are ignored."""
    path, _ = db
    dlg = make_dialogue(path, tmp_path)
    dlg.handle("my friend's wifi is not working, he is one zero zero zero zero one")
    dlg.handle("yes")
    assert dlg.last_diagnosis.state is WifiState.NEVER_APPLIED   # reads exactly the confirmed number


# ------------------------------------------------------ review-fix regressions
@pytest.mark.parametrize("text", [
    "no, I bought a new phone",
    "yes, the same phone",
])
def test_contradictory_device_answer_is_unclear(text):
    assert parse_new_device(normalize(text)) is None


@pytest.mark.parametrize("text,expected", [
    ("I haven't changed it", False),
    ("I have a new phone", True),
    ("no, same phone", False),
    ("no new phone", False),
    ("yes, I got a new laptop", True),
])
def test_device_answer_evidence(text, expected):
    assert parse_new_device(normalize(text)) is expected


def test_new_device_without_mac_on_file_is_not_called_mismatch(db, tmp_path):
    path, _ = db
    dlg = make_dialogue(path, tmp_path)
    run_walk(dlg, "200010", "yes")
    assert dlg.last_diagnosis.state.value == "mac_not_on_file"
    assert "cannot confirm" in dlg.last_diagnosis.response.lower()


def test_account_fine_response_does_not_claim_device_is_registered(db, tmp_path):
    path, _ = db
    dlg = make_dialogue(path, tmp_path)
    run_walk(dlg, "100031", "no")
    response = dlg.last_diagnosis.response.lower()
    assert "registered to your device" not in response
    assert "does not confirm" in response


def test_dialogue_times_out_after_inactivity(db, tmp_path):
    path, _ = db
    now = [100.0]
    dlg = WifiDiagnosisDialogue(
        intent_fn=lambda _text: True, db_path=path,
        log_dir=str(tmp_path / "logs"), turn_timeout_seconds=10,
        clock=lambda: now[0],
    )
    first = dlg.handle(COMPLAINT)
    assert first.handled and dlg.state is DialogueState.ASK_ENROLMENT
    now[0] += 8
    assert dlg.handle("thank you").text is None  # silence/noise must not extend the active turn
    now[0] += 3
    expired = dlg.handle("one zero zero zero zero one")
    assert expired.handled and "timed out" in expired.text.lower()
    assert dlg.state is DialogueState.IDLE


def test_infrastructure_guard_keeps_mixed_personal_complaints_in_walk():
    from intent_router import is_infrastructure_only

    assert is_infrastructure_only("the router in the library is broken")
    assert not is_infrastructure_only(
        "my laptop cannot connect to campus wifi; maybe the router is the problem"
    )
    assert not is_infrastructure_only("my wifi stopped working and the signal is weak")


def test_mixed_yes_and_no_device_answer_is_unclear():
    assert parse_new_device(normalize("yes, I have not changed my phone")) is None
