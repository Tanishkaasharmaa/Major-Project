"""
Deterministic dialogue for the Wi-Fi account walk. No Pipecat, no LLM.

    IDLE -> ASK_ENROLMENT -> CONFIRM_ENROLMENT -> (ASK_DEVICE) -> IDLE

handle(user_text) returns a DialogueResult:
    handled=False            -> not our turn; let the FAQ path answer it
    handled=True, text=None  -> our turn, say nothing (e.g. Whisper noise)
    handled=True, text="..." -> say this, verbatim, via TTS (never via the LLM)

Everything the caller says is only ever used to (a) capture six digits and
(b) answer yes/no. Nothing the caller says can select a record directly: the
digits go through the auth stub, which is the only thing that can create a
SessionIdentity.
"""
import json
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from session import SessionIdentity, establish_identity_stub
from student_tools import (get_verification_status, get_wifi_status,
                           lookup_student)
from wifi_diagnosis import (DEVICE_QUESTION, PROVISIONED_STATES, Diagnosis,
                            build_diagnosis, classify_account,
                            finish_provisioned)

ENROLMENT_LENGTH = 6
MAX_FAILURES = 3          # failed capture/confirm attempts before escalating to a ticket

DIGIT_WORDS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
               "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
WORD_TO_DIGIT = {"zero": "0", "oh": "0", "nought": "0", "one": "1", "two": "2",
                 "three": "3", "four": "4", "five": "5", "six": "6",
                 "seven": "7", "eight": "8", "nine": "9"}

YES_WORDS = {"yes", "yeah", "yep", "yup", "correct", "right", "sure", "ok",
             "okay", "affirmative", "indeed"}
NO_WORDS = {"no", "nope", "nah", "wrong", "incorrect", "negative", "not",
            "never", "havent", "hasnt", "didnt", "dont", "wasnt"}
NEW_DEVICE_WORDS = {"new", "changed", "different", "replaced", "bought", "upgraded", "another"}
SAME_DEVICE_WORDS = {"same", "old", "original", "unchanged"}
CANCEL_PHRASES = ("never mind", "nevermind", "cancel", "forget it", "stop")
# Typical Whisper hallucinations on silence/noise: ignore mid-flow, don't count as a failed attempt.
NOISE_PHRASES = {"thank you", "thanks", "thank you very much", "you", "bye",
                 "bye bye", "thanks for watching", "thank you for watching"}


class DialogueState(str, Enum):
    IDLE = "idle"
    ASK_ENROLMENT = "ask_enrolment"
    CONFIRM_ENROLMENT = "confirm_enrolment"
    ASK_DEVICE = "ask_device"


@dataclass
class DialogueResult:
    handled: bool
    text: Optional[str] = None


NOT_HANDLED = DialogueResult(False, None)
SILENT = DialogueResult(True, None)


# ------------------------------------------------------------------ parsing
def normalize(text: str) -> str:
    text = (text or "").lower().replace("'", "").replace("\u2019", "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def extract_digits(text: str) -> str:
    """
    Pull a digit string out of an utterance. Handles '1 0 0 0 0 1', '100001',
    '100,001', 'one zero zero zero zero one' and 'double zero'. Words that are
    not digits are ignored (and reset any pending 'double').
    """
    tokens = re.findall(r"[a-z]+|\d+", (text or "").lower())
    out, repeat = [], 1
    for tok in tokens:
        if tok in ("double", "triple"):
            repeat = 2 if tok == "double" else 3
            continue
        if tok.isdigit():
            digits = tok
        elif tok in WORD_TO_DIGIT:
            digits = WORD_TO_DIGIT[tok]
        else:
            repeat = 1
            continue
        out.append(digits * repeat)
        repeat = 1
    return "".join(out)


def speak_digits(digits: str) -> str:
    return ", ".join(DIGIT_WORDS[d] for d in digits)


def parse_yes_no(norm: str) -> Optional[bool]:
    tokens = set(norm.split())
    yes, no = bool(tokens & YES_WORDS), bool(tokens & NO_WORDS)
    if yes == no:        # both or neither
        return None
    return yes


def parse_new_device(norm: str) -> Optional[bool]:
    """Interpret device-change evidence; contradictory answers are unclear."""
    tokens = set(norm.split())
    has_yes = bool(tokens & YES_WORDS)
    has_no = bool(tokens & NO_WORDS)
    if has_yes and has_no:
        return None
    explicit = parse_yes_no(norm)

    # Do not treat negated changes ("haven't changed", "no new phone") as
    # positive evidence that the caller switched devices.
    negated_change = bool(re.search(
        r"\b(?:not|no|never|havent|hasnt|didnt|dont|wasnt)\s+"
        r"(?:(?:a|any)\s+)?(?:new|different|another)\b|"
        r"\b(?:not|no|never|havent|hasnt|didnt|dont|wasnt)\s+"
        r"(?:got|bought|changed|replaced|upgraded|have|own)\b", norm
    ))
    new = bool(tokens & NEW_DEVICE_WORDS) and not negated_change
    same = bool(tokens & SAME_DEVICE_WORDS)

    lexical = None if new == same else new
    if explicit is not None and lexical is not None and explicit != lexical:
        return None
    if explicit is not None:
        return explicit
    return lexical


def _is_cancel(norm: str) -> bool:
    return any(norm == p or norm.startswith(p + " ") for p in CANCEL_PHRASES)


# ------------------------------------------------------------ stub actions
def log_stub_action(action: str, enrolment: Optional[str], detail: dict, log_dir: str = "logs"):
    """Real tickets (Phase 3) and MAC update (Phase 5) don't exist yet: record the intent."""
    os.makedirs(log_dir, exist_ok=True)
    line = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "action": action,
            "enrolment": enrolment, "detail": detail}
    with open(os.path.join(log_dir, "stub_actions.log"), "a") as f:
        f.write(json.dumps(line) + "\n")
    print(f"[STUB ACTION] {action} for {enrolment}: {detail}")


# ---------------------------------------------------------------- dialogue
class WifiDiagnosisDialogue:
    def __init__(self, intent_fn: Callable[[str], bool], db_path: Optional[str] = None,
                 log_dir: str = "logs", turn_timeout_seconds: float = 45.0,
                 clock: Callable[[], float] = time.monotonic):
        self.intent_fn = intent_fn
        self.db_path = db_path
        self.log_dir = log_dir
        self.turn_timeout_seconds = turn_timeout_seconds
        self._clock = clock
        self._last_activity = None
        self.last_diagnosis: Optional[Diagnosis] = None
        self.device_question_asked = False
        self._reset()

    # ---- state ----
    def _reset(self):
        self.state = DialogueState.IDLE
        self.identity: Optional[SessionIdentity] = None
        self.candidate: Optional[str] = None
        self.failures = 0
        self._wifi = None
        self._verification = None
        self._last_activity = None

    def _escalate(self, reason: str) -> DialogueResult:
        log_stub_action("ticket_stub", self.identity.enrolment if self.identity else None,
                        {"reason": reason}, self.log_dir)
        self._reset()
        return DialogueResult(True, "I'm sorry, I'm having trouble with that. "
                                    "I'll note this down so our team can follow up.")

    def _fail(self, reason: str, retry_text: str, next_state: DialogueState) -> DialogueResult:
        self.failures += 1
        if self.failures >= MAX_FAILURES:
            return self._escalate(reason)
        self.state = next_state
        return DialogueResult(True, retry_text)

    # ---- entry point ----
    def handle(self, user_text: str) -> DialogueResult:
        text = (user_text or "").strip()
        norm = normalize(text)

        if self.state is DialogueState.IDLE:
            return self._handle_idle(text)

        now = self._clock()
        if (self._last_activity is not None
                and now - self._last_activity > self.turn_timeout_seconds):
            self._reset()
            return DialogueResult(
                True,
                "That account check timed out. Please tell me again if you need help with your Wi-Fi account.",
            )
        if _is_cancel(norm):
            self._reset()
            return DialogueResult(True, "Okay, no problem. Let me know if there is anything else I can help with.")
        if not norm or norm in NOISE_PHRASES:
            return SILENT
        self._last_activity = now

        if self.state is DialogueState.ASK_ENROLMENT:
            return self._handle_enrolment(text)
        if self.state is DialogueState.CONFIRM_ENROLMENT:
            return self._handle_confirm(text, norm)
        if self.state is DialogueState.ASK_DEVICE:
            return self._handle_device(norm)
        return NOT_HANDLED  # unreachable, defensive

    # ---- states ----
    def _handle_idle(self, text: str) -> DialogueResult:
        if not self.intent_fn(text):
            return NOT_HANDLED
        self._reset()
        self.device_question_asked = False
        self.last_diagnosis = None
        digits = extract_digits(text)
        if len(digits) == ENROLMENT_LENGTH:      # caller already said the number
            return self._read_back(digits)
        self.state = DialogueState.ASK_ENROLMENT
        self._last_activity = self._clock()
        return DialogueResult(True, "I can check your Wi-Fi account for you. "
                                    "Please say your six digit enrolment number, one digit at a time.")

    def _read_back(self, digits: str) -> DialogueResult:
        self.candidate = digits
        self.state = DialogueState.CONFIRM_ENROLMENT
        self._last_activity = self._clock()
        return DialogueResult(True, f"I heard {speak_digits(digits)}. Is that correct?")

    def _handle_enrolment(self, text: str) -> DialogueResult:
        digits = extract_digits(text)
        if len(digits) == ENROLMENT_LENGTH:
            return self._read_back(digits)
        return self._fail(
            "could not capture a six digit enrolment number",
            "Sorry, I need exactly six digits. "
            "Please say your enrolment number again, one digit at a time.",
            DialogueState.ASK_ENROLMENT)

    def _handle_confirm(self, text: str, norm: str) -> DialogueResult:
        digits = extract_digits(text)
        if len(digits) == ENROLMENT_LENGTH:      # caller restated the number instead of saying no
            return self._read_back(digits)
        answer = parse_yes_no(norm)
        if answer is None:
            return self._fail("confirmation unclear", "Sorry, was that a yes or a no?",
                              DialogueState.CONFIRM_ENROLMENT)
        if not answer:
            self.candidate = None
            return self._fail(
                "enrolment number repeatedly rejected by caller",
                "Sorry about that. Please say your enrolment number again, one digit at a time.",
                DialogueState.ASK_ENROLMENT)
        return self._lookup_and_diagnose()

    def _lookup_and_diagnose(self) -> DialogueResult:
        identity = establish_identity_stub(self.candidate, self.db_path)
        if identity is None:
            self.candidate = None
            return self._fail(
                "enrolment number not found",
                "I couldn't find an account with that number. "
                "Please say your enrolment number again, one digit at a time.",
                DialogueState.ASK_ENROLMENT)
        self.identity = identity
        if lookup_student(identity, self.db_path) is None:   # vanished between calls
            return self._escalate("student record not readable")
        self._wifi = get_wifi_status(identity, self.db_path)
        self._verification = get_verification_status(identity, self.db_path)
        state = classify_account(self._wifi, self._verification)
        if state is None:
            self.state = DialogueState.ASK_DEVICE
            self._last_activity = self._clock()
            self.device_question_asked = True
            self.failures = 0
            return DialogueResult(True, DEVICE_QUESTION)
        return self._finish(state)

    def _handle_device(self, norm: str) -> DialogueResult:
        answer = parse_new_device(norm)
        if answer is None:
            return self._fail("device answer unclear",
                              "Sorry, have you changed your phone or laptop since you registered? "
                              "Please say yes or no.",
                              DialogueState.ASK_DEVICE)
        return self._finish(finish_provisioned(self._wifi, answer))

    def _finish(self, state) -> DialogueResult:
        diagnosis = build_diagnosis(state, self._wifi, self._verification)
        self.last_diagnosis = diagnosis
        if diagnosis.action != "none":
            log_stub_action(diagnosis.action, self.identity.enrolment,
                            {"state": state.value, **diagnosis.detail}, self.log_dir)
        self._reset()
        return DialogueResult(True, diagnosis.response)
