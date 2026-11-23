"""
The 7-state Wi-Fi account diagnosis (Project_Context section 10.3).

Pure functions, no I/O, no LLM. Evaluation order matters:

  1. never applied           -> 0 questions
  2. rejected                -> 0 questions   (checked BEFORE pending: a rejected
                                               application also has incomplete approvals)
  3. verification pending    -> 0 questions
  4. provisioning backlog    -> 0 questions
  -- account is provisioned (status Created and createdBy set) --
  5. MAC mismatch            -> 1 question  ("new device since you registered?" = yes)
  6. working, never notified -> 1 question  (answer = no, notified = false)
  7. account fine            -> 1 question  (answer = no, notified = true)

Only the three provisioned states need the device question, because a MAC
mismatch is a difference between the registered device and the caller's
current device, and the DB only knows the first of those.

Spoken responses are scripted (2-3 short sentences, no digits, no formatting).
"""
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class WifiState(str, Enum):
    NEVER_APPLIED = "never_applied"
    VERIFICATION_PENDING = "verification_pending"
    REJECTED = "rejected"
    PROVISIONING_BACKLOG = "provisioning_backlog"
    MAC_MISMATCH = "mac_mismatch"
    NOT_NOTIFIED = "not_notified"
    ACCOUNT_FINE = "account_fine"


PROVISIONED_STATES = {WifiState.MAC_MISMATCH, WifiState.NOT_NOTIFIED, WifiState.ACCOUNT_FINE}


@dataclass
class Diagnosis:
    state: WifiState
    response: str
    action: str = "none"            # name of a stubbed backend action, or "none"
    detail: Dict[str, Any] = field(default_factory=dict)


DEVICE_QUESTION = (
    "I can see your Wi-Fi account is active. "
    "Have you got a new phone or laptop since you first registered for Wi-Fi?"
)

APPROVER_PHRASE = {
    "provost": "your hall provost",
    "chairman": "your department chairman",
}


# ------------------------------------------------------------ classification
def is_provisioned(wifi: Dict[str, Any]) -> bool:
    """Created = status says Created AND somebody at the Computer Centre created it."""
    return wifi.get("status", "").strip().lower() == "created" and bool(wifi.get("created_by"))


def classify_account(wifi: Dict[str, Any], verification: Dict[str, Any]) -> Optional[WifiState]:
    """
    Returns one of states 1-4, or None when the account is provisioned and the
    device question is needed to choose between states 5-7.
    """
    if not wifi.get("has_wifi_entry") or not wifi.get("applied"):
        return WifiState.NEVER_APPLIED
    if verification.get("rejected"):
        return WifiState.REJECTED
    if not verification.get("required_approved"):
        return WifiState.VERIFICATION_PENDING
    if not is_provisioned(wifi):
        return WifiState.PROVISIONING_BACKLOG
    return None


def finish_provisioned(wifi: Dict[str, Any], new_device: bool) -> WifiState:
    if new_device:
        return WifiState.MAC_MISMATCH
    if not wifi.get("notified"):
        return WifiState.NOT_NOTIFIED
    return WifiState.ACCOUNT_FINE


# ------------------------------------------------------------------ speech
def _speakable(text: str, limit: int = 160) -> str:
    """Make free text from the DB safe to read aloud."""
    text = re.sub(r"[*#_<>\[\]{}|\\`~^]", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].rstrip(",;:-") + "."
    return text


def build_diagnosis(state: WifiState, wifi: Dict[str, Any], verification: Dict[str, Any]) -> Diagnosis:
    approver = verification.get("required_approver", "chairman")

    if state is WifiState.NEVER_APPLIED:
        return Diagnosis(
            state,
            "I checked your account, and you haven't applied for Wi-Fi yet. "
            "To apply, open the student portal and submit the Wi-Fi application. "
            "Once your approver signs off, the Computer Centre will set up your access.",
        )

    if state is WifiState.REJECTED:
        who = APPROVER_PHRASE.get(verification.get("rejected_by_role"), "your approver")
        comment = _speakable(verification.get("rejection_comment", ""))
        if comment:
            reason = f"Their note says: {comment} "
        else:
            reason = "No reason was recorded, so please ask them directly. "
        return Diagnosis(
            state,
            f"Your Wi-Fi application was rejected by {who}. {reason}"
            "Once that is sorted out, you can apply again.",
            detail={"rejected_by": verification.get("rejected_by_role")},
        )

    if state is WifiState.VERIFICATION_PENDING:
        who = APPROVER_PHRASE.get(approver, "your approver")
        return Diagnosis(
            state,
            f"Your application is submitted, but it is still waiting for approval from {who}. "
            "Wi-Fi can only be set up after they approve it. "
            "Please follow up with them.",
            detail={"pending_approver": approver},
        )

    if state is WifiState.PROVISIONING_BACKLOG:
        return Diagnosis(
            state,
            "Good news, your application has been approved. "
            "It is now waiting in the Computer Centre's queue to be activated, "
            "so nothing is wrong with your device. I'll flag it so the team can speed it up.",
            action="expedite_ticket_stub",
        )

    if state is WifiState.MAC_MISMATCH:
        return Diagnosis(
            state,
            "That explains it. Wi-Fi is registered to one device only, "
            "and your new device has a different hardware address. "
            "The registered device needs to be updated, and I'll flag that for you.",
            action="mac_update_stub",
        )

    if state is WifiState.NOT_NOTIFIED:
        return Diagnosis(
            state,
            "Your Wi-Fi account has actually been created, and it should already be working. "
            "You may simply never have been told. "
            "Please try connecting to the campus Wi-Fi again.",
        )

    # ACCOUNT_FINE
    return Diagnosis(
        state,
        "Your Wi-Fi account is active and registered to your device, "
        "so the account itself looks fine. "
        "The problem may be with the connection, so I'll note this down for the team to look into.",
        action="technical_followup_stub",
    )
