"""
Session identity.

Project rule: nothing the caller says, and nothing the LLM outputs, may select
which student record is read. Only the authentication flow can create a
SessionIdentity, and every data tool takes a SessionIdentity (never a raw
string), so the rule is enforced by the type, not by good intentions.

PHASE 2 STUB: "the caller said an enrolment number that exists" is treated as
sufficient. This is NOT real authentication. In Phase 5 only the body of
establish_identity_stub() is replaced (enrolment -> SMS OTP by keypad); the
SessionIdentity type and every tool signature stay exactly the same.
"""
from dataclasses import dataclass
from typing import Optional

STUB_METHOD = "enrolment_only_stub"


@dataclass(frozen=True)
class SessionIdentity:
    """Immutable. Created only by the authentication flow."""
    enrolment: str
    method: str


def establish_identity_stub(enrolment: str, db_path: Optional[str] = None) -> Optional[SessionIdentity]:
    # imported here to avoid a circular import (student_tools imports this module)
    from student_tools import student_exists

    if student_exists(enrolment, db_path):
        return SessionIdentity(enrolment=enrolment, method=STUB_METHOD)
    return None
