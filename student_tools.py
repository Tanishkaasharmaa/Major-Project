"""
Read-only student tools (Tier 0/1) against the mock SQLite DB.

Design rules (from Project_Context sections 9 and 10):
- Every tool takes a SessionIdentity, never a raw enrolment string.
- Tools return only what a spoken response needs. No mobile number, email
  address or raw rows ever leave this module.
- The real schema stores heterogeneous / optional shapes (verification[] has
  four differently-shaped objects, services[] has one entry per service).
  All of that is normalised HERE so no other code ever touches raw JSON.
"""
import json
import os
import re
import sqlite3
from typing import Any, Dict, Optional

from session import SessionIdentity

DB_PATH = os.environ.get("STUDENT_DB_PATH", "db/students.sqlite")


# ------------------------------------------------------------------ helpers
def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _loads(value, default):
    """Parse JSON text; anything missing, malformed or null becomes `default`."""
    if value is None:
        return default
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return default
    return default if parsed is None else parsed


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return False


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _require_identity(identity) -> None:
    if not isinstance(identity, SessionIdentity):
        raise TypeError(
            "Student tools accept only a SessionIdentity created by the "
            "authentication flow, never a raw enrolment string."
        )


def _row(identity: SessionIdentity, db_path: Optional[str]):
    conn = _connect(db_path)
    try:
        return conn.execute(
            "SELECT * FROM students WHERE enroll = ?", (identity.enrolment,)
        ).fetchone()
    finally:
        conn.close()


def _is_wifi(service_entry) -> bool:
    if not isinstance(service_entry, dict):
        return False
    name = re.sub(r"[^a-z0-9]", "", str(service_entry.get("service", "")).lower())
    return name == "wifi"


# ------------------------------------------------------- auth-stub support
def student_exists(enrolment: str, db_path: Optional[str] = None) -> bool:
    """Used ONLY by the authentication stub in session.py."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM students WHERE enroll = ?", (enrolment,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()


# -------------------------------------------------------------------- tools
def lookup_student(identity: SessionIdentity, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Minimal profile. Returns None if the record vanished."""
    _require_identity(identity)
    row = _row(identity, db_path)
    if row is None:
        return None
    profile = _loads(row["profile"], {})
    contact = _loads(row["contact"], {})
    mobile_id = contact.get("mobileID", {}) if isinstance(contact, dict) else {}
    hall = _text(profile.get("hall")) if isinstance(profile, dict) else ""
    return {
        "enrolment": identity.enrolment,
        "first_name": _text(row["full_name"]).split(" ")[0] if row["full_name"] else "",
        "has_hall": bool(hall),
        "mobile_verified": _truthy(mobile_id.get("isVerified")) if isinstance(mobile_id, dict) else False,
    }


def get_wifi_status(identity: SessionIdentity, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Normalised view of the WiFi entry in services[]."""
    _require_identity(identity)
    row = _row(identity, db_path)
    status = {
        "has_wifi_entry": False,
        "applied": False,
        "status": "",
        "created_by": "",
        "mac_registered": False,
        "notified": False,
    }
    if row is None:
        return status
    services = _loads(row["services"], [])
    if not isinstance(services, list):
        return status
    for entry in services:
        if _is_wifi(entry):
            status.update(
                has_wifi_entry=True,
                applied=_truthy(entry.get("applied")),
                status=_text(entry.get("status")),
                created_by=_text(entry.get("createdBy")),
                mac_registered=bool(_text(entry.get("mac"))),
                notified=_truthy(entry.get("notified")),
            )
            break
    return status


def get_verification_status(identity: SessionIdentity, db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Flattens profile.verification[] (four differently-shaped objects) into one
    dict, and works out which approver this student needs.

    ASSUMPTIONS to confirm with the Computer Centre (Phase 2 decisions 4):
    - pst_rejected = provost rejection, chm_rejected = chairman rejection.
    - A student with a hall on file needs the provost; others need the chairman.
    """
    _require_identity(identity)
    row = _row(identity, db_path)
    out = {
        "provost_approved": False,
        "chairman_approved": False,
        "rejected": False,
        "rejected_by_role": "",
        "rejection_comment": "",
        "required_approver": "chairman",
        "required_approved": False,
    }
    if row is None:
        return out

    profile = _loads(row["profile"], {})
    if not isinstance(profile, dict):
        profile = {}
    has_hall = bool(_text(profile.get("hall")))
    out["required_approver"] = "provost" if has_hall else "chairman"

    items = profile.get("verification")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue  # malformed entry: ignore, never crash
            if "provost" in item:
                out["provost_approved"] = out["provost_approved"] or _truthy(item["provost"])
            if "chairman" in item:
                out["chairman_approved"] = out["chairman_approved"] or _truthy(item["chairman"])
            if _truthy(item.get("pst_rejected")) and not out["rejected"]:
                out.update(rejected=True, rejected_by_role="provost",
                           rejection_comment=_text(item.get("comment")))
            if _truthy(item.get("chm_rejected")) and not out["rejected"]:
                out.update(rejected=True, rejected_by_role="chairman",
                           rejection_comment=_text(item.get("comment")))

    out["required_approved"] = (
        out["provost_approved"] if out["required_approver"] == "provost"
        else out["chairman_approved"]
    )
    return out


def get_notification_history(identity: SessionIdentity, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Whether the student was ever told their Wi-Fi was activated."""
    _require_identity(identity)
    row = _row(identity, db_path)
    out = {"notified": False, "by_email": False, "by_sms": False}
    if row is None:
        return out
    services = _loads(row["services"], [])
    if isinstance(services, list):
        for entry in services:
            if _is_wifi(entry):
                out.update(
                    notified=_truthy(entry.get("notified")),
                    by_email=_truthy(entry.get("notifiedByEmail")),
                    by_sms=_truthy(entry.get("notifiedBySMS")),
                )
                break
    return out
