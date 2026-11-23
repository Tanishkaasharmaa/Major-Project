"""
Seeds the mock student DB in the real schema's shapes.

35 records = 7 Wi-Fi states x 5 records each (enrolments 100001-100035).
The first record of each state is the "golden" one used in spoken walkthroughs:

    100001 never_applied        100016 provisioning_backlog   100031 account_fine
    100006 verification_pending 100021 mac_mismatch
    100011 rejected             100026 not_notified

The state label is NOT stored in the DB (so tests can't cheat by reading it).
It lives in SEED_MANIFEST, which the tests use as the expected answer.

Edge cases (enrolments 2000xx) are only seeded with include_edge=True; the unit
tests use them, the demo DB does not.

Usage:  python db/seed_students.py
"""
import json
import os
import sqlite3
import sys

DB_PATH = os.environ.get("STUDENT_DB_PATH", "db/students.sqlite")
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")

STATES = [
    "never_applied",
    "verification_pending",
    "rejected",
    "provisioning_backlog",
    "mac_mismatch",
    "not_notified",
    "account_fine",
]
VARIANTS_PER_STATE = 5

HALLS = ["N.R.S.C. Hall", "Sir Syed Hall (North)", "Sulaiman Hall", "Aftab Hall", "Mohsinul Mulk Hall"]
FIRST_NAMES = ["Aamir", "Bushra", "Danish", "Farah", "Imran", "Nadia", "Rehan", "Sana", "Tariq", "Zoya"]
REJECT_COMMENTS = [
    "Hostel address could not be verified.",
    "Admission slip is not readable, please upload again.",
    "",  # rejected with no comment recorded
    "Faculty number does not match our records.",
    "Profile photo is missing.",
]


# ---------------------------------------------------------------- builders
def verification_items(provost=False, chairman=False, pst_rejected=False,
                       chm_rejected=False, comment="", rejected_by=None):
    """Heterogeneous array, same shape as the real record."""
    return [
        {"provost": provost,
         "verifiedBy": "Provost" if provost else None,
         "verifiedOn": "2016-09-12T07:36:53.680Z" if provost else None},
        {"chairman": chairman,
         "verifiedBy": "Chairman" if chairman else None,
         "verifiedOn": "2016-08-23T05:27:48.507Z" if chairman else None},
        {"pst_rejected": pst_rejected,
         "rejectedBy": rejected_by if pst_rejected else None,
         "rejectedOn": "2016-09-01T10:00:00.000Z" if pst_rejected else None,
         "comment": comment if pst_rejected else ""},
        {"chm_rejected": chm_rejected,
         "rejectedBy": rejected_by if chm_rejected else None,
         "rejectedOn": "2016-09-01T10:00:00.000Z" if chm_rejected else None,
         "comment": comment if chm_rejected else ""},
    ]


def wifi_entry(applied=True, status="Created", created_by="cc_admin", mac="",
               notified=True):
    return {
        "service": "WiFi",
        "applied": applied,
        "appliedOn": "2016-09-17T03:01:49.829Z" if applied else None,
        "status": status,
        "mac": mac,
        "createdBy": created_by,
        "createdOn": "2016-09-17T05:51:43.717Z" if created_by else None,
        "notified": notified,
        "notifiedOn": "2016-09-17T05:51:43.717Z" if notified else None,
        "notifiedByEmail": notified,
        "notifiedBySMS": notified,
    }


def other_services():
    out = []
    for name in ["GCL Lab", "Research Lab", "Email"]:
        out.append({"service": name, "applied": False, "appliedOn": None,
                    "status": "Not Applied", "createdBy": "", "createdOn": None,
                    "notified": False, "notifiedOn": None,
                    "notifiedByEmail": False, "notifiedBySMS": False})
    return out


def make_mac(n):
    return "70-77-81-30-%02X-%02X" % ((n * 7) % 256, (n * 13) % 256)


def student_doc(enroll, name, hall, verification, services, mobile_verified=True):
    n = int(enroll)
    return {
        "enroll": enroll,
        "fullName": name,
        "email": f"{enroll}@mock.amu.test",
        "mobile": "98" + str(10000000 + n % 90000000),
        "profile": {"isCompleted": True, "verification": verification,
                    "hall": hall, "department": "Mechanical Engineering"},
        "contact": {"emailID": {"isVerified": True},
                    "mobileID": {"isVerified": mobile_verified}},
        "services": services,
        "resetToken": None,
        "resetTokenExpiresIn": None,
    }


def build_state_record(state, variant):
    """Return (doc, manifest_entry) for one of the 35 standard records."""
    idx = STATES.index(state)
    enroll = str(100001 + idx * VARIANTS_PER_STATE + variant)
    name = f"{FIRST_NAMES[(idx + variant) % len(FIRST_NAMES)]} Mock{variant}"
    has_hall = variant % 2 == 0
    hall = HALLS[variant % len(HALLS)] if has_hall else ""
    manifest = {"state": state, "new_device": None, "approver": None}
    mac = make_mac(int(enroll))
    mobile_verified = True

    if state == "never_applied":
        services = [wifi_entry(applied=False, status="Not Applied", created_by="",
                               notified=False)] + other_services()
        if variant % 2:  # WiFi entry listed last for some records
            services = other_services() + [services[0]]
        verification = verification_items()

    elif state == "verification_pending":
        approver = "provost" if has_hall else "chairman"
        manifest["approver"] = approver
        verification = verification_items(
            provost=False, chairman=(approver == "provost" and variant % 4 == 0))
        services = [wifi_entry(applied=True, status="Applied", created_by="",
                               notified=False)] + other_services()

    elif state == "rejected":
        by_provost = variant % 2 == 0
        comment = REJECT_COMMENTS[variant % len(REJECT_COMMENTS)]
        manifest["approver"] = "provost" if by_provost else "chairman"
        manifest["comment"] = comment
        verification = verification_items(
            pst_rejected=by_provost, chm_rejected=not by_provost,
            comment=comment, rejected_by="Provost" if by_provost else "Chairman")
        services = [wifi_entry(applied=True, status="Rejected", created_by="",
                               notified=False)] + other_services()

    elif state == "provisioning_backlog":
        verification = verification_items(provost=True, chairman=True)
        status = ["Verified", "Pending Creation"][variant % 2]
        services = [wifi_entry(applied=True, status=status, created_by="",
                               notified=False)] + other_services()

    else:  # the three "Created" states
        verification = verification_items(provost=True, chairman=True)
        notified = state != "not_notified"
        services = [wifi_entry(applied=True, status="Created", created_by="cc_admin",
                               mac=mac, notified=notified)] + other_services()
        manifest["new_device"] = state == "mac_mismatch"
        if state == "account_fine" and variant == 4:
            mobile_verified = False  # needed later (Phase 5: "no mobile on file")

    doc = student_doc(enroll, name, hall, verification, services, mobile_verified)
    return doc, manifest


def build_edge_cases():
    """(doc, manifest_entry) pairs for unit tests only."""
    cases = []

    def add(n, desc, hall, verification, services, state, new_device=None,
            approver=None):
        doc = student_doc(str(200000 + n), f"Edge {n}", hall, verification, services)
        cases.append((doc, {"state": state, "new_device": new_device,
                            "approver": approver, "desc": desc}))

    add(1, "rejected and approval pending at the same time", "N.R.S.C. Hall",
        verification_items(provost=False, pst_rejected=True,
                           comment="Address mismatch.", rejected_by="Provost"),
        [wifi_entry(True, "Rejected", "", notified=False)] + other_services(),
        "rejected", approver="provost")
    add(2, "chairman rejection with empty comment (day scholar)", "",
        verification_items(chm_rejected=True, comment="", rejected_by="Chairman"),
        [wifi_entry(True, "Rejected", "", notified=False)] + other_services(),
        "rejected", approver="chairman")
    add(3, "no WiFi entry in services at all", "N.R.S.C. Hall",
        verification_items(),
        other_services(), "never_applied")
    add(4, "services is JSON null", "N.R.S.C. Hall", verification_items(),
        None, "never_applied")
    add(5, "status Created but createdBy empty", "N.R.S.C. Hall",
        verification_items(provost=True, chairman=True),
        [wifi_entry(True, "Created", "", mac="AA-BB-CC-00-00-05")] + other_services(),
        "provisioning_backlog")
    add(6, "createdBy set but status not Created", "N.R.S.C. Hall",
        verification_items(provost=True, chairman=True),
        [wifi_entry(True, "Pending", "cc_admin")] + other_services(),
        "provisioning_backlog")
    add(7, "malformed verification entries mixed with a valid provost approval",
        "N.R.S.C. Hall",
        ["garbage", None, {}, {"provost": True, "verifiedBy": "Provost"}, 42],
        [wifi_entry(True, "Created", "cc_admin", mac="AA-BB-CC-00-00-07")] + other_services(),
        "account_fine", new_device=False)
    add(8, "verification is null, applied, hall resident", "N.R.S.C. Hall",
        None,
        [wifi_entry(True, "Applied", "", notified=False)] + other_services(),
        "verification_pending", approver="provost")
    add(9, "status in upper case", "N.R.S.C. Hall",
        verification_items(provost=True, chairman=True),
        [wifi_entry(True, "CREATED", "cc_admin", mac="AA-BB-CC-00-00-09")] + other_services(),
        "account_fine", new_device=False)
    add(10, "created but no MAC on file, caller has new device", "N.R.S.C. Hall",
        verification_items(provost=True, chairman=True),
        [wifi_entry(True, "Created", "cc_admin", mac="")] + other_services(),
        "mac_not_on_file", new_device=True)
    return cases


def all_records(include_edge=False):
    records = []
    for state in STATES:
        for v in range(VARIANTS_PER_STATE):
            records.append(build_state_record(state, v))
    if include_edge:
        records.extend(build_edge_cases())
    return records


def seed(db_path=None, include_edge=False):
    """(Re)create the DB and return {enroll: manifest_entry}."""
    db_path = db_path or DB_PATH
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
    manifest = {}
    for doc, entry in all_records(include_edge):
        conn.execute(
            "INSERT INTO students VALUES (?,?,?,?,?,?,?,?,?)",
            (doc["enroll"], doc["fullName"], doc["email"], doc["mobile"],
             json.dumps(doc["profile"]), json.dumps(doc["contact"]),
             json.dumps(doc["services"]), doc["resetToken"],
             doc["resetTokenExpiresIn"]),
        )
        manifest[doc["enroll"]] = entry
    conn.commit()
    conn.close()
    return manifest


# Expected diagnosis for every standard record (used by tests and walkthroughs).
SEED_MANIFEST = {doc["enroll"]: entry for doc, entry in all_records(False)}

if __name__ == "__main__":
    m = seed(include_edge="--edge" in sys.argv)
    print(f"Seeded {len(m)} records at {DB_PATH}")
    for state in STATES:
        ids = [e for e, v in m.items() if v["state"] == state and not e.startswith("2")]
        print(f"  {state:22s} golden {ids[0]}  (others up to {ids[-1]})")
