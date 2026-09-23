import sqlite3, json, random

random.seed(42)

DB_PATH = "/content/drive/MyDrive/voice-agent-mvp/db/students.sqlite"

STATES = [
    "never_applied",
    "verification_pending",
    "rejected",
    "verified_backlogged",
    "mac_mismatch",
    "working_not_notified",
    "working_fine",
]

def make_record(i, state):
    enrolment = f"AMU{2023000+i}"
    mobile = "9" + str(random.randint(100000000, 999999999))[:9]
    services, verification = [], []

    if state == "never_applied":
        services.append({"type": "wifi", "applied": False})
    elif state == "verification_pending":
        services.append({"type": "wifi", "applied": True, "status": "pending", "createdBy": None})
        verification.append({"type": "provost", "approved": None})
    elif state == "rejected":
        services.append({"type": "wifi", "applied": True, "status": "rejected", "createdBy": None})
        verification.append({"type": "pst_rejected", "value": True, "comment": "Hostel address not verified"})
    elif state == "verified_backlogged":
        services.append({"type": "wifi", "applied": True, "status": "verified", "createdBy": None})
        verification.append({"type": "provost", "approved": True})
    elif state == "mac_mismatch":
        services.append({"type": "wifi", "applied": True, "status": "created",
                          "createdBy": "cc_admin", "mac": "AA:BB:CC:11:22:33", "notified": True})
    elif state == "working_not_notified":
        services.append({"type": "wifi", "applied": True, "status": "created",
                          "createdBy": "cc_admin", "mac": "AA:BB:CC:44:55:66", "notified": False})
    elif state == "working_fine":
        services.append({"type": "wifi", "applied": True, "status": "created",
                          "createdBy": "cc_admin", "mac": "AA:BB:CC:77:88:99", "notified": True})

    return (
        enrolment, f"Student {i}", mobile, random.choice([0, 1]) if i % 10 == 0 else 1,
        f"student{i}@amu.ac.in", 1,
        json.dumps(services), json.dumps(verification), None, None
    )

def main():
    conn = sqlite3.connect(DB_PATH)
    with open("db/schema.sql") as f:
        conn.executescript(f.read())

    records = []
    i = 1
    for state in STATES:
        for _ in range(5):
            records.append(make_record(i, state))
            i += 1

    conn.executemany("INSERT OR REPLACE INTO students VALUES (?,?,?,?,?,?,?,?,?,?)", records)
    conn.commit()
    conn.close()
    print(f"Seeded {len(records)} records at {DB_PATH}")

if __name__ == "__main__":
    main()