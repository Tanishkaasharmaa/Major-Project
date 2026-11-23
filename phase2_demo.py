"""
Phase 2 evidence: one scripted call per Wi-Fi state against the mock DB.
No voice, no GPU:   python phase2_demo.py          (add --quiet for the table only)

Prints each call's transcript and a final table: expected state vs diagnosed
state, and how many questions the agent asked beyond the enrolment number.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "db"))
import seed_students  # noqa: E402

from dialogue import DIGIT_WORDS, DialogueState, WifiDiagnosisDialogue  # noqa: E402

GOLDEN = {state: str(100001 + i * seed_students.VARIANTS_PER_STATE)
          for i, state in enumerate(seed_students.STATES)}
QUIET = "--quiet" in sys.argv


def say(enrol):
    return " ".join(DIGIT_WORDS[d] for d in enrol)


def run_call(db_path, log_dir, enrol, device_answer):
    dlg = WifiDiagnosisDialogue(intent_fn=lambda t: True, db_path=db_path, log_dir=log_dir)
    transcript = []

    def turn(caller):
        transcript.append(("Caller", caller))
        reply = dlg.handle(caller)
        if reply.text:
            transcript.append(("Agent ", reply.text))

    turn("My Wi-Fi doesn't work")
    turn(say(enrol))
    turn("yes")
    if dlg.state is DialogueState.ASK_DEVICE:
        turn(device_answer)
    return dlg, transcript


def main():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "students.sqlite")
        manifest = seed_students.seed(db_path)
        rows = []
        for state in seed_students.STATES:
            enrol = GOLDEN[state]
            expected = manifest[enrol]
            answer = "yes, I got a new phone" if expected["new_device"] else "no, same phone"
            dlg, transcript = run_call(db_path, os.path.join(tmp, "logs"), enrol, answer)
            if not QUIET:
                print(f"\n=== {state}  (enrolment {enrol}) ===")
                for who, text in transcript:
                    print(f"{who}: {text}")
            got = dlg.last_diagnosis.state.value if dlg.last_diagnosis else "NONE"
            extra_questions = 1 if dlg.device_question_asked else 0
            rows.append((state, enrol, got, extra_questions, got == state))

    print("\n" + "=" * 78)
    print(f"{'expected state':24s}{'enrolment':>10s}  {'diagnosed':24s}{'extra Qs':>9s}  result")
    for state, enrol, got, q, ok in rows:
        print(f"{state:24s}{enrol:>10s}  {got:24s}{q:>9d}  {'PASS' if ok else 'FAIL'}")
    passed = sum(r[4] for r in rows)
    print(f"\n{passed}/{len(rows)} states diagnosed correctly")
    sys.exit(0 if passed == len(rows) else 1)


if __name__ == "__main__":
    main()
