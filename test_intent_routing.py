"""
Calibrates the intent router (MIN_SIM, MARGIN) against intent_eval_set.py.
Needs the E5 model, so run it in Colab:   python test_intent_routing.py

It scores every labeled utterance once, then sweeps the two parameters offline
(fast), and prints the best settings plus every utterance still misrouted.

Failure costs are asymmetric:
  false_walk  = an FAQ/infrastructure question hijacked into "say your enrolment number" (annoying)
  missed_walk = a personal complaint sent to the FAQ path (the walk's whole value is lost)
We rank by total correct, then fewer false_walk, then fewer missed_walk.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from intent_eval_set import INTENT_EVAL_SET
from intent_router import (INFRA_PATTERN, WALK_EXAMPLES, EmbeddingIntentRouter,
                           load_negative_examples)

model = SentenceTransformer("intfloat/e5-large-v2")
router = EmbeddingIntentRouter(model, load_negative_examples())

rows = []   # (text, label, infra_guard_hit, walk_best, other_best)
for text, label in INTENT_EVAL_SET:
    w, o = router.scores(text)
    rows.append((text, label, bool(INFRA_PATTERN.search(text)), w, o))


def evaluate(min_sim, margin):
    correct = false_walk = missed_walk = 0
    wrong = []
    for text, label, infra, w, o in rows:
        predicted_walk = (not infra) and w >= min_sim and (w - o) >= margin
        if predicted_walk == (label == "walk"):
            correct += 1
        else:
            if predicted_walk:
                false_walk += 1
            else:
                missed_walk += 1
            wrong.append((text, label, round(w, 3), round(o, 3)))
    return correct, false_walk, missed_walk, wrong


results = []
for min_sim in np.arange(0.70, 0.93, 0.01):
    for margin in np.arange(-0.02, 0.081, 0.005):
        c, fw, mw, wrong = evaluate(min_sim, margin)
        results.append((c, -fw, -mw, round(float(min_sim), 2), round(float(margin), 3), wrong))

results.sort(key=lambda r: (r[0], r[1], r[2]), reverse=True)
total = len(rows)

print(f"\n{total} labeled utterances "
      f"({sum(1 for r in rows if r[1] == 'walk')} walk / {sum(1 for r in rows if r[1] == 'other')} other)")
print("\n=== TOP 8 SETTINGS ===")
for c, nfw, nmw, ms, mg, _ in results[:8]:
    print(f"MIN_SIM={ms:.2f}  MARGIN={mg:+.3f}  correct={c}/{total}  false_walk={-nfw}  missed_walk={-nmw}")

best = results[0]
print(f"\n=== MISROUTED at MIN_SIM={best[3]:.2f}, MARGIN={best[4]:+.3f} ===")
for text, label, w, o in best[5]:
    print(f"expected {label:5s}  walk_sim={w}  other_sim={o}  | {text}")

print("\nIf the best 'correct' is poor, look at the misrouted list: add those phrasings to "
      "WALK_EXAMPLES (or EXTRA_NEGATIVES in intent_router.py) and rerun, or fix a wrong label "
      "in intent_eval_set.py.")
print("Then set MIN_SIM and MARGIN at the top of intent_router.py to the best values above.")
