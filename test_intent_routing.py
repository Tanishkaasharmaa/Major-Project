"""Offline calibration for the personal Wi-Fi intent router.

Run explicitly with `python test_intent_routing.py`. Importing this module
from pytest does not load the embedding model or execute the calibration.
"""


def main():
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from intent_eval_set import INTENT_EVAL_SET
    from intent_router import (EmbeddingIntentRouter, is_infrastructure_only,
                               load_negative_examples)

    model = SentenceTransformer("intfloat/e5-large-v2")
    router = EmbeddingIntentRouter(model, load_negative_examples())

    rows = []   # (text, label, infra_guard_hit, walk_best, other_best)
    for text, label in INTENT_EVAL_SET:
        w, o = router.scores(text)
        rows.append((text, label, is_infrastructure_only(text), w, o))


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
    print("Use these values in the agent environment (they are read at startup):")
    print(f"PowerShell: $env:VOCAL_INTENT_MIN_SIM='{best[3]:.2f}'; $env:VOCAL_INTENT_MARGIN='{best[4]:.3f}'")
    print(f"bash: export VOCAL_INTENT_MIN_SIM='{best[3]:.2f}' VOCAL_INTENT_MARGIN='{best[4]:.3f}'")


if __name__ == "__main__":
    main()
