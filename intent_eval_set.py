"""
Routing test set: should an utterance enter the personal account walk ("walk")
or stay on the FAQ / fallback path ("other")?

Built from the 84-query FAQ eval set so there is one source of truth:
  - FAQ 6, 8, 9 (suddenly stopped / verified but not working / never notified)
    are descriptions of the caller's OWN account  -> "walk"
  - every other FAQ question and every out-of-scope (None) query -> "other"
Plus hand-written extras below. Edit the labels if the boundary should move.
"""
from eval_set import EVAL_SET

WALK_FAQ_IDS = {6, 8, 9}

INTENT_EVAL_SET = [
    (query, "walk" if faq_id in WALK_FAQ_IDS else "other")
    for query, faq_id in EVAL_SET
]

EXTRA = [
    # personal complaints -> walk
    ("hello my wifi is not working since morning", "walk"),
    ("wifi not connecting please help", "walk"),
    ("I cannot connect my new phone to the university wifi", "walk"),
    ("my phone is not connecting to the hostel wifi", "walk"),
    ("the wifi on my laptop stopped working today", "walk"),
    ("I am not able to use wifi on my mobile", "walk"),
    ("I never got any message that my wifi is ready", "walk"),
    ("wifi connection failed on my device", "walk"),
    ("my wifi application was approved but I still have no wifi", "walk"),
    ("internet is not working on my phone", "walk"),
    # not for the walk
    ("what is the wifi password for guests", "other"),
    ("the router on the second floor is not working", "other"),
    ("the wifi signal is weak everywhere in the hostel", "other"),
    ("how do I change my email id", "other"),
    ("can you tell me a joke", "other"),
    ("what time is it", "other"),
    ("thank you", "other"),
    ("hello", "other"),
]

INTENT_EVAL_SET = INTENT_EVAL_SET + EXTRA


# Mixed personal/infrastructure wording. Personal account symptoms remain a
# walk candidate even when the caller speculates about a router or signal.
MIXED = [
    ("my laptop cannot connect to campus wifi; maybe the router is the problem", "walk"),
    ("wifi stopped working on my phone and the signal looks weak", "walk"),
    ("my account was approved but the access point near me seems down", "walk"),
    ("the router is broken for everyone in our hostel corridor", "other"),
    ("the access point in the library has no signal for anyone", "other"),
]
INTENT_EVAL_SET.extend(MIXED)
