"""
Decides whether a caller utterance is a PERSONAL Wi-Fi account complaint
("my wifi doesn't work", "I got a new phone and can't connect") that should
enter the account walk, or anything else (FAQ question, router/infrastructure
problem, chit-chat) that stays on the existing FAQ path.

Method: compare the utterance (E5, "query: " prefix on both sides because this is
a symmetric similarity task) against two example sets:
    walk examples   vs   everything-else examples
and enter the walk only if it is close enough to the walk set AND clearly closer
to it than to the other set:

    walk_best >= MIN_SIM   and   (walk_best - other_best) >= MARGIN

MIN_SIM and MARGIN below are PLACEHOLDERS. Calibrate them with
test_intent_routing.py (same approach as the FAQ threshold: measure, don't guess).

A tiny keyword guard sends obvious infrastructure complaints (router, access
point, cable, signal strength ...) to the FAQ/ticket path regardless of scores.
"""
import json
import re
from typing import Iterable, List, Optional, Tuple

import numpy as np

MIN_SIM = 0.80      # placeholder: calibrate
MARGIN = 0.02       # placeholder: calibrate

WALK_EXAMPLES = [
    "my wifi is not working",
    "my wifi doesn't work",
    "I can't connect to the wifi",
    "wifi stopped working on my phone",
    "I am unable to connect to the campus wifi",
    "my internet is not working on my laptop",
    "I got a new phone and the wifi won't connect",
    "I don't have wifi access",
    "wifi is not connecting on my device",
    "my wifi was working yesterday but not today",
    "I applied for wifi but I still can't use it",
    "I can't use the university wifi",
    "I am not able to log in to the wifi",
    "internet is not working on my mobile",
    "I need help with my wifi account",
]

# Infrastructure / chit-chat negatives (FAQ questions are loaded from faqs.json).
EXTRA_NEGATIVES = [
    "the router in our hall is broken",
    "there is no wifi coverage in the library",
    "wifi is down for everyone on the floor",
    "the wifi in my department building is very slow for all of us",
    "the access point near the canteen is dead",
    "a lan port in the lab is not working",
    "please install a new router in the hostel corridor",
    "I want to change my mobile number in Samarth",
    "what is the weather today",
    "can you tell me a joke",
    "hello",
    "who are you",
]

# FAQ entries whose callers are really describing THEIR OWN account problem:
# they enter the walk, so their questions are not used as negatives.
WALK_FAQ_IDS = {6, 8, 9}

INFRA_PATTERN = re.compile(
    r"\b(router|access point|cable|signal|lan port|lan point|wifi box)\b", re.IGNORECASE)


def load_negative_examples(faq_path: str = "knowledge/faqs.json",
                           exclude_ids: Iterable[int] = WALK_FAQ_IDS) -> List[str]:
    with open(faq_path) as f:
        faqs = json.load(f)
    exclude = set(exclude_ids)
    questions = [faq["question"] for i, faq in enumerate(faqs) if i not in exclude]
    return questions + EXTRA_NEGATIVES


class EmbeddingIntentRouter:
    def __init__(self, embed_model, negative_examples: List[str],
                 walk_examples: Optional[List[str]] = None,
                 min_sim: float = MIN_SIM, margin: float = MARGIN,
                 prefix: str = "query: "):
        self.embed_model = embed_model
        self.prefix = prefix
        self.min_sim = min_sim
        self.margin = margin
        walk_examples = walk_examples or WALK_EXAMPLES
        self._walk = self._encode(walk_examples)
        self._other = self._encode(negative_examples)

    def _encode(self, texts: List[str]) -> np.ndarray:
        return np.asarray(
            self.embed_model.encode([self.prefix + t for t in texts], normalize_embeddings=True))

    def scores(self, text: str) -> Tuple[float, float]:
        vec = np.asarray(self.embed_model.encode(self.prefix + text, normalize_embeddings=True))
        return float((self._walk @ vec).max()), float((self._other @ vec).max())

    def is_account_complaint(self, text: str) -> bool:
        if not text or INFRA_PATTERN.search(text):
            return False
        walk_best, other_best = self.scores(text)
        return walk_best >= self.min_sim and (walk_best - other_best) >= self.margin
