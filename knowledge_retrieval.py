"""Grounded FAQ retrieval processor for the Pipecat pipeline.

A new user-message object is processed once, even when consecutive turns have
identical text. Every retrieved result is independently checked against the
cosine-distance threshold before it is included in the LLM context.
"""
import os
from typing import Any, Dict, Optional

from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

SYSTEM_TEMPLATE = """You are a voice assistant for the AMU Computer Centre help desk, answering Wi-Fi questions over a phone call.

Answer ONLY using the information in CONTEXT below. Keep your answer to one or two short spoken sentences. No lists, no markdown, no headings.

If CONTEXT does not contain the answer, respond with exactly this sentence and nothing else:
"I'm sorry, I don't have that information on file. I'll note this down so our team can follow up."

CONTEXT:
{context}
"""


def validate_faq_collection(collection, embed_model, expected_model: str = "intfloat/e5-large-v2") -> Dict[str, Any]:
    """Fail fast when the persistent FAQ collection is absent, empty or incompatible."""
    count = collection.count()
    if count <= 0:
        raise RuntimeError(
            "The Chroma collection 'wifi_faqs' is empty. Run `python knowledge/ingest.py` from the project root."
        )

    metadata = collection.metadata or {}
    metric = metadata.get("hnsw:space")
    if metric != "cosine":
        raise RuntimeError(
            f"FAQ collection uses {metric!r} distance, expected 'cosine'. "
            "Rebuild it with `python knowledge/ingest.py`."
        )

    stored_model = metadata.get("embedding_model")
    if stored_model != expected_model:
        raise RuntimeError(
            f"FAQ collection metadata identifies {stored_model!r}, but the agent expects "
            f"{expected_model!r}. Rebuild it with `python knowledge/ingest.py`."
        )

    expected_dim = getattr(embed_model, "get_sentence_embedding_dimension", lambda: None)()
    stored_dim = metadata.get("embedding_dimension")
    if stored_dim is None:
        raise RuntimeError(
            "FAQ collection has no embedding_dimension metadata. Rebuild it with "
            "`python knowledge/ingest.py`."
        )
    stored_count = metadata.get("faq_count")
    if stored_count is not None and int(stored_count) != int(count):
        raise RuntimeError(
            f"FAQ collection count mismatch: metadata={stored_count}, actual={count}. "
            "Rebuild it with `python knowledge/ingest.py`."
        )
    if expected_dim is not None and int(stored_dim) != int(expected_dim):
        raise RuntimeError(
            f"FAQ embedding dimension mismatch: collection={stored_dim}, model={expected_dim}. "
            "Rebuild it with `python knowledge/ingest.py`."
        )

    return {"count": count, "metric": metric, "embedding_model": stored_model,
            "embedding_dimension": stored_dim or expected_dim}


class KnowledgeRetrievalProcessor(FrameProcessor):
    def __init__(self, context, collection, embed_model, distance_threshold, top_k=3,
                 log_path="logs/unanswered_queries.log", claims=None):
        super().__init__()
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if distance_threshold < 0:
            raise ValueError("distance_threshold cannot be negative")
        self.context = context
        self.collection = collection
        self.embed_model = embed_model
        self.distance_threshold = distance_threshold
        self.top_k = top_k
        self.log_path = log_path
        self.claims = claims
        self._last_processed_message = None
        self._last_processed_text = None
        parent = os.path.dirname(log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        messages = self.context.messages
        if messages:
            last = messages[-1]
            if isinstance(last, dict):
                claimed = self.claims is not None and self.claims.is_claimed(last)
                is_new_turn = (
                    last is not self._last_processed_message
                    or last.get("content") != self._last_processed_text
                )
            else:
                claimed, is_new_turn = False, False
            if (not claimed and isinstance(last, dict) and last.get("role") == "user"
                    and isinstance(last.get("content"), str) and is_new_turn):
                self._last_processed_message = last
                self._last_processed_text = last["content"]
                await self._inject_context(last)

        await self.push_frame(frame, direction)

    def _log_unanswered(self, query_text: str) -> None:
        parent = os.path.dirname(self.log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as log_file:
            log_file.write(query_text.strip().replace("\n", " ") + "\n")

    async def _inject_context(self, user_message: dict):
        query_text = user_message["content"]
        embedding = self.embed_model.encode(
            "query: " + query_text, normalize_embeddings=True
        ).tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(self.top_k, self.collection.count()),
            include=["documents", "distances"],
        )

        docs = (results.get("documents") or [[]])[0] or []
        distances = (results.get("distances") or [[]])[0] or []
        relevant = [
            doc for doc, distance in zip(docs, distances)
            if doc and distance is not None and float(distance) <= self.distance_threshold
        ]

        if not relevant:
            retrieved_text = "No relevant information found."
            self._log_unanswered(query_text)
            print(f"[TICKET-RAISE STUB] Would raise a ticket for: {query_text!r}")
        else:
            retrieved_text = "\n".join(f"- {doc}" for doc in relevant)

        system_message = self.context.messages[0]
        system_message["content"] = SYSTEM_TEMPLATE.format(context=retrieved_text)
        # Keep the same user-message object so TurnClaims and downstream processors
        # can still identify the current turn by object identity.
        self.context.messages[:] = [system_message, user_message]
