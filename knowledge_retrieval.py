import os
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

SYSTEM_TEMPLATE = """You are a voice assistant for the AMU Computer Centre help desk, answering Wi-Fi questions over a phone call.

Answer ONLY using the information in CONTEXT below. Keep your answer to one or two short spoken sentences. No lists, no markdown, no headings.

If CONTEXT does not contain the answer, respond with exactly this sentence and nothing else:
"I'm sorry, I don't have that information on file. I'll note this down so our team can follow up."

CONTEXT:
{context}
"""

class KnowledgeRetrievalProcessor(FrameProcessor):
    def __init__(self, context, collection, embed_model, distance_threshold, top_k=3,
                 log_path="logs/unanswered_queries.log"):
        super().__init__()
        self.context = context
        self.collection = collection
        self.embed_model = embed_model
        self.distance_threshold = distance_threshold
        self.top_k = top_k
        self.log_path = log_path
        self._last_processed = None
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        messages = self.context.messages
        if messages:
            last = messages[-1]
            if last.get("role") == "user" and last.get("content") != self._last_processed:
                self._last_processed = last["content"]
                await self._inject_context(last["content"])

        await self.push_frame(frame, direction)

    async def _inject_context(self, query_text: str):
        embedding = self.embed_model.encode(query_text).tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=self.top_k,
            include=["documents", "distances"],
        )
        docs = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        best_distance = distances[0] if distances else 999.0

        if not docs or best_distance > self.distance_threshold:
            retrieved_text = "No relevant information found."
            with open(self.log_path, "a") as f:
                f.write(query_text.strip() + "\n")
            print(f"[TICKET-RAISE STUB] Would raise a ticket for: {query_text!r}")
        else:
            retrieved_text = "\n".join(f"- {d}" for d in docs)

        self.context.messages[0]["content"] = SYSTEM_TEMPLATE.format(context=retrieved_text)