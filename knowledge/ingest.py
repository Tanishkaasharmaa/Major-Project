"""Build the persistent Chroma FAQ collection.

Run from the project root: `python knowledge/ingest.py`.
The existing collection is replaced only after the FAQ file has been validated
and all embeddings have been generated successfully.
"""
import json
import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = os.environ.get(
    "VOCAL_CHROMA_PATH",
    "/content/drive/MyDrive/voice-agent-mvp/chroma_db",
)
COLLECTION_NAME = "wifi_faqs"
EMBEDDING_MODEL = "intfloat/e5-large-v2"


def main():
    project_root = Path(__file__).resolve().parents[1]
    faq_path = project_root / "knowledge" / "faqs.json"
    with faq_path.open(encoding="utf-8") as faq_file:
        faqs = json.load(faq_file)

    if not isinstance(faqs, list) or not faqs:
        raise ValueError(f"{faq_path} must contain a non-empty JSON list")
    for index, faq in enumerate(faqs):
        if not isinstance(faq, dict) or not faq.get("question") or not faq.get("answer"):
            raise ValueError(f"FAQ entry {index} must contain non-empty question and answer fields")

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = ["passage: " + faq["question"].strip() for faq in faqs]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()
    dimension = len(embeddings[0])
    if any(len(vector) != dimension for vector in embeddings):
        raise ValueError("Embedding model returned inconsistent vector dimensions")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    existing = client.list_collections()
    existing_names = {
        item if isinstance(item, str) else item.name
        for item in existing
    }
    if COLLECTION_NAME in existing_names:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": dimension,
            "faq_count": len(faqs),
        },
    )
    collection.add(
        ids=[str(i) for i in range(len(faqs))],
        embeddings=embeddings,
        documents=[faq["answer"].strip() for faq in faqs],
        metadatas=[{"question": faq["question"].strip()} for faq in faqs],
    )
    print(f"Ingested {collection.count()} FAQ entries into {CHROMA_PATH}")


if __name__ == "__main__":
    main()
