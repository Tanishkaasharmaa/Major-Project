import json, chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "/content/drive/MyDrive/voice-agent-mvp/chroma_db"

model = SentenceTransformer("BAAI/bge-m3")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("wifi_faqs")

with open("knowledge/faqs.json") as f:
    faqs = json.load(f)

for idx, faq in enumerate(faqs):
    embedding = model.encode(faq["question"] + " " + faq["answer"]).tolist()
    collection.add(
        ids=[str(idx)],
        embeddings=[embedding],
        documents=[faq["answer"]],
        metadatas=[{"question": faq["question"]}]
    )

print(f"Ingested {len(faqs)} FAQ entries into {CHROMA_PATH}")