import json, chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "/content/drive/MyDrive/voice-agent-mvp/chroma_db"

model = SentenceTransformer("intfloat/e5-large-v2")
client = chromadb.PersistentClient(path=CHROMA_PATH)

try:
    client.delete_collection("wifi_faqs")
except Exception:
    pass
collection = client.create_collection("wifi_faqs", metadata={"hnsw:space": "cosine"})

with open("knowledge/faqs.json") as f:
    faqs = json.load(f)

texts = ["passage: " + faq["question"] for faq in faqs]
embeddings = model.encode(texts, normalize_embeddings=True).tolist()

collection.add(
    ids=[str(i) for i in range(len(faqs))],
    embeddings=embeddings,
    documents=[faq["answer"] for faq in faqs],
    metadatas=[{"question": faq["question"]} for faq in faqs],
)
print(f"Ingested {len(faqs)} FAQ entries into {CHROMA_PATH}")