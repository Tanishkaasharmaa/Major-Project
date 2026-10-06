import json
import requests
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "/content/drive/MyDrive/voice-agent-mvp/chroma_db"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b-instruct"
DISTANCE_THRESHOLD = 1.0
NUM_CTX = 4096          # raise from Ollama's 2048 default — see note below
NUM_PREDICT = 120       # cap response length; keeps answers short + bounds latency

SYSTEM_TEMPLATE = """You are a voice assistant for the AMU Computer Centre help desk, answering Wi-Fi questions over a phone call.

Answer ONLY using the information in CONTEXT below. Keep your answer to one or two short spoken sentences. No lists, no markdown, no headings.

If CONTEXT does not contain the answer, respond with exactly this sentence and nothing else:
"I'm sorry, I don't have that information on file. I'll note this down so our team can follow up."

CONTEXT:
{context}
"""

FALLBACK_TEXT = "I'm sorry, I don't have that information on file. I'll note this down so our team can follow up."

# --- setup (once) ---
embed_model = SentenceTransformer("BAAI/bge-m3")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("wifi_faqs")


def embed_fn(query_text):
    return embed_model.encode(query_text).tolist()


def retrieve(query_text, top_k=3):
    embedding = embed_fn(query_text)
    res = collection.query(
        query_embeddings=[embedding], n_results=top_k,
        include=["documents", "distances"],
    )
    docs = res["documents"][0] if res["documents"] else []
    distances = res["distances"][0] if res["distances"] else []
    best_distance = distances[0] if distances else 999.0
    return docs, best_distance


def evaluate(threshold, embed_fn, collection, eval_set):
    correct, false_fallback, false_answer, wrong_match = 0, 0, 0, 0
    for query, expected_id in eval_set:
        emb = embed_fn(query)
        res = collection.query(query_embeddings=[emb], n_results=1, include=["distances"])
        best_id = int(res["ids"][0][0]) if (res["ids"] and res["ids"][0]) else None
        best_dist = res["distances"][0][0] if (res["distances"] and res["distances"][0]) else 999.0
        is_grounded = best_dist <= threshold

        if expected_id is None:
            if is_grounded:
                false_answer += 1
            else:
                correct += 1
        else:
            if not is_grounded:
                false_fallback += 1
            elif best_id != expected_id:
                wrong_match += 1
            else:
                correct += 1

    total = len(eval_set)
    print(f"threshold={threshold:.2f}: correct={correct}/{total}  "
          f"false_fallback={false_fallback}  false_answer={false_answer}  wrong_match={wrong_match}")


def sweep_thresholds(eval_set, start=0.3, stop=1.2, step=0.05):
    print("\n=== THRESHOLD EVALUATION SWEEP ===")
    t = start
    while t <= stop + 1e-5:
        evaluate(round(t, 2), embed_fn, collection, eval_set)
        t += step


def ask(query_text):
    docs, best_distance = retrieve(query_text)
    is_grounded = bool(docs) and best_distance <= DISTANCE_THRESHOLD
    context_text = "\n".join(f"- {d}" for d in docs) if is_grounded else "No relevant information found."

    # single-turn only — no accumulated history, so no context-length risk here
    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(context=context_text)},
        {"role": "user", "content": query_text},
    ]

    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
        },
    })
    resp.raise_for_status()
    data = resp.json()
    answer = data["message"]["content"].strip()

    # Ollama returns prompt_eval_count / eval_count — actual token usage for this call
    prompt_tokens = data.get("prompt_eval_count", "?")
    response_tokens = data.get("eval_count", "?")

    return {
        "question": query_text,
        "distance": round(best_distance, 3),
        "grounded": is_grounded,
        "answer": answer,
        "matches_fallback": answer.strip() == FALLBACK_TEXT,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
    }


TEST_QUESTIONS = {
    "close_match": [
        "How do I apply for Wi-Fi access?",
        "My Wi-Fi application was rejected, what do I do?",
        "What are the Computer Centre's support hours?",
        "How do I reset my Wi-Fi password?",
        "Is there a cost for Wi-Fi access?",
    ],
    "paraphrased": [
        "My phone won't connect to the wifi anymore, why?",
        "I got a new laptop, how do I get it working on the network?",
        "I applied ages ago and still nothing, what's happening?",
    ],
    "nonsense": [
        "What's the capital of France?",
        "Can you recommend a good pizza place?",
        "asdkj banana purple elephant",
    ],
}

if __name__ == "__main__":
    log_lines = []
    for category, questions in TEST_QUESTIONS.items():
        print(f"\n=== {category.upper()} ===")
        for q in questions:
            result = ask(q)
            flag = "GROUNDED" if result["grounded"] else "FALLBACK"
            print(f"[{flag}] dist={result['distance']} | tokens(in/out)={result['prompt_tokens']}/{result['response_tokens']}")
            print(f"  Q: {result['question']}")
            print(f"  A: {result['answer']}\n")
            log_lines.append(result)

    # quick pass/fail summary against expectations
    print("\n=== SUMMARY ===")
    close_ok = all(not r["matches_fallback"] for r in log_lines[:5])
    nonsense_ok = all(r["matches_fallback"] for r in log_lines[-3:])
    print(f"All close-match questions got real answers: {close_ok}")
    print(f"All nonsense questions got the fallback: {nonsense_ok}")

from eval_set import EVAL_SET

if __name__ == "__main__":
    # Run the evaluation sweep across candidate thresholds
    sweep_thresholds(EVAL_SET, start=0.3, stop=1.2, step=0.05)
