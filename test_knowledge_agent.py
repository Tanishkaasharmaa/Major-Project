# evaluate_embeddings.py
import json
import gc
import torch
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from eval_set import EVAL_SET

with open("knowledge/faqs.json") as f:
    FAQS = json.load(f)

# ============================================================
# Candidate embedding models to compare.
# Prefixes matter a lot for bge/e5-style models — these are my
# best understanding from each model's card, but VERIFY against
# the actual model card before trusting results from a model
# you haven't used before. Wrong prefixes silently hurt accuracy
# rather than erroring, so don't skip this check.
# ============================================================
MODEL_CONFIGS = [
    {"name": "BAAI/bge-m3", "query_prefix": "", "passage_prefix": ""},
    {"name": "BAAI/bge-small-en-v1.5",
     "query_prefix": "Represent this sentence for searching relevant passages: ",
     "passage_prefix": ""},
    {"name": "BAAI/bge-large-en-v1.5",
     "query_prefix": "Represent this sentence for searching relevant passages: ",
     "passage_prefix": ""},
    {"name": "intfloat/e5-large-v2",
     "query_prefix": "query: ", "passage_prefix": "passage: "},
]

THRESHOLD_START, THRESHOLD_STOP, THRESHOLD_STEP = 0.0, 1.0, 0.01


def build_collection(model, passage_prefix, collection_name):
    """In-memory, ephemeral client — doesn't touch your real chroma_db on Drive."""
    client = chromadb.Client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    texts = [passage_prefix + faq["question"] for faq in FAQS]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()
    collection.add(
        ids=[str(i) for i in range(len(FAQS))],
        embeddings=embeddings,
        documents=[faq["answer"] for faq in FAQS],
        metadatas=[{"question": faq["question"]} for faq in FAQS],
    )
    return collection


def make_embed_fn(model, query_prefix):
    def embed_fn(query_text):
        vec = model.encode(query_prefix + query_text, normalize_embeddings=True)
        return vec.tolist()
    return embed_fn


def evaluate(threshold, embed_fn, collection, eval_set):
    correct = false_fallback = false_answer = wrong_match = 0
    fails = []
    for query, expected_id in eval_set:
        emb = embed_fn(query)
        res = collection.query(query_embeddings=[emb], n_results=1, include=["distances"])
        best_id = int(res["ids"][0][0]) if res["ids"] and res["ids"][0] else None
        best_dist = res["distances"][0][0] if res["distances"] and res["distances"][0] else 999.0
        is_grounded = best_dist <= threshold

        ok = False
        if expected_id is None:
            if is_grounded:
                false_answer += 1
            else:
                correct += 1
                ok = True
        else:
            if not is_grounded:
                false_fallback += 1
            elif best_id != expected_id:
                wrong_match += 1
            else:
                correct += 1
                ok = True

        if not ok:
            fails.append({
                "query": query, "expected_id": expected_id, "got_id": best_id,
                "distance": round(best_dist, 3), "was_grounded": is_grounded,
            })

    return {"correct": correct, "false_fallback": false_fallback,
            "false_answer": false_answer, "wrong_match": wrong_match,
            "total": len(eval_set), "fails": fails}


def sweep_and_pick_best(embed_fn, collection, eval_set):
    results = []
    t = THRESHOLD_START
    while t <= THRESHOLD_STOP + 1e-9:
        r = evaluate(round(t, 3), embed_fn, collection, eval_set)
        results.append((round(t, 3), r))
        t += THRESHOLD_STEP

    # Rank by: most correct, then fewest false_answer (confidently-wrong is the
    # worse failure mode than an over-cautious fallback), then fewest wrong_match.
    best_t, best_r = max(
        results,
        key=lambda x: (x[1]["correct"], -x[1]["false_answer"], -x[1]["wrong_match"])
    )
    return best_t, best_r, results


def free_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Run the comparison
# ============================================================
summary = []

for cfg in MODEL_CONFIGS:
    print(f"\n{'='*60}\nLoading {cfg['name']} ...")
    model = SentenceTransformer(cfg["name"])

    collection = build_collection(model, cfg["passage_prefix"], f"eval_{cfg['name'].replace('/', '_')}")
    embed_fn = make_embed_fn(model, cfg["query_prefix"])

    best_t, best_r, all_results = sweep_and_pick_best(embed_fn, collection, EVAL_SET)

    print(f"Best threshold for {cfg['name']}: {best_t}")
    print(f"  correct={best_r['correct']}/{best_r['total']}  "
          f"false_fallback={best_r['false_fallback']}  "
          f"false_answer={best_r['false_answer']}  "
          f"wrong_match={best_r['wrong_match']}")

    summary.append({
        "model": cfg["name"], "best_threshold": best_t,
        "correct": best_r["correct"], "total": best_r["total"],
        "false_fallback": best_r["false_fallback"],
        "false_answer": best_r["false_answer"],
        "wrong_match": best_r["wrong_match"],
        "fails": best_r["fails"],
        "embed_fn": embed_fn, "collection": collection,  # kept for diagnosis below
    })

    free_model(model)

# ============================================================
# Comparison table
# ============================================================
print(f"\n\n{'='*60}\n=== MODEL COMPARISON (best threshold each) ===\n{'='*60}")
print(f"{'Model':<32} {'Thresh':>7} {'Correct':>9} {'FalseFB':>8} {'FalseAns':>9} {'WrongM':>7}")
for s in sorted(summary, key=lambda x: -x["correct"]):
    print(f"{s['model']:<32} {s['best_threshold']:>7.2f} "
          f"{s['correct']:>5}/{s['total']:<3} {s['false_fallback']:>8} "
          f"{s['false_answer']:>9} {s['wrong_match']:>7}")

winner = max(summary, key=lambda x: (x["correct"], -x["false_answer"], -x["wrong_match"]))
print(f"\nBest overall: {winner['model']} @ threshold {winner['best_threshold']}")

# ============================================================
# Detailed failure diagnosis for the winning model
# ============================================================
print(f"\n{'='*60}\n=== FAILURES for {winner['model']} @ {winner['best_threshold']} ===\n{'='*60}")
for f in winner["fails"]:
    expected_q = FAQS[f["expected_id"]]["question"] if f["expected_id"] is not None else "(nothing — nonsense query)"
    got_q = FAQS[f["got_id"]]["question"] if f["was_grounded"] and f["got_id"] is not None else "(fallback triggered)"
    print(f"dist={f['distance']}")
    print(f"  query:    {f['query']}")
    print(f"  expected: {expected_q}")
    print(f"  got:      {got_q}\n")

# ============================================================
# Optional: does a cross-encoder reranker fix the wrong_match cases?
# Only tests queries the winning model got wrong — cheap, targeted check.
# Set RUN_RERANK_CHECK = False to skip (loads one more model, adds time).
# ============================================================
RUN_RERANK_CHECK = True

if RUN_RERANK_CHECK and winner["fails"]:
    print(f"\n{'='*60}\n=== RERANKER CHECK on failures ===\n{'='*60}")
    reranker = CrossEncoder("BAAI/bge-reranker-base")
    collection = winner["collection"]
    embed_fn = winner["embed_fn"]

    for f in winner["fails"]:
        if f["expected_id"] is None:
            continue  # reranking a nonsense query against candidates isn't the test here
        emb = embed_fn(f["query"])
        candidates = collection.query(query_embeddings=[emb], n_results=5,
                                       include=["documents", "metadatas"])
        cand_questions = [m["question"] for m in candidates["metadatas"][0]]
        cand_ids = candidates["ids"][0]

        pairs = [(f["query"], q) for q in cand_questions]
        scores = reranker.predict(pairs)
        rerank_best_idx = scores.argmax()
        rerank_best_id = int(cand_ids[rerank_best_idx])

        fixed = "FIXED" if rerank_best_id == f["expected_id"] else "still wrong"
        print(f"[{fixed}] query: {f['query']}")
        print(f"  embedding picked: {FAQS[f['got_id']]['question'] if f['got_id'] is not None else '(none)'}")
        print(f"  reranker picked:  {FAQS[rerank_best_id]['question']}")
        print(f"  expected:         {FAQS[f['expected_id']]['question']}\n")

    free_model(reranker)