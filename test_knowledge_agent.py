# # evaluate_embeddings.py
# import json
# import gc
# import torch
# import chromadb
# from sentence_transformers import SentenceTransformer, CrossEncoder
# from eval_set import EVAL_SET

# with open("knowledge/faqs.json") as f:
#     FAQS = json.load(f)

# # ============================================================
# # Candidate embedding models to compare.
# # Prefixes matter a lot for bge/e5-style models — these are my
# # best understanding from each model's card, but VERIFY against
# # the actual model card before trusting results from a model
# # you haven't used before. Wrong prefixes silently hurt accuracy
# # rather than erroring, so don't skip this check.
# # ============================================================
# MODEL_CONFIGS = [
#     {"name": "BAAI/bge-m3", "query_prefix": "", "passage_prefix": ""},
#     {"name": "BAAI/bge-small-en-v1.5",
#      "query_prefix": "Represent this sentence for searching relevant passages: ",
#      "passage_prefix": ""},
#     {"name": "BAAI/bge-large-en-v1.5",
#      "query_prefix": "Represent this sentence for searching relevant passages: ",
#      "passage_prefix": ""},
#     {"name": "intfloat/e5-large-v2",
#      "query_prefix": "query: ", "passage_prefix": "passage: "},
# ]

# THRESHOLD_START, THRESHOLD_STOP, THRESHOLD_STEP = 0.0, 1.0, 0.01


# def build_collection(model, passage_prefix, collection_name):
#     """In-memory, ephemeral client — doesn't touch your real chroma_db on Drive."""
#     client = chromadb.Client()
#     try:
#         client.delete_collection(collection_name)
#     except Exception:
#         pass
#     collection = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})

#     texts = [passage_prefix + faq["question"] for faq in FAQS]
#     embeddings = model.encode(texts, normalize_embeddings=True).tolist()
#     collection.add(
#         ids=[str(i) for i in range(len(FAQS))],
#         embeddings=embeddings,
#         documents=[faq["answer"] for faq in FAQS],
#         metadatas=[{"question": faq["question"]} for faq in FAQS],
#     )
#     return collection


# def make_embed_fn(model, query_prefix):
#     def embed_fn(query_text):
#         vec = model.encode(query_prefix + query_text, normalize_embeddings=True)
#         return vec.tolist()
#     return embed_fn


# def evaluate(threshold, embed_fn, collection, eval_set):
#     correct = false_fallback = false_answer = wrong_match = 0
#     fails = []
#     for query, expected_id in eval_set:
#         emb = embed_fn(query)
#         res = collection.query(query_embeddings=[emb], n_results=1, include=["distances"])
#         best_id = int(res["ids"][0][0]) if res["ids"] and res["ids"][0] else None
#         best_dist = res["distances"][0][0] if res["distances"] and res["distances"][0] else 999.0
#         is_grounded = best_dist <= threshold

#         ok = False
#         if expected_id is None:
#             if is_grounded:
#                 false_answer += 1
#             else:
#                 correct += 1
#                 ok = True
#         else:
#             if not is_grounded:
#                 false_fallback += 1
#             elif best_id != expected_id:
#                 wrong_match += 1
#             else:
#                 correct += 1
#                 ok = True

#         if not ok:
#             fails.append({
#                 "query": query, "expected_id": expected_id, "got_id": best_id,
#                 "distance": round(best_dist, 3), "was_grounded": is_grounded,
#             })

#     return {"correct": correct, "false_fallback": false_fallback,
#             "false_answer": false_answer, "wrong_match": wrong_match,
#             "total": len(eval_set), "fails": fails}


# def sweep_and_pick_best(embed_fn, collection, eval_set):
#     results = []
#     t = THRESHOLD_START
#     while t <= THRESHOLD_STOP + 1e-9:
#         r = evaluate(round(t, 3), embed_fn, collection, eval_set)
#         results.append((round(t, 3), r))
#         t += THRESHOLD_STEP

#     # Rank by: most correct, then fewest false_answer (confidently-wrong is the
#     # worse failure mode than an over-cautious fallback), then fewest wrong_match.
#     best_t, best_r = max(
#         results,
#         key=lambda x: (x[1]["correct"], -x[1]["false_answer"], -x[1]["wrong_match"])
#     )
#     return best_t, best_r, results


# def free_model(model):
#     del model
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()


# # ============================================================
# # Run the comparison
# # ============================================================
# summary = []

# for cfg in MODEL_CONFIGS:
#     print(f"\n{'='*60}\nLoading {cfg['name']} ...")
#     model = SentenceTransformer(cfg["name"])

#     collection = build_collection(model, cfg["passage_prefix"], f"eval_{cfg['name'].replace('/', '_')}")
#     embed_fn = make_embed_fn(model, cfg["query_prefix"])

#     best_t, best_r, all_results = sweep_and_pick_best(embed_fn, collection, EVAL_SET)

#     print(f"Best threshold for {cfg['name']}: {best_t}")
#     print(f"  correct={best_r['correct']}/{best_r['total']}  "
#           f"false_fallback={best_r['false_fallback']}  "
#           f"false_answer={best_r['false_answer']}  "
#           f"wrong_match={best_r['wrong_match']}")

#     summary.append({
#         "model": cfg["name"], "best_threshold": best_t,
#         "correct": best_r["correct"], "total": best_r["total"],
#         "false_fallback": best_r["false_fallback"],
#         "false_answer": best_r["false_answer"],
#         "wrong_match": best_r["wrong_match"],
#         "fails": best_r["fails"],
#         "embed_fn": embed_fn, "collection": collection,  # kept for diagnosis below
#     })

#     free_model(model)

# # ============================================================
# # Comparison table
# # ============================================================
# print(f"\n\n{'='*60}\n=== MODEL COMPARISON (best threshold each) ===\n{'='*60}")
# print(f"{'Model':<32} {'Thresh':>7} {'Correct':>9} {'FalseFB':>8} {'FalseAns':>9} {'WrongM':>7}")
# for s in sorted(summary, key=lambda x: -x["correct"]):
#     print(f"{s['model']:<32} {s['best_threshold']:>7.2f} "
#           f"{s['correct']:>5}/{s['total']:<3} {s['false_fallback']:>8} "
#           f"{s['false_answer']:>9} {s['wrong_match']:>7}")

# winner = max(summary, key=lambda x: (x["correct"], -x["false_answer"], -x["wrong_match"]))
# print(f"\nBest overall: {winner['model']} @ threshold {winner['best_threshold']}")

# # ============================================================
# # Detailed failure diagnosis for the winning model
# # ============================================================
# print(f"\n{'='*60}\n=== FAILURES for {winner['model']} @ {winner['best_threshold']} ===\n{'='*60}")
# for f in winner["fails"]:
#     expected_q = FAQS[f["expected_id"]]["question"] if f["expected_id"] is not None else "(nothing — nonsense query)"
#     got_q = FAQS[f["got_id"]]["question"] if f["was_grounded"] and f["got_id"] is not None else "(fallback triggered)"
#     print(f"dist={f['distance']}")
#     print(f"  query:    {f['query']}")
#     print(f"  expected: {expected_q}")
#     print(f"  got:      {got_q}\n")

# # ============================================================
# # Optional: does a cross-encoder reranker fix the wrong_match cases?
# # Only tests queries the winning model got wrong — cheap, targeted check.
# # Set RUN_RERANK_CHECK = False to skip (loads one more model, adds time).
# # ============================================================
# RUN_RERANK_CHECK = True

# if RUN_RERANK_CHECK and winner["fails"]:
#     print(f"\n{'='*60}\n=== RERANKER CHECK on failures ===\n{'='*60}")
#     reranker = CrossEncoder("BAAI/bge-reranker-base")
#     collection = winner["collection"]
#     embed_fn = winner["embed_fn"]

#     for f in winner["fails"]:
#         if f["expected_id"] is None:
#             continue  # reranking a nonsense query against candidates isn't the test here
#         emb = embed_fn(f["query"])
#         candidates = collection.query(query_embeddings=[emb], n_results=5,
#                                        include=["documents", "metadatas"])
#         cand_questions = [m["question"] for m in candidates["metadatas"][0]]
#         cand_ids = candidates["ids"][0]

#         pairs = [(f["query"], q) for q in cand_questions]
#         scores = reranker.predict(pairs)
#         rerank_best_idx = scores.argmax()
#         rerank_best_id = int(cand_ids[rerank_best_idx])

#         fixed = "FIXED" if rerank_best_id == f["expected_id"] else "still wrong"
#         print(f"[{fixed}] query: {f['query']}")
#         print(f"  embedding picked: {FAQS[f['got_id']]['question'] if f['got_id'] is not None else '(none)'}")
#         print(f"  reranker picked:  {FAQS[rerank_best_id]['question']}")
#         print(f"  expected:         {FAQS[f['expected_id']]['question']}\n")

#     free_model(reranker)



import json
import gc
import time
import math
import statistics
import torch
import chromadb

from sentence_transformers import SentenceTransformer, CrossEncoder
from eval_set import EVAL_SET


# ============================================================
# 1. LOAD FAQ DATA
# ============================================================

with open("knowledge/faqs.json", "r", encoding="utf-8") as f:
    FAQS = json.load(f)


# ============================================================
# 2. EMBEDDING MODEL CONFIGURATIONS
# ============================================================

MODEL_CONFIGS = [

    {
        "name": "BAAI/bge-m3",
        "query_prefix": "",
        "passage_prefix": "",
        "query_prompt_name": None
    },

    {
        "name": "Snowflake/snowflake-arctic-embed-m",
        "query_prefix":
            "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
        "query_prompt_name": None
    },

    {
        "name": "Snowflake/snowflake-arctic-embed-l",
        "query_prefix":
            "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
        "query_prompt_name": None
    },

    {
        "name": "BAAI/bge-large-en-v1.5",
        "query_prefix":
            "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
        "query_prompt_name": None
    },

    {
        "name": "intfloat/e5-large-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "query_prompt_name": None
    }

]


# ============================================================
# 3. EXPERIMENT CONFIGURATION
# ============================================================

THRESHOLD_START = 0.0
THRESHOLD_STOP = 1.0
THRESHOLD_STEP = 0.01

TOP_K = 1

RUN_RERANK_CHECK = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("EMBEDDING MODEL EVALUATION")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Number of FAQs: {len(FAQS)}")
print(f"Number of evaluation queries: {len(EVAL_SET)}")
print(f"Number of models: {len(MODEL_CONFIGS)}")


# ============================================================
# 4. GPU SYNCHRONIZATION
# ============================================================

def synchronize_gpu():

    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ============================================================
# 5. BUILD CHROMADB COLLECTION
# ============================================================

def build_collection(model, passage_prefix, collection_name):

    # In-memory ChromaDB client.
    # Does not modify your existing persistent chroma_db.

    client = chromadb.Client()

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Apply the appropriate passage prefix
    texts = [
        passage_prefix + faq["question"]
        for faq in FAQS
    ]

    # Generate FAQ embeddings
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    ).tolist()

    # Store embeddings and FAQ answers
    collection.add(

        ids=[
            str(i)
            for i in range(len(FAQS))
        ],

        embeddings=embeddings,

        documents=[
            faq["answer"]
            for faq in FAQS
        ],

        metadatas=[
            {"question": faq["question"]}
            for faq in FAQS
        ]
    )

    return client, collection


# ============================================================
# 6. QUERY EMBEDDING FUNCTION
# ============================================================

def make_embed_fn(model, query_prefix, query_prompt_name=None):

    def embed_fn(query_text):

        if query_prompt_name is not None:

            vec = model.encode(
                query_text,
                normalize_embeddings=True,
                prompt_name=query_prompt_name
            )

        else:

            vec = model.encode(
                query_prefix + query_text,
                normalize_embeddings=True
            )

        return vec.tolist()

    return embed_fn


# ============================================================
# 7. RETRIEVE EACH QUERY ONCE AND MEASURE LATENCY
# ============================================================

def retrieve_once(embed_fn, collection, eval_set):

    observations = []

    for index, (query, expected_id) in enumerate(eval_set):

        # ----------------------------------------------------
        # A. QUERY EMBEDDING LATENCY
        # ----------------------------------------------------

        synchronize_gpu()

        start = time.perf_counter()

        emb = embed_fn(query)

        synchronize_gpu()

        embedding_time = (
            time.perf_counter() - start
        ) * 1000


        # ----------------------------------------------------
        # B. CHROMADB VECTOR SEARCH LATENCY
        # ----------------------------------------------------

        start = time.perf_counter()

        res = collection.query(

            query_embeddings=[emb],

            n_results=TOP_K,

            include=[
                "distances"
            ]
        )

        search_time = (
            time.perf_counter() - start
        ) * 1000


        # ----------------------------------------------------
        # C. EXTRACT RETRIEVED RESULT
        # ----------------------------------------------------

        best_id = (
            int(res["ids"][0][0])
            if res["ids"] and res["ids"][0]
            else None
        )

        best_dist = (
            res["distances"][0][0]
            if res["distances"] and res["distances"][0]
            else 999.0
        )


        # ----------------------------------------------------
        # D. STORE OBSERVATION
        # ----------------------------------------------------

        observations.append({

            "query": query,

            "expected_id": expected_id,

            "got_id": best_id,

            "distance": best_dist,

            "embedding_ms": embedding_time,

            "search_ms": search_time,

            "total_ms": embedding_time + search_time

        })


        if (index + 1) % 10 == 0:

            print(
                f"Processed {index + 1}/{len(eval_set)} queries"
            )

    return observations


# ============================================================
# 8. PERCENTILE CALCULATION
# ============================================================

def percentile(values, p):

    values = sorted(values)

    if not values:
        return 0.0

    index = math.ceil((p / 100) * len(values)) - 1

    index = max(0, min(index, len(values) - 1))

    return values[index]


# ============================================================
# 9. LATENCY STATISTICS
# ============================================================

def latency_statistics(observations):

    embedding = [
        x["embedding_ms"]
        for x in observations
    ]

    search = [
        x["search_ms"]
        for x in observations
    ]

    total = [
        x["total_ms"]
        for x in observations
    ]

    return {

        "embedding_avg": statistics.mean(embedding),

        "search_avg": statistics.mean(search),

        "total_avg": statistics.mean(total),

        "total_p50": percentile(total, 50),

        "total_p95": percentile(total, 95),

        "total_max": max(total),

        "throughput": (
            1000 / statistics.mean(total)
            if statistics.mean(total) > 0
            else 0
        )

    }


# ============================================================
# 10. EVALUATE RETRIEVAL RESULTS AT A THRESHOLD
# ============================================================

def evaluate_cached(threshold, observations):

    correct = 0

    false_fallback = 0

    false_answer = 0

    wrong_match = 0

    fails = []

    for item in observations:

        expected_id = item["expected_id"]

        best_id = item["got_id"]

        best_dist = item["distance"]

        is_grounded = best_dist <= threshold

        ok = False


        # ----------------------------------------------------
        # QUERY WITH NO EXPECTED FAQ
        # ----------------------------------------------------

        if expected_id is None:

            if is_grounded:

                false_answer += 1

            else:

                correct += 1

                ok = True


        # ----------------------------------------------------
        # QUERY WITH AN EXPECTED FAQ
        # ----------------------------------------------------

        else:

            if not is_grounded:

                false_fallback += 1

            elif best_id != expected_id:

                wrong_match += 1

            else:

                correct += 1

                ok = True


        # ----------------------------------------------------
        # FAILURE DIAGNOSIS
        # ----------------------------------------------------

        if not ok:

            fails.append({

                "query": item["query"],

                "expected_id": expected_id,

                "got_id": best_id,

                "distance": round(best_dist, 3),

                "was_grounded": is_grounded

            })


    return {

        "correct": correct,

        "false_fallback": false_fallback,

        "false_answer": false_answer,

        "wrong_match": wrong_match,

        "total": len(observations),

        "fails": fails

    }


# ============================================================
# 11. THRESHOLD SWEEP
# ============================================================

def sweep_and_pick_best(observations):

    results = []

    t = THRESHOLD_START

    while t <= THRESHOLD_STOP + 1e-9:

        r = evaluate_cached(
            round(t, 3),
            observations
        )

        results.append((round(t, 3), r))

        t += THRESHOLD_STEP


    # Ranking:
    # 1. Maximum correct answers
    # 2. Minimum false answers
    # 3. Minimum wrong matches

    best_t, best_r = max(

        results,

        key=lambda x: (

            x[1]["correct"],

            -x[1]["false_answer"],

            -x[1]["wrong_match"]

        )

    )

    return best_t, best_r, results


# ============================================================
# 12. MEMORY CLEANUP
# ============================================================

def free_model(model):

    del model

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# 13. MAIN MODEL EVALUATION LOOP
# ============================================================

summary = []

all_model_failures = {}

for cfg in MODEL_CONFIGS:

    print("\n" + "=" * 70)

    print(f"LOADING MODEL: {cfg['name']}")

    print("=" * 70)


    # --------------------------------------------------------
    # A. LOAD MODEL
    # --------------------------------------------------------

    model = SentenceTransformer(
        cfg["name"],
        device=DEVICE
    )


    # --------------------------------------------------------
    # B. BUILD FAQ VECTOR DATABASE
    # --------------------------------------------------------

    collection_name = (
        "eval_" + cfg["name"].replace("/", "_").replace("-", "_")
    )

    client, collection = build_collection(

        model,

        cfg["passage_prefix"],

        collection_name

    )


    # --------------------------------------------------------
    # C. CREATE QUERY EMBEDDING FUNCTION
    # --------------------------------------------------------

    embed_fn = make_embed_fn(

        model,

        cfg["query_prefix"],

        cfg.get("query_prompt_name")

    )


    # --------------------------------------------------------
    # D. WARM-UP
    # --------------------------------------------------------

    print("\nWarming up model and vector database...")

    warmup_query = EVAL_SET[0][0]

    warmup_emb = embed_fn(warmup_query)

    synchronize_gpu()

    collection.query(

        query_embeddings=[warmup_emb],

        n_results=TOP_K,

        include=["distances"]

    )

    print("Warm-up complete.")


    # --------------------------------------------------------
    # E. ACTUAL RETRIEVAL BENCHMARK
    # --------------------------------------------------------

    print("\nStarting retrieval benchmark...")

    observations = retrieve_once(

        embed_fn,

        collection,

        EVAL_SET

    )


    # --------------------------------------------------------
    # F. THRESHOLD OPTIMIZATION
    # --------------------------------------------------------

    best_t, best_r, all_results = sweep_and_pick_best(
        observations
    )


    # --------------------------------------------------------
    # G. LATENCY ANALYSIS
    # --------------------------------------------------------

    latency = latency_statistics(observations)


    # --------------------------------------------------------
    # H. DISPLAY MODEL RESULTS
    # --------------------------------------------------------

    print("\n" + "-" * 60)

    print(f"MODEL: {cfg['name']}")

    print("-" * 60)

    print(f"Best threshold       : {best_t:.2f}")

    print(
        f"Correct retrievals   : "
        f"{best_r['correct']}/{best_r['total']}"
    )

    print(
        f"False fallbacks      : "
        f"{best_r['false_fallback']}"
    )

    print(
        f"False answers        : "
        f"{best_r['false_answer']}"
    )

    print(
        f"Wrong matches        : "
        f"{best_r['wrong_match']}"
    )


    print("\n--- RETRIEVAL LATENCY ---")

    print(
        f"Average embedding    : "
        f"{latency['embedding_avg']:.2f} ms"
    )

    print(
        f"Average vector search: "
        f"{latency['search_avg']:.2f} ms"
    )

    print(
        f"Average total        : "
        f"{latency['total_avg']:.2f} ms"
    )

    print(
        f"P50 latency          : "
        f"{latency['total_p50']:.2f} ms"
    )

    print(
        f"P95 latency          : "
        f"{latency['total_p95']:.2f} ms"
    )

    print(
        f"Maximum latency      : "
        f"{latency['total_max']:.2f} ms"
    )

    print(
        f"Sequential throughput: "
        f"{latency['throughput']:.2f} queries/sec"
    )


    # --------------------------------------------------------
    # I. SAVE RESULTS WITHOUT RETAINING MODEL IN MEMORY
    # --------------------------------------------------------

    summary.append({

        "model": cfg["name"],

        "best_threshold": best_t,

        "correct": best_r["correct"],

        "total": best_r["total"],

        "false_fallback": best_r["false_fallback"],

        "false_answer": best_r["false_answer"],

        "wrong_match": best_r["wrong_match"],

        **latency

    })

    all_model_failures[cfg["name"]] = best_r["fails"]


    # --------------------------------------------------------
    # J. RELEASE MEMORY
    # --------------------------------------------------------

    del observations

    del embed_fn

    del collection

    del client

    free_model(model)

    print(f"\nFinished: {cfg['name']}")

    print("Model memory released.")


# ============================================================
# 14. FINAL MODEL COMPARISON TABLE
# ============================================================

print("\n\n")

print("=" * 120)

print(" " * 35 + "FINAL MODEL COMPARISON")

print("=" * 120)


header = (

    f"{'Model':<42}"

    f"{'Accuracy':>10}"

    f"{'Embed':>10}"

    f"{'Search':>10}"

    f"{'Total':>10}"

    f"{'P50':>10}"

    f"{'P95':>10}"

)

print(header)

print("-" * 120)


for s in sorted(

    summary,

    key=lambda x: (

        -x["correct"],

        x["false_answer"],

        x["wrong_match"]

    )

):

    print(

        f"{s['model']:<42}"

        f"{s['correct']}/{s['total']:>5}"

        f"{s['embedding_avg']:>9.2f}"

        f"{s['search_avg']:>9.2f}"

        f"{s['total_avg']:>9.2f}"

        f"{s['total_p50']:>9.2f}"

        f"{s['total_p95']:>9.2f}"

    )


print("-" * 120)


# ============================================================
# 15. SELECT BEST MODEL
# ============================================================

winner = max(

    summary,

    key=lambda x: (

        x["correct"],

        -x["false_answer"],

        -x["wrong_match"],

        -x["total_avg"]

    )

)


print("\n" + "=" * 70)

print("BEST OVERALL MODEL")

print("=" * 70)

print(f"Model             : {winner['model']}")

print(
    f"Accuracy          : "
    f"{winner['correct']}/{winner['total']}"
)

print(
    f"Threshold         : "
    f"{winner['best_threshold']}"
)

print(
    f"Average latency   : "
    f"{winner['total_avg']:.2f} ms"
)

print(
    f"P95 latency       : "
    f"{winner['total_p95']:.2f} ms"
)


# ============================================================
# 16. FAILURE DIAGNOSIS FOR WINNING MODEL
# ============================================================

print("\n" + "=" * 70)

print("FAILURE DIAGNOSIS")

print("=" * 70)

winner_fails = all_model_failures[winner["model"]]

for f in winner_fails:

    expected_q = (

        FAQS[f["expected_id"]]["question"]

        if f["expected_id"] is not None

        else "(nothing - unsupported query)"

    )

    got_q = (

        FAQS[f["got_id"]]["question"]

        if f["was_grounded"] and f["got_id"] is not None

        else "(fallback triggered)"

    )

    print(f"\nDistance: {f['distance']}")

    print(f"Query    : {f['query']}")

    print(f"Expected : {expected_q}")

    print(f"Got      : {got_q}")


# ============================================================
# 17. OPTIONAL CROSS-ENCODER RERANKER CHECK
# ============================================================

if RUN_RERANK_CHECK and winner_fails:

    print("\n" + "=" * 70)

    print("CROSS-ENCODER RERANKER CHECK")

    print("=" * 70)


    # Find configuration for winning model

    winner_cfg = next(

        cfg for cfg in MODEL_CONFIGS

        if cfg["name"] == winner["model"]

    )


    # Reload winning embedding model

    winner_model = SentenceTransformer(

        winner_cfg["name"],

        device=DEVICE

    )


    # Rebuild its collection

    rerank_client, rerank_collection = build_collection(

        winner_model,

        winner_cfg["passage_prefix"],

        "rerank_winner_collection"

    )


    winner_embed_fn = make_embed_fn(

        winner_model,

        winner_cfg["query_prefix"],

        winner_cfg.get("query_prompt_name")

    )


    # Load cross-encoder

    reranker = CrossEncoder(

        "BAAI/bge-reranker-base",

        device=DEVICE

    )


    fixed_count = 0

    tested_count = 0


    for f in winner_fails:

        if f["expected_id"] is None:

            continue


        # Generate query embedding

        emb = winner_embed_fn(f["query"])


        # Retrieve top 5 candidates

        candidates = rerank_collection.query(

            query_embeddings=[emb],

            n_results=min(5, len(FAQS)),

            include=["documents", "metadatas"]

        )


        cand_questions = [

            m["question"]

            for m in candidates["metadatas"][0]

        ]

        cand_ids = candidates["ids"][0]


        # Cross-encoder scoring

        pairs = [

            (f["query"], q)

            for q in cand_questions

        ]


        scores = reranker.predict(pairs)


        rerank_best_idx = scores.argmax()

        rerank_best_id = int(cand_ids[rerank_best_idx])


        fixed = rerank_best_id == f["expected_id"]

        tested_count += 1

        if fixed:

            fixed_count += 1


        print(

            f"\n[{'FIXED' if fixed else 'STILL WRONG'}] "

            f"{f['query']}"

        )

        print(

            f"Embedding picked: "

            f"{FAQS[f['got_id']]['question']}"

        )

        print(

            f"Reranker picked : "

            f"{FAQS[rerank_best_id]['question']}"

        )

        print(

            f"Expected        : "

            f"{FAQS[f['expected_id']]['question']}"

        )


    print("\n--- RERANKER SUMMARY ---")

    print(f"Tested wrong matches: {tested_count}")

    print(f"Fixed matches       : {fixed_count}")

    print(f"Still incorrect     : {tested_count - fixed_count}")


    # Release reranker memory

    del winner_embed_fn

    del reranker

    del rerank_collection

    del rerank_client

    free_model(winner_model)


# ============================================================
# 18. END OF EXPERIMENT
# ============================================================

print("\n" + "=" * 70)

print("ALL EVALUATIONS COMPLETED")

print("=" * 70)
