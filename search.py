import os
import psycopg
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from embeddings import embed_one

load_dotenv()

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def tokenize(text):
    return text.lower().split()

def load_corpus():
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    cur.execute("SELECT chunk_id, content, source, question, category FROM support_chunks")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    metas = [{"source": r[2], "question": r[3], "category": r[4]} for r in rows]
    return ids, texts, metas

# Load once when the module is imported, not on every request
ALL_IDS, ALL_CHUNKS, ALL_METAS = load_corpus()
BM25 = BM25Okapi([tokenize(c) for c in ALL_CHUNKS])

def vector_search(query, k=10):
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    query_embedding = embed_one(query, task_type="RETRIEVAL_QUERY")
    cur.execute(
        """
        SELECT chunk_id, content, source, question, category
        FROM support_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_embedding, k)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [(r[0], r[1], {"source": r[2], "question": r[3], "category": r[4]}) for r in rows]

def bm25_search(query, k=10):
    scores = BM25.get_scores(tokenize(query))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(ALL_IDS[i], ALL_CHUNKS[i], ALL_METAS[i]) for i in top_idx]

def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    scores, doc_lookup = {}, {}
    for rank, (doc_id, text, meta) in enumerate(vector_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        doc_lookup[doc_id] = (text, meta)
    for rank, (doc_id, text, meta) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        doc_lookup[doc_id] = (text, meta)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_id, doc_lookup[doc_id][0], doc_lookup[doc_id][1]) for doc_id, _ in ranked]

def hybrid_search(query, k=10):
    vec_results = vector_search(query, k=k)
    bm_results = bm25_search(query, k=k)
    return reciprocal_rank_fusion(vec_results, bm_results)[:k]

def rerank(query, candidates, top_n=5):
    pairs = [[query, text] for _, text, _ in candidates]
    scores = reranker.predict(pairs)
    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [(doc_id, text, meta) for (doc_id, text, meta), _ in scored[:top_n]]