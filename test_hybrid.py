from search import hybrid_search, rerank

query = "Can I return shoes after 10 days?"
candidates = hybrid_search(query, k=10)
top = rerank(query, candidates, top_n=3)

for doc_id, text, meta in top:
    print(doc_id, "|", meta["source"], "|", text[:100])