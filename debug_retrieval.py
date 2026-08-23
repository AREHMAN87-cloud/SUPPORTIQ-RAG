from search import bm25_search

query = "Can I return shoes after 10 days?"
results = bm25_search(query, k=15)

for rank, (doc_id, text, meta) in enumerate(results):
    marker = " <-- POLICY DOC" if meta["source"].endswith(".txt") else ""
    print(f"{rank+1}. {doc_id} | {meta['source']}{marker}")