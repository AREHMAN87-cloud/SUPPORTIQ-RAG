from embeddings import embed_one
import psycopg
import os
from dotenv import load_dotenv

load_dotenv()
conn = psycopg.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

query = "Can I return shoes after 10 days?"
query_embedding = embed_one(query, task_type="RETRIEVAL_QUERY")

cur.execute(
    """
    SELECT chunk_id, content, source, 1 - (embedding <=> %s::vector) AS similarity
    FROM support_chunks
    ORDER BY embedding <=> %s::vector
    LIMIT 3
    """,
    (query_embedding, query_embedding)
)

for row in cur.fetchall():
    print("Chunk:", row[0])
    print("Source:", row[2])
    print("Similarity:", round(row[3], 4))
    print("Content:", row[1][:150])
    print("---")

cur.close()
conn.close()