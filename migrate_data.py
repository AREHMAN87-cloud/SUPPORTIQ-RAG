import os
from dotenv import load_dotenv
import psycopg
import chromadb
from embeddings import embed_one

load_dotenv()

# --- Connect to your old Chroma collection ---
CHROMA_PATH = r"C:\Users\Hp\Downloads\Supportiq_chroma"  # e.g. downloaded copy of supportiq_chroma folder
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection("supportiq_kb")

old_data = collection.get(include=["documents", "metadatas"])
chunks = old_data["documents"]
metadatas = old_data["metadatas"]
ids = old_data["ids"]

print(f"Found {len(chunks)} chunks in Chroma to migrate")

# --- Connect to Postgres ---
conn = psycopg.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

inserted, skipped = 0, 0

for i, (chunk_id, text, meta) in enumerate(zip(ids, chunks, metadatas)):
    try:
        # Re-embed at 768 dims since your old Chroma embeddings were 3072-dim
        embedding = embed_one(text, task_type="RETRIEVAL_DOCUMENT")

        cur.execute(
            """
            INSERT INTO support_chunks (chunk_id, content, embedding, source, question, category)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO NOTHING
            """,
            (
                meta.get("source", "unknown") + f"_{i}",  # unique fallback if chunk_id collides
                text,
                embedding,
                meta.get("source", ""),
                meta.get("question", ""),
                meta.get("category", "")
            )
        )
        inserted += 1
        if inserted % 20 == 0:
            conn.commit()
            print(f"Inserted {inserted}/{len(chunks)}")
    except Exception as e:
        print(f"Skipped chunk {i}: {e}")
        skipped += 1

conn.commit()
cur.close()
conn.close()

print(f"Done. Inserted: {inserted}, Skipped: {skipped}")