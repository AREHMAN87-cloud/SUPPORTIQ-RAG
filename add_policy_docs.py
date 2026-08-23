
import os
import glob
import psycopg
from dotenv import load_dotenv
from embeddings import embed_one

load_dotenv()

def chunk_text(text, chunk_size=200, overlap=30):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks

conn = psycopg.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

policy_files = glob.glob("policy_docs/*.txt")
print(f"Found {len(policy_files)} policy files")

inserted = 0
for path in policy_files:
    filename = os.path.basename(path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    for idx, chunk in enumerate(chunk_text(text)):
        embedding = embed_one(chunk, task_type="RETRIEVAL_DOCUMENT")
        chunk_id = f"{filename}_{idx}"

        cur.execute(
            """
            INSERT INTO support_chunks (chunk_id, content, embedding, source, question, category)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO NOTHING
            """,
            (chunk_id, chunk, embedding, filename, "", "policy")
        )
        inserted += 1

conn.commit()
cur.close()
conn.close()
print(f"Inserted {inserted} policy chunks")