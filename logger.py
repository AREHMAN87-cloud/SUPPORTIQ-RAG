import json
import os
from datetime import datetime, timezone

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "queries.jsonl")

def log_query(question, sources, answer, retrieval_ms, generation_ms):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "sources": sources,
        "answer": answer,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")