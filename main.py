
from fastapi import FastAPI
from pydantic import BaseModel
from search import hybrid_search, rerank
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import time
from logger import log_query

load_dotenv()
app = FastAPI()
llm_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"status": "SupportIQ API is running"}


@app.post("/query")
def query(request: QueryRequest):
    t0 = time.time()
    candidates = hybrid_search(request.question, k=20)
    top_chunks = rerank(request.question, candidates, top_n=5)
    t1 = time.time()

    context = "\n\n".join(text for _, text, _ in top_chunks)
    sources = list(set(meta["source"] for _, _, meta in top_chunks))

    prompt = f"""Answer the customer's question using ONLY the context below.
If the answer isn't in the context, say you don't have that information.

Context:
{context}

Question: {request.question}

Answer:"""

    response = llm_client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )
    t2 = time.time()

    log_query(
        question=request.question,
        sources=sources,
        answer=response.text,
        retrieval_ms=round((t1 - t0) * 1000, 1),
        generation_ms=round((t2 - t1) * 1000, 1)
    )

    return {
        "question": request.question,
        "answer": response.text,
        "sources": sources
    }