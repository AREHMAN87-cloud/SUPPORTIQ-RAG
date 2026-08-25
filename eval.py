import json
import time
from search import hybrid_search, rerank
from google import genai
import os
from dotenv import load_dotenv

" eval.py — runs every question through your real /query pipeline and scores both retrieval and generation:"
load_dotenv()
llm_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

with open("eval_dataset.json", "r") as f:
    eval_set = json.load(f)

def run_query(question):
    candidates = hybrid_search(question, k=20)
    top_chunks = rerank(question, candidates, top_n=5)
    context = "\n\n".join(text for _, text, _ in top_chunks)
    sources = list(set(meta["source"] for _, _, meta in top_chunks))

    prompt = f"""Answer using ONLY the context below. If the answer isn't in the context, say you don't have that information.

Context:
{context}

Question: {question}

Answer:"""
    response = llm_client.models.generate_content(model="gemini-3.1-flash-lite", contents=prompt)
    return response.text, sources

def judge_answer(question, answer, expected_source):
    """Uses the LLM itself to judge whether the answer is appropriate."""
    if expected_source == "NONE":
        judge_prompt = f"""Question: {question}
Answer given: {answer}

Was the answer appropriately honest about not knowing, rather than making something up? Reply with only YES or NO."""
    else:
        judge_prompt = f"""Question: {question}
Answer given: {answer}

Does this answer seem like a confident, specific, relevant response (not a generic "please provide more info" deflection, not a refusal)? Reply with only YES or NO."""

    result = llm_client.models.generate_content(model="gemini-3.1-flash-lite", contents=judge_prompt)
    return result.text.strip().upper().startswith("YES")

retrieval_hits = 0
generation_pass = 0
results = []

for item in eval_set:
    question = item["question"]
    expected = item["expected_source"]

    answer, sources = run_query(question)

    retrieval_hit = (expected == "NONE" and len(sources) == 0) or (expected in sources)
    gen_pass = judge_answer(question, answer, expected)

    retrieval_hits += retrieval_hit
    generation_pass += gen_pass

    results.append({
        "question": question,
        "expected_source": expected,
        "retrieved_sources": sources,
        "retrieval_hit": retrieval_hit,
        "answer": answer,
        "generation_pass": gen_pass
    })

    print(f"[{'HIT' if retrieval_hit else 'MISS'}] [{'PASS' if gen_pass else 'FAIL'}] {question}")
    time.sleep(1)  # stay within free-tier rate limits

n = len(eval_set)
print(f"\n=== Results ===")
print(f"Retrieval accuracy: {retrieval_hits}/{n} ({100*retrieval_hits/n:.0f}%)")
print(f"Generation quality:  {generation_pass}/{n} ({100*generation_pass/n:.0f}%)")

with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)