# SupportIQ — Agentic RAG for Customer Support

**SupportIQ** is an AI-powered customer support backend designed for e-commerce and retail use cases. It combines Retrieval-Augmented Generation (RAG) with hybrid search and an LLM-powered agent to answer customer questions, retrieve company policies, track orders, evaluate return eligibility, and create support tickets.

The system exposes REST APIs through FastAPI and uses Google Gemini for embeddings and response generation. Its retrieval pipeline combines semantic vector search, BM25 keyword search, Reciprocal Rank Fusion (RRF), and cross-encoder reranking to improve the relevance of retrieved information.

## Key Features

- **Hybrid Retrieval:** Combines dense vector search with BM25 keyword retrieval to capture semantic meaning and exact keyword matches.
- **Reciprocal Rank Fusion:** Merges vector and keyword search rankings into a unified candidate list.
- **Cross-Encoder Reranking:** Reorders retrieved candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Grounded Response Generation:** Uses retrieved company documentation as context and instructs the LLM to acknowledge when information is unavailable.
- **Agentic Tool Calling:** Uses a LangChain/LangGraph agent to select tools based on the customer's request.
- **Customer Support Tools:** Supports knowledge-base search, order status lookup, return eligibility checks, and support ticket creation.
- **Conversational Memory:** Uses an in-memory checkpointer to maintain conversation state by thread ID.
- **Source Tracking and Logging:** Returns document source names for RAG queries and logs query responses and timing information.
- **Evaluation Pipeline:** Evaluates retrieval performance and generated-answer quality using a question dataset and an LLM-based judge.
- **Docker Support:** Includes a Dockerfile for containerizing the FastAPI application.

## Architecture

```mermaid
flowchart TD
    A[Customer Request] --> B{API Endpoint}
    B -->|POST /query| C[Hybrid RAG Pipeline]
    B -->|POST /chat| D[LangChain Agent]

    C --> E[Vector Search]
    C --> F[BM25 Keyword Search]

    E --> G[Reciprocal Rank Fusion]
    F --> G

    G --> H[Cross-Encoder Reranking]
    H --> I[Top Retrieved Context]
    I --> J[Gemini Response Generation]
    J --> K[Answer and Sources]

    D --> L{Select Tool}
    L --> M[Knowledge Base Search]
    L --> N[Order Status]
    L --> O[Return Eligibility]
    L --> P[Support Ticket Creation]

    M --> C
    D --> Q[Conversation Checkpointer]
```

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| API framework | FastAPI |
| Agent orchestration | LangChain and LangGraph |
| LLM provider | Google Gemini |
| Embeddings | `gemini-embedding-001` |
| Vector database | PostgreSQL with pgvector |
| Keyword retrieval | BM25 via `rank-bm25` |
| Reranking | Sentence Transformers CrossEncoder |
| Database driver | Psycopg |
| API validation | Pydantic |
| Environment configuration | python-dotenv |
| Containerization | Docker |
| Evaluation | Custom retrieval evaluation and LLM-as-judge |

## How the RAG Pipeline Works

SupportIQ uses a multi-stage retrieval pipeline to improve the relevance of the context supplied to the language model.

1. **Query embedding:** The incoming question is converted into a 768-dimensional embedding using Google's `gemini-embedding-001` model.
2. **Vector search:** PostgreSQL with pgvector retrieves semantically similar knowledge chunks using vector distance.
3. **Keyword search:** BM25 ranks chunks based on lexical overlap with the question.
4. **Rank fusion:** Reciprocal Rank Fusion combines both ranked result lists.
5. **Reranking:** A cross-encoder scores the query and candidate-text pairs and selects the top five chunks.
6. **Context-grounded generation:** Gemini generates an answer using the retrieved context, with instructions not to invent unsupported information.
7. **Source reporting:** The `/query` endpoint returns the generated answer alongside the source documents associated with the retrieved chunks.

This design combines semantic similarity with exact-term matching before passing the strongest candidates to the LLM.

## Agent Tools

The agent in `agent.py` is configured with four tools from `tools.py`.

| Tool | Purpose |
|---|---|
| `search_knowledge_base` | Retrieves relevant information about policies, FAQs, shipping, warranties, returns, and account issues. |
| `get_order_status` | Looks up an order by its order ID. |
| `check_return_eligibility` | Checks delivery status and applies a 14-day return window. |
| `create_support_ticket` | Generates a support ticket identifier for an unresolved customer issue. |

The agent uses Gemini and an in-memory conversation checkpointer. A `thread_id` lets callers associate successive requests with the same conversation.

**Implementation note:** Order records and ticket creation are currently simulated in Python rather than connected to a live order-management system or persistent ticketing platform.

## API Endpoints

The application exposes three endpoints.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Returns the API status. |
| `POST` | `/query` | Executes the standalone hybrid RAG pipeline and returns an answer with source names. |
| `POST` | `/chat` | Sends a message to the tool-using agent and returns its response and conversation thread ID. |

### 1. Health Check

```http
GET /
```

Example response:

```json
{
  "status": "SupportIQ API is running"
}
```

### 2. Knowledge-Base Query

```http
POST /query
Content-Type: application/json
```

Request:

```json
{
  "question": "What is the return policy?"
}
```

Response structure:

```json
{
  "question": "What is the return policy?",
  "answer": "The answer generated from the retrieved context.",
  "sources": ["returns_policy.pdf"]
}
```

The source name above is illustrative; actual values depend on the documents stored in the database.

### 3. Agentic Customer Support

```http
POST /chat
Content-Type: application/json
```

Request:

```json
{
  "message": "Can you check my order PK-39281?",
  "thread_id": "customer-session-001"
}
```

Response structure:

```json
{
  "message": "Can you check my order PK-39281?",
  "response": "The agent's response based on the selected tool.",
  "thread_id": "customer-session-001"
}
```

The example order ID corresponds to sample data currently defined in `tools.py`.

## Project Structure

```text
SUPPORTIQ-RAG/
├── policy_docs/          # Company policy documents
├── agent.py              # Agent configuration and conversation handling
├── main.py               # FastAPI application and API endpoints
├── search.py             # Vector search, BM25, RRF and reranking
├── embeddings.py         # Gemini embedding generation
├── tools.py              # Customer support agent tools
├── logger.py             # Query logging
├── db.py                 # Database-related module
├── add_policy_docs.py    # Policy document ingestion utility
├── migrate_data.py       # Data migration utility
├── eval.py               # RAG evaluation pipeline
├── eval_dataset.json     # Evaluation questions and expected sources
├── eval_results.json     # Saved evaluation results
├── debug_retrieval.py    # Retrieval debugging utility
├── test_agent.py         # Agent tests
├── test_db.py            # Database tests
├── test_embed.py         # Embedding tests
├── test_hybrid.py        # Hybrid retrieval tests
├── test_search.py        # Search tests
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker image configuration
├── .dockerignore
└── .gitignore
```

## Getting Started

### Prerequisites

Before running SupportIQ, prepare:

- Python 3.12 or a compatible environment.
- A PostgreSQL database with the pgvector extension.
- A Google Gemini API key.
- The project's knowledge-base data loaded into the expected database schema.
- Docker, if you prefer containerized execution.

### 1. Clone the Repository

```bash
git clone https://github.com/AREHMAN87-cloud/SUPPORTIQ-RAG.git
cd SUPPORTIQ-RAG
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
DATABASE_URL=postgresql://username:password@host:5432/database_name
```

Replace the example values with your actual credentials. Do not commit `.env` or expose API keys in source control.

### 5. Prepare the Database

Configure PostgreSQL with pgvector and ensure the `support_chunks` table contains the fields expected by `search.py`:

- `chunk_id`
- `content`
- `source`
- `question`
- `category`
- `embedding`

The embedding column must use a compatible pgvector type for the 768-dimensional embeddings generated by the application.

The retrieval module loads the corpus from this table during initialization. Consequently, the database and its expected schema must be available before starting the API.

Review `add_policy_docs.py` and `migrate_data.py` for the repository's data-ingestion and migration workflows.

### 6. Start the API

```bash
uvicorn main:app --reload
```

The API should be available at:

- Base URL: `http://127.0.0.1:8000`
- Interactive Swagger documentation: `http://127.0.0.1:8000/docs`

Use the Swagger interface or an HTTP client to test `/query` and `/chat`.

### 7. Run with Docker

Build the image:

```bash
docker build -t supportiq-rag .
```

Run the container with your configured credentials:

```bash
docker run --rm -p 8000:8000 \
  -e GEMINI_API_KEY="your_gemini_api_key" \
  -e DATABASE_URL="your_database_connection_string" \
  supportiq-rag
```

The PostgreSQL database must be accessible from the container. If the database is running on the host machine, configure the database hostname appropriately for your Docker environment.

## Evaluation

SupportIQ includes an evaluation workflow in `eval.py` that runs questions from `eval_dataset.json` through the retrieval and generation pipeline.

The evaluation tracks two metrics:

- **Retrieval accuracy:** Whether the expected source appears in the retrieved source set, including cases where no source is expected.
- **Generation quality:** Whether an LLM-based judge considers the generated answer acceptable under the evaluation prompt.

Results are printed to the console and saved to `eval_results.json`.

In the current implementation, generation quality is an LLM-judged pass rate rather than a comprehensive factuality or hallucination benchmark. Results depend on the evaluation dataset, judge model, and test conditions.

To run the evaluation:

```bash
python eval.py
```

The database and Gemini API credentials must be configured first.

## Current Limitations and Future Improvements

- **Simulated business integrations:** Replace the in-memory order dictionary and generated ticket IDs with real order-management and ticketing APIs.
- **Persistent conversational memory:** Replace the in-memory checkpointer with persistent storage for conversations across restarts.
- **Retrieval tuning:** Improve corpus coverage, chunking, metadata filtering, and retrieval thresholds based on evaluation results.
- **Expanded evaluation:** Add more test cases, source-grounded factuality checks, latency tracking, and regression testing.
- **Production observability:** Add structured logs, error monitoring, and request-level metrics.
- **Operational hardening:** Add authentication, rate limiting, stronger input validation, and appropriate deployment security controls.

## Learning Outcomes

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation (RAG).
- Hybrid dense and sparse information retrieval.
- Reciprocal Rank Fusion and cross-encoder reranking.
- LLM-powered agents and tool calling.
- FastAPI-based backend development.
- PostgreSQL and vector similarity search.
- Docker-based application packaging.
- Retrieval evaluation and LLM-as-judge workflows.

## Author

**Abdur Rehman**

GitHub: [@AREHMAN87-cloud](https://github.com/AREHMAN87-cloud)

---

If you find this project useful, consider starring the repository or opening an issue with suggestions for improvement.
