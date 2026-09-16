# HCL-Hackathon — Wealth Advisor Assistant

A grounded RAG assistant for private banking relationship managers. A
LangChain tool-calling agent answers questions about clients, portfolios,
and policy by querying structured client data in Supabase Postgres and
doing semantic search over unstructured documents (fund fact sheets,
policies, call notes, correspondence) via pgvector. Served through a
FastAPI backend with a Streamlit chat frontend.

If a question isn't answerable from what the tools return, the agent is
instructed to say so explicitly rather than guess — see the system prompt
in `rag/agent.py`.

## Architecture

```
app/
├── main.py                FastAPI backend — /api/query, /api/health
└── streamlit_app.py        Streamlit chat UI, calls the FastAPI backend over HTTP

rag/
├── agent.py                LangChain tool-calling agent + system prompt
├── tools.py                 The two tools: retrieve_documents, query_client_database
├── sql_store.py             Read-only SQL execution + schema introspection
├── db.py                    Supabase Postgres connection pool
└── check_chunks.py         One-off script to sanity-check ingested documents/chunks

retrieval/
└── vectorstore.py          Embeds queries, does pgvector similarity search over document_chunks

data/
├── clients_portfolio.json   Source client + portfolio data
└── transactions.csv         Source transaction data

backend/                    Earlier implementation (Anthropic-based agent). Not
                             imported by app/ — kept for reference only, see
                             "Known issues" below.
```

**Request flow:** Streamlit → `POST /api/query` on FastAPI → LangChain
`AgentExecutor` (`rag/agent.py`) → the model decides between
`retrieve_documents` (pgvector semantic search, `retrieval/vectorstore.py`)
and `query_client_database` (agent-written SQL, `rag/sql_store.py`) → answer
+ intermediate steps returned to the frontend, shown in a "retrieval / tool
trace" expander.

## Evaluation

`backend/eval/run_golden_qa.py` is set up to evaluate the agent against a
golden question-answer set: for each question it runs the agent, checks
whether the tool calls actually touched the expected source
documents/tables (retrieval recall), and prints the agent's answer next to
the expected answer for a human to compare (free-text correctness isn't
auto-graded — a quick eyeball is faster and more trustworthy than a shaky
grader on hackathon timelines).
Run it with:
```bash
python -m eval.run_golden_qa
```
Results (per-question answers, sources touched vs. required, retrieval
recall) get written to `eval/results.json`.

## Prerequisites

- Python 3.10+
- A Supabase project with the `vector` extension enabled, and the
  `clients`, `portfolio_holdings`, `transactions`, `documents`, and
  `document_chunks` tables created (`backend/db/schema.sql` has the DDL —
  run it in the Supabase SQL editor)
- An OpenAI API key (the agent uses `ChatOpenAI`, default model
  `gpt-4o-mini`)

## Environment variables

Create a `.env` file in the **project root** (not inside `app/` or `rag/`):

```
SUPABASE_DB_URL=postgresql://...   # Supabase "Session pooler" URI, not the direct/IPv6 one
OPENAI_API_KEY=sk-...

# optional, all have defaults:
AGENT_MODEL=gpt-4o-mini
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RETRIEVAL_TOP_K=5
BACKEND_URL=http://localhost:8000   # used by streamlit_app.py to find the API
```

> Use the Supabase **Session pooler** connection string specifically —
> `app/main.py`'s health check comments note that the direct/IPv6-only
> connection string is a common cause of connection failures.

## Setup

```bash
git clone https://github.com/skipppypeanutbutter/HCL-Hackathon.git
cd HCL-Hackathon
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Everything runs from the **project root** — `rag` and `retrieval` are
imported as top-level packages, so running from inside `app/` or `rag/`
will break the imports.

## Database setup

1. In the Supabase SQL editor, enable the `vector` extension and run
   `backend/db/schema.sql` to create the tables and the
   `match_document_chunks()` function (still the correct schema for this
   app, even though the ingestion code that originally shipped with it now
   lives under the legacy `backend/` folder).
2. Load the structured data (clients, holdings, transactions) and the
   embedded documents into Supabase. The ingestion scripts for this are
   under `backend/ingestion/` — see "Known issues" below, their default
   `data/` path needs a one-line fix to point at the root `data/` folder
   before running them from this layout.
3. Sanity-check what actually landed in Supabase:
   ```bash
   python rag/check_chunks.py
   ```

## Running locally

Two processes, two terminals, both from the project root:

```bash
# Terminal 1 — API
uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend
streamlit run app/streamlit_app.py
```

- Frontend: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/health` — confirms both the
  vector store and the SQL store are reachable, and reports chunk/client
  counts

## Deploying

This is two independently deployable services (FastAPI + Streamlit)
talking over HTTP, plus a managed Supabase database — nothing here needs a
persistent local filesystem, so any container host works.

**Backend (FastAPI)**
- Any container platform works. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
- Set `SUPABASE_DB_URL` and `OPENAI_API_KEY` as environment variables/secrets
  on the host — never commit `.env`.
- Update the CORS `allow_origins` in `app/main.py` (currently hardcoded to
  `http://localhost:8501`) to your deployed frontend's real URL.

**Frontend (Streamlit)**
- Streamlit Community Cloud is the fastest path — point it at
  `app/streamlit_app.py`.
- Set `BACKEND_URL` to your deployed FastAPI URL (as a secret/env var, not
  hardcoded).

**Database**
- Already hosted (Supabase) — nothing to deploy, just make sure the
  deployed backend's IP/network can reach it and that ingestion has been
  run against the same project you're pointing production at.

A single Dockerfile running both processes (e.g. via `supervisord` or a
shell script that backgrounds uvicorn and foregrounds streamlit) also
works if you'd rather have one deployable unit — not set up in this repo
yet.


## Tests

```bash
pytest backend/tests/ -v
```

(No test coverage yet for `rag/`/`retrieval/` — worth adding if there's
time, at minimum a repeat of the embedding-only tests in
`backend/tests/test_semantic_retrieval.py` pointed at `retrieval/vectorstore.py`.)

## Team / Hackathon

Built for the HCLTech Hackathon (banking domain — Private Banking / Wealth
Management use cases).
