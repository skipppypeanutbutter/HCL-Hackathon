"""
Tools available to the RAG agent.

Two tools, two fundamentally different data shapes - this is the whole
"vector DB vs. structured query" split:

- retrieve_documents: semantic search over UNSTRUCTURED text (fund fact
  sheets, policy, call notes, correspondence). Use when the answer is
  somewhere in prose and you're looking for the passage that's most
  *similar* to the question.

- query_client_database: a general SQL query tool over STRUCTURED data
  (clients, their holdings, transactions). Use when the answer is an
  exact fact, filter, aggregate, or comparison over known fields - not
  a similarity match. Instead of hardcoding one Python function per
  question shape ("high risk", "low risk", "HNW clients", "who holds
  fund X"...), the LLM writes the SQL itself against a schema it's
  given, the same way your old chains.py had the LLM write Cypher
  against the Neo4j schema instead of hand-coding a function per
  question type.
"""
from langchain_core.tools import tool

from retrieval.vectorstore import get_retriever
from rag.sql_store import run_readonly_sql, get_schema_description


@tool
def retrieve_documents(query: str) -> str:
    """
    Search fund fact sheets, policy documents, RM call notes, and client
    correspondence for passages relevant to the query. Returns the top
    matching passages with their source filenames so the answer can cite
    them. Use this for product, policy, or general (non-client-specific)
    questions - anything where the answer is prose, not a structured field.
    """
    retriever = get_retriever(k=5)
    results = retriever.invoke(query)

    print(f"RETRIEVAL QUERY: {query!r}")
    print(f"RETRIEVAL COUNT: {len(results)}")

    for document in results:
        print(
            "SOURCE:",
            document.metadata.get("source"),
            "DISTANCE:",
            document.metadata.get("distance"),
        )
    if not results:
        return "NO_RESULTS: nothing relevant was found in the document store."

    # Citation tag is tied to the actual chunk (source + chunk_index), NOT
    # a per-call counter like [1], [2]. The agent calls retrieve_documents
    # more than once per turn (it did twice in your last log), and a
    # counter that resets to [1] every call means two DIFFERENT passages
    # both end up labeled "[1]" - the model then has no way to cite them
    # unambiguously, and rag/citations.py's extraction would silently
    # collide/overwrite one with the other. A tag built from the chunk's
    # own identity is stable and unique no matter how many times or in
    # what order the tool gets called.
    formatted = []
    for doc in results:
        source = doc.metadata.get("source", "unknown")
        doc_type = doc.metadata.get("doc_type", "")
        chunk_index = doc.metadata.get("chunk_index")
        tag = f"{source}#p{chunk_index}" if chunk_index is not None else source
        formatted.append(f"[{tag}] (source: {source}, type: {doc_type})\n{doc.page_content}")
    return "\n\n".join(formatted)


_SCHEMA = get_schema_description()

# NOTE: an f-string is NOT captured as a function's __doc__ by Python (only
# a literal string constant is) - so the schema has to be spliced in and
# assigned to __doc__ explicitly, then wrapped with tool() as a plain call
# rather than `@tool` decorator syntax, so LangChain picks up the real
# description instead of an empty one.
_QUERY_CLIENT_DB_DOC = """
Run a read-only SQL SELECT query against the structured client data in
Supabase Postgres. Use this for ANY question about clients, their
holdings, or their transactions that isn't a document-search question -
risk profiles, net worth, nationality, specific fund holdings,
transaction history, filters, sorting, counts, aggregates, comparisons
across clients, joins, etc. Only SELECT is allowed.

clients.raw_profile is a JSONB column holding the rest of each client's
original profile (things like aum_sgd, risk_score_1_to_10, occupation,
notes, suitability_flag) that don't have their own column - pull those
out with the ->> operator and cast numeric ones, e.g.
(raw_profile->>'aum_sgd')::numeric. The schema below lists which keys
were actually seen inside it - use those, don't guess a key name.

Schema:
{schema}

Examples of questions -> SQL:
  "which client has the lowest risk score" ->
    SELECT client_id, name, (raw_profile->>'risk_score_1_to_10')::int AS risk_score
    FROM clients ORDER BY risk_score ASC LIMIT 1
  "who is client CL001" ->
    SELECT * FROM clients WHERE client_id = 'CL001'
  "which clients hold the APEX autocallable note" ->
    SELECT DISTINCT c.client_id, c.name FROM clients c
    JOIN portfolio_holdings h ON c.client_id = h.client_id
    WHERE h.product_name ILIKE '%Autocallable%'
  "clients with AUM over 2 million SGD" ->
    SELECT client_id, name, (raw_profile->>'aum_sgd')::numeric AS aum_sgd
    FROM clients WHERE (raw_profile->>'aum_sgd')::numeric > 2000000
    ORDER BY aum_sgd DESC
  "total holding value by risk profile" ->
    SELECT c.risk_profile, SUM(h.holding_value) AS total_value
    FROM clients c JOIN portfolio_holdings h ON c.client_id = h.client_id
    GROUP BY c.risk_profile
  "which clients have a suitability issue flagged" ->
    SELECT client_id, name, raw_profile->>'suitability_flag' AS suitability_flag
    FROM clients
    WHERE raw_profile->>'suitability_flag' LIKE 'POTENTIAL MISMATCH%'
       OR raw_profile->>'suitability_flag' LIKE 'SUITABILITY EXCEPTION%'
       OR raw_profile->>'suitability_flag' LIKE 'REVIEW RECOMMENDED%'
    -- NOTE: match on the FLAG PREFIX (the all-caps phrase at the start
    -- of the value), never a bare '%mismatch%' - that also matches the
    -- string "No mismatch identified". Also note Postgres's plain LIKE
    -- is case-SENSITIVE (unlike SQLite) - use ILIKE when case shouldn't
    -- matter, e.g. free-text search over product_name or notes.
  "recent settled transactions for CL001" ->
    SELECT txn_date, txn_type, product_name, amount, currency
    FROM transactions WHERE client_id = 'CL001' AND status = 'Settled'
    ORDER BY txn_date DESC

Write the SQL yourself based on what's actually asked - do not guess a
client_id, and do not try to answer from memory. If the query you'd
need isn't answerable from this schema, say so instead of guessing.
""".format(schema=_SCHEMA)


def _query_client_database(sql: str) -> str:
    return run_readonly_sql(sql)


_query_client_database.__doc__ = _QUERY_CLIENT_DB_DOC
_query_client_database.__name__ = "query_client_database"
query_client_database = tool(_query_client_database)

TOOLS = [retrieve_documents, query_client_database]