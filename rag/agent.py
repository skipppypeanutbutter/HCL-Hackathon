"""
Tool-calling RAG agent for the Wealth Advisor Assistant.

Equivalent to your old agents.py, but using `create_tool_calling_agent`
(the modern, model-agnostic replacement for `create_openai_functions_agent`)
and three tools instead of the career-graph/course/personalized set.

One agent, three tools - see tools.py. The grounding + abstention rule
lives HERE, in the system prompt, instead of a hardcoded decision diamond:
the model is told to answer only from tool output and to say so plainly
when nothing relevant came back. This is what satisfies the mandatory
"grounded response + explicit abstention" requirement in an agentic setup.
"""
import os
from dotenv import load_dotenv

# MUST run before any project import (rag.tools -> retrieval.vectorstore ->
# rag.db reads DATABASE_URL/SUPABASE_DB_URL from os.environ at import
# time). Calling load_dotenv() after `from rag.tools import TOOLS` means
# .env hasn't been loaded yet when rag/db.py goes looking for it.
load_dotenv()

from langchain_classic.agents import (
    AgentExecutor,
    create_tool_calling_agent,
)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from rag.tools import TOOLS


AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are a wealth management assistant for relationship managers (RMs).

You have two tools:
1. retrieve_documents - semantic search over fund fact sheets, policy documents, call notes,
   and correspondence. Use for anything where the answer is prose you need to find by meaning.
2. query_client_database - write and run a SQL SELECT against the clients/portfolio_holdings/
   transactions tables in Supabase. Use for anything about a client's data: risk profile, net worth, holdings,
   transactions, filters, sorting, counts, aggregates, or comparisons across clients. The
   tool's own description has the full schema and worked examples - write the SQL yourself
   for whatever is actually asked. Do not wait for the question to use a specific keyword
   like "high" or "low" before you think to use this tool - any question about client data
   goes through SQL, not guesswork.

RULES (follow these exactly):
- Only answer using information returned by your tools. Never use outside knowledge about
  specific funds, clients, or policies.
- Never invent or guess a client_id, a filter value, or a fact about a client. If you need an
  exact ID or spelling you don't have, query for it (e.g. SELECT client_id, name FROM clients)
  rather than assuming.
- When a query asks for "the" single highest/lowest/largest/smallest something, write the SQL
  with ORDER BY ... LIMIT 1 (or GROUP BY + aggregate) so the database computes the answer -
  never fetch a list of rows and try to eyeball or re-rank the extreme yourself. That kind of
  manual comparison across many rows is exactly where mistakes happen.
- CITATIONS: every passage retrieve_documents returns starts with a bracketed tag, like
  [suitability_policy.pdf#p2]. When a sentence in your answer draws on that passage, put that
  EXACT tag immediately after the sentence, e.g.:
    "Exotic structured products require a signed risk acknowledgement [suitability_policy.pdf#p2]."
  Copy the tag character-for-character from the tool output - never renumber it, shorten it, or
  invent one. If a sentence draws on more than one passage, put all the tags after it, e.g.
  "...[fact_sheet.pdf#p1][suitability_policy.pdf#p2]". For query_client_database results, cite
  as "from the client database" instead (there's no page/chunk tag for a SQL row).
- If none of your tools return relevant information (a tool returns "NO_RESULTS", "ERROR", or
  what it returned doesn't actually address the question), you MUST respond exactly:
  "I don't have enough information in the available documents to answer that."
  Do not guess, and do not soften this into a partial answer.
- Keep answers concise and RM-facing - they need a quick, defensible answer, not an essay.
"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

_llm = ChatOpenAI(model=AGENT_MODEL, temperature=0)

_agent = create_tool_calling_agent(llm=_llm, tools=TOOLS, prompt=_prompt)

# return_intermediate_steps=True gives you the tool calls + outputs, which
# is what you show in the "retrieval / tool trace" expander in Streamlit
# and what rag/citations.py uses to build a reliable structured citations
# list independent of whether the model's inline [tag] citations above are
# perfectly formatted - the intermediate steps are ground truth for what
# was actually retrieved.
agent_executor = AgentExecutor(
    agent=_agent,
    tools=TOOLS,
    return_intermediate_steps=True,
    verbose=True,
)

print("Wealth Advisor RAG agent initialized.")