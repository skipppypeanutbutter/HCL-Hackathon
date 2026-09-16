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
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from rag.tools import TOOLS

load_dotenv()

AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are a wealth management assistant for relationship managers (RMs).

You have three tools:
1. retrieve_documents - semantic search over fund fact sheets, policy documents, call notes, correspondence
2. lookup_client_portfolio - exact lookup of a client's holdings and risk profile by client_id
3. lookup_client_transactions - exact lookup of a client's recent transactions by client_id

RULES (follow these exactly):
- Only answer using information returned by your tools. Never use outside knowledge about
  specific funds, clients, or policies.
- Always cite your source for every factual claim: name the document (for retrieve_documents)
  or say "from the client record" (for the lookup tools).
- If a question is about a specific client, call lookup_client_portfolio and/or
  lookup_client_transactions FIRST - do not try to answer client-specific questions from
  retrieve_documents alone.
- If none of your tools return relevant information (a tool returns "NO_RESULTS", or what it
  returned doesn't actually address the question), you MUST respond exactly:
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
# and what satisfies the traceability bonus item almost for free.
agent_executor = AgentExecutor(
    agent=_agent,
    tools=TOOLS,
    return_intermediate_steps=True,
    verbose=True,
)

print("Wealth Advisor RAG agent initialized.")