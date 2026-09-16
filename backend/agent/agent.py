"""
Multi-step tool-calling agent.

The model gets both tools and decides how many times to call them
before answering. This is what the multi-hop golden QA questions need,
e.g. "walk me through Carlos Bautista's DCI trade" requires call notes,
a compliance email, the transaction ledger, and the client record,
none of which alone answers the question.

Run with: python -m agent.agent "your question here"
"""

import sys
import json
from anthropic import Anthropic

from agent.tools import sql_query, vector_search, TOOL_DEFINITIONS

client = Anthropic()

SYSTEM_PROMPT = """You are an assistant for a private banking compliance and \
advisory team. You answer questions using the sql_query and vector_search \
tools. Call them as many times as needed, including multiple rounds, before \
answering.

Rules:
- Never state a fact you have not retrieved through a tool call in this \
conversation. If something is missing (e.g. no risk acknowledgement form \
found for a client), say so explicitly rather than assuming it exists.
- For questions involving a specific client, look up their client_id first \
if you don't already have it, then use it to filter vector_search.
- For questions about concentration limits, guideline thresholds, or policy \
definitions, retrieve the actual policy text with vector_search rather than \
assuming what the guideline says.
- When you have enough evidence, give a direct answer and list which \
documents or records it's based on.
"""

MAX_TOOL_ROUNDS = 6


def run_tool(name: str, tool_input: dict):
    if name == "sql_query":
        return sql_query(tool_input["query"])
    if name == "vector_search":
        return vector_search(tool_input["question"], client_id=tool_input.get("client_id"))
    raise ValueError(f"Unknown tool: {name}")


def ask(question: str) -> dict:
    """Returns {"answer": str, "tool_calls": [...]} so the eval script
    can inspect what was retrieved, not just the final text.
    """
    messages = [{"role": "user", "content": question}]
    tool_call_log = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            final_text = "".join(block.text for block in response.content if block.type == "text")
            return {"answer": final_text, "tool_calls": tool_call_log}

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = run_tool(block.name, block.input)
            tool_call_log.append({"tool": block.name, "input": block.input, "result": result})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, default=str),
            })

        messages.append({"role": "user", "content": tool_results})

    return {"answer": "Ran out of tool-call rounds without a final answer.", "tool_calls": tool_call_log}


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Which clients hold Complex Products above the 20% concentration guideline?"
    result = ask(question)
    print("ANSWER:\n", result["answer"])
    print("\nTOOL CALLS:")
    for call in result["tool_calls"]:
        print(f"  {call['tool']}({call['input']})")
