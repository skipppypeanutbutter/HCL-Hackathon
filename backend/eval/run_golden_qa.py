"""
Run the agent against golden_qa_dataset.json and report:
- retrieval: did the tool calls touch the expected source documents
- generation: print the agent's answer next to the expected answer
  for manual comparison (correctness grading of free-text answers
  isn't something to fully automate on hackathon day, a human glance
  is faster and more trustworthy than a shaky auto-grader)

TODO confirm the golden_qa_dataset.json key names below match the
real file (question / expected_answer / required_source_documents).

Run with: python -m eval.run_golden_qa
"""

import json
from pathlib import Path

from agent.agent import ask

DATA_DIR = Path("data")


def source_names_from_tool_calls(tool_calls: list[dict]) -> set[str]:
    """Pull out which documents/tables actually got touched, from the
    vector_search results' titles/source_path and any table names
    mentioned in sql_query queries.
    """
    touched = set()
    for call in tool_calls:
        if call["tool"] == "vector_search":
            for row in call["result"]:
                if isinstance(row, dict) and "title" in row:
                    touched.add(row["title"])
        elif call["tool"] == "sql_query":
            query_lower = call["input"]["query"].lower()
            for table in ("clients", "portfolio_holdings", "transactions"):
                if table in query_lower:
                    touched.add(table)
    return touched


def run_eval():
    with open(DATA_DIR / "golden_qa_dataset.json") as f:
        golden_qa = json.load(f)

    results = []
    for item in golden_qa:
        question = item["question"]
        expected_answer = item.get("expected_answer", item.get("answer"))
        required_sources = set(item.get("required_source_documents", []))

        print(f"\n{'=' * 80}\nQ: {question}")
        outcome = ask(question)
        touched = source_names_from_tool_calls(outcome["tool_calls"])

        overlap = required_sources & touched
        missing = required_sources - touched

        print(f"AGENT ANSWER:\n{outcome['answer']}")
        print(f"\nEXPECTED ANSWER:\n{expected_answer}")
        print(f"\nSOURCES TOUCHED: {touched}")
        print(f"REQUIRED SOURCES: {required_sources}")
        print(f"MISSED SOURCES: {missing or 'none'}")

        results.append({
            "question": question,
            "answer": outcome["answer"],
            "expected_answer": expected_answer,
            "sources_touched": list(touched),
            "sources_missed": list(missing),
            "retrieval_recall": len(overlap) / len(required_sources) if required_sources else None,
        })

    with open("eval/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nSaved {len(results)} results to eval/results.json")


if __name__ == "__main__":
    run_eval()
