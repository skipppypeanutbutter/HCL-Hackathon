"""
Extracts structured citations from an agent run's intermediate_steps.

Why this exists: the model is instructed (see agent.py's SYSTEM_PROMPT) to
copy the bracketed citation tag - like [fact_sheet.pdf#p3] - out of each
retrieve_documents passage and into its final answer. That's usually
reliable with gpt-4o-mini, but "usually" isn't good enough for a
mandatory citations requirement - a model can drop or mangle a tag,
especially under a cheaper/smaller model or a long multi-tool turn.

This module is the belt-and-suspenders half: it re-derives the citation
list directly from what retrieve_documents actually returned (ground
truth - the real source/doc_type of every chunk that was retrieved),
not from the model's transcription of it, and flags which of those tags
the model's final answer actually used. Put BOTH the model's inline-cited
prose AND this structured list in the API response, so the frontend
always has something reliable to render as a "Sources" list even if the
prose citation is imperfect.
"""
import re
from typing import Any, Dict, List

# Matches the "[tag] (source: X, type: Y)" header tools.py's
# retrieve_documents puts before every passage it returns.
_CITATION_BLOCK_RE = re.compile(
    r"\[(?P<tag>[^\[\]]+?)\]\s*\(source:\s*(?P<source>[^,]+),\s*type:\s*(?P<doc_type>[^)]+)\)"
)


def extract_citations(intermediate_steps: List[Any], final_answer: str) -> List[Dict[str, Any]]:
    """
    intermediate_steps: the list of (AgentAction, observation) tuples
        AgentExecutor returns when return_intermediate_steps=True.
    final_answer: result["output"] - the model's final answer text.

    Returns a deduped list of
        {"tag": ..., "source": ..., "doc_type": ..., "used": bool}
    in first-retrieved order. "used" is True if that exact "[tag]" string
    shows up in final_answer, so the frontend can show "cited" passages
    differently from "retrieved but not used" ones.
    """
    seen: Dict[str, Dict[str, Any]] = {}
    for step in intermediate_steps:
        try:
            action, observation = step
        except (TypeError, ValueError):
            continue

        if getattr(action, "tool", None) != "retrieve_documents":
            continue
        if not isinstance(observation, str):
            continue

        for match in _CITATION_BLOCK_RE.finditer(observation):
            tag = match.group("tag").strip()
            if tag in seen:
                continue
            seen[tag] = {
                "tag": tag,
                "source": match.group("source").strip(),
                "doc_type": match.group("doc_type").strip(),
                "used": f"[{tag}]" in final_answer,
            }
    return list(seen.values())