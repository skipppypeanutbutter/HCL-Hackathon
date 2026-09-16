"""
Thin wrapper around whichever LLM provider you end up using.

Keep exactly one function, call_llm(), as the single entry point.
Swap the implementation depending on which API key you have on the
day (Claude, OpenAI, Gemini) without touching main.py.
"""

import os


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call the LLM and return its text response.

    Default implementation uses Anthropic's API. Set ANTHROPIC_API_KEY
    in your environment before running. Swap the body of this function
    to use a different provider if needed, the signature stays the same.
    """
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


def call_llm_structured(system_prompt: str, user_prompt: str, schema_hint: str) -> str:
    """Ask the LLM to return JSON matching a described shape.

    schema_hint should be a short plain-text description of the fields
    you want back. Pair this with json.loads() and a Pydantic model's
    model_validate() in the route to get validated structured output.
    """
    strict_system = (
        f"{system_prompt}\n\n"
        f"Respond with only valid JSON, no markdown fences, no preamble. "
        f"Shape: {schema_hint}"
    )
    return call_llm(strict_system, user_prompt)
