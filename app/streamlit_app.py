"""
Streamlit frontend for the Wealth Advisor Assistant.

Replaces the Django `chatbot/chat.html` template + its JS fetch calls.
Talks to the FastAPI backend over plain HTTP - run both, in two terminals,
from the project root:

    uvicorn app.main:app --reload --port 8000
    streamlit run app/streamlit_app.py
"""
import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Wealth Advisor Assistant", page_icon="\U0001F3DB️")
st.title("Wealth Advisor Assistant")
st.caption(
    "Ask about a client's portfolio, a fund, or policy."
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def _render_answer_with_citations(answer: str, citations: list) -> None:
    """
    citations: the list the backend returns per QueryResponse.citations -
    dicts with tag/source/doc_type/used, e.g.
        {"tag": "suitability_policy.pdf#p2", "source": "suitability_policy.pdf",
         "doc_type": "policy", "used": True}

    The model's raw answer has inline tags like
    "...requires a signed acknowledgement [suitability_policy.pdf#p2]."
    Those are exact and reliable (rag/citations.py re-derives them from the
    real tool output, not just the model's formatting), but a raw filename
    in brackets mid-sentence isn't pleasant to read. This renders the
    answer with those swapped for small numbered footnote markers, then
    lists the numbered sources underneath - like a real research citation
    - while leaving the underlying data (tag/source/doc_type) fully
    visible for anyone who wants to check where a number leads.
    """
    used = [c for c in citations if c.get("used")]

    # Number sources in the order they first appear in the answer text,
    # not the order the tool happened to return them in.
    order = sorted(used, key=lambda c: answer.find(f"[{c['tag']}]"))
    tag_to_number = {c["tag"]: i + 1 for i, c in enumerate(order)}

    display_text = answer
    for tag, number in tag_to_number.items():
        display_text = display_text.replace(f"[{tag}]", f"`[{number}]`")

    st.markdown(display_text)

    if order:
        st.markdown("**Sources**")
        for c in order:
            n = tag_to_number[c["tag"]]
            st.caption(f"[{n}] {c['source']} — {c['doc_type']} ({c['tag']})")

    unused = [c for c in citations if not c.get("used")]
    if used or unused:
        with st.expander("Retrieval / tool trace"):
            if used:
                st.markdown("**Cited in the answer above:**")
                for c in order:
                    st.write(f"[{tag_to_number[c['tag']]}] {c['tag']} - {c['source']} ({c['doc_type']})")
            if unused:
                st.markdown("**Retrieved but not cited** (the model saw these and chose not to use them):")
                for c in unused:
                    st.write(f"{c['tag']} - {c['source']} ({c['doc_type']})")


def _render_message(msg: dict) -> None:
    citations = msg.get("citations")
    if citations:
        _render_answer_with_citations(msg["content"], citations)
    else:
        st.markdown(msg["content"])
    # Raw tool trace, kept separate from the citations expander above so
    # SQL-only turns (no document citations at all) still show what ran.
    if msg.get("steps"):
        with st.expander("Raw tool calls"):
            for step in msg["steps"]:
                st.code(step, language=None)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _render_message(msg)
        else:
            st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/api/query",
                    json={"text": prompt, "session_id": "streamlit"},
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                answer = data.get("output", "No response generated")
                steps = data.get("intermediate_steps", [])
                citations = data.get("citations", [])
            except Exception as e:
                answer = f"Error reaching the backend: {e}"
                steps = []
                citations = []

        new_msg = {"role": "assistant", "content": answer, "steps": steps, "citations": citations}
        _render_message(new_msg)

    st.session_state.messages.append(new_msg)