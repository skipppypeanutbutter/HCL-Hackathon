"""
Streamlit frontend for the Wealth Advisor Assistant.

Replaces the Django `chatbot/chat.html` template + its JS fetch calls.
Talks to the FastAPI backend over plain HTTP - run both, in two terminals,
from the project root:

    uvicorn app.backend.main:app --reload --port 8000
    streamlit run app/frontend/streamlit_app.py
"""
import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Wealth Advisor Assistant", page_icon="\U0001F3DB️")
st.title("Wealth Advisor Assistant")
st.caption(
    "Ask about a client's portfolio, a fund, or policy. Answers are grounded in the "
    "supplied documents - if there isn't enough evidence, it will say so instead of guessing."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("steps"):
            with st.expander("Retrieval / tool trace"):
                for step in msg["steps"]:
                    st.code(step, language=None)

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
            except Exception as e:
                answer = f"Error reaching the backend: {e}"
                steps = []

        st.markdown(answer)
        if steps:
            with st.expander("Retrieval / tool trace"):
                for step in steps:
                    st.code(step, language=None)

    st.session_state.messages.append({"role": "assistant", "content": answer, "steps": steps})