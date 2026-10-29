import sys
from pathlib import Path
from typing import Any,Dict,List
import streamlit as st

# Streamlit executes this file as a script, so expose the project-level backend.
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.core import run_llm


def _format_sources(context_docs: List[Any]) -> List[str]:
    return [
        str((meta.get("source") or "unknown"))
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]


st.set_page_config(page_title="LangChain Documentation Helper", layout="centered")
st.title("Langchain Documentation Helper")

with st.sidebar:
     st.subheader("Session")
     if st.button("Clear chat", use_container_width=True):
         st.session_state.pop("messages", None) 
         st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about Langchain docs. I'll retrieve context and cite sources.",
            "sources": [],
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

prompt = st.chat_input("Ask a question about Langchain...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving docs and generating answer..."):
                result:Dict[str,Any] = run_llm(prompt)
                answer = str(result.get("answer", "")).strip() or "(No answer returned.)"
                sources = _format_sources(result.get("context", []))
                st.markdown(answer)
                if sources:
                    with st.expander("sources"):
                        for s in sources:
                            st.markdown(f"- {s}")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )

        except Exception as e:
            st.error("Failed to generate a response.")
            st.exception(e)
