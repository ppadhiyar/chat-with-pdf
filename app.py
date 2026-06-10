import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from rag_chain import ask, build_qa_chain

load_dotenv()

st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="centered")
st.title("📄 Chat with Your PDF")
st.caption("Upload a PDF and ask questions about it — powered by LangChain + GPT-4o mini")

if "chain" not in st.session_state:
    st.session_state.chain = None
if "page_count" not in st.session_state:
    st.session_state.page_count = 0
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file and st.button("Process PDF", type="primary"):
        with st.spinner("Reading and indexing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            chain, page_count = build_qa_chain(tmp_path)
            st.session_state.chain = chain
            st.session_state.page_count = page_count
            st.session_state.messages = []
            os.unlink(tmp_path)
        st.success(f"Indexed {page_count} page(s). Start chatting!")

    if st.session_state.chain:
        st.divider()
        st.metric("Pages indexed", st.session_state.page_count)
        if st.button("Clear chat"):
            st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("pages"):
            st.caption(f"Sources: page(s) {', '.join(map(str, msg['pages']))}")

if st.session_state.chain is None:
    st.info("Upload and process a PDF using the sidebar to get started.")
else:
    if question := st.chat_input("Ask a question about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = ask(st.session_state.chain, question)
                    answer = result["answer"]
                    pages = result["source_pages"]
                except Exception as e:
                    err = str(e)
                    if "429" in err or "ResourceExhausted" in err or "quota" in err.lower():
                        answer = "⚠️ Gemini API rate limit reached. The free tier resets daily at midnight Pacific time. Try again later or add billing at https://ai.google.dev"
                    else:
                        answer = f"⚠️ Error: {err}"
                    pages = []
            st.markdown(answer)
            if pages:
                st.caption(f"Sources: page(s) {', '.join(map(str, pages))}")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "pages": pages,
            }
        )
