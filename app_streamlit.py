import os
import time
from pathlib import Path
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv

from rag_store import RAGStore
from llm import chat_completion
from tts import synthesize_speech

st.set_page_config(page_title="Workshop 3 Chatbot", page_icon="💬", layout="centered")
load_dotenv()

@st.cache_resource(show_spinner=False)
def get_store():
    store = RAGStore(persist_dir="chroma_db")
    kb_path = os.getenv("KB_PATH", "data/mock_kb.json")
    # Initialize once
    if not Path("chroma_db").exists() or not any(Path("chroma_db").iterdir()):
        items = store.load_from_json(kb_path)
        store.upsert_items(items)
    return store

SYSTEM_PROMPT = (
    "You are a helpful workshop chatbot. Use the provided CONTEXT to answer. "
    "If the answer is not in the context, say you will answer from general knowledge, "
    "but prefer the context first. Keep answers concise and clear."
)

def build_prompt(context_chunks: List[str], user_msg: str) -> List[Dict[str, str]]:
    """
    Build prompt with system + context so that GPT MUST use KB data if available.
    """
    if context_chunks:
        context_text = "\n\n".join(context_chunks)
        context_message = (
            "You are a workshop chatbot. Below is the knowledge base (KB) context retrieved "
            "from internal data. Always use this information to answer the user's question "
            "if it is relevant. If the answer cannot be found in the KB, then you may answer "
            "from general knowledge.\n\n"
            f"=== KNOWLEDGE CONTEXT START ===\n{context_text}\n=== CONTEXT END ==="
        )
    else:
        context_message = (
            "No relevant internal KB context was found. Answer from general knowledge."
        )

    return [
        {"role": "system", "content": context_message},
        {"role": "user", "content": user_msg},
    ]


store = get_store()

st.title("💬 Workshop 3 — RAG Chatbot with TTS")
st.caption("ChromaDB + OpenAI + HuggingFace (Coqui) VITS")

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, audio?}]

with st.sidebar:
    st.header("⚙️ Settings")
    openai_model = st.text_input("OpenAI model", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    k = st.slider("Top-K retrieval", 1, 8, 4)
    st.markdown("**How to run**")
    st.code("pip install -r requirements.txt\ncp .env.example .env\n# edit .env\nstreamlit run app_streamlit.py")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("audio"):
            st.audio(m["audio"])

user_msg = st.chat_input("Ask me anything about the workshop or your docs...")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            results = store.similarity_search(user_msg, k=k)
            # context_chunks = [doc for (_id, doc, _meta) in results]
            context_chunks = [r["document"] for r in results]
            messages = build_prompt(context_chunks, user_msg)
            answer = chat_completion(messages, model=openai_model)
            audio_path = synthesize_speech(answer)
            st.markdown(answer)
            st.audio(audio_path)

    st.session_state.messages.append({"role": "assistant", "content": answer, "audio": audio_path})
