import os
import json
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def load_vectorstore(path="data/mock_kb.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)["docs"]

    texts, metadatas = [], []
    for d in data:
        text = f"Q: {d['question']}\nA: {d['answer']}"
        texts.append(text)
        metadatas.append({"id": d["id"], "tags": d.get("tags", []), "updated_at": d.get("updated_at", "")})

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY_EMBED"),
        openai_api_base=os.getenv("OPENAI_BASE_URL")
    )

    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)
