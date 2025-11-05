import os
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions
import numpy as np


DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


# ==========================
# Data structure
# ==========================
@dataclass
class KBItem:
    id: str
    text: str
    metadata: Dict[str, Any]


# ==========================
# RAGStore (fixed for Chroma v0.5+)
# ==========================
class RAGStore:
    def __init__(self, persist_dir: str = "chroma_db"):
        """
        Initialize Chroma persistent client (new API for >=0.5)
        """
        load_dotenv()
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        # ✅ NEW Chroma client style (no Settings() anymore)
        from chromadb import PersistentClient
        self.client = PersistentClient(path=self.persist_dir)

        self.embedder = CustomEmbeddingWrapper()

        # ✅ collection auto-created if not exists
        self.collection = self.client.get_or_create_collection(
            name="workshop_kb",
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"},
        )

    # ==========================
    # Load from JSON
    # ==========================
    def load_from_json(self, path: str) -> List[KBItem]:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        docs = payload.get("docs", [])
        items: List[KBItem] = []

        for d in docs:
            q = d.get("question", "").strip()
            a = d.get("answer", "").strip()
            if not q or not a:
                print(f"⚠️ Skipped doc without question/answer: {d.get('id')}")
                continue

            text = f"Q: {q}\nA: {a}"

            tags = d.get("tags", [])
            tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)

            items.append(
                KBItem(
                    id=str(d.get("id", f"doc-{len(items)+1}")),
                    text=text,
                    metadata={
                        "tags": tags_str,
                        "updated_at": d.get("updated_at", ""),
                        "document_name": d.get("question", "")[:50],
                    },
                )
            )

        print(f"✅ Loaded {len(items)} FAQ records from {path}")
        return items


    # ==========================
    # Upsert into Chroma
    # ==========================
    def upsert_items(self, items: List[KBItem]) -> None:
        if not items:
            return
        self.collection.upsert(
            ids=[it.id for it in items],
            documents=[it.text for it in items],
            metadatas=[it.metadata for it in items],
        )

    # ==========================
    # Similarity Search with Logging
    # ==========================
    def similarity_search(self, query: str, k: int = 4):
        res = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        for idx, (d, m, dist) in enumerate(zip(docs, metas, dists), start=1):
            similarity = round(1 - float(dist), 4)
            results.append({
                "id": m.get("document_name", f"doc-{idx}"),
                "document": d,
                "metadata": m,
                "distance": float(dist),
                "similarity": similarity,
                "preview": (d[:200] + "…") if isinstance(d, str) and len(d) > 200 else d,
            })

        # ✅ Log nicely for debug
        print("\n[Retrieval Log]")
        print(f"🔍 Query: {query}")
        for r in results:
            print(f"→ {r['id']} | sim={r['similarity']} | dist={r['distance']}")
            print(f"   Preview: {r['preview'][:120]}...\n")

        return results


# ==========================
# Embedding wrapper (OpenAI)
# ==========================
class CustomEmbeddingWrapper:
    """
    ✅ Universal Embedding Wrapper (for all Chroma versions)
    Compatible with both old and new ChromaDB APIs.
    """

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY_EMBED"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # --- Required for Chroma >= 0.5
    def name(self) -> str:
        """Return embedding function name."""
        return "openai-text-embedding"

    # --- Core embedding function (used by newer Chroma)
    def __call__(self, input: str | list[str]):
        """Generic entrypoint for embedding both queries and documents."""
        if isinstance(input, str):
            input = [input]
        response = self.client.embeddings.create(model=self.model, input=input)
        return [np.array(e.embedding, dtype=float) for e in response.data]

    # --- Backward compatibility for Chroma < 0.5.0
    def embed_query(self, input: str | list[str]):
        """For Chroma versions that still call embed_query."""
        return self.__call__(input)

    def embed_documents(self, input: list[str]):
        """For Chroma versions that still call embed_documents."""
        return self.__call__(input)
