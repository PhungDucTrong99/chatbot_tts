# Workshop 3: Chatbot with ChromaDB, OpenAI SDK, and HuggingFace TTS (VITS)

This repo contains a complete, minimal example of a RAG-powered chatbot that:
- Stores & retrieves knowledge using **ChromaDB**
- Generates responses via the **OpenAI SDK**
- Converts responses to audio via a **HuggingFace (Coqui) VITS** model

You can run either a **CLI** app or a simple **Streamlit** web UI.

---

## 1) Setup

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
cp .env.example .env
# Edit .env, set OPENAI_API_KEY and (optionally) TTS_MODEL/TTS_SPEAKER
```

> The default VITS model is `tts_models/en/vctk/vits` with speaker `p225`.
> The first run will download the model from HuggingFace Hub (via the `TTS` package).

Mock KB lives in `data/mock_kb.json`. You can replace it with your own docs
and point to it via `KB_PATH=/path/to/your.json` in `.env`.

---

## 2) Run (Streamlit)

```bash
streamlit run app_streamlit.py
```
Ask questions in the chat. Each assistant reply is also synthesized as audio and playable in the UI.

---

## 3) Code Map

- `rag_store.py` — wraps ChromaDB. Uses OpenAI **text-embedding-3-small** for embeddings by default.
- `llm.py` — OpenAI Chat Completions wrapper (default model `gpt-4o-mini`).
- `tts.py` — HuggingFace/Coqui **VITS** speech synthesis, configurable via `.env`.
- `app_cli.py` — terminal chat app with retrieval + TTS + logging
- `app_streamlit.py` — minimal web UI for chat + audio playback
- `data/mock_kb.json` — example FAQs for the KB
- `requirements.txt`, `.env.example`, `README.md`

---

## 4) Customize

- **Add docs**: convert your sources to small chunks and upsert them via `RAGStore.upsert_items(...)`.
- **Change model**: set `OPENAI_MODEL` in `.env`.
- **Change voice**: set `TTS_MODEL` and `TTS_SPEAKER` in `.env`. Many VITS/VCTK speakers are available.
- **Logs**: check `logs/` for JSONL transcripts with path to generated audio.

---

## 5) Notes

- The TTS step requires PyTorch and may download a few hundred MB the first time.
- If you run on CPU, generation will be slower than with GPU, but it still works.
- For production, consider using a smaller/faster TTS model, caching, and a background worker for audio.
