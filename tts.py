import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Coqui TTS (loads models from HuggingFace Hub too)
from TTS.api import TTS

load_dotenv()
# dùng model vctk/vits, with language EN
TTS_MODEL = os.getenv("TTS_MODEL", "tts_models/en/vctk/vits")
TTS_SPEAKER = os.getenv("TTS_SPEAKER", "p226")

_cache = {}

def _get_tts():
    global _cache
    if "engine" not in _cache:
        # This will download the model on first run
        _cache["engine"] = TTS(model_name=TTS_MODEL)
    return _cache["engine"]

def synthesize_speech(text: str, out_dir: str = "audio_out", speaker: Optional[str] = None) -> str:
    engine = _get_tts()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    speaker = speaker or TTS_SPEAKER
    # Save to WAV for broad compatibility
    out_path = Path(out_dir) / f"tts_{abs(hash(text))}.wav"
    engine.tts_to_file(text=text, speaker=speaker, file_path=str(out_path))
    return str(out_path)
