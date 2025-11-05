from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client_chat = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY_GPT4"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
client_embed = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY_EMBED"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

OPENAI_MODEL = "GPT-4o-mini"

def chat_completion(messages, model=None, temperature=0.2):
    resp = client_chat.chat.completions.create(
        model=model or OPENAI_MODEL,
        messages=messages,
        temperature=temperature
    )
    return resp.choices[0].message.content
