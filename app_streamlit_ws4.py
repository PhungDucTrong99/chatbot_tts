# =====================================================
# 🧹 Clean proxy BEFORE import openai (critical for Streamlit)
# =====================================================
import os
for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(var, None)

# =====================================================
# Imports (sau khi đã xoá proxy)
# =====================================================
import openai
import json
from dotenv import load_dotenv
from tools_ws4 import get_weather, get_time, get_profile

load_dotenv()

client = openai.OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://aiportalapi.stu-platform.live/jpe"),
    api_key=os.getenv("OPENAI_API_KEY_GPT4", "sk-your-key")
)

MODEL = os.getenv("OPENAI_MODEL", "GPT-4o-mini")

FUNCTIONS = {
    "get_weather": get_weather,
    "get_time": get_time,
    "get_profile": get_profile,
}

def chat_with_function(messages):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {"type": "object","properties": {"location": {"type": "string"}},"required": ["location"]},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time for a location",
                    "parameters": {"type": "object","properties": {"location": {"type": "string"}},"required": ["location"]},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_profile",
                    "description": "Get short profile for a person",
                    "parameters": {"type": "object","properties": {"name": {"type": "string"}},"required": ["name"]},
                },
            },
        ],
    )

    message = response.choices[0].message
    if getattr(message, "tool_calls", None):
        call = message.tool_calls[0]
        func_name = call.function.name
        args = json.loads(call.function.arguments or "{}")

        func = FUNCTIONS.get(func_name)
        result = func(**args) if func else f"[Error] Unknown function: {func_name}"

        print(f"🧩 Function call: {func_name}({args}) → {result}")
        messages.append(message)
        messages.append({"role": "tool", "content": result})
        return chat_with_function(messages)
    return message.content
