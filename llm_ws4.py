import openai
import os
import json
from dotenv import load_dotenv
from tools_ws4 import get_weather, get_time, get_profile


# ==============================
# Clean proxy variables (important for macOS / VPN)
# ==============================
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)

# ==========================
# Load environment variables
# ==========================
load_dotenv()

# ==========================
# OpenAI client (new SDK style)
# ==========================
client = openai.OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://aiportalapi.stu-platform.live/jpe"),
    api_key=os.getenv("OPENAI_API_KEY_GPT4", "sk-your-api-key-here")
)

MODEL = os.getenv("OPENAI_MODEL", "GPT-4o-mini")

# ==========================
# Function registry
# ==========================
FUNCTIONS = {
    "get_weather": get_weather,
    "get_time": get_time,
    "get_profile": get_profile,
}

# ==========================
# Chat with Function Calling
# ==========================
def chat_with_function(messages):
    """
    Chat completion with recursive function calling.
    Supports multiple tool calls until the final natural language output.
    """
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
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                        },
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time in a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City or country name"},
                        },
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_profile",
                    "description": "Get a short profile for a given name",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Person's name"},
                        },
                        "required": ["name"],
                    },
                },
            },
        ],
    )

    message = response.choices[0].message

    # Check if model called any tool
    if getattr(message, "tool_calls", None):
        call = message.tool_calls[0]
        func_name = call.function.name
        args = json.loads(call.function.arguments or "{}")

        func = FUNCTIONS.get(func_name)
        if not func:
            return f"[Error] Unknown tool called: {func_name}"

        try:
            result = func(**args)
        except Exception as e:
            result = f"[Error executing {func_name}]: {e}"

        # Log function call
        print(f"\n🧩 Function Call: {func_name}({args})")
        print(f"→ Result: {result}\n")

        # Append and continue the chat
        messages.append(message)
        messages.append({"role": "tool", "content": result})
        return chat_with_function(messages)
    else:
        return message.content


# ==========================
# Test CLI
# ==========================
if __name__ == "__main__":
    print("=== Workshop 4 Function Calling Test ===")
    query = input("You: ")
    messages = [{"role": "user", "content": query}]
    answer = chat_with_function(messages)
    print("Bot:", answer)
