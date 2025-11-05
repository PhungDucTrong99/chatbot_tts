import datetime, random

def get_weather(location: str) -> str:
    weather = random.choice(["sunny", "rainy", "cloudy", "windy"])
    return f"The weather in {location} is {weather} today."

def get_time(location: str) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"The current time in {location} is {now}."

def get_profile(name: str) -> str:
    return f"{name} is a senior leader at FPT Software. He has over 20 years of experience in IT management."
