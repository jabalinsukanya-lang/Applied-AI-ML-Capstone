import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("LLM_API_KEY")

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost",
    "X-Title": "Applied AI ML Capstone"
}

payload = {
    "model": "tencent/hy3:free",
    "messages": [
        {
            "role": "user",
            "content": "Reply with only the word: hello"
        }
    ],
    "temperature": 0,
    "max_tokens": 150
}

response = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=60
)

import json

print("Status Code:", response.status_code)

data = response.json()

print(json.dumps(data, indent=4))