import requests
import json

url = "http://localhost:11434/api/generate"

headers = {
    "Content-Type": "application/json"
}

model = "llama3"

prompt = "Should I invest in Tesla? Why this might not be a good idea? Give me 3 reasons."

payload = {
    "prompt": prompt,
    "model": model,
    "stream": False
}

response = requests.post(url, json=payload, headers=headers)

if response.status_code == 200:
    data = response.json()
    print("Generated Text:")
    print(data['response'])
else:
    print(f"Error: {response.status_code} - {response.text}")