import requests

api_key = "your_api_key_here"
url = "https://open.bigmodel.cn/api/paas/v4/"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": "glm-4.6",  # or whatever model name is available
    "messages": [{"role": "user", "content": "Your prompt here"}]
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
