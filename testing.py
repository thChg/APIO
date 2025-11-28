import httpx, os

headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
try:
    r = httpx.get("https://api.openai.com/v1/models/gpt-4.1-mini", headers=headers, timeout=10)
    print(r.json())
except Exception as e:
    print("Direct HTTP request failed:", e)
