import time, requests

URL = "http://localhost:8080/chat"

t0 = time.time()
r = requests.post(URL, json={
    "messages": [{"role":"user","content":"Say hello in one sentence."}],
    "max_tokens": 64,
    "temperature": 0.2
}, timeout=300)
dt = time.time() - t0
r.raise_for_status()
print("latency_s:", round(dt, 3))
print(r.json()["choices"][0]["message"]["content"])

