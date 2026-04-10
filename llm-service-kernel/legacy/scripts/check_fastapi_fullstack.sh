#!/usr/bin/env bash
set -euo pipefail

LOCAL_PORT="${1:-18080}"
BASE_URL="http://127.0.0.1:${LOCAL_PORT}"

echo "== Health =="
curl -fsS "${BASE_URL}/health"
echo
echo

python - <<PY
import json
import urllib.request

base = "${BASE_URL}"

def post(prompt: str):
    body = {
        "model": "qwen2.5-0.5b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Tenant-Id": "tenantA"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))

tests = [
    "who are you?",
    "who are you?",
    "what are the challenges addressed in starnuma",
]

for prompt in tests:
    resp = post(prompt)
    route = (resp.get("_route") or {}).get("route_taken")
    cache = resp.get("_cache") or {}
    rag = resp.get("_rag") or {}
    print(json.dumps({
        "prompt": prompt,
        "route_taken": route,
        "semantic_enabled": cache.get("semantic_enabled"),
        "semantic_hit": cache.get("semantic_hit"),
        "rag_enabled": rag.get("enabled"),
        "rag_used": rag.get("used"),
        "rag_top_score": rag.get("top_score"),
    }, indent=2))
PY
