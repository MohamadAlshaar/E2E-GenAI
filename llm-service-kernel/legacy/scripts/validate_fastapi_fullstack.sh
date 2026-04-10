#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-28080}"
PF_LOG="${PF_LOG:-/tmp/llm-service-kernel-port-forward.log}"

cleanup() {
  if [[ -n "${PF_PID:-}" ]]; then
    kill "$PF_PID" >/dev/null 2>&1 || true
    wait "$PF_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# Fail fast if the chosen local port is already in use
if ss -ltn "( sport = :$PORT )" | grep -q LISTEN; then
  echo "ERROR: local port $PORT is already in use"
  exit 1
fi

kubectl port-forward -n llm-service svc/llm-service-kernel "${PORT}:8080" >"$PF_LOG" 2>&1 &
PF_PID=$!

READY=0
for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done

if [[ "$READY" != "1" ]]; then
  echo "ERROR: port-forward did not become ready"
  echo "--- port-forward log ---"
  cat "$PF_LOG" || true
  exit 1
fi

python - <<PY
import json
import urllib.request

BASE = "http://127.0.0.1:${PORT}"

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
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Tenant-Id": "tenantA"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))

with urllib.request.urlopen(BASE + "/health", timeout=30) as resp:
    health = json.loads(resp.read().decode("utf-8"))

print("HEALTH")
print(json.dumps(health, indent=2))

tests = [
    "who are you?",
    "who are you?",
    "what are the challenges addressed in starnuma",
]

print("\\nREQUESTS")
for prompt in tests:
    resp = post(prompt)
    route = (resp.get("_route") or {}).get("route_taken")
    cache = resp.get("_cache") or {}
    rag = resp.get("_rag") or {}
    print(json.dumps({
        "prompt": prompt,
        "route_taken": route,
        "semantic_hit": cache.get("semantic_hit"),
        "rag_used": rag.get("used"),
        "rag_top_score": rag.get("top_score"),
    }, indent=2))
PY

