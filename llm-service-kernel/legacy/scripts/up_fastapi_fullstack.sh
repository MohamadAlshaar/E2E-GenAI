#!/usr/bin/env bash
set -euo pipefail

SERVICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_ROOT="${MINIKUBE_REMOTE_ROOT:-/mnt/llm-host}"
LOCAL_PORT="${FASTAPI_LOCAL_PORT:-18080}"

echo "== Checking Minikube-mounted assets =="
minikube ssh "test -f '$REMOTE_ROOT/all-MiniLM-L6-v2/config.json'"
minikube ssh "test -f '$REMOTE_ROOT/bge-base-en-v1.5/config.json'"
minikube ssh "test -f '$REMOTE_ROOT/Qwen2.5-0.5B-Instruct/config.json'"
minikube ssh "test -d '$REMOTE_ROOT/rag_store_tenants'"

echo "== Applying fullstack config =="
kubectl apply -f "$SERVICE_ROOT/deploy/k8s-fastapi/fastapi-configmap.fullstack.yaml"
kubectl apply -f "$SERVICE_ROOT/deploy/k8s-fastapi/fastapi-deployment.fullstack.yaml"
kubectl apply -f "$SERVICE_ROOT/deploy/k8s-fastapi/fastapi-service.yaml"

echo "== Restarting FastAPI deployment =="
kubectl rollout restart deployment/llm-service-kernel -n llm-service
kubectl rollout status deployment/llm-service-kernel -n llm-service --timeout=180s

echo "== Current pod =="
kubectl get pods -n llm-service -l app=llm-service-kernel

echo "== Starting port-forward on localhost:${LOCAL_PORT} =="
pkill -f "port-forward -n llm-service svc/llm-service-kernel ${LOCAL_PORT}:8080" >/dev/null 2>&1 || true
nohup kubectl port-forward -n llm-service svc/llm-service-kernel "${LOCAL_PORT}:8080" >/tmp/llm-service-kernel-port-forward.log 2>&1 &

READY=0
for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${LOCAL_PORT}/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done

if [[ "$READY" != "1" ]]; then
  echo "ERROR: FastAPI did not become reachable on localhost:${LOCAL_PORT}"
  echo "--- port-forward log ---"
  cat /tmp/llm-service-kernel-port-forward.log || true
  exit 1
fi

echo
echo "FastAPI fullstack is up."
echo "Health: http://127.0.0.1:${LOCAL_PORT}/health"
echo "CLI:    python scripts/chat_cli.py --base-url http://127.0.0.1:${LOCAL_PORT} --show-debug"
