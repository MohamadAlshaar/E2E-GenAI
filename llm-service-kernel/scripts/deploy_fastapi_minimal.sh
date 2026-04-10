#!/usr/bin/env bash
set -euo pipefail

SERVICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

kubectl apply -f "$SERVICE_ROOT/deploy/k8s-fastapi/fastapi-configmap.minimal.yaml"
kubectl apply -f "$SERVICE_ROOT/deploy/k8s-fastapi/fastapi-deployment.minimal.yaml"
kubectl apply -f "$SERVICE_ROOT/deploy/k8s-fastapi/fastapi-service.yaml"

kubectl rollout restart deployment/llm-service-kernel -n llm-service
kubectl rollout status deployment/llm-service-kernel -n llm-service --timeout=180s

kubectl get pods -n llm-service -l app=llm-service-kernel
