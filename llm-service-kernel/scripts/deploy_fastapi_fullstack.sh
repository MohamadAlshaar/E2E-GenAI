#!/usr/bin/env bash
set -euo pipefail

SERVICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAMESPACE="${NAMESPACE:-llm-service}"
BOOTSTRAP_JOB="llm-service-kernel-bootstrap"

kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1 || kubectl create namespace "${NAMESPACE}"

echo "Applying storage layer..."
kubectl apply -f "${SERVICE_ROOT}/deploy/k8s-storage/mongo.yaml"
kubectl apply -f "${SERVICE_ROOT}/deploy/k8s-storage/milvus.yaml"
kubectl apply -f "${SERVICE_ROOT}/deploy/k8s-storage/seaweedfs.yaml"

echo "Applying FastAPI PVCs/config/service..."
kubectl apply -f "${SERVICE_ROOT}/deploy/k8s-fastapi/rag-store-tenants-pvc.yaml"
kubectl apply -f "${SERVICE_ROOT}/deploy/k8s-fastapi/tenant-ingest-input-pvc.yaml"
kubectl apply -f "${SERVICE_ROOT}/deploy/k8s-fastapi/fastapi-configmap.fullstack.yaml"
kubectl apply -f "${SERVICE_ROOT}/deploy/k8s-fastapi/fastapi-secret.fullstack.yaml"
kubectl apply -f "${SERVICE_ROOT}/deploy/k8s-fastapi/fastapi-service.yaml"

echo "Running bootstrap job..."
kubectl delete job "${BOOTSTRAP_JOB}" -n "${NAMESPACE}" --ignore-not-found=true
kubectl apply -f "${SERVICE_ROOT}/deploy/k8s-fastapi/fastapi-bootstrap-job.yaml"
kubectl wait --for=condition=complete "job/${BOOTSTRAP_JOB}" -n "${NAMESPACE}" --timeout=600s

echo "Deploying FastAPI..."
kubectl apply -f "${SERVICE_ROOT}/deploy/k8s-fastapi/fastapi-deployment.fullstack.yaml"
kubectl rollout restart deployment/llm-service-kernel -n "${NAMESPACE}"
kubectl rollout status deployment/llm-service-kernel -n "${NAMESPACE}" --timeout=300s

echo
kubectl get pods -n "${NAMESPACE}" -o wide
