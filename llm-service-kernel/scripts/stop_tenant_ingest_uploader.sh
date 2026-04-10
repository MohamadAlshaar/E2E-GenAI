#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-llm-service}"
POD_NAME="${POD_NAME:-tenant-ingest-uploader}"

kubectl delete pod "${POD_NAME}" -n "${NAMESPACE}" --ignore-not-found=true
