#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/upload_tenant_pdfs.sh <tenant_id> <local_pdf_dir>"
  exit 1
fi

TENANT_ID="$1"
LOCAL_DIR="$2"

if [ ! -d "${LOCAL_DIR}" ]; then
  echo "Local directory does not exist: ${LOCAL_DIR}"
  exit 1
fi

NAMESPACE="${NAMESPACE:-llm-service}"
POD_NAME="${POD_NAME:-tenant-ingest-uploader}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/start_tenant_ingest_uploader.sh"

kubectl exec -n "${NAMESPACE}" "${POD_NAME}" -- sh -c "mkdir -p /tenant_ingest_input/${TENANT_ID}"
kubectl cp "${LOCAL_DIR}/." "${NAMESPACE}/${POD_NAME}:/tenant_ingest_input/${TENANT_ID}/"

echo
echo "Uploaded PDFs for tenant: ${TENANT_ID}"
kubectl exec -n "${NAMESPACE}" "${POD_NAME}" -- sh -c "find /tenant_ingest_input/${TENANT_ID} -maxdepth 3 -type f | sort"
