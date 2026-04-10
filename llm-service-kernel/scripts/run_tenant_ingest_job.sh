#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/run_tenant_ingest_job.sh <tenant_id> [docs_subdir]"
  echo "Example: bash scripts/run_tenant_ingest_job.sh tenantC"
  exit 1
fi

TENANT_ID="$1"
DOCS_SUBDIR="${2:-$1}"

NAMESPACE="${NAMESPACE:-llm-service}"
IMAGE="${IMAGE:-llm-service-kernel:fastapi-selfcontained}"
JOB_BASENAME="${JOB_BASENAME:-tenant-ingest}"
TIMESTAMP="$(date +%Y%m%d%H%M%S)"
JOB_NAME="${JOB_BASENAME}-${TENANT_ID,,}-${TIMESTAMP}"

cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: ${NAMESPACE}
spec:
  backoffLimit: 1
  template:
    metadata:
      labels:
        app: tenant-ingest
        tenant: "${TENANT_ID}"
    spec:
      restartPolicy: Never
      containers:
        - name: ingest
          image: ${IMAGE}
          imagePullPolicy: Never
          command:
            - python
            - /app/scripts/ingest_tenant_to_milvus.py
            - --tenant
            - ${TENANT_ID}
            - --docs-dir
            - /tenant_ingest_input/${DOCS_SUBDIR}
            - --manifest-root
            - /rag_store_tenants
          envFrom:
            - configMapRef:
                name: llm-service-kernel-config
            - secretRef:
                name: llm-service-kernel-secret
          volumeMounts:
            - name: ingest-input
              mountPath: /tenant_ingest_input
              readOnly: true
            - name: rag-store-tenants
              mountPath: /rag_store_tenants
      volumes:
        - name: ingest-input
          persistentVolumeClaim:
            claimName: tenant-ingest-input-pvc
        - name: rag-store-tenants
          persistentVolumeClaim:
            claimName: rag-store-tenants-pvc
EOF

kubectl wait --for=condition=complete "job/${JOB_NAME}" -n "${NAMESPACE}" --timeout=1800s
echo
kubectl logs -n "${NAMESPACE}" "job/${JOB_NAME}"
echo
echo "Job completed: ${JOB_NAME}"
