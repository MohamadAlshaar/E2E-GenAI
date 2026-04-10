#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-llm-service}"
POD_NAME="${POD_NAME:-tenant-ingest-uploader}"
UPLOADER_IMAGE="${UPLOADER_IMAGE:-llm-service-kernel:fastapi-selfcontained}"

status="$(kubectl get pod "${POD_NAME}" -n "${NAMESPACE}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
if [ "${status}" = "Running" ]; then
  echo "Uploader pod already running: ${POD_NAME}"
  exit 0
fi

if kubectl get pod "${POD_NAME}" -n "${NAMESPACE}" >/dev/null 2>&1; then
  kubectl delete pod "${POD_NAME}" -n "${NAMESPACE}" --wait=true
fi

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: ${POD_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: tenant-ingest-uploader
spec:
  restartPolicy: Always
  containers:
    - name: uploader
      image: ${UPLOADER_IMAGE}
      imagePullPolicy: IfNotPresent
      command: ["sh", "-c", "mkdir -p /tenant_ingest_input && sleep 86400"]
      volumeMounts:
        - name: ingest-input
          mountPath: /tenant_ingest_input
  volumes:
    - name: ingest-input
      persistentVolumeClaim:
        claimName: tenant-ingest-input-pvc
EOF

kubectl wait --for=condition=Ready "pod/${POD_NAME}" -n "${NAMESPACE}" --timeout=180s
echo "Uploader pod ready: ${POD_NAME}"
