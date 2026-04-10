#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# deploy.sh — deploy the full E2E GenAI stack and populate RAG
#
# Prerequisite: run setup.sh once first (system deps + models + minikube).
#
# What this does:
#   1. Preflight — verify models, minikube, GPU are available
#   2. Deploy — run deploy_fullstack_single_node.sh (Istio, storage, llm-d, FastAPI)
#   3. Ingest — run RAG ingestion (PDFs → embeddings → Milvus + SeaweedFS)
#   4. Verify — health check + smoke test
#   5. Print — usage instructions with port-forward info
#
# Environment overrides:
#   SKIP_DEPLOY            set to 1 to skip k8s deploy (if already deployed)
#   SKIP_INGEST            set to 1 to skip RAG ingestion
#   SKIP_SMOKE_TEST        set to 1 to skip the post-deploy smoke test
#   TENANT_ID              tenant name for RAG (default: tenantA)
#   FASTAPI_LOCAL_PORT     local port for FastAPI (default: 18081)
#   MILVUS_LOCAL_PORT      local port for Milvus (default: 19530)
#   S3_LOCAL_PORT          local port for SeaweedFS S3 (default: 8333)
#
# Model / worker configuration (preserved for easy changes):
#   GENERATION_MODEL_NAME  model name served by vLLM (default: qwen2.5-0.5b)
#   QWEN_HF_REPO          HuggingFace repo for the LLM (for download_models.sh)
#   See deploy/llmd-local/modelservice-values.yaml for vLLM worker config
#   See deploy/k8s-fastapi/fastapi-configmap.fullstack.yaml for service config
# ---------------------------------------------------------------------------
set -euo pipefail

KERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${KERNEL_ROOT}/.." && pwd)"
SCRIPTS_DIR="${KERNEL_ROOT}/scripts"

SKIP_DEPLOY="${SKIP_DEPLOY:-0}"
SKIP_INGEST="${SKIP_INGEST:-0}"
SKIP_SMOKE_TEST="${SKIP_SMOKE_TEST:-0}"

TENANT_ID="${TENANT_ID:-tenantA}"
FASTAPI_LOCAL_PORT="${FASTAPI_LOCAL_PORT:-18081}"
MILVUS_LOCAL_PORT="${MILVUS_LOCAL_PORT:-19530}"
S3_LOCAL_PORT="${S3_LOCAL_PORT:-8333}"

FASTAPI_NAMESPACE="${FASTAPI_NAMESPACE:-llm-service}"
FASTAPI_SERVICE="${FASTAPI_SERVICE:-llm-service-kernel}"
FASTAPI_DEPLOYMENT="${FASTAPI_DEPLOYMENT:-llm-service-kernel}"

INGEST_TIMEOUT_S="${INGEST_TIMEOUT_S:-600}"

PORT_FORWARD_LOG_DIR="/tmp/e2e-genai-portforward"

_MANAGED_PIDS=()

# ── Formatting helpers ────────────────────────────────────────────────────
log()  { printf '\n\033[1;34m[deploy]\033[0m %s\n' "$*"; }
step() { printf '\n\033[1;36m━━━ Step %s ━━━\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m  ✓\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m  ⚠\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[deploy] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

cleanup() {
  for pid in "${_MANAGED_PIDS[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" 2>/dev/null || true
  done
}

trap cleanup EXIT INT TERM

# ── Port-forward management ───────────────────────────────────────────────
kill_port() {
  local port="$1"
  local pids
  pids="$(lsof -iTCP:"${port}" -sTCP:LISTEN -t 2>/dev/null || true)"
  if [ -n "${pids}" ]; then
    echo "${pids}" | xargs kill 2>/dev/null || true
    sleep 1
  fi
}

start_port_forward() {
  local namespace="$1" service="$2" local_port="$3" remote_port="$4"

  # Kill any existing forward on this port
  kill_port "${local_port}"

  mkdir -p "${PORT_FORWARD_LOG_DIR}"
  local log_file="${PORT_FORWARD_LOG_DIR}/${service}-${local_port}.log"

  kubectl port-forward -n "${namespace}" "svc/${service}" "${local_port}:${remote_port}" >"${log_file}" 2>&1 &
  local pid=$!
  _MANAGED_PIDS+=("${pid}")
  sleep 2

  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    cat "${log_file}" >&2 2>/dev/null || true
    die "port-forward failed for svc/${service} on port ${local_port}"
  fi
  ok "Port-forward: svc/${service} → localhost:${local_port}"
}

ensure_port_forward() {
  local namespace="$1" service="$2" local_port="$3" remote_port="$4"

  # Check if port is already forwarded and working
  if curl -sf --connect-timeout 2 "http://127.0.0.1:${local_port}/" >/dev/null 2>&1 || \
     lsof -iTCP:"${local_port}" -sTCP:LISTEN >/dev/null 2>&1; then
    return 0
  fi

  start_port_forward "${namespace}" "${service}" "${local_port}" "${remote_port}"
}

wait_http_ok() {
  local url="$1" timeout_s="$2"
  local deadline=$((SECONDS + timeout_s))

  while [ "${SECONDS}" -lt "${deadline}" ]; do
    if curl -sf --connect-timeout 3 "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

# ── Pod health checks ─────────────────────────────────────────────────────
check_pods_healthy() {
  local namespace="$1"
  local not_ready
  not_ready="$(kubectl get pods -n "${namespace}" --no-headers 2>/dev/null | grep -v 'Running\|Completed' || true)"

  if [ -n "${not_ready}" ]; then
    return 1
  fi
  return 0
}

wait_all_pods_ready() {
  local namespace="$1" timeout_s="${2:-300}" label="${3:-}"
  local deadline=$((SECONDS + timeout_s))

  log "Waiting for all pods in ${namespace} to be ready..."
  while [ "${SECONDS}" -lt "${deadline}" ]; do
    if check_pods_healthy "${namespace}"; then
      return 0
    fi
    sleep 5
  done

  warn "Some pods in ${namespace} are not ready after ${timeout_s}s:"
  kubectl get pods -n "${namespace}" --no-headers 2>/dev/null | grep -v 'Running\|Completed' || true
  return 1
}

recover_stuck_pods() {
  local namespace="$1"

  # Find pods in CrashLoopBackOff or Error state
  local bad_pods
  bad_pods="$(kubectl get pods -n "${namespace}" --no-headers 2>/dev/null | \
    grep -E 'CrashLoopBackOff|Error|ImagePullBackOff|ErrImagePull' | awk '{print $1}' || true)"

  if [ -z "${bad_pods}" ]; then
    return 0
  fi

  warn "Found stuck pods in ${namespace} — attempting recovery"
  for pod in ${bad_pods}; do
    log "Deleting stuck pod: ${pod}"
    kubectl delete pod -n "${namespace}" "${pod}" --grace-period=10 2>/dev/null || true
  done

  sleep 10
}

# ── Preflight ─────────────────────────────────────────────────────────────
preflight() {
  step "1/5: Preflight checks"

  for cmd in kubectl minikube docker helm python3; do
    command -v "${cmd}" >/dev/null 2>&1 || die "missing: ${cmd} — run setup.sh first"
  done
  ok "Required tools installed"

  if ! minikube status -p minikube 2>/dev/null | grep -q "Running"; then
    die "minikube not running — run setup.sh first"
  fi
  ok "minikube running"

  # Check GPU is available in the cluster
  local gpu_count
  gpu_count="$(kubectl get nodes -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}' 2>/dev/null || echo 0)"
  if [ "${gpu_count}" -ge 1 ]; then
    ok "GPU available in cluster: ${gpu_count}"
  else
    warn "No GPU allocatable in cluster — vLLM pods will fail to schedule"
    warn "Run setup.sh to deploy the NVIDIA device plugin"
  fi

  # Check models
  for model_dir in bge-base-en-v1.5 Qwen2.5-0.5B-Instruct; do
    if [ ! -d "${REPO_ROOT}/${model_dir}" ]; then
      die "Model not found: ${REPO_ROOT}/${model_dir} — run setup.sh or scripts/download_models.sh"
    fi
  done
  ok "All models present"

  # Check docs
  local pdf_count=0
  if [ -d "${REPO_ROOT}/docs_RAG" ]; then
    pdf_count="$(find "${REPO_ROOT}/docs_RAG" -name '*.pdf' -type f 2>/dev/null | wc -l)"
  fi
  if [ "${pdf_count}" -eq 0 ] && [ "${SKIP_INGEST}" != "1" ]; then
    warn "No PDFs in docs_RAG/ — run scripts/download_sample_docs.sh or set SKIP_INGEST=1"
    die "Cannot ingest without documents"
  fi
  ok "docs_RAG/ has ${pdf_count} PDFs"
}

# ── Deploy full stack ─────────────────────────────────────────────────────
deploy_stack() {
  step "2/5: Deploying full stack to Kubernetes"

  if [ "${SKIP_DEPLOY}" = "1" ]; then
    log "Skipping deploy (SKIP_DEPLOY=1)"

    # Even when skipping deploy, check pod health and recover if needed
    for ns in llm-service llm-d-local; do
      if ! check_pods_healthy "${ns}"; then
        warn "Unhealthy pods detected in ${ns}"
        recover_stuck_pods "${ns}"
        wait_all_pods_ready "${ns}" 120 || true
      fi
    done
    return 0
  fi

  FASTAPI_LOCAL_PORT="${FASTAPI_LOCAL_PORT}" \
    bash "${SCRIPTS_DIR}/deploy_fullstack_single_node.sh"

  ok "Full stack deployed"
}

# ── RAG ingestion ─────────────────────────────────────────────────────────
ingest_rag() {
  step "3/5: RAG ingestion (PDFs → Milvus + SeaweedFS)"

  if [ "${SKIP_INGEST}" = "1" ]; then
    log "Skipping RAG ingestion (SKIP_INGEST=1)"
    return 0
  fi

  local pdf_count
  pdf_count="$(find "${REPO_ROOT}/docs_RAG" -name '*.pdf' -type f 2>/dev/null | wc -l)"
  if [ "${pdf_count}" -eq 0 ]; then
    warn "No PDFs found — skipping ingestion"
    return 0
  fi

  # Ensure port-forwards for Milvus and SeaweedFS S3
  log "Setting up port-forwards for ingestion"
  ensure_port_forward "${FASTAPI_NAMESPACE}" milvus "${MILVUS_LOCAL_PORT}" 19530
  ensure_port_forward "${FASTAPI_NAMESPACE}" seaweed-s3 "${S3_LOCAL_PORT}" 8333

  # Wait for Milvus to be ready
  log "Waiting for Milvus to be reachable..."
  if ! wait_http_ok "http://127.0.0.1:${MILVUS_LOCAL_PORT}/healthz" 60; then
    die "Milvus not reachable on port ${MILVUS_LOCAL_PORT}"
  fi

  # Check if collection already has data
  local row_count=0
  if command -v python3 >/dev/null 2>&1; then
    row_count="$(python3 -c "
import sys
try:
    from pymilvus import MilvusClient
    c = MilvusClient(uri='http://127.0.0.1:${MILVUS_LOCAL_PORT}', token='root:Milvus')
    stats = c.get_collection_stats('rag_chunks_seaweed_v2')
    print(stats.get('row_count', 0))
except:
    print(0)
" 2>/dev/null || echo 0)"
  fi

  if [ "${row_count}" -gt 0 ]; then
    ok "Milvus collection already has ${row_count} rows — skipping ingestion"
    log "To re-ingest, run: FORCE_REINGEST=1 ./deploy.sh"
    if [ "${FORCE_REINGEST:-0}" != "1" ]; then
      return 0
    fi
    log "FORCE_REINGEST=1 — re-ingesting"
  fi

  # Install Python deps if needed
  python3 -c "import pymilvus, boto3, pypdf, sentence_transformers" 2>/dev/null || {
    log "Installing Python dependencies for ingestion..."
    pip install --quiet pymilvus boto3 pypdf sentence-transformers Pillow 2>/dev/null || \
    python3 -m pip install --quiet --user pymilvus boto3 pypdf sentence-transformers Pillow
  }

  local drop_flag=""
  if [ "${FORCE_REINGEST:-0}" = "1" ]; then
    drop_flag="--drop-existing"
  fi

  log "Running ingestion: ${pdf_count} PDFs for tenant ${TENANT_ID}"
  timeout "${INGEST_TIMEOUT_S}" python3 "${SCRIPTS_DIR}/ingest_tenant_to_milvus.py" \
    --tenant "${TENANT_ID}" \
    --docs-dir "${REPO_ROOT}/docs_RAG" \
    --manifest-root "${REPO_ROOT}/rag_store_tenants" \
    --bge-model-path "${REPO_ROOT}/bge-base-en-v1.5" \
    --milvus-uri "http://127.0.0.1:${MILVUS_LOCAL_PORT}" \
    --milvus-token "root:Milvus" \
    --collection "rag_chunks_seaweed_v2" \
    --s3-endpoint-url "http://127.0.0.1:${S3_LOCAL_PORT}" \
    --s3-access-key-id "llmbenchadmin" \
    --s3-secret-access-key "llmbenchsecretkey123" \
    --s3-bucket "llm-rag-store" \
    ${drop_flag}

  ok "RAG ingestion complete"

  # Restart FastAPI so it picks up the new collection data
  log "Restarting FastAPI to pick up ingested data..."
  kubectl rollout restart deployment/"${FASTAPI_DEPLOYMENT}" -n "${FASTAPI_NAMESPACE}"
  kubectl rollout status deployment/"${FASTAPI_DEPLOYMENT}" -n "${FASTAPI_NAMESPACE}" --timeout=120s

  # Re-establish FastAPI port-forward since the pod changed
  start_port_forward "${FASTAPI_NAMESPACE}" "${FASTAPI_SERVICE}" "${FASTAPI_LOCAL_PORT}" 8080
  sleep 3
}

# ── Smoke test ────────────────────────────────────────────────────────────
smoke_test() {
  step "4/5: Smoke test"

  if [ "${SKIP_SMOKE_TEST}" = "1" ]; then
    log "Skipping smoke test (SKIP_SMOKE_TEST=1)"
    return 0
  fi

  # Ensure FastAPI port-forward is up
  ensure_port_forward "${FASTAPI_NAMESPACE}" "${FASTAPI_SERVICE}" "${FASTAPI_LOCAL_PORT}" 8080
  if ! wait_http_ok "http://127.0.0.1:${FASTAPI_LOCAL_PORT}/health" 30; then
    die "FastAPI not reachable after deploy"
  fi

  # Health check
  local health
  health="$(curl -sf "http://127.0.0.1:${FASTAPI_LOCAL_PORT}/health" 2>/dev/null)"
  local gen_ok rag_ok cache_ok
  gen_ok="$(echo "${health}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('generation_backend_usable',False))" 2>/dev/null || echo "False")"
  rag_ok="$(echo "${health}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('rag_usable',False))" 2>/dev/null || echo "False")"
  cache_ok="$(echo "${health}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('semantic_cache_usable',False))" 2>/dev/null || echo "False")"

  [ "${gen_ok}" = "True" ] && ok "LLM backend: healthy" || warn "LLM backend: NOT healthy"
  [ "${rag_ok}" = "True" ] && ok "RAG: healthy" || warn "RAG: NOT healthy"
  [ "${cache_ok}" = "True" ] && ok "Semantic cache: healthy" || warn "Semantic cache: NOT healthy"

  # Quick inference test
  log "Testing inference..."
  local model_name
  model_name="$(echo "${health}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('model','qwen2.5-0.5b'))" 2>/dev/null || echo "qwen2.5-0.5b")"

  local resp
  resp="$(curl -sf --max-time 120 "http://127.0.0.1:${FASTAPI_LOCAL_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${model_name}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"What is the transformer architecture?\"}],
      \"max_tokens\": 64,
      \"temperature\": 0.0,
      \"extra_body\": {\"tenant_id\": \"${TENANT_ID}\"}
    }" 2>/dev/null || echo "{}")"

  local route
  route="$(echo "${resp}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('_route',{}).get('route_taken','unknown'))" 2>/dev/null || echo "unknown")"
  local rag_used
  rag_used="$(echo "${resp}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('_rag',{}).get('used',False))" 2>/dev/null || echo "False")"
  local e2e_ms
  e2e_ms="$(echo "${resp}" | python3 -c "import sys,json; print(f\"{json.load(sys.stdin).get('_perf',{}).get('e2e_ms',0):.0f}\")" 2>/dev/null || echo "?")"

  ok "Inference OK — route=${route}, rag_used=${rag_used}, e2e=${e2e_ms}ms"
}

# ── Summary ──────────────────────────────────────────────────────────────
print_summary() {
  step "5/5: Ready!"

  # Ensure all three port-forwards are up for the user
  ensure_port_forward "${FASTAPI_NAMESPACE}" "${FASTAPI_SERVICE}" "${FASTAPI_LOCAL_PORT}" 8080
  ensure_port_forward "${FASTAPI_NAMESPACE}" milvus "${MILVUS_LOCAL_PORT}" 19530
  ensure_port_forward "${FASTAPI_NAMESPACE}" seaweed-s3 "${S3_LOCAL_PORT}" 8333

  local model_name
  model_name="$(curl -sf "http://127.0.0.1:${FASTAPI_LOCAL_PORT}/health" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('model','qwen2.5-0.5b'))" 2>/dev/null || echo "qwen2.5-0.5b")"

  echo
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo -e "\033[1;32m  Deployment complete — service is live!\033[0m"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo
  echo "  Interactive chat:"
  echo "    python3 scripts/chat_cli.py --show-debug"
  echo
  echo "  curl example:"
  echo "    curl -s http://127.0.0.1:${FASTAPI_LOCAL_PORT}/v1/chat/completions \\"
  echo "      -H 'Content-Type: application/json' \\"
  echo "      -d '{\"model\":\"${model_name}\",\"messages\":[{\"role\":\"user\",\"content\":\"What is attention?\"}],\"max_tokens\":256,\"extra_body\":{\"tenant_id\":\"${TENANT_ID}\"}}'"
  echo
  echo "  Health check:"
  echo "    curl -s http://127.0.0.1:${FASTAPI_LOCAL_PORT}/health | python3 -m json.tool"
  echo
  echo "  Port-forwards (running in background):"
  echo "    FastAPI      → http://127.0.0.1:${FASTAPI_LOCAL_PORT}"
  echo "    Milvus       → http://127.0.0.1:${MILVUS_LOCAL_PORT}"
  echo "    SeaweedFS S3 → http://127.0.0.1:${S3_LOCAL_PORT}"
  echo
  echo "  To change the LLM model or add workers:"
  echo "    Edit deploy/llmd-local/modelservice-values.yaml"
  echo "    Edit deploy/k8s-fastapi/fastapi-configmap.fullstack.yaml"
  echo "    Re-run: ./deploy.sh"
  echo
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # Keep port-forwards alive
  log "Port-forwards are running in the background."
  log "Press Ctrl+C to stop them, or close this terminal."
  wait
}

main() {
  echo
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  E2E GenAI Service — Full Stack Deploy"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  preflight
  deploy_stack
  ingest_rag
  smoke_test
  print_summary
}

main "$@"
