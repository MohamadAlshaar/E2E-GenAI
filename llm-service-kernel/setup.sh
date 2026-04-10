#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup.sh — one-time machine setup for the E2E GenAI Service
#
# Run this once on a fresh GPU machine. It will:
#   1. Install system dependencies (Docker, NVIDIA toolkit, minikube, etc.)
#   2. Start minikube with GPU passthrough
#   3. Download required ML models from HuggingFace
#   4. Download sample PDF papers for RAG
#
# After setup completes, run:  ./deploy.sh
#
# Environment overrides:
#   SKIP_HOST_BOOTSTRAP     set to 1 if system deps are already installed
#   SKIP_MINIKUBE_START     set to 1 if minikube is already running
#   SKIP_MODEL_DOWNLOAD     set to 1 if models are already present
#   SKIP_DOCS_DOWNLOAD      set to 1 if docs_RAG/ is already populated
#   MINIKUBE_CPUS           CPU count for minikube (default: 8)
#   MINIKUBE_MEMORY         memory in MB for minikube (default: 16384)
#   MINIKUBE_DISK_SIZE      disk for minikube (default: 80g)
# ---------------------------------------------------------------------------
set -euo pipefail

KERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${KERNEL_ROOT}/.." && pwd)"
SCRIPTS_DIR="${KERNEL_ROOT}/scripts"

SKIP_HOST_BOOTSTRAP="${SKIP_HOST_BOOTSTRAP:-0}"
SKIP_MINIKUBE_START="${SKIP_MINIKUBE_START:-0}"
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"
SKIP_DOCS_DOWNLOAD="${SKIP_DOCS_DOWNLOAD:-0}"

log() { printf '\n\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[setup] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }
ok() { printf '\033[1;32m  ✓\033[0m %s\n' "$*"; }

# ── Preflight checks ──────────────────────────────────────────────────────
preflight() {
  log "Running preflight checks"

  # Check OS
  if [ -f /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    ok "OS: ${PRETTY_NAME:-${ID:-unknown}}"
  fi

  # Check NVIDIA driver
  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_name
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'unknown')"
    local driver_ver
    driver_ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo '?')"
    ok "GPU: ${gpu_name} (driver ${driver_ver})"
  else
    warn "nvidia-smi not found — NVIDIA driver must be installed before running this script"
    warn "Install it with: sudo apt install nvidia-driver-XXX (or your distro's equivalent)"
    die "NVIDIA driver required"
  fi

  # Check disk space (need ~10GB for models + images)
  local avail_gb
  avail_gb="$(df -BG --output=avail "${REPO_ROOT}" | tail -1 | tr -d ' G')"
  if [ "${avail_gb}" -lt 10 ]; then
    die "Less than 10GB disk space available (${avail_gb}GB). Need space for models and container images."
  fi
  ok "Disk: ${avail_gb}GB available"

  # Check RAM
  local total_mem_mb
  total_mem_mb="$(free -m | awk '/Mem:/ {print $2}')"
  if [ "${total_mem_mb}" -lt 12000 ]; then
    warn "Only ${total_mem_mb}MB RAM detected — minikube needs at least 16GB for the full stack"
  fi
  ok "RAM: ${total_mem_mb}MB"
}

# ── Step 1: System dependencies ──────────────────────────────────────────
install_system_deps() {
  if [ "${SKIP_HOST_BOOTSTRAP}" = "1" ]; then
    log "Skipping host bootstrap (SKIP_HOST_BOOTSTRAP=1)"
    return 0
  fi

  # Check if everything is already installed
  local all_present=1
  for cmd in docker minikube kubectl helm istioctl; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
      all_present=0
      break
    fi
  done

  if [ "${all_present}" = "1" ]; then
    ok "All system dependencies already installed"
    return 0
  fi

  log "Installing system dependencies via bootstrap_host_ubuntu.sh"
  # Don't start minikube yet — we do that separately with GPU config
  START_MINIKUBE=0 \
  ENABLE_MINIKUBE_STORAGE_ADDONS=0 \
  APPLY_GATEWAY_API=0 \
  INSTALL_ISTIO=0 \
  CREATE_NAMESPACES=0 \
    bash "${SCRIPTS_DIR}/bootstrap_host_ubuntu.sh"

  ok "System dependencies installed"
}

# ── Step 2: Start minikube ───────────────────────────────────────────────
start_minikube() {
  if [ "${SKIP_MINIKUBE_START}" = "1" ]; then
    log "Skipping minikube start (SKIP_MINIKUBE_START=1)"
    return 0
  fi

  if minikube status -p minikube >/dev/null 2>&1; then
    ok "minikube already running"
    return 0
  fi

  log "Starting minikube with GPU passthrough"
  minikube start \
    --driver=docker \
    --container-runtime=docker \
    --kubernetes-version="${KUBERNETES_VERSION:-v1.33.1}" \
    --cpus="${MINIKUBE_CPUS:-8}" \
    --memory="${MINIKUBE_MEMORY:-16384}" \
    --disk-size="${MINIKUBE_DISK_SIZE:-80g}" \
    --gpus=all

  # Enable storage addons
  minikube addons enable default-storageclass
  minikube addons enable storage-provisioner

  ok "minikube started with GPU"
}

# ── Step 3: Download models ──────────────────────────────────────────────
download_models() {
  if [ "${SKIP_MODEL_DOWNLOAD}" = "1" ]; then
    log "Skipping model download (SKIP_MODEL_DOWNLOAD=1)"
    return 0
  fi

  # Check if all models are already present
  if [ -f "${REPO_ROOT}/all-MiniLM-L6-v2/config.json" ] && \
     [ -f "${REPO_ROOT}/bge-base-en-v1.5/config.json" ] && \
     [ -f "${REPO_ROOT}/Qwen2.5-0.5B-Instruct/config.json" ]; then
    ok "All models already present"
    return 0
  fi

  log "Downloading ML models from HuggingFace (~3.2GB)"
  bash "${SCRIPTS_DIR}/download_models.sh"
  ok "Models downloaded"
}

# ── Step 4: Download sample docs ─────────────────────────────────────────
download_docs() {
  if [ "${SKIP_DOCS_DOWNLOAD}" = "1" ]; then
    log "Skipping docs download (SKIP_DOCS_DOWNLOAD=1)"
    return 0
  fi

  local pdf_count=0
  if [ -d "${REPO_ROOT}/docs_RAG" ]; then
    pdf_count="$(find "${REPO_ROOT}/docs_RAG" -name '*.pdf' -type f 2>/dev/null | wc -l)"
  fi

  if [ "${pdf_count}" -ge 10 ]; then
    ok "docs_RAG/ already has ${pdf_count} PDFs"
    return 0
  fi

  log "Downloading sample PDF papers for RAG (~127MB)"
  bash "${SCRIPTS_DIR}/download_sample_docs.sh"
  ok "Sample docs downloaded"
}

# ── Step 5: Ensure rag_store_tenants seed ─────────────────────────────────
ensure_tenant_seed() {
  local manifest="${REPO_ROOT}/rag_store_tenants/tenantA/manifest.json"
  if [ -f "${manifest}" ]; then
    ok "Tenant seed manifest already exists"
    return 0
  fi

  log "Creating default tenant seed manifest"
  mkdir -p "${REPO_ROOT}/rag_store_tenants/tenantA"

  local num_pdfs=0
  if [ -d "${REPO_ROOT}/docs_RAG" ]; then
    num_pdfs="$(find "${REPO_ROOT}/docs_RAG" -name '*.pdf' -type f 2>/dev/null | wc -l)"
  fi

  cat > "${manifest}" <<MANIFEST
{
  "tenant_id": "tenantA",
  "docs_dir": "/app/docs_RAG",
  "rag_store_dir": "/rag_store_tenants/tenantA",
  "embed_model_path": "/app/fastapi_runtime_assets/models/bge-base-en-v1.5",
  "embed_dim": 768,
  "chunk_size": 800,
  "chunk_overlap": 120,
  "num_pdf_files": ${num_pdfs},
  "kb_version": "pending_ingest"
}
MANIFEST

  ok "Created ${manifest}"
}

# ── Summary ──────────────────────────────────────────────────────────────
print_summary() {
  echo
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo -e "\033[1;32m  Setup complete!\033[0m"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo
  echo "  Next step — deploy the full stack:"
  echo
  echo "    cd $(basename "${KERNEL_ROOT}")"
  echo "    ./deploy.sh"
  echo
  echo "  Then chat:"
  echo
  echo "    python3 scripts/chat_cli.py --show-debug"
  echo
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

main() {
  echo
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  E2E GenAI Service — One-time Setup"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  preflight
  install_system_deps
  start_minikube
  download_models
  download_docs
  ensure_tenant_seed
  print_summary
}

main "$@"
