#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup.sh — one-time machine setup for the E2E GenAI Service
#
# Run this once on a fresh GPU machine. It will:
#   1. Install NVIDIA drivers if missing
#   2. Install system dependencies (Docker, NVIDIA toolkit, minikube, etc.)
#   3. Start minikube with GPU passthrough
#   4. Deploy NVIDIA device plugin into minikube
#   5. Download required ML models from HuggingFace
#   6. Download sample PDF papers for RAG
#
# After setup completes, run:  ./deploy.sh
#
# Environment overrides:
#   SKIP_NVIDIA_DRIVER      set to 1 to skip NVIDIA driver installation
#   SKIP_HOST_BOOTSTRAP     set to 1 if system deps are already installed
#   SKIP_MINIKUBE_START     set to 1 if minikube is already running
#   SKIP_MODEL_DOWNLOAD     set to 1 if models are already present
#   SKIP_DOCS_DOWNLOAD      set to 1 if docs_RAG/ is already populated
#   MINIKUBE_CPUS           CPU count for minikube (default: 8)
#   MINIKUBE_MEMORY         memory in MB for minikube (default: 16384)
#   MINIKUBE_DISK_SIZE      disk for minikube (default: 80g)
#   NVIDIA_DRIVER_VERSION   specific driver version (default: auto-detect)
# ---------------------------------------------------------------------------
set -euo pipefail

KERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${KERNEL_ROOT}/.." && pwd)"
SCRIPTS_DIR="${KERNEL_ROOT}/scripts"

SKIP_NVIDIA_DRIVER="${SKIP_NVIDIA_DRIVER:-0}"
SKIP_HOST_BOOTSTRAP="${SKIP_HOST_BOOTSTRAP:-0}"
SKIP_MINIKUBE_START="${SKIP_MINIKUBE_START:-0}"
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"
SKIP_DOCS_DOWNLOAD="${SKIP_DOCS_DOWNLOAD:-0}"

NVIDIA_DRIVER_VERSION="${NVIDIA_DRIVER_VERSION:-}"
NVIDIA_DEVICE_PLUGIN_VERSION="${NVIDIA_DEVICE_PLUGIN_VERSION:-v0.18.2}"

log()  { printf '\n\033[1;34m[setup]\033[0m %s\n' "$*"; }
step() { printf '\n\033[1;36m━━━ %s ━━━\033[0m\n' "$*"; }
warn() { printf '\033[1;33m  ⚠\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[setup] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }
ok()   { printf '\033[1;32m  ✓\033[0m %s\n' "$*"; }

sudo_cmd() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

# ── Preflight checks ──────────────────────────────────────────────────────
preflight() {
  step "Preflight checks"

  # Check OS
  if [ -f /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    ok "OS: ${PRETTY_NAME:-${ID:-unknown}}"
  fi

  # Check for a GPU (lspci)
  if command -v lspci >/dev/null 2>&1; then
    local gpu_line
    gpu_line="$(lspci | grep -i 'nvidia' | head -1 || true)"
    if [ -n "${gpu_line}" ]; then
      ok "GPU hardware detected: ${gpu_line##*: }"
    else
      warn "No NVIDIA GPU detected in lspci — this service requires a GPU"
    fi
  fi

  # Check disk space (need ~15GB for models + images + minikube)
  local avail_gb
  avail_gb="$(df -BG --output=avail "${REPO_ROOT}" | tail -1 | tr -d ' G')"
  if [ "${avail_gb}" -lt 15 ]; then
    die "Only ${avail_gb}GB disk space available. Need at least 15GB for models, images, and minikube."
  fi
  ok "Disk: ${avail_gb}GB available"

  # Check RAM
  local total_mem_mb
  total_mem_mb="$(free -m | awk '/Mem:/ {print $2}')"
  if [ "${total_mem_mb}" -lt 12000 ]; then
    warn "Only ${total_mem_mb}MB RAM — minikube needs at least 16GB for the full stack"
  fi
  ok "RAM: ${total_mem_mb}MB"
}

# ── Step 1: NVIDIA driver ─────────────────────────────────────────────────
install_nvidia_driver() {
  step "Step 1/6: NVIDIA driver"

  if [ "${SKIP_NVIDIA_DRIVER}" = "1" ]; then
    log "Skipping NVIDIA driver install (SKIP_NVIDIA_DRIVER=1)"
    return 0
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_name driver_ver
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'unknown')"
    driver_ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo '?')"
    ok "NVIDIA driver already installed: ${gpu_name} (driver ${driver_ver})"
    return 0
  fi

  log "NVIDIA driver not found — installing automatically"

  # Ensure we can use sudo
  if [ "$(id -u)" -ne 0 ] && ! sudo -n true 2>/dev/null; then
    log "sudo access required to install NVIDIA drivers."
    log "You may be prompted for your password."
  fi

  # Install prerequisites
  sudo_cmd apt-get update -y
  sudo_cmd apt-get install -y --no-install-recommends \
    ubuntu-drivers-common pciutils software-properties-common

  if [ -n "${NVIDIA_DRIVER_VERSION}" ]; then
    # User specified a version
    log "Installing user-specified driver: nvidia-driver-${NVIDIA_DRIVER_VERSION}"
    sudo_cmd apt-get install -y "nvidia-driver-${NVIDIA_DRIVER_VERSION}"
  else
    # Auto-detect the best driver
    log "Auto-detecting best NVIDIA driver..."
    local recommended
    recommended="$(ubuntu-drivers devices 2>/dev/null | grep 'recommended' | awk '{print $3}' || true)"

    if [ -z "${recommended}" ]; then
      # Fallback: try to find any nvidia-driver package
      recommended="$(ubuntu-drivers list 2>/dev/null | grep 'nvidia-driver' | sort -V | tail -1 || true)"
    fi

    if [ -z "${recommended}" ]; then
      die "Could not auto-detect an NVIDIA driver. Install manually with: sudo apt install nvidia-driver-XXX"
    fi

    log "Installing recommended driver: ${recommended}"
    sudo_cmd apt-get install -y "${recommended}"
  fi

  # Load the driver module
  sudo_cmd modprobe nvidia 2>/dev/null || true

  # Verify
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    local gpu_name driver_ver
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'unknown')"
    driver_ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo '?')"
    ok "NVIDIA driver installed: ${gpu_name} (driver ${driver_ver})"
  else
    echo
    warn "NVIDIA driver installed but nvidia-smi not working yet."
    warn "A REBOOT is required to load the driver."
    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "\033[1;33m  Please reboot and re-run: ./setup.sh\033[0m"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo
    exit 0
  fi
}

# ── Step 2: System dependencies ──────────────────────────────────────────
install_system_deps() {
  step "Step 2/6: System dependencies"

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

  # Also check nvidia-ctk (container toolkit)
  if ! command -v nvidia-ctk >/dev/null 2>&1; then
    all_present=0
  fi

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

# ── Step 3: Start minikube ───────────────────────────────────────────────
start_minikube() {
  step "Step 3/6: Minikube cluster"

  if [ "${SKIP_MINIKUBE_START}" = "1" ]; then
    log "Skipping minikube start (SKIP_MINIKUBE_START=1)"
    return 0
  fi

  if minikube status -p minikube 2>/dev/null | grep -q "Running"; then
    ok "minikube already running"

    # Still ensure GPU is available inside minikube
    ensure_nvidia_device_plugin
    return 0
  fi

  log "Starting minikube with GPU passthrough"

  # Ensure docker is usable (may need newgrp after fresh install)
  if ! docker info >/dev/null 2>&1; then
    if sudo docker info >/dev/null 2>&1; then
      warn "Docker requires sudo — you may need to log out and back in, or run: newgrp docker"
      die "Docker not usable without sudo. Log out/in and re-run setup.sh"
    else
      die "Docker not running. Start it with: sudo systemctl start docker"
    fi
  fi

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

  # Deploy NVIDIA device plugin
  ensure_nvidia_device_plugin

  ok "minikube started with GPU"
}

# ── NVIDIA device plugin ─────────────────────────────────────────────────
ensure_nvidia_device_plugin() {
  if kubectl get daemonset nvidia-device-plugin-daemonset -n kube-system >/dev/null 2>&1; then
    # Check if it has ready pods
    local desired ready
    desired="$(kubectl get daemonset nvidia-device-plugin-daemonset -n kube-system -o jsonpath='{.status.desiredNumberScheduled}' 2>/dev/null || echo 0)"
    ready="$(kubectl get daemonset nvidia-device-plugin-daemonset -n kube-system -o jsonpath='{.status.numberReady}' 2>/dev/null || echo 0)"
    if [ "${ready}" -ge 1 ] && [ "${ready}" -ge "${desired}" ]; then
      ok "NVIDIA device plugin already running (${ready}/${desired} ready)"
      return 0
    fi
    log "NVIDIA device plugin exists but not ready (${ready}/${desired}) — waiting..."
  else
    log "Deploying NVIDIA device plugin ${NVIDIA_DEVICE_PLUGIN_VERSION}"
    kubectl apply -f "https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/${NVIDIA_DEVICE_PLUGIN_VERSION}/deployments/static/nvidia-device-plugin.yml"
  fi

  # Wait for it to be ready
  log "Waiting for NVIDIA device plugin to be ready..."
  local deadline=$((SECONDS + 120))
  while [ "${SECONDS}" -lt "${deadline}" ]; do
    local ready
    ready="$(kubectl get daemonset nvidia-device-plugin-daemonset -n kube-system -o jsonpath='{.status.numberReady}' 2>/dev/null || echo 0)"
    if [ "${ready}" -ge 1 ]; then
      ok "NVIDIA device plugin ready"

      # Verify GPU is allocatable
      local gpu_count
      gpu_count="$(kubectl get nodes -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}' 2>/dev/null || echo 0)"
      if [ "${gpu_count}" -ge 1 ]; then
        ok "GPU allocatable in cluster: ${gpu_count}"
      else
        warn "nvidia-device-plugin running but no GPUs allocatable yet — may need a moment"
      fi
      return 0
    fi
    sleep 5
  done

  warn "NVIDIA device plugin not ready after 120s — GPU workloads may fail"
}

# ── Step 4: Download models ──────────────────────────────────────────────
download_models() {
  step "Step 4/6: ML models"

  if [ "${SKIP_MODEL_DOWNLOAD}" = "1" ]; then
    log "Skipping model download (SKIP_MODEL_DOWNLOAD=1)"
    return 0
  fi

  # Check if all models are already present
  if [ -f "${REPO_ROOT}/bge-base-en-v1.5/config.json" ] && \
     [ -f "${REPO_ROOT}/Qwen2.5-0.5B-Instruct/config.json" ]; then
    ok "All models already present"
    return 0
  fi

  # Ensure pip/huggingface_hub are available
  if ! command -v python3 >/dev/null 2>&1; then
    log "Installing python3..."
    sudo_cmd apt-get install -y python3 python3-pip
  fi

  log "Downloading ML models from HuggingFace (~3.2GB)"
  bash "${SCRIPTS_DIR}/download_models.sh"
  ok "Models downloaded"
}

# ── Step 5: Download sample docs ─────────────────────────────────────────
download_docs() {
  step "Step 5/6: Sample documents"

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

# ── Step 6: Ensure rag_store_tenants seed ─────────────────────────────────
ensure_tenant_seed() {
  step "Step 6/6: Tenant seed"

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
  install_nvidia_driver
  install_system_deps
  start_minikube
  download_models
  download_docs
  ensure_tenant_seed
  print_summary
}

main "$@"
