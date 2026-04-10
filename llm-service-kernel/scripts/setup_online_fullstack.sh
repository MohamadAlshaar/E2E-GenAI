#!/usr/bin/env bash
set -euo pipefail

KERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${KERNEL_ROOT}/.." && pwd)"

MINIKUBE_PROFILE="${MINIKUBE_PROFILE:-minikube}"
FASTAPI_IMAGE="${FASTAPI_IMAGE:-llm-service-kernel:fastapi-selfcontained}"
MODEL_SOURCE_DIR="${MODEL_SOURCE_DIR:-${REPO_ROOT}/Qwen2.5-0.5B-Instruct}"

PREP_ASSETS="${PREP_ASSETS:-1}"
VENDOR_PYPDF="${VENDOR_PYPDF:-1}"

PULL_RUNTIME_IMAGES="${PULL_RUNTIME_IMAGES:-1}"
LOAD_RUNTIME_IMAGES_TO_MINIKUBE="${LOAD_RUNTIME_IMAGES_TO_MINIKUBE:-1}"
PULL_OPTIONAL_VLLM="${PULL_OPTIONAL_VLLM:-0}"

# If your repo contains llm-d tar files, leave this at 1.
# If you want to pull llm-d images from the internet instead, set PREPULL_LLMD_ONLINE=1
# and provide LLMD_ROUTING_IMAGE_SOURCE and LLMD_CUDA_IMAGE_SOURCE.
LOAD_LLMD_TARS="${LOAD_LLMD_TARS:-1}"
PREPULL_LLMD_ONLINE="${PREPULL_LLMD_ONLINE:-0}"
LLMD_ROUTING_IMAGE_SOURCE="${LLMD_ROUTING_IMAGE_SOURCE:-}"
LLMD_CUDA_IMAGE_SOURCE="${LLMD_CUDA_IMAGE_SOURCE:-}"

BUILD_FASTAPI_IMAGE="${BUILD_FASTAPI_IMAGE:-1}"
LOAD_FASTAPI_IMAGE_TO_MINIKUBE="${LOAD_FASTAPI_IMAGE_TO_MINIKUBE:-1}"
DEPLOY_LLMD="${DEPLOY_LLMD:-1}"
DEPLOY_FASTAPI="${DEPLOY_FASTAPI:-1}"

log() {
  printf '[setup_online_fullstack] %s\n' "$*"
}

die() {
  printf '[setup_online_fullstack] ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

image_exists_on_host() {
  docker image inspect "${FASTAPI_IMAGE}" >/dev/null 2>&1
}

image_exists_in_minikube() {
  minikube -p "${MINIKUBE_PROFILE}" ssh -- "docker image inspect '${FASTAPI_IMAGE}' >/dev/null 2>&1" >/dev/null 2>&1
}

build_fastapi_image_on_host() {
  log "Building FastAPI image on host Docker daemon: ${FASTAPI_IMAGE}"
  docker build -f Dockerfile.service -t "${FASTAPI_IMAGE}" .
}

load_fastapi_image_into_minikube() {
  log "Loading FastAPI image into minikube: ${FASTAPI_IMAGE}"
  minikube -p "${MINIKUBE_PROFILE}" image load "${FASTAPI_IMAGE}"
}

ensure_fastapi_image_available() {
  local host_has=0
  local mini_has=0

  if image_exists_on_host; then
    host_has=1
  fi

  if image_exists_in_minikube; then
    mini_has=1
  fi

  if [ "${BUILD_FASTAPI_IMAGE}" = "1" ]; then
    build_fastapi_image_on_host
    host_has=1
  else
    log "Skipping FastAPI image build"
  fi

  if [ "${LOAD_FASTAPI_IMAGE_TO_MINIKUBE}" = "1" ]; then
    if [ "${host_has}" = "1" ]; then
      load_fastapi_image_into_minikube
    elif [ "${mini_has}" = "1" ]; then
      log "FastAPI image already present in minikube; skipping image load"
    else
      die "FastAPI image '${FASTAPI_IMAGE}' not found on host or in minikube. Build it first or import/load it."
    fi
  else
    log "Skipping FastAPI image load into minikube"
  fi
}

require_cmd minikube
require_cmd docker
require_cmd kubectl

[ -d "${KERNEL_ROOT}" ] || die "kernel root not found: ${KERNEL_ROOT}"
[ -d "${MODEL_SOURCE_DIR}" ] || die "model source directory not found: ${MODEL_SOURCE_DIR}"

cd "${KERNEL_ROOT}"

if [ "${PULL_RUNTIME_IMAGES}" = "1" ]; then
  log "Preparing external runtime images"
  LOAD_TO_MINIKUBE="${LOAD_RUNTIME_IMAGES_TO_MINIKUBE}" \
  MINIKUBE_PROFILE="${MINIKUBE_PROFILE}" \
  PULL_OPTIONAL_VLLM="${PULL_OPTIONAL_VLLM}" \
  LOAD_LLMD_TARS="${LOAD_LLMD_TARS}" \
  PREPULL_LLMD_ONLINE="${PREPULL_LLMD_ONLINE}" \
  LLMD_ROUTING_IMAGE_SOURCE="${LLMD_ROUTING_IMAGE_SOURCE}" \
  LLMD_CUDA_IMAGE_SOURCE="${LLMD_CUDA_IMAGE_SOURCE}" \
  bash scripts/pull_required_images.sh
else
  log "Skipping runtime image preparation"
fi

if [ "${PREP_ASSETS}" = "1" ]; then
  log "Preparing FastAPI runtime assets"
  bash scripts/prepare_fastapi_runtime_assets.sh
else
  log "Skipping runtime asset preparation"
fi

if [ "${VENDOR_PYPDF}" = "1" ]; then
  log "Vendoring pypdf"
  bash scripts/vendor_local_pypdf.sh
else
  log "Skipping pypdf vendoring"
fi

ensure_fastapi_image_available

if [ "${DEPLOY_LLMD}" = "1" ]; then
  log "Deploying llm-d local backend"
  MODEL_SOURCE_DIR="${MODEL_SOURCE_DIR}" bash scripts/deploy_llmd_local.sh
else
  log "Skipping llm-d deployment"
fi

if [ "${DEPLOY_FASTAPI}" = "1" ]; then
  log "Deploying FastAPI fullstack"
  bash scripts/deploy_fastapi_fullstack.sh
else
  log "Skipping FastAPI deployment"
fi

cat <<MSG

Full-stack setup complete.

Suggested checks:
  kubectl get pods -n llm-d-local
  kubectl get pods -n llm-service
  kubectl port-forward -n llm-service svc/llm-service-kernel 8080:8080
  curl http://127.0.0.1:8080/health
  python scripts/chat_cli.py --base-url http://127.0.0.1:8080 --tenant tenantA --show-debug

MSG
