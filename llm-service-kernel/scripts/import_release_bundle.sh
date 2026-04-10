#!/usr/bin/env bash
set -euo pipefail

KERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${KERNEL_ROOT}/.." && pwd)"

BUNDLE_DIR="${1:-${REPO_ROOT}/release-bundle}"
EXTRACT_REBUILD_INPUTS="${EXTRACT_REBUILD_INPUTS:-0}"

if [ "${2:-}" = "--extract-rebuild-inputs" ]; then
  EXTRACT_REBUILD_INPUTS=1
fi

IMAGES_DIR="${BUNDLE_DIR}/images"
DOCS_DIR="${BUNDLE_DIR}/docs"
MANIFESTS_DIR="${BUNDLE_DIR}/manifests"
REBUILD_INPUTS_DIR="${BUNDLE_DIR}/rebuild-inputs"

FASTAPI_IMAGE_TAR="${IMAGES_DIR}/llm-service-kernel-fastapi-selfcontained.tar"
STORAGE_IMAGE_TAR="${IMAGES_DIR}/storage-images.tar"
LLMD_LOCAL_DIR="${IMAGES_DIR}/llmd-local"
FASTAPI_BUILD_INPUTS_TGZ="${REBUILD_INPUTS_DIR}/fastapi-build-inputs.tgz"

log() {
  printf '[import_release_bundle] %s\n' "$*"
}

die() {
  printf '[import_release_bundle] ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

load_if_exists() {
  local tar_path="$1"
  if [ -f "$tar_path" ]; then
    log "Loading image tar: $tar_path"
    docker load -i "$tar_path"
  else
    log "Skipping missing tar: $tar_path"
  fi
}

extract_rebuild_inputs() {
  [ -f "${FASTAPI_BUILD_INPUTS_TGZ}" ] || die "rebuild inputs tar not found: ${FASTAPI_BUILD_INPUTS_TGZ}"
  log "Extracting rebuild inputs into ${KERNEL_ROOT}"
  tar -C "${KERNEL_ROOT}" -xzf "${FASTAPI_BUILD_INPUTS_TGZ}"
}

copy_docs_hint() {
  if [ -d "${DOCS_DIR}" ]; then
    log "Bundle docs available under: ${DOCS_DIR}"
  fi
  if [ -d "${MANIFESTS_DIR}" ]; then
    log "Bundle manifests available under: ${MANIFESTS_DIR}"
  fi
}

show_next_steps() {
  cat <<EOF

Next steps:
  cd ${KERNEL_ROOT}
  bash scripts/deploy_fastapi_fullstack.sh

Useful checks:
  kubectl get pods -n llm-service
  kubectl logs -n llm-service job/llm-service-kernel-bootstrap
  python scripts/chat_cli.py --tenant tenantC --show-debug
EOF
}

main() {
  require_cmd docker
  require_cmd tar

  [ -d "${BUNDLE_DIR}" ] || die "bundle directory not found: ${BUNDLE_DIR}"
  [ -d "${IMAGES_DIR}" ] || die "bundle images directory not found: ${IMAGES_DIR}"

  load_if_exists "${FASTAPI_IMAGE_TAR}"
  load_if_exists "${STORAGE_IMAGE_TAR}"

  if [ -d "${LLMD_LOCAL_DIR}" ]; then
    if [ -f "${LLMD_LOCAL_DIR}/gaie-images.tar" ]; then
      load_if_exists "${LLMD_LOCAL_DIR}/gaie-images.tar"
    fi

    if [ -d "${LLMD_LOCAL_DIR}/images" ]; then
      while IFS= read -r -d '' tar_file; do
        load_if_exists "${tar_file}"
      done < <(find "${LLMD_LOCAL_DIR}/images" -maxdepth 1 -type f -name '*.tar' -print0 | sort -z)
    fi

    log "llm-d local artifact directory present: ${LLMD_LOCAL_DIR}"
  fi

  if [ "${EXTRACT_REBUILD_INPUTS}" = "1" ]; then
    extract_rebuild_inputs
  else
    log "Skipping rebuild input extraction"
  fi

  copy_docs_hint
  show_next_steps
}

main "$@"
