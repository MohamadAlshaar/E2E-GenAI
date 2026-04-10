#!/usr/bin/env bash
set -euo pipefail

KERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${KERNEL_ROOT}/.." && pwd)"

BUNDLE_DIR="${1:-${REPO_ROOT}/release-bundle}"
IMAGES_DIR="${BUNDLE_DIR}/images"
MANIFESTS_DIR="${BUNDLE_DIR}/manifests"
DOCS_DIR="${BUNDLE_DIR}/docs"
REBUILD_INPUTS_DIR="${BUNDLE_DIR}/rebuild-inputs"

FASTAPI_IMAGE="${FASTAPI_IMAGE:-llm-service-kernel:fastapi-selfcontained}"
FASTAPI_IMAGE_TAR="${IMAGES_DIR}/llm-service-kernel-fastapi-selfcontained.tar"
STORAGE_IMAGE_TAR="${IMAGES_DIR}/storage-images.tar"

EXPORT_LLMD_LOCAL="${EXPORT_LLMD_LOCAL:-1}"
PACK_REBUILD_INPUTS="${PACK_REBUILD_INPUTS:-1}"

STORAGE_IMAGES=(
  "mongo:8"
  "quay.io/coreos/etcd:v3.5.18"
  "minio/minio:RELEASE.2023-03-20T20-16-18Z"
  "milvusdb/milvus:v2.6.11"
  "chrislusf/seaweedfs:latest"
)

log() {
  printf '[export_release_bundle] %s\n' "$*"
}

die() {
  printf '[export_release_bundle] ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

check_image_exists() {
  local image="$1"
  docker image inspect "$image" >/dev/null 2>&1 || die "docker image not found locally: $image"
}

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -e "$src" ]; then
    cp -r "$src" "$dst"
  fi
}

write_readme() {
  cat > "${BUNDLE_DIR}/README.txt" <<'EOF'
This bundle contains:
- FastAPI image tar
- storage image tar
- optional llm-d local artifacts
- Kubernetes manifests
- runbook / docs
- optional rebuild inputs (fastapi_runtime_assets + vendor)

Typical target-machine flow:
1. Copy repo and release-bundle to the target machine.
2. Load images from release-bundle/images/.
3. Optionally extract rebuild inputs.
4. Deploy with:
   bash scripts/deploy_fastapi_fullstack.sh
5. Test with:
   python scripts/chat_cli.py --tenant tenantC --show-debug
EOF
}

write_target_machine_steps() {
  cat > "${BUNDLE_DIR}/TARGET_MACHINE_STEPS.txt" <<'EOF'
1. Copy the repo and release-bundle to the target machine.

2. Go to:
   cd /path/to/LLM-end-to-end-Service-main/llm-service-kernel

3. Import bundle:
   bash scripts/import_release_bundle.sh /path/to/release-bundle --extract-rebuild-inputs

4. Deploy:
   bash scripts/deploy_fastapi_fullstack.sh

5. Verify:
   kubectl get pods -n llm-service
   kubectl logs -n llm-service job/llm-service-kernel-bootstrap

6. Test:
   python scripts/chat_cli.py --tenant tenantC --show-debug
EOF
}

main() {
  require_cmd docker
  require_cmd tar
  require_cmd find

  check_image_exists "${FASTAPI_IMAGE}"
  for image in "${STORAGE_IMAGES[@]}"; do
    check_image_exists "$image"
  done

  rm -rf "${BUNDLE_DIR}"
  mkdir -p "${IMAGES_DIR}" "${MANIFESTS_DIR}" "${DOCS_DIR}" "${REBUILD_INPUTS_DIR}"

  log "Copying docs and manifests"
  copy_if_exists "${KERNEL_ROOT}/RUNBOOK_SINGLE_NODE_FULLSTACK.md" "${DOCS_DIR}/"
  copy_if_exists "${KERNEL_ROOT}/DEPLOYMENT_REQUIREMENTS.md" "${DOCS_DIR}/"
  copy_if_exists "${KERNEL_ROOT}/RELEASE_CHECKLIST.md" "${DOCS_DIR}/"
  cp -r "${KERNEL_ROOT}/deploy" "${MANIFESTS_DIR}/"

  if [ "${PACK_REBUILD_INPUTS}" = "1" ]; then
    log "Packing rebuild inputs"
    if [ -d "${KERNEL_ROOT}/fastapi_runtime_assets" ] && [ -d "${KERNEL_ROOT}/vendor" ]; then
      tar -C "${KERNEL_ROOT}" -czf "${REBUILD_INPUTS_DIR}/fastapi-build-inputs.tgz" fastapi_runtime_assets vendor
    else
      log "Skipping rebuild inputs tar because fastapi_runtime_assets/ or vendor/ is missing"
    fi
  fi

  log "Exporting FastAPI image: ${FASTAPI_IMAGE}"
  docker save -o "${FASTAPI_IMAGE_TAR}" "${FASTAPI_IMAGE}"

  log "Exporting storage images"
  docker save -o "${STORAGE_IMAGE_TAR}" "${STORAGE_IMAGES[@]}"

  if [ "${EXPORT_LLMD_LOCAL}" = "1" ] && [ -d "${KERNEL_ROOT}/deploy/llmd-local" ]; then
    log "Copying llm-d local artifacts"
    mkdir -p "${IMAGES_DIR}/llmd-local"
    cp -r "${KERNEL_ROOT}/deploy/llmd-local/." "${IMAGES_DIR}/llmd-local/"
  else
    log "Skipping llm-d local artifact copy"
  fi

  write_readme
  write_target_machine_steps

  log "Bundle created at: ${BUNDLE_DIR}"
  find "${BUNDLE_DIR}" -maxdepth 3 | sort
  du -sh "${BUNDLE_DIR}"
}

main "$@"
