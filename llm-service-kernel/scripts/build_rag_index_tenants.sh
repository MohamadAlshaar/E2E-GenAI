#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="/home/mohamad/LLM-end-to-end-Service-main"

EMBED_MODEL_PATH="${RAG_EMBED_MODEL_PATH:-${DATA_ROOT}/all-MiniLM-L6-v2}"

TENANT_DOCS_ROOT="${TENANT_DOCS_ROOT:-${DATA_ROOT}/docs_RAG_tenants}"
TENANT_STORE_ROOT="${TENANT_STORE_ROOT:-${DATA_ROOT}/rag_store_tenants}"

# Reuse your existing build script
BUILD_PY="${REPO_ROOT}/scripts/build_rag_index.py"

build_one () {
  local tenant="$1"
  local docs_dir="${TENANT_DOCS_ROOT}/${tenant}"
  local store_dir="${TENANT_STORE_ROOT}/${tenant}"

  echo
  echo "=============================="
  echo "[tenant] ${tenant}"
  echo "[docs ] ${docs_dir}"
  echo "[store] ${store_dir}"
  echo "=============================="

  export RAG_DOCS_DIR="${docs_dir}"
  export RAG_STORE_DIR="${store_dir}"
  export RAG_EMBED_MODEL_PATH="${EMBED_MODEL_PATH}"

  python3 "${BUILD_PY}"
}

mkdir -p "${TENANT_STORE_ROOT}"

build_one "tenantA"
build_one "tenantB"

echo
echo "[OK] Built tenant indices under: ${TENANT_STORE_ROOT}"
