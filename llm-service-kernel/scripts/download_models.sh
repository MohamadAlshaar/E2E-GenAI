#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# download_models.sh — download HuggingFace models required by the service
#
# Idempotent: skips any model whose directory already contains config.json.
# Can be run standalone or called from setup.sh.
#
# Environment overrides:
#   MODELS_DIR          where to place model dirs (default: repo root)
#   HF_TOKEN            optional HuggingFace token for gated models
#   DOWNLOAD_MINILM     set to 1 to also download all-MiniLM-L6-v2 (not needed by default)
#   SKIP_BGE            set to 1 to skip bge-base-en-v1.5
#   SKIP_LLM            set to 1 to skip the LLM download (MODEL_NAME from deploy/model.env)
# ---------------------------------------------------------------------------
set -euo pipefail

KERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${KERNEL_ROOT}/.." && pwd)"

# ── Load model config (single source of truth) ───────────────────────────────
MODEL_ENV="${KERNEL_ROOT}/deploy/model.env"
if [ -f "${MODEL_ENV}" ]; then
  # shellcheck source=deploy/model.env
  set -a; source "${MODEL_ENV}"; set +a
fi

MODELS_DIR="${MODELS_DIR:-${REPO_ROOT}}"
HF_TOKEN="${HF_TOKEN:-}"

# MiniLM is no longer required — BGE is used for both RAG and semantic cache.
# Set DOWNLOAD_MINILM=1 if you want to use MiniLM as the semantic cache model instead.
DOWNLOAD_MINILM="${DOWNLOAD_MINILM:-0}"
SKIP_BGE="${SKIP_BGE:-0}"
SKIP_LLM="${SKIP_LLM:-0}"

log() { printf '[download_models] %s\n' "$*"; }
die() { printf '[download_models] ERROR: %s\n' "$*" >&2; exit 1; }

require_python() {
  command -v python3 >/dev/null 2>&1 || die "python3 not found"
}

ensure_hf_hub() {
  if ! python3 -c "import huggingface_hub" >/dev/null 2>&1; then
    log "Installing huggingface_hub..."
    python3 -m pip install --user --quiet huggingface_hub
  fi
}

# download_model <repo_id> <local_dir> [allow_patterns...]
download_model() {
  local repo_id="$1"
  local local_dir="$2"
  shift 2
  local patterns=("$@")

  if [ -f "${local_dir}/config.json" ] || [ -f "${local_dir}/tokenizer.json" ]; then
    log "${repo_id} already present at ${local_dir} — skipping"
    return 0
  fi

  log "Downloading ${repo_id} → ${local_dir}"
  mkdir -p "${local_dir}"

  local token_arg=""
  if [ -n "${HF_TOKEN}" ]; then
    token_arg="token='${HF_TOKEN}',"
  fi

  local patterns_arg="None"
  if [ ${#patterns[@]} -gt 0 ]; then
    patterns_arg="["
    for p in "${patterns[@]}"; do
      patterns_arg+="'${p}',"
    done
    patterns_arg+="]"
  fi

  python3 - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${repo_id}",
    local_dir="${local_dir}",
    local_dir_use_symlinks=False,
    ${token_arg}
    allow_patterns=${patterns_arg},
)
print("done")
PYEOF

  log "${repo_id} downloaded successfully"
}

main() {
  require_python
  ensure_hf_hub

  # MiniLM is optional — only downloaded if explicitly requested.
  # BGE handles both RAG embeddings and semantic cache by default.
  if [ "${DOWNLOAD_MINILM}" = "1" ]; then
    download_model "sentence-transformers/all-MiniLM-L6-v2" "${MODELS_DIR}/all-MiniLM-L6-v2" \
      "*.json" "*.txt" "*.safetensors" "*.bin" "1_Pooling/*" "modules.json"
  fi

  if [ "${SKIP_BGE}" != "1" ]; then
    download_model "BAAI/bge-base-en-v1.5" "${MODELS_DIR}/bge-base-en-v1.5" \
      "*.json" "*.txt" "*.safetensors" "*.bin" "1_Pooling/*" "modules.json"
  fi

  if [ "${SKIP_LLM}" != "1" ]; then
    local llm_repo="${MODEL_HF_REPO:-Qwen/Qwen2.5-7B-Instruct}"
    local llm_dir="${MODELS_DIR}/${MODEL_NAME:-Qwen2.5-7B-Instruct}"
    download_model "${llm_repo}" "${llm_dir}" \
      "*.json" "*.txt" "*.safetensors" "*.bin" "merges.txt"
  fi

  log "All models ready under ${MODELS_DIR}"
}

main "$@"
