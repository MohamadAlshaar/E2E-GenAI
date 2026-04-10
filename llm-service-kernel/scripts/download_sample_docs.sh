#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# download_sample_docs.sh — download sample arxiv/PDF papers for RAG ingestion
#
# Idempotent: skips PDFs that already exist.
# Can be run standalone or called from setup.sh.
#
# Environment overrides:
#   DOCS_DIR    where to place PDFs (default: $REPO_ROOT/docs_RAG)
# ---------------------------------------------------------------------------
set -euo pipefail

KERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${KERNEL_ROOT}/.." && pwd)"

DOCS_DIR="${DOCS_DIR:-${REPO_ROOT}/docs_RAG}"

log() { printf '[download_sample_docs] %s\n' "$*"; }

# arxiv_id → filename mapping
# Format: "arxiv_id|filename"
PAPERS=(
  "1508.07909v5|1508.07909v5.pdf"
  "1512.03385v1|1512.03385v1.pdf"
  "1607.06450v1|1607.06450v1.pdf"
  "1706.03762v7|1706.03762v7.pdf"
  "2001.08361v1|2001.08361v1.pdf"
  "2002.05202v1|2002.05202v1.pdf"
  "2003.05997v5|2003.05997v5.pdf"
  "2005.11401v4|2005.11401v4.pdf"
  "2101.03961v3|2101.03961v3.pdf"
  "2104.09864v5|2104.09864v5.pdf"
  "2108.07258v3|2108.07258v3.pdf"
  "2203.15556v1|2203.15556v1.pdf"
  "2205.14135v2|2205.14135v2.pdf"
  "2208.07339v2|2208.07339v2.pdf"
  "2209.00188v3|2209.00188v3.pdf"
  "2302.13971v1|2302.13971v1.pdf"
  "2309.06180v1|2309.06180v1.pdf"
  "2401.06066v1|2401.06066v1.pdf"
  "2401.09670v3|2401.09670v3.pdf"
  "2403.02310v3|2403.02310v3.pdf"
  "2403.02694v4|2403.02694v4.pdf"
  "2403.18702v2|2403.18702v2.pdf"
  "2404.19737v1|2404.19737v1.pdf"
  "2406.18219v3|2406.18219v3.pdf"
  "2407.20272v1|2407.20272v1.pdf"
  "2407.21783v3|2407.21783v3.pdf"
  "2409.14317v1|2409.14317v1.pdf"
  "2409.16626v2|2409.16626v2.pdf"
  "2412.19437v2|2412.19437v2.pdf"
  "2506.11789v1|2506.11789v1.pdf"
  "2508.18736v1|2508.18736v1.pdf"
  "2509.24626v1|2509.24626v1.pdf"
)

# ACM papers (non-arxiv, direct URL)
ACM_PAPERS=(
  "https://dl.acm.org/doi/pdf/10.1145/2150976.2150982|2150976.2150982.pdf"
  "https://dl.acm.org/doi/pdf/10.1145/2408776.2408794|2408776.2408794.pdf"
)

download_arxiv() {
  local arxiv_id="$1"
  local filename="$2"
  local out="${DOCS_DIR}/${filename}"

  if [ -f "${out}" ] && [ -s "${out}" ]; then
    return 0
  fi

  local url="https://arxiv.org/pdf/${arxiv_id}.pdf"
  log "Downloading ${url}"
  if curl -fsSL --retry 3 --retry-delay 2 -o "${out}" "${url}" 2>/dev/null; then
    return 0
  else
    log "WARN: failed to download ${arxiv_id} — skipping"
    rm -f "${out}"
    return 0
  fi
}

main() {
  mkdir -p "${DOCS_DIR}"

  local total=${#PAPERS[@]}
  local downloaded=0
  local skipped=0

  for entry in "${PAPERS[@]}"; do
    local arxiv_id="${entry%%|*}"
    local filename="${entry##*|}"

    if [ -f "${DOCS_DIR}/${filename}" ] && [ -s "${DOCS_DIR}/${filename}" ]; then
      skipped=$((skipped + 1))
    else
      download_arxiv "${arxiv_id}" "${filename}"
      downloaded=$((downloaded + 1))
      # Be polite to arxiv
      sleep 1
    fi
  done

  for entry in "${ACM_PAPERS[@]}"; do
    local url="${entry%%|*}"
    local filename="${entry##*|}"
    local out="${DOCS_DIR}/${filename}"

    if [ -f "${out}" ] && [ -s "${out}" ]; then
      skipped=$((skipped + 1))
    else
      log "Downloading ${url}"
      curl -fsSL --retry 3 --retry-delay 2 -o "${out}" "${url}" 2>/dev/null || {
        log "WARN: failed to download ${filename} — skipping"
        rm -f "${out}"
      }
      downloaded=$((downloaded + 1))
    fi
  done

  local final_count
  final_count="$(find "${DOCS_DIR}" -name '*.pdf' -type f | wc -l)"
  log "Done — ${final_count} PDFs in ${DOCS_DIR} (${downloaded} downloaded, ${skipped} already present)"
}

main "$@"
