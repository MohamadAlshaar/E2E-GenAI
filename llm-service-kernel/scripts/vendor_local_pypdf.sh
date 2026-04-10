#!/usr/bin/env bash
set -euo pipefail

KERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${KERNEL_ROOT}/vendor"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PYPDF_VERSION="${PYPDF_VERSION:-pypdf}"

VENDOR_MODE="${VENDOR_MODE:-auto}"   # auto | local | online
TMP_VENV_DIR="${KERNEL_ROOT}/.tmp_vendor_pypdf_venv"

log() {
  printf '[vendor_local_pypdf] %s\n' "$*"
}

die() {
  printf '[vendor_local_pypdf] ERROR: %s\n' "$*" >&2
  exit 1
}

have_python() {
  command -v "${PYTHON_BIN}" >/dev/null 2>&1
}

ensure_python() {
  have_python || die "python not found: ${PYTHON_BIN}"
  "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
print("ok")
PY
}

local_pypdf_available() {
  "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import pypdf
print(pypdf.__version__)
PY
}

resolve_local_pkg_dir() {
  "${PYTHON_BIN}" - <<'PY'
import inspect
from pathlib import Path
import pypdf
print(Path(inspect.getfile(pypdf)).resolve().parent)
PY
}

resolve_local_dist_info_dir() {
  "${PYTHON_BIN}" - <<'PY'
import site
from pathlib import Path

candidates = []

for root in site.getsitepackages():
    p = Path(root)
    if p.exists():
        candidates.extend(sorted(p.glob("pypdf-*.dist-info")))

user_site = Path(site.getusersitepackages())
if user_site.exists():
    candidates.extend(sorted(user_site.glob("pypdf-*.dist-info")))

if not candidates:
    raise SystemExit("Could not find pypdf-*.dist-info")

print(candidates[0].resolve())
PY
}

clean_vendor_dir() {
  rm -rf "${VENDOR_DIR}/pypdf" "${VENDOR_DIR}"/pypdf-*.dist-info
  mkdir -p "${VENDOR_DIR}"
}

copy_into_vendor() {
  local pkg_dir="$1"
  local dist_info_dir="$2"

  clean_vendor_dir
  cp -r "${pkg_dir}" "${VENDOR_DIR}/"
  cp -r "${dist_info_dir}" "${VENDOR_DIR}/"

  log "Vendored pypdf into ${VENDOR_DIR}"
  find "${VENDOR_DIR}" -maxdepth 2 | sort
}

vendor_from_local_install() {
  log "Using locally installed pypdf from ${PYTHON_BIN}"
  local pkg_dir
  local dist_info_dir

  pkg_dir="$(resolve_local_pkg_dir)"
  dist_info_dir="$(resolve_local_dist_info_dir)"

  [ -d "${pkg_dir}" ] || die "Resolved local pypdf package dir does not exist: ${pkg_dir}"
  [ -d "${dist_info_dir}" ] || die "Resolved local pypdf dist-info dir does not exist: ${dist_info_dir}"

  copy_into_vendor "${pkg_dir}" "${dist_info_dir}"
}

vendor_from_online_install() {
  ensure_python
  log "Using online mode: creating temporary venv and installing ${PYPDF_VERSION}"

  rm -rf "${TMP_VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${TMP_VENV_DIR}"

  local vpy="${TMP_VENV_DIR}/bin/python"
  local vpip="${TMP_VENV_DIR}/bin/pip"

  "${vpip}" install --upgrade pip >/dev/null
  "${vpip}" install "${PYPDF_VERSION}" >/dev/null

  local pkg_dir
  local dist_info_dir

  pkg_dir="$("${vpy}" - <<'PY'
import inspect
from pathlib import Path
import pypdf
print(Path(inspect.getfile(pypdf)).resolve().parent)
PY
)"

  dist_info_dir="$("${vpy}" - <<'PY'
import site
from pathlib import Path

candidates = []
for root in site.getsitepackages():
    p = Path(root)
    if p.exists():
        candidates.extend(sorted(p.glob("pypdf-*.dist-info")))

if not candidates:
    raise SystemExit("Could not find pypdf-*.dist-info in temporary venv")

print(candidates[0].resolve())
PY
)"

  copy_into_vendor "${pkg_dir}" "${dist_info_dir}"

  rm -rf "${TMP_VENV_DIR}"
}

main() {
  ensure_python

  case "${VENDOR_MODE}" in
    local)
      local_pypdf_available || die "VENDOR_MODE=local but pypdf is not installed for ${PYTHON_BIN}"
      vendor_from_local_install
      ;;
    online)
      vendor_from_online_install
      ;;
    auto)
      if local_pypdf_available; then
        vendor_from_local_install
      else
        log "Local pypdf not found; falling back to online install"
        vendor_from_online_install
      fi
      ;;
    *)
      die "Unsupported VENDOR_MODE=${VENDOR_MODE}. Use: auto | local | online"
      ;;
  esac
}

main "$@"
