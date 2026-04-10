#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_ROOT="${REPO_ROOT}/llm-service-kernel"
VENV_DIR="${REPO_ROOT}/LLM"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip setuptools wheel
pip install -r "${APP_ROOT}/requirements-service.txt"

mkdir -p "${APP_ROOT}/state"

if [ ! -f "${APP_ROOT}/.env" ] && [ -f "${APP_ROOT}/.env.example" ]; then
  cp "${APP_ROOT}/.env.example" "${APP_ROOT}/.env"
fi

echo "Bootstrap complete."
echo "Activate with: source ${VENV_DIR}/bin/activate"
