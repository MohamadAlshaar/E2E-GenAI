#!/usr/bin/env bash
set -euo pipefail

APP_MODULE="${APP_MODULE:-src.service.main:app}"
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8080}"

exec python -m uvicorn "${APP_MODULE}" --host "${APP_HOST}" --port "${APP_PORT}"

