#!/usr/bin/env bash
set -euo pipefail

SERVICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$SERVICE_ROOT/.." && pwd)"
REMOTE_ROOT="${1:-/mnt/llm-host}"

echo "Mounting host project into Minikube:"
echo "  host:   $PROJECT_ROOT"
echo "  remote: $REMOTE_ROOT"
echo
echo "Keep this terminal open while using the fullstack profile."
exec minikube mount "$PROJECT_ROOT:$REMOTE_ROOT"
