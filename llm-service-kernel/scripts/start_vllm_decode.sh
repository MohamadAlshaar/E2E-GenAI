#!/usr/bin/env bash
set -euo pipefail

append_path_if_exists() {
  local var_name="$1"
  local path_value="$2"

  [ -n "${path_value}" ] || return 0
  [ -d "${path_value}" ] || return 0

  local current="${!var_name:-}"
  if [ -z "${current}" ]; then
    export "${var_name}=${path_value}"
  else
    case ":${current}:" in
      *":${path_value}:"*) ;;
      *) export "${var_name}=${current}:${path_value}" ;;
    esac
  fi
}

append_path_if_exists LD_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu"
append_path_if_exists LD_LIBRARY_PATH "/usr/local/cuda/lib64"
append_path_if_exists LD_LIBRARY_PATH "/usr/local/cuda/compat"
append_path_if_exists LD_LIBRARY_PATH "/usr/local/cuda-12.9/compat"
append_path_if_exists LD_LIBRARY_PATH "/usr/local/nvidia/lib"
append_path_if_exists LD_LIBRARY_PATH "/usr/local/nvidia/lib64"
append_path_if_exists LD_LIBRARY_PATH "/opt/nvshmem-v3.5.19-1/lib"

append_path_if_exists LIBRARY_PATH "/usr/lib/x86_64-linux-gnu"
append_path_if_exists LIBRARY_PATH "/usr/local/cuda/lib64"
append_path_if_exists LIBRARY_PATH "/usr/local/cuda/compat"
append_path_if_exists LIBRARY_PATH "/usr/local/cuda-12.9/compat"
append_path_if_exists LIBRARY_PATH "/usr/local/nvidia/lib"
append_path_if_exists LIBRARY_PATH "/usr/local/nvidia/lib64"
append_path_if_exists LIBRARY_PATH "/opt/nvshmem-v3.5.19-1/lib"

echo "[start_vllm_decode] LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" >&2
echo "[start_vllm_decode] LIBRARY_PATH=${LIBRARY_PATH:-}" >&2

exec vllm serve /model-cache \
  --host 0.0.0.0 \
  --port 8200 \
  --served-model-name qwen2.5-0.5b \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager