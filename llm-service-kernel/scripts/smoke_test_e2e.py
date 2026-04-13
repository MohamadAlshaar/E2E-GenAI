#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
import uuid
import urllib.error
import urllib.request
from typing import Any


FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:18081").rstrip("/")
LLMD_BASE_URL = os.getenv("LLMD_BASE_URL", "").rstrip("/")
LLMD_API_MODE = os.getenv("LLMD_API_MODE", "completions").strip().lower()
MODEL_NAME = os.getenv("MODEL_NAME", "")
HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "30"))
RAG_TEST_PROMPT = os.getenv("RAG_TEST_PROMPT", "What does tenantA say about StarNUMA?")
NEGATIVE_RAG_PROMPT = os.getenv("NEGATIVE_RAG_PROMPT", "What is 2 plus 2?")


def record(ok: bool, label: str, detail: str = "") -> bool:
    if ok:
        print(f"PASS {label}" + (f" - {detail}" if detail else ""))
    else:
        print(f"FAIL {label}" + (f" - {detail}" if detail else ""))
    return ok


def http_json(method: str, url: str, payload: dict | None = None) -> tuple[int, Any, str | None]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, data=data, method=method.upper(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8")
            try:
                body = json.loads(raw)
            except Exception:
                body = raw
            return resp.status, body, None
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except Exception:
            body = raw
        return exc.code, body, str(exc)
    except Exception as exc:
        return 0, None, str(exc)


def recursive_find_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = recursive_find_key(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = recursive_find_key(item, key)
            if found is not None:
                return found
    return None


def route_taken(resp: Any) -> str | None:
    route = recursive_find_key(resp, "route_taken")
    if route is None and isinstance(resp, dict) and isinstance(resp.get("_route"), dict):
        route = resp["_route"].get("route_taken")
    return str(route) if route is not None else None


def rag_used(resp: Any) -> bool:
    used = recursive_find_key(resp, "rag_used")
    if used is not None:
        return bool(used)
    return route_taken(resp) in {"rag_plus_backend", "rag_plus_vllm"}


def sources_present(resp: Any) -> bool:
    sources = recursive_find_key(resp, "sources")
    return isinstance(sources, list) and len(sources) > 0


def make_chat_payload(prompt: str) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32,
        "temperature": 0,
    }


def make_completions_payload(prompt: str) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 16,
        "temperature": 0,
    }


def llmd_payload() -> tuple[str, dict[str, Any]]:
    if LLMD_API_MODE == "chat":
        return "/v1/chat/completions", make_chat_payload("Reply with exactly: ok")
    return "/v1/completions", make_completions_payload("Reply with exactly: ok")


def main() -> int:
    ok_all = True

    if LLMD_BASE_URL:
        status, _, err = http_json("GET", f"{LLMD_BASE_URL}/v1/models")
        ok_all &= record(status == 200, "llm-d /v1/models", err or f"status={status}")

        path, payload = llmd_payload()
        status, _, err = http_json("POST", f"{LLMD_BASE_URL}{path}", payload)
        ok_all &= record(status == 200, f"llm-d {path}", err or f"status={status}")

    status, health, err = http_json("GET", f"{FASTAPI_BASE_URL}/health")
    ok_all &= record(status == 200 and isinstance(health, dict) and health.get("ok") is True,
                     "FastAPI /health", err or f"status={status}")

    nonce_a = uuid.uuid4().hex[:8]
    backend_prompt = f"Backend-only validation nonce {nonce_a}. Reply exactly with backend-ok"

    status_a, resp_a, err_a = http_json(
        "POST",
        f"{FASTAPI_BASE_URL}/v1/chat/completions",
        make_chat_payload(backend_prompt),
    )
    route_a = route_taken(resp_a)
    ok_all &= record(
        status_a == 200 and route_a in {"plain_backend", "plain_vllm", "rag_plus_backend", "rag_plus_vllm"},
        "Case A backend only",
        err_a or f"status={status_a}, route={route_a}",
    )

    nonce_b = uuid.uuid4().hex[:8]
    sem_prompt = f"Semantic cache validation nonce {nonce_b}. Reply exactly with cache-ok"

    status_b1, resp_b1, err_b1 = http_json(
        "POST",
        f"{FASTAPI_BASE_URL}/v1/chat/completions",
        make_chat_payload(sem_prompt),
    )
    status_b2, resp_b2, err_b2 = http_json(
        "POST",
        f"{FASTAPI_BASE_URL}/v1/chat/completions",
        make_chat_payload(sem_prompt),
    )
    route_b1 = route_taken(resp_b1)
    route_b2 = route_taken(resp_b2)

    ok_all &= record(
        status_b1 == 200 and route_b1 in {"plain_backend", "plain_vllm", "rag_plus_backend", "rag_plus_vllm"},
        "Case B semantic cache first call",
        err_b1 or f"status={status_b1}, route={route_b1}",
    )
    ok_all &= record(
        status_b2 == 200 and route_b2 == "semantic_cache",
        "Case B semantic cache second call",
        err_b2 or f"status={status_b2}, route={route_b2}",
    )

    health_dict = health if isinstance(health, dict) else {}
    rag_should_run = bool(health_dict.get("rag_enabled")) and bool(health_dict.get("rag_runtime_enabled")) and bool(health_dict.get("rag_collection_exists"))

    if rag_should_run:
        status_c, resp_c, err_c = http_json(
            "POST",
            f"{FASTAPI_BASE_URL}/v1/chat/completions",
            make_chat_payload(RAG_TEST_PROMPT),
        )
        ok_all &= record(
            status_c == 200 and rag_used(resp_c) and sources_present(resp_c),
            "Case C RAG positive",
            err_c or f"status={status_c}, route={route_taken(resp_c)}, sources={sources_present(resp_c)}",
        )

        status_d, resp_d, err_d = http_json(
            "POST",
            f"{FASTAPI_BASE_URL}/v1/chat/completions",
            make_chat_payload(NEGATIVE_RAG_PROMPT),
        )
        route_d = route_taken(resp_d)
        ok_all &= record(
            status_d == 200 and route_d in {"plain_backend", "plain_vllm", "semantic_cache"},
            "Case D RAG negative",
            err_d or f"status={status_d}, route={route_d}",
        )
    else:
        print("PASS Case C RAG positive - skipped (RAG not provisioned)")
        print("PASS Case D RAG negative - skipped (RAG not provisioned)")

    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())