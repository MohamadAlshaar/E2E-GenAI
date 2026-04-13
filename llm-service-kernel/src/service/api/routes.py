from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from src.service.api.schemas import ChatRequest
from src.service.auth.deps import get_auth_context
from src.service.auth.types import AuthContext
from src.service.bootstrap import refresh_startup_checks

router = APIRouter()


def _body_to_dict(body: ChatRequest) -> dict:
    if hasattr(body, "model_dump"):
        return body.model_dump()
    if hasattr(body, "dict"):
        return body.dict()
    return dict(body)


def _current_checks(request: Request) -> dict:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        return getattr(request.app.state, "startup_checks", {}) or {}
    # Return cached checks — do not refresh on every probe (refresh_startup_checks
    # makes blocking network calls that freeze the event loop and cause liveness timeouts)
    return getattr(runtime, "startup_checks", {}) or {}


def _refreshed_checks(request: Request) -> dict:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        return getattr(request.app.state, "startup_checks", {}) or {}
    checks = refresh_startup_checks(runtime)
    request.app.state.startup_checks = checks
    return checks


@router.get("/health")
async def health(request: Request):
    s = request.app.state.settings
    sem_cache = getattr(request.app.state, "semantic_cache", None)
    rag_router = getattr(request.app.state, "rag_router", None)
    startup_checks = _current_checks(request)

    semcache_mongo = startup_checks.get("semantic_cache_mongo", {})
    semcache_milvus = startup_checks.get("semantic_cache_milvus", {})
    rag_milvus = startup_checks.get("rag_milvus", {})
    rag_manifest_root = startup_checks.get("rag_manifest_root_dir", {})
    rag_seed_manifest = startup_checks.get("rag_seed_manifest_dir", {})
    generation_backend = startup_checks.get("generation_backend", {})
    generation_status = startup_checks.get("generation_status", {})
    semantic_status = startup_checks.get("semantic_cache_status", {})
    rag_status = startup_checks.get("rag_status", {})
    ready_status = startup_checks.get("ready_status", {})

    return {
        "ok": True,
        "ready": bool(ready_status.get("ok")),
        "ready_reason": ready_status.get("reason"),
        "model_backend": s.MODEL_BACKEND,
        "model_base_url": s.model_base_url,
        "model": s.SERVED_MODEL_NAME,
        "auth_enabled": s.AUTH_ENABLED,
        "tenant_claim": s.TENANT_CLAIM,
        "exact_cache_enabled": s.EXACT_CACHE_ENABLED,
        "semantic_cache_enabled": s.SEM_CACHE_ENABLED,
        "semantic_cache_runtime_enabled": bool(
            sem_cache is not None and getattr(sem_cache, "enabled", False)
        ),
        "semantic_cache_init_error": getattr(sem_cache, "init_error", None)
        if sem_cache is not None
        else None,
        "semantic_cache_usable": bool(semantic_status.get("usable")),
        "semantic_cache_reason": semantic_status.get("reason"),
        "semantic_cache_mongo_reachable": semcache_mongo.get("reachable"),
        "semantic_cache_milvus_reachable": semcache_milvus.get("reachable"),
        "semantic_cache_collection_exists": semcache_milvus.get("collection_exists"),
        "semantic_cache_schema_ok": semcache_milvus.get("schema_ok"),
        "semantic_cache_collection": s.SEM_CACHE_MILVUS_COLLECTION,
        "rag_enabled": s.RAG_ENABLED,
        "rag_runtime_enabled": bool(rag_router is not None and getattr(rag_router, "enabled", False)),
        "rag_init_error": getattr(rag_router, "init_error", None) if rag_router is not None else None,
        "rag_usable": bool(rag_status.get("usable")),
        "rag_reason": rag_status.get("reason"),
        "rag_store_root_dir": s.RAG_STORE_ROOT_DIR,
        "rag_manifest_root_dir": s.RAG_MANIFEST_ROOT_DIR,
        "rag_manifest_root_exists": rag_manifest_root.get("exists"),
        "rag_manifest_root_non_empty": rag_manifest_root.get("non_empty"),
        "rag_seed_manifest_dir": rag_seed_manifest.get("path"),
        "rag_seed_manifest_exists": rag_seed_manifest.get("exists"),
        "rag_seed_manifest_non_empty": rag_seed_manifest.get("non_empty"),
        "rag_milvus_reachable": rag_milvus.get("reachable"),
        "rag_collection_exists": rag_milvus.get("collection_exists"),
        "rag_collection": s.MILVUS_COLLECTION,
        "rag_score_threshold": s.RAG_SCORE_THRESHOLD,
        "benchmark_shadow_mode": s.BENCHMARK_SHADOW_MODE,
        "rag_retrieve_every_request": s.RAG_RETRIEVE_EVERY_REQUEST,
        "cache_scope": s.CACHE_SCOPE,
        "dev_tenant_id": s.DEV_TENANT_ID,
        "generation_backend_reachable": generation_backend.get("reachable"),
        "generation_backend_status_code": generation_backend.get("status_code"),
        "generation_backend_reason": generation_backend.get("reason"),
        "generation_backend_usable": bool(generation_status.get("usable")),
        "generation_backend_usable_reason": generation_status.get("reason"),
        "startup_checks": startup_checks,
    }


@router.get("/ready")
async def ready(request: Request):
    startup_checks = _refreshed_checks(request)
    ready_status = startup_checks.get("ready_status", {}) or {}
    status_code = 200 if bool(ready_status.get("ok")) else 503

    payload = {
        "ok": bool(ready_status.get("ok")),
        "reason": ready_status.get("reason"),
        "generation_backend_usable": bool(
            (startup_checks.get("generation_status", {}) or {}).get("usable")
        ),
        "semantic_cache_usable": bool(
            (startup_checks.get("semantic_cache_status", {}) or {}).get("usable")
        ),
        "rag_usable": bool(
            (startup_checks.get("rag_status", {}) or {}).get("usable")
        ),
    }
    return JSONResponse(status_code=status_code, content=payload)


@router.get("/v1/models")
async def list_models(request: Request):
    s = request.app.state.settings
    return {
        "object": "list",
        "data": [
            {
                "id": s.SERVED_MODEL_NAME,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    body: ChatRequest,
    auth: AuthContext = Depends(get_auth_context),
):
    orch = request.app.state.orchestrator
    return await orch.handle_chat_completion(_body_to_dict(body), auth)


@router.post("/chat")
async def chat_alias(
    request: Request,
    body: ChatRequest,
    auth: AuthContext = Depends(get_auth_context),
):
    orch = request.app.state.orchestrator
    return await orch.handle_chat_completion(_body_to_dict(body), auth)
