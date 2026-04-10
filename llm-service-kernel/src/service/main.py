from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.service.api.routes import router as api_router
from src.service.bootstrap import build_runtime, shutdown_runtime
from src.service.observability.viz import router as viz_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = build_runtime()

    app.state.runtime = runtime
    app.state.settings = runtime.settings
    app.state.vllm = runtime.vllm
    app.state.exact_cache = runtime.exact_cache
    app.state.semantic_cache = runtime.semantic_cache
    app.state.rag_router = runtime.rag_router
    app.state.auth_verifier = runtime.auth_verifier
    app.state.orchestrator = runtime.orchestrator
    app.state.startup_checks = runtime.startup_checks

    yield

    await shutdown_runtime(runtime)


app = FastAPI(lifespan=lifespan)
app.include_router(api_router)
app.include_router(viz_router)