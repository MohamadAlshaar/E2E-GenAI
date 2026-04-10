#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import hashlib
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]          # llm-service-kernel/
PROJECT_ROOT = REPO_ROOT.parent                          # LLM-end-to-end-Service-main/
DEFAULT_DATA_ROOT = PROJECT_ROOT                         # where docs_RAG_tenants 

DATA_ROOT = Path(os.getenv("DATA_ROOT", str(DEFAULT_DATA_ROOT)))

TENANT_DOCS_ROOT = Path(os.getenv("TENANT_DOCS_ROOT", str(DATA_ROOT / "docs_RAG_tenants")))
TENANT_STORE_ROOT = Path(os.getenv("TENANT_STORE_ROOT", str(DATA_ROOT / "rag_store_tenants")))

BUILD_SCRIPT = Path(os.getenv("BUILD_RAG_SCRIPT", str(REPO_ROOT / "scripts" / "build_rag_index.py")))

EMBED_MODEL_PATH = os.getenv("RAG_EMBED_MODEL_PATH", str(DATA_ROOT / "all-MiniLM-L6-v2"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
EMBED_BATCH_SIZE = int(os.getenv("RAG_EMBED_BATCH_SIZE", "8"))
EMBED_DIM = int(os.getenv("RAG_EMBED_DIM", "384"))

# : build only selected tenants
TENANTS_ENV = os.getenv("TENANTS", "").strip()  # "tenantA,tenantB"


def compute_kb_version(docs_dir: Path) -> str:
    pdf_files = sorted(docs_dir.rglob("*.pdf"))
    h = hashlib.sha256()
    for p in pdf_files:
        stat = p.stat()
        h.update(str(p.relative_to(docs_dir)).encode("utf-8"))
        h.update(str(stat.st_size).encode("utf-8"))
        h.update(str(int(stat.st_mtime)).encode("utf-8"))
    return h.hexdigest()[:16]


def load_manifest(store_dir: Path) -> Optional[dict]:
    mp = store_dir / "manifest.json"
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return None


def needs_rebuild(tenant: str, docs_dir: Path, store_dir: Path) -> bool:
    manifest = load_manifest(store_dir)
    if manifest is None:
        return True

    expected_kb = compute_kb_version(docs_dir)

    # Rebuild if KB changed or build config changed
    if manifest.get("kb_version") != expected_kb:
        return True
    if str(manifest.get("embed_model_path", "")) != str(EMBED_MODEL_PATH):
        return True
    if int(manifest.get("chunk_size", -1)) != CHUNK_SIZE:
        return True
    if int(manifest.get("chunk_overlap", -1)) != CHUNK_OVERLAP:
        return True
    if int(manifest.get("embed_dim", -1)) != EMBED_DIM:
        return True

    return False


def list_tenants() -> List[str]:
    if TENANTS_ENV:
        return [t.strip() for t in TENANTS_ENV.split(",") if t.strip()]
    if not TENANT_DOCS_ROOT.exists():
        return []
    return sorted([p.name for p in TENANT_DOCS_ROOT.iterdir() if p.is_dir()])


def build_one(tenant: str) -> int:
    docs_dir = TENANT_DOCS_ROOT / tenant
    store_dir = TENANT_STORE_ROOT / tenant
    if not docs_dir.exists():
        print(f"[skip] tenant={tenant}: docs dir missing: {docs_dir}")
        return 0

    store_dir.mkdir(parents=True, exist_ok=True)

    if not needs_rebuild(tenant, docs_dir, store_dir):
        print(f"[ok] tenant={tenant}: up-to-date (skip)")
        return 0

    print(f"[build] tenant={tenant}")
    print(f"       docs : {docs_dir}")
    print(f"       store: {store_dir}")

    env = os.environ.copy()
    env["RAG_DOCS_DIR"] = str(docs_dir)
    env["RAG_STORE_DIR"] = str(store_dir)
    env["RAG_EMBED_MODEL_PATH"] = str(EMBED_MODEL_PATH)
    env["RAG_CHUNK_SIZE"] = str(CHUNK_SIZE)
    env["RAG_CHUNK_OVERLAP"] = str(CHUNK_OVERLAP)
    env["RAG_EMBED_BATCH_SIZE"] = str(EMBED_BATCH_SIZE)
    env["RAG_EMBED_DIM"] = str(EMBED_DIM)

    # Call hte existing builder
    return subprocess.call([sys.executable, str(BUILD_SCRIPT)], env=env)


def main() -> int:
    TENANT_STORE_ROOT.mkdir(parents=True, exist_ok=True)
    tenants = list_tenants()
    if not tenants:
        print(f"[error] no tenants found under {TENANT_DOCS_ROOT} (or TENANTS not set)")
        return 2

    rc = 0
    for t in tenants:
        r = build_one(t)
        if r != 0:
            rc = r
    print(f"[done] stores under: {TENANT_STORE_ROOT}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

