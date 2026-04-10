#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
from typing import List

import faiss
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

REPO_ROOT = Path(__file__).resolve().parents[2]

DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", REPO_ROOT / "docs_RAG"))
RAG_STORE_DIR = Path(os.getenv("RAG_STORE_DIR", REPO_ROOT / "rag_store"))
EMBED_MODEL_PATH = os.getenv(
    "RAG_EMBED_MODEL_PATH",
    str(REPO_ROOT / "all-MiniLM-L6-v2"),
)

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
EMBED_BATCH_SIZE = int(os.getenv("RAG_EMBED_BATCH_SIZE", "8"))
EMBED_DIM = int(os.getenv("RAG_EMBED_DIM", "384"))  # all-MiniLM-L6-v2 = 384

FAISS_INDEX_PATH = RAG_STORE_DIR / "default__vector_store.json"
MANIFEST_PATH = RAG_STORE_DIR / "manifest.json"


def compute_kb_version(pdf_files: List[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(pdf_files):
        stat = p.stat()
        h.update(str(p.relative_to(DOCS_DIR)).encode("utf-8"))
        h.update(str(stat.st_size).encode("utf-8"))
        h.update(str(int(stat.st_mtime)).encode("utf-8"))
    return h.hexdigest()[:16]


def main() -> None:
    if not DOCS_DIR.exists():
        raise SystemExit(f"docs dir does not exist: {DOCS_DIR}")

    RAG_STORE_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(DOCS_DIR.rglob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"no PDFs found in {DOCS_DIR}")

    print(f"[RAG] Found {len(pdf_files)} PDF files in {DOCS_DIR}")

    # Local embedding model
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=str(EMBED_MODEL_PATH),
        device="cpu",
        embed_batch_size=EMBED_BATCH_SIZE,
    )

    # Chunking
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # Load PDFs
    documents = SimpleDirectoryReader(
        input_dir=str(DOCS_DIR),
        required_exts=[".pdf"],
        recursive=True,
        filename_as_id=True,
        raise_on_error=False,
    ).load_data()

    if not documents:
        raise SystemExit("no documents were loaded")

    print(f"[RAG] Loaded {len(documents)} document objects")

    # New FAISS index
    faiss_index = faiss.IndexFlatL2(EMBED_DIM)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build vector index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    # Persist all storage pieces to disk
    index.storage_context.persist(persist_dir=str(RAG_STORE_DIR))

    kb_version = compute_kb_version(pdf_files)

    manifest = {
        "docs_dir": str(DOCS_DIR),
        "rag_store_dir": str(RAG_STORE_DIR),
        "embed_model_path": str(EMBED_MODEL_PATH),
        "embed_dim": EMBED_DIM,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "num_pdf_files": len(pdf_files),
        "num_documents": len(documents),
        "kb_version": kb_version,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[RAG] Index build complete")
    print(f"[RAG] Persist dir: {RAG_STORE_DIR}")
    print(f"[RAG] kb_version: {kb_version}")
    print(f"[RAG] Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
