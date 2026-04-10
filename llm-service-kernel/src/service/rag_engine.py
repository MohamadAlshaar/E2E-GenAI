######### not used ############
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore


class LocalRAG:
    def __init__(self):
        repo_root = Path(__file__).resolve().parents[3]

        self.rag_store_dir = Path(
            os.getenv("RAG_STORE_DIR", str(repo_root / "rag_store"))
        )
        self.embed_model_path = os.getenv(
            "RAG_EMBED_MODEL_PATH",
            str(repo_root / "bge-base-en-v1.5"),
        )
        self.top_k = int(os.getenv("RAG_TOP_K", "4"))

        manifest_path = self.rag_store_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"RAG manifest not found: {manifest_path}. Run build_rag_index.py first."
            )

        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.kb_version = self.manifest.get("kb_version", "unknown")

        Settings.embed_model = HuggingFaceEmbedding(
            model_name=str(self.embed_model_path),
            device="cpu",
            embed_batch_size=8,
        )

        vector_store = FaissVectorStore.from_persist_dir(str(self.rag_store_dir))
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=str(self.rag_store_dir),
        )

        self.index = load_index_from_storage(storage_context=storage_context)
        self.retriever = self.index.as_retriever(similarity_top_k=self.top_k)

    def retrieve(self, query: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        t0 = time.perf_counter()
        nodes = self.retriever.retrieve(query)
        retrieve_ms = (time.perf_counter() - t0) * 1000.0

        out: List[Dict[str, Any]] = []
        for rank, node in enumerate(nodes, start=1):
            metadata = dict(getattr(node, "metadata", {}) or {})
            text = node.get_text().strip()

            out.append(
                {
                    "rank": rank,
                    "score": float(getattr(node, "score", 0.0) or 0.0),
                    "text": text,
                    "metadata": metadata,
                }
            )

        meta = {
            "retrieve_ms": retrieve_ms,
            "num_chunks": len(out),
            "top_score": float(out[0]["score"]) if out else 0.0,
        }
        return out, meta

    def format_context(self, items: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for item in items:
            meta = item["metadata"]
            source = (
                meta.get("file_name")
                or meta.get("filename")
                or meta.get("source")
                or "unknown"
            )
            page = meta.get("page_label") or meta.get("page") or "?"
            parts.append(
                f"[Source {item['rank']}: {source}, page {page}]\n{item['text']}"
            )
        return "\n\n".join(parts)
