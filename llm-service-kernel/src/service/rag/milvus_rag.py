from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from pymilvus import MilvusClient

from src.service.embeddings.bge import BGEEmbedder
from src.service.rag.seaweed_chunk_store import SeaweedChunkStore
from src.service.utils.hashing import sha256_text


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class MilvusTenantRAG:
    """
    Milvus RAG retriever.

    Backward-compatible behavior:
      - If the collection has inline `text`, use it directly.
      - If the collection has `object_key`, fetch chunk text from SeaweedFS.
      - If the collection has `pdf_object_key`, surface it in metadata/debug output.
      - Only request output fields that actually exist in the collection schema.
    """

    def __init__(
        self,
        *,
        tenant_id: str,
        milvus: MilvusClient,
        collection: str,
        embedder: BGEEmbedder,
        top_k: int,
        vector_field: str = "embedding",
        tenant_field: str = "tenant_id",
        metric_type: str = "COSINE",
        search_params: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
    ):
        self.tenant_id = tenant_id
        self._milvus = milvus
        self._collection = collection
        self._embedder = embedder
        self._top_k = int(top_k)
        self._vector_field = vector_field
        self._tenant_field = tenant_field
        self._metric_type = metric_type.upper()
        self._search_params = search_params or {"ef": 64}

        self.kb_version: str = "milvus"
        self._chunk_store = SeaweedChunkStore.from_env()

        self._available_fields = self._discover_collection_fields()
        self._output_fields = self._build_output_fields(output_fields)

    def _discover_collection_fields(self) -> Set[str]:
        desc = self._milvus.describe_collection(collection_name=self._collection)
        fields = desc.get("fields", []) or []
        out: Set[str] = set()
        for f in fields:
            name = f.get("name")
            if name:
                out.add(str(name))
        return out

    def _build_output_fields(self, requested: Optional[List[str]]) -> List[str]:
        preferred = requested or [
            "source",
            "page",
            "text",
            "chunk_id",
            "object_key",
            "pdf_object_key",
            "text_sha256",
        ]
        return [f for f in preferred if f in self._available_fields]

    def retrieve(self, query: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        t0 = time.perf_counter()
        qvec = self._embedder.embed_query(query)

        flt = f'{self._tenant_field} == "{self.tenant_id}"'

        res = self._milvus.search(
            collection_name=self._collection,
            data=[qvec],
            filter=flt,
            limit=self._top_k,
            output_fields=self._output_fields,
            search_params={
                "metric_type": self._metric_type,
                "params": dict(self._search_params),
            },
            anns_field=self._vector_field,
        )

        retrieve_ms = (time.perf_counter() - t0) * 1000.0
        hits = res[0] if isinstance(res, list) and res and isinstance(res[0], list) else res

        out: List[Dict[str, Any]] = []
        for rank, h in enumerate(hits, start=1):
            if isinstance(h, dict):
                score = h.get("score", h.get("distance", 0.0))
                entity = h.get("entity", h)
            else:
                score = getattr(h, "score", getattr(h, "distance", 0.0))
                entity = getattr(h, "entity", {}) or {}

            src = entity.get("source", "unknown")
            page = entity.get("page", "?")
            chunk_id = entity.get("chunk_id", "")
            object_key = entity.get("object_key", "")
            pdf_object_key = entity.get("pdf_object_key", "")
            text = (entity.get("text", "") or "").strip()

            if not text and object_key:
                chunk_payload = self._chunk_store.get_chunk(str(object_key))
                if chunk_payload is not None:
                    text = str(chunk_payload.get("text") or "").strip()
                    src = chunk_payload.get("source") or src
                    page = chunk_payload.get("page") or page
                    chunk_id = chunk_payload.get("chunk_id") or chunk_id
                    pdf_object_key = chunk_payload.get("pdf_object_key") or pdf_object_key

            out.append(
                {
                    "rank": rank,
                    "score": float(score) if score is not None else 0.0,
                    "text": text,
                    "metadata": {
                        "file_name": src,
                        "page_label": page,
                        "chunk_id": chunk_id,
                        "object_key": object_key,
                        "pdf_object_key": pdf_object_key,
                        "tenant_id": self.tenant_id,
                    },
                }
            )

        fp_parts: List[str] = []
        for item in out:
            md = item["metadata"]
            text = str(item.get("text") or "")
            object_key = str(md.get("object_key") or "")
            text_part = sha256_text(text) if text else object_key or "no_text"
            fp_parts.append(f'{md.get("file_name")}:{md.get("page_label")}:{text_part}')

        context_fingerprint = _sha("|".join(fp_parts)) if fp_parts else _sha("no_context")

        meta = {
            "retrieve_ms": retrieve_ms,
            "num_chunks": len(out),
            "top_score": float(out[0]["score"]) if out else 0.0,
            "context_fingerprint": context_fingerprint,
        }
        return out, meta

    def format_context(self, items: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for item in items:
            meta = item["metadata"]
            source = meta.get("file_name", "unknown")
            page = meta.get("page_label", "?")
            parts.append(f"[Source {item['rank']}: {source}, page {page}]\n{item['text']}")
        return "\n\n".join(parts)
