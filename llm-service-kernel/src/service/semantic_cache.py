######## not using this ############


import json
import os
import sqlite3
import time
import hashlib
from typing import Optional, Dict, Any, List

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class SemanticCache:
    def __init__(
        self,
        db_path: str = "semantic_cache.db",
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embed_model_path: Optional[str] = None,
        threshold: float = 0.90,
        ttl_seconds: int = 24 * 3600,
    ):
        self.db_path = db_path
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds

        # Prefer local path
        self.embedder = self._load_embedder(embed_model_name, embed_model_path)
        self.dim = self.embedder.get_sentence_embedding_dimension()

        self.index = faiss.IndexFlatIP(self.dim)  # cosine via normalized vectors
        self.row_to_cache_id: List[int] = []

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        self._load_index_from_db()

    def _load_embedder(
        self,
        embed_model_name: str,
        embed_model_path: Optional[str],
    ) -> SentenceTransformer:
        # 1) explicit local path
        if embed_model_path:
            if os.path.exists(embed_model_path):
                return SentenceTransformer(embed_model_path)
            raise FileNotFoundError(
                f"SEM_CACHE_EMBED_MODEL_PATH does not exist: {embed_model_path}"
            )

        # 2) model name (s if already cached locally)
        try:
            return SentenceTransformer(embed_model_name)
        except Exception as e:
            raise RuntimeError(
                "Failed to load embedding model. Network may be blocked and the model "
                "is not in local cache. Pre-download the model or set "
                "SEM_CACHE_EMBED_MODEL_PATH to a local directory.\n"
                f"Original error: {e}"
            )

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                query_embedding TEXT NOT NULL,
                response_json TEXT NOT NULL,

                model_name TEXT NOT NULL,
                temperature REAL NOT NULL,
                top_p REAL NOT NULL,
                cache_scope TEXT NOT NULL,

                kb_version TEXT NOT NULL,
                system_prompt_hash TEXT NOT NULL,

                created_at REAL NOT NULL,
                ttl_seconds INTEGER NOT NULL
            )
            """
        )
        self.conn.commit()

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v if n == 0 else v / n

    def _embed(self, text: str) -> np.ndarray:
        vec = self.embedder.encode(text, convert_to_numpy=True).astype(np.float32)
        return self._normalize(vec)

    def _load_index_from_db(self):
        cur = self.conn.cursor()
        cur.execute("SELECT id, query_embedding FROM cache_entries")
        rows = cur.fetchall()
        if not rows:
            return

        vecs = []
        ids = []
        for row in rows:
            vec = np.array(json.loads(row["query_embedding"]), dtype=np.float32)
            vec = self._normalize(vec)
            vecs.append(vec)
            ids.append(int(row["id"]))

        mat = np.vstack(vecs).astype(np.float32)
        self.index.add(mat)
        self.row_to_cache_id.extend(ids)

    def _expired(self, created_at: float, ttl_seconds: int) -> bool:
        return (time.time() - created_at) > ttl_seconds

    def _get_entry(self, cache_id: int):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM cache_entries WHERE id = ?", (cache_id,))
        return cur.fetchone()

    def lookup(
        self,
        query_text: str,
        model_name: str,
        temperature: float,
        top_p: float,
        cache_scope: str,
        kb_version: str,
        system_prompt_hash: str,
        top_k: int = 3,
    ) -> Optional[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return None

        q = self._embed(query_text).reshape(1, -1)
        scores, idxs = self.index.search(q, min(top_k, self.index.ntotal))

        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or float(score) < self.threshold:
                continue

            cache_id = self.row_to_cache_id[int(idx)]
            row = self._get_entry(cache_id)
            if row is None:
                continue

            # Hard guards
            if row["model_name"] != model_name:
                continue
            if row["cache_scope"] != cache_scope:
                continue
            if row["kb_version"] != kb_version:
                continue
            if row["system_prompt_hash"] != system_prompt_hash:
                continue
            if abs(float(row["temperature"]) - float(temperature)) > 1e-9:
                continue
            if abs(float(row["top_p"]) - float(top_p)) > 1e-9:
                continue
            if self._expired(float(row["created_at"]), int(row["ttl_seconds"])):
                continue

            return {
                "cache_id": cache_id,
                "score": float(score),
                "response_json": json.loads(row["response_json"]),
            }

        return None

    def insert(
        self,
        query_text: str,
        response_json: Dict[str, Any],
        model_name: str,
        temperature: float,
        top_p: float,
        cache_scope: str,
        kb_version: str,
        system_prompt_hash: str,
    ) -> int:
        vec = self._embed(query_text)
        now = time.time()

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO cache_entries (
                query_text, query_embedding, response_json,
                model_name, temperature, top_p, cache_scope,
                kb_version, system_prompt_hash,
                created_at, ttl_seconds
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                query_text,
                json.dumps(vec.tolist()),
                json.dumps(response_json),
                model_name,
                float(temperature),
                float(top_p),
                cache_scope,
                kb_version,
                system_prompt_hash,
                now,
                self.ttl_seconds,
            ),
        )
        self.conn.commit()
        cache_id = int(cur.lastrowid)

        self.index.add(vec.reshape(1, -1))
        self.row_to_cache_id.append(cache_id)

        return cache_id
