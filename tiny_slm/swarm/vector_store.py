"""Fast in-memory vector store (FastEmbed or TF-IDF fallback)."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Chunk:
    text: str
    url: str = ""
    title: str = ""
    subgoal: str = ""


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]{2,}", (text or "").lower())


class _TfidfBackend:
    """Lightweight bag-of-words cosine when FastEmbed is unavailable."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        docs = [_tokenize(t) for t in texts]
        for toks in docs:
            for t in toks:
                if t not in self._vocab:
                    self._vocab[t] = len(self._vocab)
        n = len(texts)
        d = max(1, len(self._vocab))
        mat = np.zeros((n, d), dtype=np.float32)
        for i, toks in enumerate(docs):
            if not toks:
                continue
            for t in toks:
                mat[i, self._vocab[t]] += 1.0
            mat[i] /= max(1.0, np.linalg.norm(mat[i]))
        return mat


class _FastEmbedBackend:
    def __init__(self) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = list(self._model.embed(list(texts)))
        arr = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return arr / norms


def _get_backend():
    try:
        return _FastEmbedBackend(), "fastembed"
    except Exception:
        return _TfidfBackend(), "tfidf"


@dataclass
class VectorStore:
    chunks: List[Chunk] = field(default_factory=list)
    vectors: Optional[np.ndarray] = None
    backend_name: str = "none"
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _backend: object = field(default=None, repr=False)

    def _ensure_backend(self) -> None:
        if self._backend is None:
            self._backend, self.backend_name = _get_backend()

    def add(self, chunks: Sequence[Chunk]) -> int:
        fresh = [c for c in chunks if (c.text or "").strip()]
        if not fresh:
            return 0
        with self._lock:
            self._ensure_backend()
            texts = [c.text for c in fresh]
            vecs = self._backend.embed(texts)  # type: ignore[union-attr]
            if self.vectors is None or len(self.chunks) == 0:
                self.chunks = list(fresh)
                self.vectors = vecs
            else:
                self.chunks.extend(fresh)
                self.vectors = np.vstack([self.vectors, vecs])
            return len(fresh)

    def search(self, query: str, top_k: int = 5) -> List[ScoredChunk]:
        with self._lock:
            if not self.chunks or self.vectors is None:
                return []
            self._ensure_backend()
            q = self._backend.embed([query])[0]  # type: ignore[union-attr]
            scores = self.vectors @ q
            k = min(top_k, len(self.chunks))
            idx = np.argpartition(-scores, kth=k - 1)[:k]
            idx = idx[np.argsort(-scores[idx])]
            return [ScoredChunk(chunk=self.chunks[i], score=float(scores[i])) for i in idx]

    def stats(self) -> dict:
        return {
            "chunks": len(self.chunks),
            "backend": self.backend_name,
            "dim": int(self.vectors.shape[1]) if self.vectors is not None else 0,
        }
