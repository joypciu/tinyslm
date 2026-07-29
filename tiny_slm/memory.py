"""Long-context memory: store ~2M tokens and retrieve relevant chunks.

A ~4M-param model cannot attend over 2,000,000 tokens on 2–3 GB RAM
(KV cache alone would be gigabytes). Instead we keep a disk/RAM corpus of
up to 2M tokens and inject only the top-k retrieved chunks into the
neural context window (MQA + RoPE).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def approx_tokens(text: str) -> int:
    # Rough BPE-ish estimate for English / chat text
    return max(1, len(text) // 4) if text else 0


_WORD = re.compile(r"[a-z0-9_]+", re.I)


def tokenize(text: str) -> List[str]:
    return _WORD.findall(text.lower())


@dataclass
class Chunk:
    id: int
    text: str
    source: str = "chat"
    tokens: int = 0


@dataclass
class LongContextMemory:
    """Inverted-index memory sized for ~2,000,000 tokens."""

    max_tokens: int = 2_000_000
    chunk_chars: int = 480
    chunks: List[Chunk] = field(default_factory=list)
    inverted: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
    doc_freq: Counter = field(default_factory=Counter)
    total_tokens: int = 0
    _next_id: int = 0

    def clear(self) -> None:
        self.chunks.clear()
        self.inverted.clear()
        self.doc_freq.clear()
        self.total_tokens = 0
        self._next_id = 0

    def _index_chunk(self, chunk: Chunk) -> None:
        terms = set(tokenize(chunk.text))
        for t in terms:
            self.inverted[t].append(chunk.id)
            self.doc_freq[t] += 1

    def _evict_oldest(self) -> None:
        while self.chunks and self.total_tokens > self.max_tokens:
            old = self.chunks.pop(0)
            self.total_tokens -= old.tokens
        # Rebuild index after eviction (rare; keeps code simple/CPU-light)
        self.inverted = defaultdict(list)
        self.doc_freq = Counter()
        for c in self.chunks:
            self._index_chunk(c)

    def add_text(self, text: str, source: str = "chat") -> int:
        """Ingest text split into overlapping-ish chunks. Returns tokens added."""
        text = (text or "").strip()
        if not text:
            return 0
        added = 0
        step = max(200, self.chunk_chars - 80)
        for i in range(0, len(text), step):
            piece = text[i : i + self.chunk_chars].strip()
            if len(piece) < 20:
                continue
            tok = approx_tokens(piece)
            chunk = Chunk(id=self._next_id, text=piece, source=source, tokens=tok)
            self._next_id += 1
            self.chunks.append(chunk)
            self._index_chunk(chunk)
            self.total_tokens += tok
            added += tok
            if self.total_tokens > self.max_tokens:
                self._evict_oldest()
        return added

    def add_turn(self, user: str, assistant: str) -> None:
        self.add_text(f"User: {user}\nAssistant: {assistant}", source="dialog")

    def retrieve(self, query: str, top_k: int = 4, max_chars: int = 900) -> str:
        """BM25-lite retrieval over the memory bank."""
        q_terms = tokenize(query)
        if not q_terms or not self.chunks:
            return ""
        N = max(1, len(self.chunks))
        avgdl = max(1.0, sum(c.tokens for c in self.chunks) / N)
        k1, b = 1.4, 0.75
        scores: Dict[int, float] = defaultdict(float)
        q_tf = Counter(q_terms)
        for term, qf in q_tf.items():
            postings = self.inverted.get(term)
            if not postings:
                continue
            df = self.doc_freq.get(term, 1)
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            for cid in postings:
                # cid may be stale after eviction — map by position
                pass
            # Use chunk list index via id map
        id_map = {c.id: c for c in self.chunks}
        for term, qf in q_tf.items():
            postings = self.inverted.get(term)
            if not postings:
                continue
            df = max(1, self.doc_freq.get(term, 1))
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            for cid in postings:
                chunk = id_map.get(cid)
                if chunk is None:
                    continue
                tf = tokenize(chunk.text).count(term)
                dl = max(1, chunk.tokens)
                denom = tf + k1 * (1 - b + b * dl / avgdl)
                scores[cid] += idf * (tf * (k1 + 1) / denom) * (1 + 0.1 * qf)

        if not scores:
            # fallback: most recent chunks
            recent = self.chunks[-top_k:]
            blob = "\n---\n".join(c.text for c in recent)
            return blob[:max_chars]

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        parts = []
        size = 0
        for cid, _ in ranked:
            chunk = id_map[cid]
            if size + len(chunk.text) > max_chars and parts:
                break
            parts.append(chunk.text)
            size += len(chunk.text)
        return "\n---\n".join(parts)

    def stats(self) -> dict:
        return {
            "chunks": len(self.chunks),
            "tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
            "fill_pct": round(100.0 * self.total_tokens / max(1, self.max_tokens), 2),
        }
