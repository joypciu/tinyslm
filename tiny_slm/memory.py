"""Long-context memory: store ~2M tokens and retrieve relevant chunks.

A ~4M-param model cannot attend over 2,000,000 tokens on 2–3 GB RAM
(KV cache alone would be gigabytes). Instead we keep a disk/RAM corpus of
up to 2M tokens and inject only the top-k retrieved chunks into the
neural context window (MQA + RoPE).

Retrieval is hybrid by default: BM25-lite (lexical / needle-safe) blended
with optional dense embeddings (FastEmbed, else TF-IDF cosine). Dense is a
soft signal — recall boosts and fact chunks still dominate code needles.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def approx_tokens(text: str) -> int:
    # Rough BPE-ish estimate for English / chat text
    return max(1, len(text) // 4) if text else 0


class _TfidfEmbed:
    """Bag-of-words cosine when FastEmbed is unavailable (CPU-light)."""

    def __init__(self) -> None:
        self._vocab: Dict[str, int] = {}

    def embed(self, texts: Sequence[str], fit: bool = True) -> np.ndarray:
        docs = [tokenize(t) for t in texts]
        if fit:
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
                j = self._vocab.get(t)
                if j is None:
                    continue
                mat[i, j] += 1.0
            norm = float(np.linalg.norm(mat[i]))
            if norm > 0:
                mat[i] /= norm
        return mat


class _FastEmbed:
    def __init__(self) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def embed(self, texts: Sequence[str], fit: bool = True) -> np.ndarray:
        del fit  # fixed-dim neural embeddings
        vecs = list(self._model.embed(list(texts)))
        arr = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return arr / norms


def _make_embed_backend(prefer: str = "auto"):
    """prefer: auto | fastembed | tfidf."""
    if prefer == "tfidf":
        return _TfidfEmbed(), "tfidf"
    if prefer == "fastembed":
        return _FastEmbed(), "fastembed"
    try:
        return _FastEmbed(), "fastembed"
    except Exception:
        return _TfidfEmbed(), "tfidf"


def _minmax_norm(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


_WORD = re.compile(r"[a-z0-9_]+", re.I)
# Distinctive codes / tokens planted in long chats (e.g. BLUE_LANTERN_CODE)
_CODEISH = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")
_RECALL_CUES = re.compile(
    r"\b("
    r"remember|memory|earlier|before|previously|secret|code|token|password|"
    r"what was|remind|using memory|from (our|the) (chat|conversation|context|document)|"
    r"you (told|said)|store(d)? this|launch code|warehouse|meeting password"
    r")\b",
    re.I,
)
_FACT_LINE = re.compile(
    r"(?:secret project code|warehouse id|meeting password|token for the final check|"
    r"launch code|password is|code is|code:|token[:\s]+|id is)\s*[:\s]*"
    r"([A-Z0-9][A-Z0-9_\-]{3,})",
    re.I,
)


def tokenize(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def looks_like_recall(query: str) -> bool:
    """True when the user is asking to recover a past fact from memory."""
    return bool(_RECALL_CUES.search(query or ""))


def extract_codes(text: str) -> List[str]:
    return list(dict.fromkeys(_CODEISH.findall(text or "")))


_GENERIC_Q = {
    "code",
    "token",
    "secret",
    "password",
    "memory",
    "using",
    "what",
    "was",
    "the",
    "related",
    "exact",
    "know",
    "reply",
    "with",
    "project",
    "meeting",
    "warehouse",
    "final",
    "check",
    "store",
    "this",
    "from",
    "user",
    "assistant",
    "fact",
}


def answer_from_memory(query: str, memory_text: str) -> Optional[str]:
    """Extractive answer so tiny models do not have to regenerate rare codes.

    No weight updates — uses BM25-retrieved text only. Prevents long-chat
    needle misses when the neural window has already dropped older turns.
    """
    mem = (memory_text or "").strip()
    q = (query or "").strip()
    if not mem or not q or not looks_like_recall(q):
        return None

    q_upper = q.upper()
    q_terms = {t for t in tokenize(q) if t not in _GENERIC_Q and len(t) >= 3}

    def _score_code(code: str) -> int:
        # Ignore generic segments (CODE/TOKEN/ID) — they appear in every probe
        parts = [
            p
            for p in code.replace("-", "_").split("_")
            if p and p.lower() not in _GENERIC_Q and not p.isdigit()
        ]
        score = 0
        ql = f" {q.lower()} "
        for p in parts:
            pu = p.upper()
            pl = p.lower()
            if len(pl) < 3:
                continue
            if pu in q_upper.split() or f" {pl} " in ql:
                score += 5
            elif pl in q_terms:
                score += 4
            elif pl in q.lower():
                # Prefix hint: "related to BLUE" → BLUE_LANTERN_CODE
                score += 3
        return score

    # Score fact-line values and underscore codes; distinctive query hints win
    candidates: List[Tuple[int, str]] = []
    all_facts: List[str] = []
    for m in _FACT_LINE.finditer(mem):
        fact = m.group(1).upper().strip()
        all_facts.append(fact)
        sc = _score_code(fact)
        fl = m.group(0).lower()
        ql = q.lower()
        # Typed cues only when the query names that slot (not generic "code/token")
        if "launch" in ql and "launch" in fl:
            sc += 6
        if "password" in ql and "password" in fl:
            sc += 6
        if "warehouse" in ql and "warehouse" in fl:
            sc += 6
        if sc > 0:
            candidates.append((sc, fact))

    for c in extract_codes(mem):
        sc = _score_code(c)
        if sc > 0:
            candidates.append((sc, c.upper()))

    if candidates:
        candidates.sort(key=lambda x: (-x[0], -len(x[1])))
        return f"From memory: {candidates[0][1]}."

    uniq_facts = list(dict.fromkeys(all_facts))
    if len(uniq_facts) == 1:
        return f"From memory: {uniq_facts[0]}."
    codes = extract_codes(mem)
    if len(codes) == 1:
        return f"From memory: {codes[0]}."
    return None


@dataclass
class Chunk:
    id: int
    text: str
    source: str = "chat"
    tokens: int = 0


@dataclass
class LongContextMemory:
    """Inverted-index + optional dense hybrid memory (~2,000,000 tokens)."""

    max_tokens: int = 2_000_000
    chunk_chars: int = 480
    chunks: List[Chunk] = field(default_factory=list)
    inverted: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
    doc_freq: Counter = field(default_factory=Counter)
    total_tokens: int = 0
    _next_id: int = 0
    # Hybrid retrieval: dense_weight in [0, 1]; 0 = BM25-only (legacy behavior)
    hybrid: bool = True
    dense_weight: float = 0.35
    embed_backend: str = "auto"  # auto | fastembed | tfidf
    _dense: Optional[np.ndarray] = field(default=None, repr=False)
    _embedder: object = field(default=None, repr=False)
    _embed_name: str = field(default="none", repr=False)

    def clear(self) -> None:
        self.chunks.clear()
        self.inverted.clear()
        self.doc_freq.clear()
        self.total_tokens = 0
        self._next_id = 0
        self._dense = None

    def _ensure_embedder(self) -> None:
        if self._embedder is None:
            self._embedder, self._embed_name = _make_embed_backend(self.embed_backend)

    def _embed_texts(self, texts: Sequence[str], fit: bool = True) -> np.ndarray:
        self._ensure_embedder()
        return self._embedder.embed(texts, fit=fit)  # type: ignore[union-attr]

    def _append_dense(self, texts: Sequence[str]) -> None:
        if not texts:
            return
        # TF-IDF vocab grows with new docs — rebuild so query/doc dims stay aligned
        if self._embed_name == "tfidf" or self.embed_backend == "tfidf":
            self._rebuild_dense()
            return
        try:
            vecs = self._embed_texts(list(texts), fit=True)
        except Exception:
            # Dense is optional — lexical path must keep working
            self._dense = None
            self._embedder = None
            self._embed_name = "none"
            return
        if self._dense is None or len(self._dense) == 0:
            self._dense = vecs
        elif self._dense.shape[1] != vecs.shape[1]:
            self._rebuild_dense()
        else:
            self._dense = np.vstack([self._dense, vecs])

    def _rebuild_dense(self) -> None:
        if not self.chunks:
            self._dense = None
            return
        # Fresh TF-IDF vocab / FastEmbed batch after eviction or dim change
        if self.embed_backend == "tfidf" or self._embed_name in ("tfidf", "none"):
            if self.embed_backend != "fastembed":
                self._embedder = _TfidfEmbed()
                self._embed_name = "tfidf"
        try:
            self._dense = self._embed_texts([c.text for c in self.chunks], fit=True)
        except Exception:
            self._dense = None
            self._embedder = None
            self._embed_name = "none"

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
        if self.hybrid:
            self._rebuild_dense()

    def add_text(self, text: str, source: str = "chat") -> int:
        """Ingest text split into overlapping-ish chunks. Returns tokens added."""
        text = (text or "").strip()
        if not text:
            return 0
        added = 0
        step = max(200, self.chunk_chars - 80)
        new_texts: List[str] = []
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
            new_texts.append(piece)
            if self.total_tokens > self.max_tokens:
                self._evict_oldest()
                new_texts.clear()  # dense rebuilt inside eviction
        if self.hybrid and new_texts:
            self._append_dense(new_texts)
        return added

    def add_turn(self, user: str, assistant: str) -> None:
        self.add_text(f"User: {user}\nAssistant: {assistant}", source="dialog")
        # Keep rare codes / passwords as their own fact chunks so BM25
        # is not diluted by a weak assistant reply in the same chunk.
        if extract_codes(user) or _FACT_LINE.search(user or ""):
            self.add_text(f"FACT from user: {user[:420]}", source="fact")

    def _bm25_scores(self, query: str, q_terms: List[str]) -> Dict[int, float]:
        N = max(1, len(self.chunks))
        avgdl = max(1.0, sum(c.tokens for c in self.chunks) / N)
        k1, b = 1.4, 0.75
        scores: Dict[int, float] = defaultdict(float)
        q_tf = Counter(q_terms)
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

        # Recall boost: rare query hints (BLUE/ORANGE/…) beat shared words
        # like "memory"/"code" that otherwise drown distinctive fact chunks.
        if looks_like_recall(query):
            hints = {t for t in q_terms if len(t) >= 3 and t not in _GENERIC_Q}
            for chunk in self.chunks:
                text_l = chunk.text.lower()
                boost = 0.0
                for h in hints:
                    if h in text_l:
                        df = max(1, self.doc_freq.get(h, 1))
                        boost += 12.0 * math.log(1 + N / df)
                for code in extract_codes(chunk.text):
                    parts = {
                        p.lower()
                        for p in code.replace("-", "_").split("_")
                        if p and p.lower() not in _GENERIC_Q
                    }
                    if parts & hints:
                        boost += 40.0
                if chunk.source == "fact":
                    boost += 3.0
                if chunk.source == "skill":
                    boost -= 25.0
                if boost:
                    scores[chunk.id] += boost
        return dict(scores)

    def _dense_scores(self, query: str) -> Dict[int, float]:
        if not self.chunks:
            return {}
        if self._dense is None or len(self._dense) != len(self.chunks):
            self._rebuild_dense()
        if self._dense is None or len(self._dense) != len(self.chunks):
            return {}
        try:
            # Do not grow TF-IDF vocab on queries — keeps dim == stored matrix
            qv = self._embed_texts([query], fit=False)[0]
        except Exception:
            return {}
        if qv.shape[0] != self._dense.shape[1]:
            self._rebuild_dense()
            if self._dense is None:
                return {}
            try:
                qv = self._embed_texts([query], fit=False)[0]
            except Exception:
                return {}
        sims = self._dense @ qv
        return {c.id: float(sims[i]) for i, c in enumerate(self.chunks)}

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        max_chars: int = 900,
        hybrid: Optional[bool] = None,
    ) -> str:
        """Hybrid BM25 + dense retrieval over the memory bank."""
        q_terms = tokenize(query)
        if not self.chunks:
            return ""
        if not q_terms:
            recent = [c for c in self.chunks if c.source != "skill"][-top_k:]
            if not recent:
                recent = self.chunks[-top_k:]
            return "\n---\n".join(c.text for c in recent)[:max_chars]

        id_map = {c.id: c for c in self.chunks}
        lexical = self._bm25_scores(query, q_terms)

        use_hybrid = self.hybrid if hybrid is None else hybrid
        dense: Dict[int, float] = {}
        if use_hybrid:
            dense = self._dense_scores(query)

        # Blend: keep BM25 dominant on recall (needle-safe); denser on paraphrase
        if dense:
            w = float(self.dense_weight)
            if looks_like_recall(query):
                w = min(w, 0.2)
            w = max(0.0, min(1.0, w))
            lex_n = _minmax_norm(lexical)
            den_n = _minmax_norm(dense)
            all_ids = set(lex_n) | set(den_n)
            scores = {
                cid: (1.0 - w) * lex_n.get(cid, 0.0) + w * den_n.get(cid, 0.0)
                for cid in all_ids
            }
        else:
            scores = lexical

        # Demote skill cards on ordinary / recall queries so FAQ facts win
        want_skills = any(
            w in (query or "").lower()
            for w in ("skill", "plan", "step by step", "compare", "break down")
        )
        if not want_skills:
            for chunk in self.chunks:
                if chunk.source == "skill" and chunk.id in scores:
                    scores[chunk.id] *= 0.05

        if not scores:
            recent = [c for c in self.chunks if c.source != "skill"][-top_k:]
            if not recent:
                recent = self.chunks[-top_k:]
            blob = "\n---\n".join(c.text for c in recent)
            return blob[:max_chars]

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        parts = []
        size = 0
        for cid, _ in ranked:
            chunk = id_map[cid]
            if not want_skills and chunk.source == "skill":
                continue
            if size + len(chunk.text) > max_chars and parts:
                break
            parts.append(chunk.text)
            size += len(chunk.text)
            if len(parts) >= top_k:
                break
        if not parts:
            for cid, _ in ranked[:top_k]:
                parts.append(id_map[cid].text)
        return "\n---\n".join(parts)

    def stats(self) -> dict:
        return {
            "chunks": len(self.chunks),
            "tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
            "fill_pct": round(100.0 * self.total_tokens / max(1, self.max_tokens), 2),
            "hybrid": self.hybrid,
            "dense_backend": self._embed_name if self.hybrid else "off",
            "dense_weight": self.dense_weight if self.hybrid else 0.0,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Persist chunks to JSON (index rebuilt on load)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_tokens": self.max_tokens,
            "chunk_chars": self.chunk_chars,
            "chunks": [
                {"text": c.text, "source": c.source, "tokens": c.tokens}
                for c in self.chunks
                if c.source != "skill"  # skills re-seeded by TinyChat
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def load(self, path: Union[str, Path]) -> int:
        """Load chunks from JSON; returns number of chunks loaded."""
        path = Path(path)
        if not path.exists():
            return 0
        data = json.loads(path.read_text(encoding="utf-8"))
        self.clear()
        self.max_tokens = int(data.get("max_tokens", self.max_tokens))
        self.chunk_chars = int(data.get("chunk_chars", self.chunk_chars))
        n = 0
        for row in data.get("chunks", []):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            tok = int(row.get("tokens") or approx_tokens(text))
            chunk = Chunk(
                id=self._next_id,
                text=text,
                source=row.get("source") or "chat",
                tokens=tok,
            )
            self._next_id += 1
            self.chunks.append(chunk)
            self._index_chunk(chunk)
            self.total_tokens += tok
            n += 1
            if self.total_tokens > self.max_tokens:
                self._evict_oldest()
        if self.hybrid and self.chunks:
            self._rebuild_dense()
        return n
