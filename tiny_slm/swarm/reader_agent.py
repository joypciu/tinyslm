"""Reader sub-agent: vector-retrieve passages relevant to a sub-goal."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from tiny_slm.swarm.vector_store import ScoredChunk, VectorStore


@dataclass
class ReaderNote:
    subgoal: str
    bullets: List[str]
    sources: List[str]


def _best_sentences(text: str, limit: int = 2) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    out = []
    for p in parts:
        p = p.strip()
        if len(p) < 40:
            continue
        if len(p) > 220:
            p = p[:217].rstrip() + "..."
        out.append(p)
        if len(out) >= limit:
            break
    return out


def run_reader_agent(store: VectorStore, subgoal: str, top_k: int = 4) -> ReaderNote:
    scored: List[ScoredChunk] = store.search(subgoal, top_k=top_k)
    bullets: List[str] = []
    sources: List[str] = []
    seen = set()
    for sc in scored:
        if sc.score < 0.05 and store.backend_name == "fastembed":
            continue
        for sent in _best_sentences(sc.chunk.text, limit=1):
            key = re.sub(r"[^a-z0-9]+", "", sent.lower())[:48]
            if key in seen:
                continue
            seen.add(key)
            bullets.append(sent)
        src = sc.chunk.title or sc.chunk.url
        if src and src not in sources:
            sources.append(src)
        if len(bullets) >= 4:
            break
    if not bullets and scored:
        bullets = [scored[0].chunk.text[:200]]
        if scored[0].chunk.url:
            sources.append(scored[0].chunk.title or scored[0].chunk.url)
    return ReaderNote(subgoal=subgoal, bullets=bullets, sources=sources)
