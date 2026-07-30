"""Reader sub-agent: vector-retrieve passages relevant to a sub-goal."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from tiny_slm.swarm.vector_store import ScoredChunk, VectorStore

_NOISE = re.compile(
    r"(?i)\b(sign in|sign up|subscribe|cookie|sitemap|open in app|refcard|trending|"
    r"edit profile|medium\.com|get app|follow us|save this for your next|min read|"
    r"table of contents|geeksforgeeks home|skip to)\b"
)


@dataclass
class ReaderNote:
    subgoal: str
    bullets: List[str]
    sources: List[str]


def _best_sentences(text: str, limit: int = 2) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    out = []
    for p in parts:
        p = p.strip(" -•\t")
        if len(p) < 55:
            continue
        # Drop mid-chunk truncations and link-farm titles
        if re.match(r"^[a-z]", p) and not p[:1].isupper():
            # allow if it starts with a known tech acronym
            if not re.match(r"^(JWT|API|HTTP|SQL|Redis|Python)\b", p):
                continue
        if _NOISE.search(p):
            continue
        if p.count(" ") < 7:
            continue
        if len(p) > 240:
            p = p[:237].rstrip() + "..."
        out.append(p)
        if len(out) >= limit:
            break
    return out


def run_reader_agent(store: VectorStore, subgoal: str, top_k: int = 5) -> ReaderNote:
    scored: List[ScoredChunk] = store.search(subgoal, top_k=top_k)
    bullets: List[str] = []
    sources: List[str] = []
    seen = set()
    for sc in scored:
        if sc.score < 0.05 and store.backend_name == "fastembed":
            continue
        if _NOISE.search(sc.chunk.text):
            continue
        for sent in _best_sentences(sc.chunk.text, limit=2):
            key = re.sub(r"[^a-z0-9]+", "", sent.lower())[:48]
            if key in seen:
                continue
            seen.add(key)
            bullets.append(sent)
        src = sc.chunk.title or sc.chunk.url
        if src and src not in sources and not _NOISE.search(src):
            sources.append(src)
        if len(bullets) >= 5:
            break
    if not bullets and scored:
        clean = re.sub(r"\s+", " ", scored[0].chunk.text)[:220]
        if clean and not _NOISE.search(clean):
            bullets = [clean]
        if scored[0].chunk.url:
            sources.append(scored[0].chunk.title or scored[0].chunk.url)
    return ReaderNote(subgoal=subgoal, bullets=bullets, sources=sources)
