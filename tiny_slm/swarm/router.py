"""Route complex / research asks to the multi-agent web swarm."""

from __future__ import annotations

import re


_COMPLEX_PHRASES = (
    "deep dive",
    "from the web",
    "look up sources",
    "end to end",
    "end-to-end",
    "from scratch",
    "tradeoff",
    "trade-off",
    "architecture",
    "investigate",
    "research",
    "autonomous",
    "crawl",
    "how do i build",
    "how to build",
    "design a",
    "implement a",
    "compare and",
)

_COMPLEX_WORDS = (
    r"\bproject\b",
    r"\bresearch\b",
    r"\binvestigate\b",
    r"\barchitecture\b",
    r"\bcrawl\b",
    r"\bsources\b",
    r"\bapproach(?:es)?\b",
    r"\bbest practices?\b",
)


def looks_complex_query(user: str) -> bool:
    u = (user or "").strip()
    if not u:
        return False
    low = u.lower()
    if len(u) >= 120:
        return True
    if any(p in low for p in _COMPLEX_PHRASES):
        return True
    if sum(1 for w in ("and", "then", "also", "plus", "with") if w in low) >= 3 and len(u) >= 80:
        return True
    return any(re.search(w, low) for w in _COMPLEX_WORDS)


def should_spawn_swarm(user: str, *, has_card: bool = False) -> bool:
    """Spawn swarm only when no short card already answered and query is complex."""
    if has_card:
        return False
    return looks_complex_query(user)
