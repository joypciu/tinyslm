"""Heuristic decomposition of a complex goal into parallel sub-goals."""

from __future__ import annotations

import re
from typing import List


def decompose(goal: str, max_subgoals: int = 4) -> List[str]:
    g = (goal or "").strip()
    if not g:
        return ["general overview"]
    low = g.lower()
    subs: List[str] = []

    # Compare A vs B
    parts = re.split(r"\bvs\.?\b|\bversus\b|\bcompare\b", g, flags=re.I)
    parts = [p.strip(" :?-") for p in parts if p and len(p.strip()) > 2]
    if len(parts) >= 2 and ("compare" in low or " vs" in low or "versus" in low):
        a, b = parts[0][:80], parts[1][:80]
        subs = [
            f"Key facts and strengths of {a}",
            f"Key facts and strengths of {b}",
            f"Tradeoffs between {a} and {b}",
        ]
        return subs[:max_subgoals]

    if any(w in low for w in ("build", "implement", "design", "project", "architecture")):
        subs = [
            f"Requirements and constraints for: {g[:100]}",
            f"Existing approaches and tools for: {g[:100]}",
            f"Practical implementation steps for: {g[:100]}",
            f"Risks, pitfalls, and best practices for: {g[:100]}",
        ]
        return subs[:max_subgoals]

    if any(w in low for w in ("research", "investigate", "deep dive", "latest", "news")):
        subs = [
            f"Current overview: {g[:120]}",
            f"Recent developments: {g[:120]}",
            f"Key sources and citations: {g[:120]}",
        ]
        return subs[:max_subgoals]

    # Generic multi-facet split
    subs = [
        f"Core answer to: {g[:140]}",
        f"Supporting details and examples for: {g[:100]}",
        f"Caveats and alternatives for: {g[:100]}",
    ]
    return subs[:max_subgoals]
