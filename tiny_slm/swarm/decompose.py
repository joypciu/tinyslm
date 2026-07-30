"""Heuristic decomposition of a complex goal into parallel sub-goals."""

from __future__ import annotations

import re
from typing import List


def _numbered_items(goal: str) -> List[str]:
    """Extract '1) ...' / '1. ...' / '1: ...' checklist facets (inline or multiline)."""
    items: List[str] = []
    # Global split on numbered markers (works for inline "Cover: 1) ... 2) ...")
    parts = re.split(r"(?:^|[\n;]|\s)\d+\s*[\)\.\-:]\s+", goal)
    if len(parts) < 3:
        parts = re.split(r"\d+\s*[\)\.\-:]\s+", goal)
    for p in parts[1:]:
        line = re.sub(r"\s+", " ", p).strip(" \n\t-;,")
        line = re.split(r"\s+\d+\s*[\)\.\-:]\s+", line)[0].strip()
        line = re.sub(r"^\d+\s*[\)\.\-:]\s*", "", line).strip()
        line = re.sub(r"(?i)\bcite sources\.?$", "", line).strip()
        if 12 <= len(line) <= 220:
            items.append(line)
        elif len(line) > 220:
            items.append(line[:200].rstrip() + "...")
    return items if len(items) >= 2 else []


def _clean_compare_side(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip(" :?-")
    # Drop long preambles — keep last noun-ish phrase if too long
    if len(t) > 90:
        # Prefer trailing clause after common verbs
        m = re.search(
            r"(?:tradeoffs?|options?|approaches?|between|for)\s+(.+)$",
            t,
            flags=re.I,
        )
        if m and len(m.group(1)) > 8:
            t = m.group(1).strip()
        t = t[-90:]
    return t[:80]


def _is_clean_pairwise_compare(goal: str, parts: List[str]) -> bool:
    """True only for short A vs B compares — not multi-way lists inside a big brief."""
    low = goal.lower()
    if len(parts) != 2:
        return False
    if _numbered_items(goal):
        return False
    # Multi-way: "A vs B vs C"
    if len(re.findall(r"\bvs\.?\b|\bversus\b", low)) >= 2:
        return False
    a, b = parts[0], parts[1]
    # First side should not be a whole research brief
    if len(a) > 100 or len(goal) > 280:
        return False
    if not ("compare" in low or re.search(r"\bvs\.?\b|\bversus\b", low)):
        return False
    return True


def decompose(goal: str, max_subgoals: int = 5) -> List[str]:
    g = (goal or "").strip()
    if not g:
        return ["general overview"]
    low = g.lower()

    # 1) Explicit numbered checklist in the user ask → one sub-agent per item
    numbered = _numbered_items(g)
    if len(numbered) >= 2:
        return [f"Research and answer: {item}" for item in numbered[:max_subgoals]]

    # 2) Clean pairwise compare only
    parts = re.split(r"\bvs\.?\b|\bversus\b|\bcompare\b", g, flags=re.I)
    parts = [p.strip(" :?-") for p in parts if p and len(p.strip()) > 2]
    if _is_clean_pairwise_compare(g, parts):
        a, b = _clean_compare_side(parts[0]), _clean_compare_side(parts[1])
        return [
            f"Key facts and strengths of {a}",
            f"Key facts and strengths of {b}",
            f"Tradeoffs between {a} and {b}",
        ][:max_subgoals]

    # 3) Multi-way architecture / options lists (A vs B vs C) inside a design brief
    if len(re.findall(r"\bvs\.?\b", low)) >= 2 or (
        "monolith" in low and "microservice" in low
    ):
        subs = [
            "Monolith vs modular monolith vs microservices tradeoffs for a small team",
            "Auth options JWT vs session and multi-tenant data isolation patterns",
            "Recommended Python API stack: framework, database, cache, queue",
            "Two-week implementation roadmap with concrete milestones",
            "Security risks, pitfalls, and how to test a multi-tenant SaaS API",
        ]
        # Only use this specialized pack when the ask looks like that product brief
        if any(w in low for w in ("saas", "tenant", "api", "auth", "roadmap", "python")):
            return subs[:max_subgoals]

    if any(w in low for w in ("build", "implement", "design", "project", "architecture", "saas")):
        return [
            f"Architecture options and tradeoffs for: {g[:90]}",
            f"Auth, tenancy, and data isolation for: {g[:90]}",
            f"Recommended tech stack for: {g[:90]}",
            f"Implementation roadmap and milestones for: {g[:90]}",
            f"Risks, security pitfalls, and testing for: {g[:90]}",
        ][:max_subgoals]

    if any(w in low for w in ("research", "investigate", "deep dive", "latest", "news")):
        return [
            f"Current overview: {g[:120]}",
            f"Recent developments and best practices: {g[:100]}",
            f"Practical recommendations and sources: {g[:100]}",
        ][:max_subgoals]

    return [
        f"Core answer to: {g[:140]}",
        f"Supporting details and examples for: {g[:100]}",
        f"Caveats and alternatives for: {g[:100]}",
    ][:max_subgoals]
