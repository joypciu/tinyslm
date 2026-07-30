"""Search sub-agent: focused web queries → URL seeds."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from tiny_slm.search import SearchHit, clean_search_query, search_web_hits


@dataclass
class SearchSeed:
    subgoal: str
    query: str
    hits: List[SearchHit]


def _focus_query(subgoal: str) -> str:
    q = re.sub(r"(?i)^research and answer:\s*", "", (subgoal or "").strip())
    q = re.sub(r"^\d+\s*[\)\.\-:]\s*", "", q).strip()
    low = q.lower()

    # Prefer domain-specific search phrases for common design facets
    if "monolith" in low or "microservice" in low:
        return "modular monolith vs microservices vs monolith tradeoffs small team"
    if "jwt" in low or ("session" in low and "auth" in low):
        return "JWT vs session authentication multi-tenant SaaS data isolation"
    if "stack" in low or ("framework" in low and ("db" in low or "database" in low)):
        return "Python FastAPI PostgreSQL Redis Celery SaaS API stack"
    if "roadmap" in low or "milestone" in low:
        return "two-week MVP implementation plan software API milestones week 1 week 2"
    if "security" in low or "risk" in low:
        return "multi-tenant SaaS API security risks OWASP testing checklist"
    return clean_search_query(q)[:160]


def run_search_agent(subgoal: str, max_results: int = 8) -> SearchSeed:
    q = _focus_query(subgoal)[:160]
    hits = search_web_hits(q, max_results=max_results)
    if not hits and " " in q:
        hits = search_web_hits(" ".join(q.split()[:8]), max_results=max_results)
    return SearchSeed(subgoal=subgoal, query=q, hits=hits)
