"""Search sub-agent: focused web queries → URL seeds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from tiny_slm.search import SearchHit, clean_search_query, search_web_hits


@dataclass
class SearchSeed:
    subgoal: str
    query: str
    hits: List[SearchHit]


def run_search_agent(subgoal: str, max_results: int = 8) -> SearchSeed:
    q = clean_search_query(subgoal)[:160]
    hits = search_web_hits(q, max_results=max_results)
    if not hits and " " in q:
        hits = search_web_hits(" ".join(q.split()[:8]), max_results=max_results)
    return SearchSeed(subgoal=subgoal, query=q, hits=hits)
