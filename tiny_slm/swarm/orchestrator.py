"""Orchestrator: decompose → parallel search/crawl/read → synthesize."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tiny_slm.swarm.crawl_agent import crawl_hits
from tiny_slm.swarm.decompose import decompose
from tiny_slm.swarm.reader_agent import ReaderNote, run_reader_agent
from tiny_slm.swarm.search_agent import run_search_agent
from tiny_slm.swarm.synthesizer import synthesize
from tiny_slm.swarm.vector_store import VectorStore

_ANSWER_CACHE: Dict[str, "SwarmResult"] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 32


@dataclass
class SwarmResult:
    answer: str
    subgoals: List[str] = field(default_factory=list)
    pages_crawled: int = 0
    chunks: int = 0
    workers: int = 0
    backend: str = ""
    digest: str = ""


def _cache_key(goal: str) -> str:
    return " ".join((goal or "").lower().split())[:240]


def _run_one_subgoal(
    subgoal: str,
    store: VectorStore,
    seen_urls: set,
    seen_lock: threading.Lock,
    max_pages: int,
) -> ReaderNote:
    seed = run_search_agent(subgoal, max_results=8)
    with seen_lock:
        local_seen = set(seen_urls)
    chunks = crawl_hits(seed.hits, subgoal, max_pages=max_pages, seen=local_seen)
    with seen_lock:
        seen_urls.update(local_seen)
    store.add(chunks)
    return run_reader_agent(store, subgoal, top_k=4)


def run_swarm(
    goal: str,
    *,
    max_workers: int = 3,
    max_subgoals: int = 3,
    max_pages_per_agent: int = 2,
    use_cache: bool = True,
) -> SwarmResult:
    """Spawn parallel sub-agents: search + crawl + vector-read, then synthesize."""
    key = _cache_key(goal)
    if use_cache:
        with _CACHE_LOCK:
            hit = _ANSWER_CACHE.get(key)
            if hit is not None:
                return hit

    subgoals = decompose(goal, max_subgoals=max_subgoals)
    store = VectorStore()
    seen_urls: set = set()
    seen_lock = threading.Lock()
    notes: List[ReaderNote] = []
    workers = min(max_workers, max(1, len(subgoals)))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(
                _run_one_subgoal,
                sg,
                store,
                seen_urls,
                seen_lock,
                max_pages_per_agent,
            ): sg
            for sg in subgoals
        }
        for fut in as_completed(futs):
            try:
                notes.append(fut.result())
            except Exception as exc:
                sg = futs[fut]
                notes.append(
                    ReaderNote(
                        subgoal=sg,
                        bullets=[f"(sub-agent failed: {type(exc).__name__})"],
                        sources=[],
                    )
                )

    # Preserve subgoal order
    order = {s: i for i, s in enumerate(subgoals)}
    notes.sort(key=lambda n: order.get(n.subgoal, 99))

    answer = synthesize(goal, notes)
    stats = store.stats()
    digest_lines = [
        f"subgoals={len(subgoals)} workers={workers} chunks={stats['chunks']} "
        f"backend={stats['backend']} urls={len(seen_urls)}"
    ]
    for n in notes:
        digest_lines.append(f"- {n.subgoal[:80]} :: bullets={len(n.bullets)}")

    result = SwarmResult(
        answer=answer,
        subgoals=subgoals,
        pages_crawled=len(seen_urls),
        chunks=int(stats["chunks"]),
        workers=workers,
        backend=str(stats["backend"]),
        digest="\n".join(digest_lines),
    )
    if use_cache and answer:
        with _CACHE_LOCK:
            _ANSWER_CACHE[key] = result
            while len(_ANSWER_CACHE) > _CACHE_MAX:
                _ANSWER_CACHE.pop(next(iter(_ANSWER_CACHE)))
    return result
