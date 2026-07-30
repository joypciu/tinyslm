"""Multi-agent web swarm: parallel search + crawl + fast vector RAG."""

from tiny_slm.swarm.orchestrator import SwarmResult, run_swarm
from tiny_slm.swarm.router import looks_complex_query, should_spawn_swarm

__all__ = [
    "SwarmResult",
    "run_swarm",
    "looks_complex_query",
    "should_spawn_swarm",
]
