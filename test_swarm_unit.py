"""Offline unit checks for the multi-agent swarm (no live network required)."""

from __future__ import annotations

from tiny_slm.search import SearchHit
from tiny_slm.swarm.crawl_agent import _strip_html, crawl_hits
from tiny_slm.swarm.decompose import decompose
from tiny_slm.swarm.reader_agent import run_reader_agent
from tiny_slm.swarm.router import looks_complex_query, should_spawn_swarm
from tiny_slm.swarm.synthesizer import synthesize
from tiny_slm.swarm.vector_store import Chunk, VectorStore


def main() -> None:
    assert looks_complex_query(
        "Research architecture tradeoffs and design a project to build a scalable API from scratch"
    )
    assert not looks_complex_query("What is RAM?")
    assert should_spawn_swarm("Investigate best practices for Python packaging", has_card=False)
    assert not should_spawn_swarm("What is RAM?", has_card=False)

    subs = decompose("Compare Redis versus Memcached for caching tradeoffs")
    assert len(subs) >= 2
    assert any("redis" in s.lower() or "Redis" in s or "trade" in s.lower() for s in subs)

    html = "<html><head><script>x()</script></head><body><p>Hello swarm world. " * 5 + "</p></body></html>"
    text = _strip_html(html)
    assert "Hello swarm" in text
    assert "script" not in text.lower() or "x()" not in text

    hits = [
        SearchHit(
            title="Redis Guide",
            body="Redis is an in-memory data structure store used as a cache and message broker.",
            href="https://example.com/redis",
        ),
        SearchHit(
            title="Memcached Overview",
            body="Memcached is a distributed memory object caching system for speeding up web apps.",
            href="https://example.com/memcached",
        ),
    ]
    # Crawl without fetching (invalid/example URLs still yield snippet chunks)
    chunks = crawl_hits(hits, "cache tradeoffs", max_pages=0)
    assert len(chunks) >= 2

    store = VectorStore()
    n = store.add(
        [
            Chunk(
                text="Redis supports rich data structures and persistence options for caching.",
                url="https://example.com/redis",
                title="Redis Guide",
                subgoal="Redis",
            ),
            Chunk(
                text="Memcached focuses on simple key-value caching with very low latency.",
                url="https://example.com/memcached",
                title="Memcached Overview",
                subgoal="Memcached",
            ),
        ]
    )
    assert n == 2
    hits2 = store.search("Redis persistence caching", top_k=1)
    assert hits2 and "redis" in hits2[0].chunk.text.lower()

    note = run_reader_agent(store, "Redis caching features", top_k=2)
    assert note.bullets
    ans = synthesize("Compare Redis and Memcached", [note])
    assert "Sources:" in ans or "Redis" in ans

    print("swarm_unit_ok True")
    print("backend", store.backend_name)
    print("OVERALL PASS")


if __name__ == "__main__":
    main()
