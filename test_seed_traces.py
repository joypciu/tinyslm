"""Tests for verified-tool success-trace seeding."""

from __future__ import annotations

import tempfile
from pathlib import Path

from tiny_slm.seed_traces import collect_verified_seeds, seed_verified_traces
from tiny_slm.traces import TraceStore


def test_collect_seeds() -> None:
    seeds = collect_verified_seeds()
    assert len(seeds) >= 12
    modes = {m for _, _, m, _ in seeds}
    assert "math" in modes and "code" in modes


def test_seed_writes_jsonl() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "traces.jsonl"
        store = TraceStore(path)
        n = seed_verified_traces(store)
        assert n >= 12
        st = store.stats()
        assert st["count"] == n
        assert st["by_mode"].get("math", 0) >= 8
        # Idempotent: second pass should add 0
        assert seed_verified_traces(store) == 0
        assert store.stats()["count"] == n


def main() -> None:
    tests = [test_collect_seeds, test_seed_writes_jsonl]
    for fn in tests:
        fn()
        print(f"  OK {fn.__name__}")
    print("test_seed_traces OK")


if __name__ == "__main__":
    main()
