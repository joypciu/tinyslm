"""Unit tests for hybrid BM25 + dense LongContextMemory retrieval."""

from __future__ import annotations

from pathlib import Path

from tiny_slm.memory import LongContextMemory, answer_from_memory


def _mem(**kwargs) -> LongContextMemory:
    # Force TF-IDF so CI/smoke never waits on FastEmbed model download
    opts = dict(hybrid=True, embed_backend="tfidf", dense_weight=0.45, chunk_chars=400)
    opts.update(kwargs)
    return LongContextMemory(**opts)


def test_needle_still_wins_on_recall() -> None:
    mem = _mem()
    mem.add_text(
        "Important project note: the launch code is ORBIT-77 and the deadline is Friday.",
        source="doc",
    )
    mem.add_text("Meeting notes about lunch menus and office plants. " * 40, source="noise")
    mem.add_text("SKILL plan: always break work into steps.", source="skill")
    got = mem.retrieve("Using memory, what is the launch code?", top_k=3)
    assert "ORBIT-77" in got, got[:200]
    assert "SKILL" not in got, got[:200]


def test_dense_helps_rank_fact_over_generic_noise() -> None:
    """Hybrid should keep the fact on top despite generic recall-word noise."""
    mem = _mem(dense_weight=0.5)
    mem.add_text(
        "Warehouse inventory label for the north dock is ORANGE_RIVER_42.",
        source="fact",
    )
    mem.add_text(
        "Using memory matters. Reply with the exact code token password from memory. " * 6,
        source="noise",
    )
    mem.add_text(
        "Today I practiced short math drills and reviewed study notes carefully. " * 6,
        source="noise",
    )
    got = mem.retrieve(
        "Using memory, what was the warehouse identifier at the north dock?",
        top_k=2,
        max_chars=500,
    )
    assert "ORANGE_RIVER_42" in got, got[:240]
    # First chunk in the joined blob should be the fact (ranked above noise)
    first = got.split("\n---\n")[0]
    assert "ORANGE_RIVER_42" in first, first[:200]
    st = mem.stats()
    assert st["dense_backend"] == "tfidf"
    assert st["hybrid"] is True


def test_hybrid_off_matches_lexical_path() -> None:
    mem = _mem(hybrid=False)
    mem.add_text("SECRET_NEEDLE_ALPHA_42 the mooncake protocol is active", source="needle")
    mem.add_text("Irrelevant weather notes for Tuesday afternoon.", source="noise")
    got = mem.retrieve("mooncake protocol SECRET_NEEDLE", top_k=2)
    assert "SECRET_NEEDLE_ALPHA_42" in got


def test_extractive_answer_still_works() -> None:
    mem = _mem()
    mem.add_turn(
        "Remember this secret project code: BLUE_LANTERN_CODE.",
        "Got it, I'll remember.",
    )
    block = mem.retrieve(
        "Using memory, what was the secret code related to BLUE?", top_k=3
    )
    ans = answer_from_memory(
        "Using memory, what was the secret code related to BLUE?", block
    )
    assert ans and "BLUE_LANTERN_CODE" in ans, ans


def test_save_load_keeps_hybrid_retrieval() -> None:
    tmp = Path("checkpoints/_test_hybrid_memory.json")
    mem = _mem()
    mem.add_text("Launch credential stored as ORBIT-77 for Friday.", source="doc")
    mem.add_text("Noise about plants and sunlight. " * 20, source="noise")
    mem.save(tmp)
    mem2 = _mem()
    n = mem2.load(tmp)
    tmp.unlink(missing_ok=True)
    assert n >= 1
    assert mem2.stats().get("hybrid") is True
    got = mem2.retrieve("Using memory, what is the launch code ORBIT?", top_k=2)
    assert "ORBIT-77" in got, got[:200]


def main() -> None:
    tests = [
        test_needle_still_wins_on_recall,
        test_dense_helps_rank_fact_over_generic_noise,
        test_hybrid_off_matches_lexical_path,
        test_extractive_answer_still_works,
        test_save_load_keeps_hybrid_retrieval,
    ]
    fails = []
    for fn in tests:
        try:
            fn()
            print(f"  OK {fn.__name__}")
        except Exception as exc:
            fails.append(f"{fn.__name__}: {exc}")
            print(f"  FAIL {fn.__name__}: {exc}")
    if fails:
        raise SystemExit(1)
    print("test_hybrid_memory OK")


if __name__ == "__main__":
    main()
