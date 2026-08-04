"""Tests for success-trace logging and distill corpus builder."""

from __future__ import annotations

from pathlib import Path

from tiny_slm.traces import (
    TraceStore,
    looks_distillable,
    strip_display_headers,
    traces_to_chat_corpus,
)


def test_looks_distillable() -> None:
    assert looks_distillable("2 + 2 equals 4.")
    assert not looks_distillable("I'm not sure I followed that — try a shorter question?")
    assert not looks_distillable("hi")
    assert not looks_distillable(
        "I don't have a verified answer for that, and I won't guess.",
        mode="faq",
    )
    assert not looks_distillable("Some long abstain text that looks ok.", mode="abstain")


def test_strip_headers() -> None:
    raw = "[ir] MODE=math NEED=[math]\n\n[model]\n2 + 2 equals 4."
    assert strip_display_headers(raw) == "2 + 2 equals 4."


def test_store_roundtrip(tmp_path: Path | None = None) -> None:
    path = (tmp_path or Path("checkpoints")) / "_test_success_traces.jsonl"
    if path.exists():
        path.unlink()
    store = TraceStore(path, enabled=True)
    assert store.record(
        "What is 2 + 2?",
        "2 + 2 equals 4.",
        mode="math",
        source="math",
        ir_tag="MODE=math NEED=[math] VERIFY=[symbolic] FACETS=[-] conf=0.95",
        verify=["symbolic"],
    )
    assert not store.record(
        "x",
        "I'm not sure I followed that — try a shorter question?",
        mode="chat",
        source="neural",
    )
    assert not store.record(
        "Prove Riemann",
        "I will not invent a proof for open research math.",
        mode="abstain",
        source="abstain",
    )
    rows = store.load()
    assert len(rows) == 1
    assert rows[0].mode == "math"
    corpus = traces_to_chat_corpus(rows, repeat=2)
    assert "<user>What is 2 + 2?<eos>" in corpus
    assert corpus.count("<assistant>2 + 2 equals 4.<eos>") == 2
    path.unlink(missing_ok=True)


def test_chat_logs_grounded_trace() -> None:
    from tiny_slm.chat import TinyChat

    path = Path("checkpoints/_test_chat_traces.jsonl")
    path.unlink(missing_ok=True)
    chat = TinyChat(auto_search=False, log_traces=True)
    chat.traces = TraceStore(path, enabled=True)
    chat.clear_memory()
    chat.reset()
    reply, _ = chat.generate_reply("What is 10 percent of 200?", temperature=0.1)
    assert "20" in reply
    rows = chat.traces.load()
    path.unlink(missing_ok=True)
    assert rows, "expected a success trace from math card"
    assert rows[-1].mode == "math"
    assert "20" in rows[-1].answer


def test_prepare_distill_script() -> None:
    import prepare_distill_traces as prep

    path = Path("checkpoints/_test_prep_traces.jsonl")
    out = Path("data/_test_distill_traces.txt")
    path.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    store = TraceStore(path)
    store.record(
        "What is RAM?",
        "RAM is short-term computer memory.",
        mode="faq",
        source="faq",
    )
    # Run builder against temp paths
    import sys

    argv = sys.argv
    try:
        sys.argv = [
            "prepare_distill_traces.py",
            "--traces",
            str(path),
            "--out",
            str(out),
            "--repeat",
            "2",
            "--rehearsal-repeat",
            "1",
        ]
        prep.main()
    finally:
        sys.argv = argv
    text = out.read_text(encoding="utf-8")
    path.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    assert "What is RAM?" in text
    assert "2 + 2 equals 4" in text  # rehearsal present


def main() -> None:
    tests = [
        test_looks_distillable,
        test_strip_headers,
        test_store_roundtrip,
        test_chat_logs_grounded_trace,
        test_prepare_distill_script,
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
    print("test_traces OK")


if __name__ == "__main__":
    main()
