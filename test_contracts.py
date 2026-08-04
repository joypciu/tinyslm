"""Tests for agent step contracts."""

from __future__ import annotations

from tiny_slm.agent import run_agent_tools
from tiny_slm.contracts import best_contract_answer, contract_memory, contract_search


def test_search_contract_quorum() -> None:
    digest = (
        "1. Python 3.12\nhttps://docs.python.org/3.12/\n"
        "Python 3.12 improves errors and speed.\n\n"
        "2. Release notes\nhttps://www.python.org/downloads/\n"
        "Latest Python release adds typing features.\n"
    )
    cr = contract_search("latest Python release", digest)
    assert cr.ok and "python" in cr.artifact.lower()
    assert "Sources:" in cr.artifact


def test_host_diversity() -> None:
    from tiny_slm.search import citation_hosts, host_diversity_ok
    from tiny_slm.tvs import evidence_quorum

    digest = (
        "1. Ada Lovelace\nhttps://en.wikipedia.org/wiki/Ada_Lovelace\n"
        "Ada Lovelace was a mathematician and early programmer.\n\n"
        "2. Biography\nhttps://www.britannica.com/biography/Ada-Lovelace\n"
        "Ada Lovelace wrote notes on Babbage's analytical engine.\n"
    )
    hosts = citation_hosts(digest)
    assert len(hosts) >= 2
    ok_h, _ = host_diversity_ok(digest, min_hosts=2)
    assert ok_h
    ok, ans, note = evidence_quorum(digest, "Who was Ada Lovelace?", min_hits=2)
    assert ok and "ada" in ans.lower()
    assert len(citation_hosts(ans)) >= 2 or "Sources:" in ans
    assert "hosts=" in note


def test_memory_contract() -> None:
    cr = contract_memory(
        "Using memory, what is the launch code?",
        "FACT from user: launch code is ORBIT-77 for Friday.",
    )
    assert cr.ok
    assert "ORBIT" in cr.artifact.upper() or "orbit" in cr.artifact.lower()


def test_best_answer_prefers_search() -> None:
    from tiny_slm.contracts import ContractResult

    ans = best_contract_answer(
        [
            ContractResult("memory", True, "noise context " * 20, "context-only"),
            ContractResult("search", True, "From the web: Ada Lovelace was a mathematician.", "quorum=2"),
        ]
    )
    assert ans and "Ada" in ans


def test_agent_tools_contract_lines() -> None:
    ctx, st = run_agent_tools(
        "Plan a short study session step by step.",
        memory_retrieve=lambda q: "",
        auto_search=False,
    )
    assert "Plan:" in ctx
    assert st.steps_done >= 1


def main() -> None:
    tests = [
        test_search_contract_quorum,
        test_host_diversity,
        test_memory_contract,
        test_best_answer_prefers_search,
        test_agent_tools_contract_lines,
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
    print("test_contracts OK")


if __name__ == "__main__":
    main()
