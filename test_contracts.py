"""Tests for agent step contracts."""

from __future__ import annotations

from tiny_slm.agent import build_plan, run_agent_tools
from tiny_slm.contracts import (
    ContractResult,
    best_contract_answer,
    contract_compare,
    contract_memory,
    contract_search,
    synthesize_from_contracts,
)


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


def test_repair_search_query() -> None:
    from tiny_slm.search import repair_search_query

    q = repair_search_query(
        "Search the web for the latest Python release news and summarize in one sentence.",
        failed_query="latest Python release news and summarize in one sentence",
    )
    assert "python" in q.lower()
    assert "summarize" not in q.lower()
    assert "release" in q.lower() or "news" in q.lower()


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


def test_repair_memory_query() -> None:
    from tiny_slm.memory import repair_memory_query

    q = repair_memory_query("Using memory, what is the launch code?")
    assert "using memory" not in q.lower()
    assert "launch" in q.lower() and "code" in q.lower()


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


def test_ir_need_shapes_plan() -> None:
    plan = build_plan("tell me something", need=["search", "reason"])
    assert plan[0] == "search"
    assert "reason" in plan


def test_synthesize_from_pass_contracts() -> None:
    ans = synthesize_from_contracts(
        [
            ContractResult("memory", True, "noise " * 30, "context-only"),
            ContractResult(
                "search",
                True,
                "From the web: Ada Lovelace pioneered computing.",
                "evidence-quorum quorum=2",
            ),
        ],
        "Who was Ada?",
    )
    assert ans and "Ada" in ans


def test_contract_compare_sides() -> None:
    cr = contract_compare(
        "Compare France and Japan capitals.",
        [
            ContractResult(
                "search",
                True,
                "France capital Paris. Japan capital Tokyo.",
                "evidence-quorum",
            )
        ],
    )
    assert cr.ok and "France" in cr.artifact and "Japan" in cr.artifact


def test_agent_memory_need() -> None:
    ctx, st = run_agent_tools(
        "Using memory, what is the launch code?",
        memory_retrieve=lambda q: "FACT from user: launch code is ORBIT-77 for Friday.",
        auto_search=False,
        need=["memory", "reason"],
    )
    assert "memory" in st.plan
    assert st.verified_answer and "ORBIT" in st.verified_answer.upper()


def main() -> None:
    tests = [
        test_search_contract_quorum,
        test_repair_search_query,
        test_host_diversity,
        test_memory_contract,
        test_repair_memory_query,
        test_best_answer_prefers_search,
        test_agent_tools_contract_lines,
        test_ir_need_shapes_plan,
        test_synthesize_from_pass_contracts,
        test_contract_compare_sides,
        test_agent_memory_need,
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
