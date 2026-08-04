"""Tests for Thought-Verify Scratchpad + evidence quorum."""

from __future__ import annotations

from tiny_slm.tvs import classify_domain, evidence_quorum, run_tvs, safe_micro_exec


def test_domains() -> None:
    assert classify_domain("integrate x**2") == "math"
    assert classify_domain("Write a Python function that adds two numbers.") == "code"
    assert classify_domain("Plan a short study session step by step.") == "agent"
    assert classify_domain("search the web for python release news") == "search"


def test_math_tvs() -> None:
    r = run_tvs("What is 2 + 2?", auto_search=False)
    assert r.ok and "4" in r.answer
    r2 = run_tvs("Prove the Riemann hypothesis with a full formal proof.", auto_search=False)
    assert r2.abstained


def test_agent_tvs_defers_research() -> None:
    r = run_tvs("Investigate the latest Python release deeply.", auto_search=False)
    assert not r.ok and not r.abstained
    assert any(s.detail == "defer-to-sara" for s in r.steps)


def test_code_tvs_and_micro_exec() -> None:
    from tiny_slm.code_verify import run_spec_asserts

    r = run_tvs("Write a Python function that adds two numbers.", auto_search=False)
    assert r.ok and "def add" in r.answer.lower()
    ok, note = safe_micro_exec("def add(a, b):\n    return a + b\n")
    assert ok, note
    ok2, note2 = run_spec_asserts(
        "def add(a, b):\n    return a + b\n",
        "Write a Python function that adds two numbers.",
    )
    assert ok2 and note2.startswith("spec-assert"), note2
    bad, _ = run_spec_asserts(
        "def add(a, b):\n    return a - b\n",
        "Write a Python function that adds two numbers.",
    )
    assert not bad
    bank = (
        "class BankAccount:\n"
        "    def __init__(self, balance=0):\n"
        "        self.balance = balance\n"
        "    def deposit(self, amount):\n"
        "        self.balance += amount\n"
        "    def withdraw(self, amount):\n"
        "        if amount > self.balance:\n"
        "            raise ValueError('overdraft')\n"
        "        self.balance -= amount\n"
    )
    okb, noteb = run_spec_asserts(bank, "Write a BankAccount class with deposit and withdraw.")
    assert okb and noteb.startswith("spec-assert"), noteb


def test_evidence_quorum() -> None:
    digest = (
        "1. Python 3.12 release\n"
        "https://example.com/a\n"
        "Python 3.12 adds better error messages and performance tweaks.\n\n"
        "2. Python release notes\n"
        "https://example.com/b\n"
        "The latest Python release focuses on typing and speed improvements.\n"
    )
    ok, ans, note = evidence_quorum(digest, "latest Python release", min_hits=2)
    assert ok, (ok, ans, note)
    assert "python" in ans.lower()
    assert "Sources:" in ans and "example.com" in ans
    bad_ok, _, bad_note = evidence_quorum(
        "1. Cooking pasta\nhttps://x\nBoil water and add salt.\n",
        "latest Python release",
        min_hits=2,
    )
    assert not bad_ok


def test_chat_tvs_header() -> None:
    from tiny_slm.chat import TinyChat

    c = TinyChat(auto_search=False, log_traces=False)
    c.clear_memory()
    c.reset()
    a, _ = c.generate_reply("integrate x**2", temperature=0.1)
    assert "[tvs]" in a or "Verified" in a or "x**3" in a.replace(" ", "")


def main() -> None:
    tests = [
        test_domains,
        test_math_tvs,
        test_agent_tvs_defers_research,
        test_code_tvs_and_micro_exec,
        test_evidence_quorum,
        test_chat_tvs_header,
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
    print("test_tvs OK")


if __name__ == "__main__":
    main()
