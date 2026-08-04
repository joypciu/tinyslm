"""Production intelligence: math verify, code verify, calibrated routing."""

from __future__ import annotations

from tiny_slm.code_verify import require_verified_code, verify_python_syntax
from tiny_slm.math_engine import looks_like_research_math, math_policy, try_solve_math
from tiny_slm.policy import decide_route


def test_basic_and_symbolic_math() -> None:
    assert "4" in (try_solve_math("What is 2 + 2?") or "")
    assert "20" in (try_solve_math("What is 10 percent of 200?") or "")
    ans = try_solve_math("integrate x**2")
    assert ans and ("x**3" in ans.replace(" ", "") or "x^3" in ans), ans
    ans2 = try_solve_math("derivative of sin(x)")
    assert ans2 and "cos" in ans2.lower(), ans2
    ans3 = try_solve_math("solve x**2 - 1 = 0 for x")
    assert ans3 and ("-1" in ans3 and "1" in ans3), ans3


def test_research_math_abstains() -> None:
    assert looks_like_research_math("Prove the Riemann hypothesis rigorously.")
    action, ans = math_policy("Prove the Riemann hypothesis rigorously.")
    assert action == "abstain" and ans is None
    r = decide_route("Prove Navier-Stokes smoothness for my PhD thesis.", auto_search=False)
    assert r.action == "abstain"
    assert r.message and "will not invent" in r.message.lower() or "checkable" in r.message.lower()


def test_router_actions() -> None:
    assert decide_route("What is RAM?", auto_search=False).action == "grounded"
    assert decide_route("What is 2 + 2?", auto_search=False).action == "grounded"
    assert decide_route("Write a Python function that adds two numbers.", auto_search=False).action in (
        "grounded",
        "agent",
    )
    assert decide_route("Plan a short study session step by step.", auto_search=False).action in (
        "grounded",
        "agent",
    )
    assert decide_route("search the web for python release news", auto_search=True).action == "search"
    assert decide_route("Who won the 2024 election?", auto_search=True).action == "search"
    assert decide_route("Invent a biography of a fake astronaut named Zorp.", auto_search=False).action == "abstain"


def test_code_verify() -> None:
    ok, _ = verify_python_syntax("def add(a, b):\n    return a + b\n")
    assert ok
    bad_ok, _ = verify_python_syntax("def add(a, b)\n    return a + b")
    assert not bad_ok
    assert require_verified_code("write python", "def add(a, b):\n    return a + b")
    assert require_verified_code("write python", "Not real code at all.") is None


def test_chat_math_and_abstain() -> None:
    from tiny_slm.chat import TinyChat

    c = TinyChat(auto_search=False, log_traces=False)
    c.clear_memory()
    c.reset()
    a, _ = c.generate_reply("integrate x**2", temperature=0.1)
    body = a.split("[model]")[-1]
    assert "x**3" in body.replace(" ", "") or "x^3" in body or "Verified" in body

    b, _ = c.generate_reply(
        "Prove the Riemann hypothesis with a full formal proof.", temperature=0.2
    )
    body_b = b.split("[model]")[-1].lower()
    assert "will not" in body_b or "checkable" in body_b or "won't guess" in body_b or "invent" in body_b
    assert "ACTION=abstain" in b or "MODE=abstain" in b


def test_chat_coding_verified() -> None:
    from tiny_slm.chat import TinyChat

    c = TinyChat(auto_search=False, log_traces=False)
    c.clear_memory()
    c.reset()
    a, _ = c.generate_reply(
        "Write a Python function that adds two numbers.", temperature=0.1
    )
    body = a.split("[model]")[-1]
    assert "def add" in body.lower()
    ok, _ = verify_python_syntax(body)
    assert ok


def main() -> None:
    tests = [
        test_basic_and_symbolic_math,
        test_research_math_abstains,
        test_router_actions,
        test_code_verify,
        test_chat_math_and_abstain,
        test_chat_coding_verified,
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
    print("test_production_intel OK")


if __name__ == "__main__":
    main()
