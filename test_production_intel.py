"""Production intelligence: math verify, code verify, calibrated routing."""

from __future__ import annotations

from tiny_slm.code_verify import require_verified_code, verify_python_syntax
from tiny_slm.math_engine import looks_like_research_math, math_policy, try_solve_math
from tiny_slm.policy import decide_route


def test_basic_and_symbolic_math() -> None:
    assert "4" in (try_solve_math("What is 2 + 2?") or "")
    assert "20" in (try_solve_math("What is 10 percent of 200?") or "")
    assert "48" in (try_solve_math("What is 15 percent of 240, then add 12?") or "")
    assert "5050" in (try_solve_math("sum from 1 to 100") or "")
    assert "12" in (try_solve_math("what is 2 + 2, then multiply by 3") or "")
    ans = try_solve_math("integrate x**2")
    assert ans and ("x**3" in ans.replace(" ", "") or "x^3" in ans), ans
    ans2 = try_solve_math("derivative of sin(x)")
    assert ans2 and "cos" in ans2.lower(), ans2
    ans3 = try_solve_math("solve x**2 - 1 = 0 for x")
    assert ans3 and ("-1" in ans3 and "1" in ans3), ans3
    ans4 = try_solve_math("eigenvalues of [[1,2],[2,1]]")
    assert ans4 and ("-1" in ans4 and "3" in ans4), ans4
    ans5 = try_solve_math("taylor series of sin(x) around 0 order 5")
    assert ans5 and ("x" in ans5) and ("x**3" in ans5.replace(" ", "") or "x^3" in ans5), ans5
    ans6 = try_solve_math("jacobian of [x**2*y, x+y] w.r.t. [x, y]")
    assert ans6 and ("2" in ans6) and ("Matrix" in ans6 or "[" in ans6), ans6
    ans7 = try_solve_math(
        "Bayes: P(B|A)=0.9, P(A)=0.01, P(B)=0.1, what is P(A|B)?"
    )
    assert ans7 and "0.09" in ans7, ans7
    ans8 = try_solve_math("hessian of x**3 + x*y**2 w.r.t. [x, y]")
    assert ans8 and "6*x" in ans8.replace(" ", ""), ans8
    assert "= 6" in (try_solve_math("gcd of 48 and 18") or "")
    assert "= 36" in (try_solve_math("lcm(12, 18)") or "")
    assert "= 45" in (try_solve_math("10 choose 2") or "")
    ans9 = try_solve_math("binomial probability n=5 p=0.5 k=2")
    assert ans9 and "0.3125" in ans9, ans9
    ans10 = try_solve_math("inverse of [[1,2],[3,4]]")
    assert ans10 and ("-2" in ans10 or "Matrix" in ans10), ans10
    assert "5" in (try_solve_math("trace of [[1,2],[3,4]]") or "")
    assert "= 12" in (try_solve_math("dot product of [1, 2, 3] and [4, 1, 2]") or "")
    assert "= 5" in (try_solve_math("norm of [3, 4]") or "")
    ans11 = try_solve_math("partial of x**2*y w.r.t. x")
    assert ans11 and "2*x*y" in ans11.replace(" ", ""), ans11
    assert "3" in (try_solve_math("mean of [1, 2, 3, 4, 5]") or "")
    assert "2" in (try_solve_math("variance of [1, 2, 3, 4, 5]") or "")
    assert "= 120" in (try_solve_math("factorial of 5") or "")
    assert "= 60" in (try_solve_math("P(5, 3)") or "")
    ans12 = try_solve_math("[[1,2],[3,4]] times [[0,1],[1,0]]")
    assert ans12 and ("2" in ans12) and ("Matrix" in ans12 or "[" in ans12), ans12
    ans13 = try_solve_math("integrate x**2 from 0 to 1")
    assert ans13 and ("1/3" in ans13.replace(" ", "") or "0.333" in ans13), ans13
    ans14 = try_solve_math("solve x+y=5, x-y=1 for x,y")
    assert ans14 and ("3" in ans14) and ("2" in ans14), ans14
    assert "= 5" in (try_solve_math("abs(3+4i)") or "")
    ans15 = try_solve_math("(1+2i)*(3-i)")
    assert ans15 and ("5" in ans15) and ("I" in ans15 or "i" in ans15.lower()), ans15
    assert "2" in (try_solve_math("log10 of 100") or "")
    assert "0" in (try_solve_math("ln of 1") or "")
    assert "1" in (try_solve_math("exp(0)") or "")
    assert "5" in (try_solve_math("sqrt of 25") or "")
    assert "2" in (try_solve_math("3rd root of 8") or "")
    assert "pi" in (try_solve_math("180 degrees to radians") or "").lower()
    assert "= 1" in (try_solve_math("17 mod 4") or "")
    action_pde, _ = math_policy("Solve the Navier-Stokes PDE for my thesis.")
    assert action_pde == "abstain"


def test_code_card_routing() -> None:
    r = decide_route("How do I reverse a string in Python?", auto_search=False)
    assert r.action == "grounded" and r.reason == "code-card"
    r2 = decide_route("Sort a list in Python.", auto_search=False)
    assert r2.action == "grounded"


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
        test_code_card_routing,
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
