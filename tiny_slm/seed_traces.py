"""Seed success traces from verified tools (LoRA distill fuel, no chat needed).

Novel for this stack: instead of waiting for live chat wins, we mint
machine-checked (user, answer) pairs from the math engine + code cards, then
append them to success_traces.jsonl for prepare_distill_traces.py / LoRA.
"""

from __future__ import annotations

from typing import List, Tuple

from tiny_slm.code_verify import run_spec_asserts
from tiny_slm.knowledge import answer_from_code_template
from tiny_slm.math_engine import try_solve_math
from tiny_slm.traces import DEFAULT_TRACE_PATH, TraceStore

# Keep small, high-signal, and always machine-checkable
_MATH_SEEDS: List[str] = [
    "What is 2 + 2?",
    "What is 10 percent of 200?",
    "sum from 1 to 100",
    "10 choose 2",
    "factorial of 5",
    "P(5, 3)",
    "gcd of 48 and 18",
    "lcm(12, 18)",
    "Bayes: P(B|A)=0.9, P(A)=0.01, P(B)=0.1, what is P(A|B)?",
    "derivative of sin(x)",
    "taylor series of sin(x) around 0 order 5",
    "eigenvalues of [[1,2],[2,1]]",
    "dot product of [1, 2, 3] and [4, 1, 2]",
    "norm of [3, 4]",
    "mean of [1, 2, 3, 4, 5]",
]

_CODE_SEEDS: List[str] = [
    "Write a Python function that adds two numbers.",
    "How do I reverse a string in Python?",
    "Sort a list in Python.",
    "Write a Python function safe_div(a, b) that returns None on divide-by-zero.",
    "Write a recursive Python function fib(n).",
]


def collect_verified_seeds() -> List[Tuple[str, str, str, List[str]]]:
    """Return (user, answer, mode, verify) for checkable seeds only."""
    out: List[Tuple[str, str, str, List[str]]] = []
    for q in _MATH_SEEDS:
        ans = try_solve_math(q)
        if ans:
            out.append((q, ans, "math", ["symbolic"]))
    for q in _CODE_SEEDS:
        code = answer_from_code_template(q)
        if not code:
            continue
        ok, note = run_spec_asserts(code, q)
        if ok:
            out.append((q, code, "code", [note or "syntax"]))
    return out


def seed_verified_traces(
    store: TraceStore | None = None,
    *,
    path=DEFAULT_TRACE_PATH,
) -> int:
    """Append verified seeds; returns number of new records written."""
    ts = store or TraceStore(path)
    n = 0
    for user, answer, mode, verify in collect_verified_seeds():
        if ts.record(user, answer, mode=mode, source="seed-verified", verify=verify):
            n += 1
    return n
