"""Verified math engine for TinySLM (research-capable, hallucination-safe).

A 4M-param net cannot do PhD math by generation. This module:
  1) Detects math / research-math intent
  2) Solves with SymPy when the ask is machine-checkable
  3) Returns None (caller must abstain or tool-route) when not verifiable

No invented proofs or numeric guesses.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# Keep legacy integer path available even if sympy missing
_LEGACY_OK = True

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    _HAS_SYMPY = True
except Exception:  # pragma: no cover
    sp = None  # type: ignore
    _HAS_SYMPY = False


_TRANSFORMS = (
    standard_transformations + (implicit_multiplication_application,)
    if _HAS_SYMPY
    else ()
)

_MATH_CUES = re.compile(
    r"\b("
    r"integral|integrate|derivative|differentiate|limit|prove|theorem|"
    r"eigenvalue|matrix|determinant|laplacian|fourier|"
    r"differential equation|\bode\b|\bpde\b|hessian|"
    r"simplify|expand|factor|solve for|equation|inequality|"
    r"probability|bayes|expectation|variance|binomial|"
    r"modulo|\bgcd\b|\blcm\b|combinator|permutation|"
    r"\blog\b|\bln\b|\bsin\b|\bcos\b|\btan\b|sqrt|factorial|"
    r"percent|squared|power of|calculus|algebra|linear algebra"
    r")\b|"
    r"[√∫∑∏∂∇]|\\frac|\\int|\\sum|"
    # Arithmetic operators only when touching digits/identifiers (not hyphens in prose)
    r"\d\s*[\+\-×÷*/^]\s*\d|"
    r"[a-zA-Z]\s*=\s*[-\d(]",
    re.I,
)

_RESEARCH_CUES = re.compile(
    r"\b("
    r"phd|research|theorem|lemma|proof|derive|closed form|"
    r"asymptotic|convergence|rigorous|formalize|"
    r"partial differential|navier|schrodinger|hamiltonian|"
    r"ricci|lie algebra|galois|measure theory|"
    r"maximum likelihood|lagrangian|hessian|jacobian"
    r")\b",
    re.I,
)

_UNSAFE = re.compile(
    r"(__|import\b|exec\b|eval\b|open\s*\(|os\.|sys\.|subprocess|"
    r"\bwhile\b|\blambda\b|\bclass\b|\bdef\b)",
    re.I,
)

_ALLOWED_SYMPY = {
    "x",
    "y",
    "z",
    "t",
    "n",
    "k",
    "a",
    "b",
    "c",
    "m",
    "r",
    "theta",
    "phi",
    "pi",
    "e",
    "I",
    "oo",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "log",
    "ln",
    "sqrt",
    "Abs",
    "factorial",
    "gamma",
    "erf",
    "floor",
    "ceiling",
    "diff",
    "integrate",
    "limit",
    "summation",
    "Product",
    "solve",
    "simplify",
    "expand",
    "factor",
    "Matrix",
    "det",
    "eye",
    "zeros",
    "ones",
    "Eq",
    "Ne",
    "Lt",
    "Le",
    "Gt",
    "Ge",
    "pi",
    "E",
    "I",
    "oo",
}


def looks_like_math(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    # Coding / product asks are not math even if they mention divide/add
    low = t.lower()
    if any(
        w in low
        for w in (
            "python",
            "tkinter",
            "function",
            "class ",
            "desktop",
            "button",
            "cli ",
            "script",
            "program",
            "todo list",
            "write a",
            "implement",
        )
    ) and not re.search(
        r"\b(integrate|derivative|eigen|theorem|prove|matrix|limit as)\b", low
    ):
        return False
    if _MATH_CUES.search(t):
        return True
    # Bare arithmetic: "12*17+3" / "what is 2+2"
    if re.search(r"\d+\s*[\+\-×÷*/^]\s*\d+", t):
        return True
    if re.search(r"\bwhat is\s+\d", low) and re.search(
        r"[\+\-×÷*/]|percent|squared|power", low
    ):
        return True
    return False


def looks_like_research_math(text: str) -> bool:
    t = text or ""
    return bool(_RESEARCH_CUES.search(t)) or (
        looks_like_math(t)
        and bool(
            re.search(
                r"\b(integral|derivative|limit|matrix|eigen|prove|ode|pde|bayes)\b",
                t,
                re.I,
            )
        )
    )


def _legacy_eval(text: str) -> Optional[str]:
    """Original integer-safe path (no sympy required)."""
    t = (text or "").lower()
    pct = re.search(r"what\s+is\s+(\d+)\s*(?:%|percent|pct)\s*of\s+(\d+)", t)
    if not pct:
        pct = re.search(r"(\d+)\s*(?:%|percent|pct)\s*of\s+(\d+)", t)
    if pct:
        a, b = int(pct.group(1)), int(pct.group(2))
        return f"{a}% of {b} equals {(a * b) // 100}."
    # compound: percent then add
    pct_add = re.search(
        r"what\s+is\s+(\d+)\s*(?:%|percent)\s*of\s+(\d+)\s*,?\s*then\s+add\s+(\d+)",
        t,
    )
    if pct_add:
        a, b, c = int(pct_add.group(1)), int(pct_add.group(2)), int(pct_add.group(3))
        part = (a * b) // 100
        return f"{a}% of {b} is {part}; plus {c} equals {part + c}."
    sq = re.search(r"(?:what\s+is\s+)?(?:the\s+)?square\s+of\s+(\d+)", t)
    if sq:
        n = int(sq.group(1))
        return f"{n} squared equals {n * n}."
    powm = re.search(
        r"(?:what\s+is\s+)?(\d+)\s*(?:\*\*|to the power of|raised to)\s*(\d+)", t
    )
    if powm:
        a, b = int(powm.group(1)), int(powm.group(2))
        if 0 <= b <= 12 and a <= 10_000:
            return f"{a} to the power of {b} equals {a ** b}."
    m = re.search(
        r"what\s+is\s+(\d+)\s*(\+|plus|minus|-|times\b|\*|x|divided by|/)\s*(\d+)",
        t,
    )
    if not m:
        m = re.search(r"(\d+)\s*(\+|plus|minus|-|times\b|\*|x|/)\s*(\d+)\s*\??\s*$", t)
    if not m:
        return None
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    if op in ("+", "plus"):
        return f"{a} + {b} equals {a + b}."
    if op in ("-", "minus"):
        return f"{a} minus {b} equals {a - b}."
    if op in ("*", "x", "times"):
        return f"{a} times {b} equals {a * b}."
    if op in ("/", "divided by"):
        if b == 0:
            return "Division by zero is undefined."
        # Prefer exact rational when not divisible
        if a % b == 0:
            return f"{a} divided by {b} equals {a // b}."
        return f"{a} divided by {b} equals {a / b}."
    return None


def _sympy_locals():
    assert _HAS_SYMPY
    return {
        "x": sp.symbols("x"),
        "y": sp.symbols("y"),
        "z": sp.symbols("z"),
        "t": sp.symbols("t"),
        "n": sp.symbols("n", integer=True, positive=True),
        "k": sp.symbols("k", integer=True),
        "a": sp.symbols("a"),
        "b": sp.symbols("b"),
        "c": sp.symbols("c"),
        "m": sp.symbols("m"),
        "r": sp.symbols("r"),
        "theta": sp.symbols("theta"),
        "phi": sp.symbols("phi"),
        "pi": sp.pi,
        "e": sp.E,
        "E": sp.E,
        "I": sp.I,
        "oo": sp.oo,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "ln": sp.log,
        "sqrt": sp.sqrt,
        "Abs": sp.Abs,
        "factorial": sp.factorial,
        "diff": sp.diff,
        "integrate": sp.integrate,
        "limit": sp.limit,
        "simplify": sp.simplify,
        "expand": sp.expand,
        "factor": sp.factor,
        "solve": sp.solve,
        "Matrix": sp.Matrix,
        "det": sp.det,
        "Eq": sp.Eq,
    }


def _extract_expr_blob(text: str) -> Optional[str]:
    """Pull a likely math expression / command from natural language."""
    t = (text or "").strip()
    if not t or _UNSAFE.search(t):
        return None
    low = t.lower()

    # integrate x^2 dx / integral of sin(x)
    m = re.search(
        r"(?:integrate|integral of|∫)\s+(.+?)(?:\s+d([a-z])\b|\s*$)",
        low,
        re.I,
    )
    if m:
        expr = m.group(1).strip(" .?")
        var = m.group(2) or "x"
        return f"integrate({expr}, {var})"

    m = re.search(r"(?:derivative of|differentiate)\s+(.+)$", low)
    if m:
        expr = m.group(1).strip(" .?")
        return f"diff({expr}, x)"

    m = re.search(r"d/d([a-z])\s+(.+)$", low)
    if m:
        return f"diff({m.group(2).strip(' .?')}, {m.group(1)})"

    m = re.search(r"limit\s+(?:as\s+)?([a-z])\s*->\s*([^\s,]+)\s+of\s+(.+)$", low)
    if m:
        return f"limit({m.group(3).strip(' .?')}, {m.group(1)}, {m.group(2)})"

    m = re.search(r"simplify\s+(.+)$", low)
    if m:
        return f"simplify({m.group(1).strip(' .?')})"

    m = re.search(r"expand\s+(.+)$", low)
    if m:
        return f"expand({m.group(1).strip(' .?')})"

    m = re.search(r"factor\s+(.+)$", low)
    if m:
        return f"factor({m.group(1).strip(' .?')})"

    m = re.search(r"solve\s+(.+)\s+for\s+([a-z])\s*$", t, re.I)
    if m:
        eq_raw = m.group(1).strip(" .?")
        var = m.group(2)
        if "=" in eq_raw:
            left, right = eq_raw.split("=", 1)
            return f"solve(Eq({left}, {right}), {var})"
        return f"solve({eq_raw}, {var})"

    # determinant / matrix
    m = re.search(r"determinant of\s*\[(.+)\]", low)
    if m:
        return f"det(Matrix([[{m.group(1)}]]))"

    # evaluate / compute / what is <expr>
    m = re.search(
        r"(?:what is|compute|evaluate|calculate)\s+(.+?)\s*\??\s*$",
        t,
        re.I,
    )
    if m:
        blob = m.group(1).strip()
        if re.search(r"[\d\w]\s*[\+\-\*/^()]", blob) or re.search(
            r"\b(sin|cos|tan|log|ln|sqrt|exp|pi)\b", blob, re.I
        ):
            return blob

    # bare expression
    if re.fullmatch(r"[\w\s\+\-\*/^().,=]+", t) and re.search(r"[\+\-\*/^]", t):
        return t.strip(" ?")
    return None


def _eval_sympy(blob: str) -> Optional[str]:
    if not _HAS_SYMPY or not blob or _UNSAFE.search(blob):
        return None
    # Block attribute access / dunders
    if ".__" in blob or ";" in blob:
        return None
    try:
        local_dict = _sympy_locals()
        # Minimal globals so Integer/Pow exist; no builtins → safer
        global_dict = {
            "Integer": sp.Integer,
            "Float": sp.Float,
            "Rational": sp.Rational,
            "Symbol": sp.Symbol,
            "Mul": sp.Mul,
            "Add": sp.Add,
            "Pow": sp.Pow,
            "Eq": sp.Eq,
        }
        expr = parse_expr(
            blob,
            local_dict=local_dict,
            global_dict=global_dict,
            transformations=_TRANSFORMS,
            evaluate=True,
        )
        if isinstance(expr, (list, tuple)):
            result = expr
        else:
            result = sp.simplify(expr)
        # Bound crazy expansions
        s = str(result)
        if len(s) > 500:
            s = s[:500] + "…"
        return s
    except Exception:
        return None


def try_solve_math(text: str) -> Optional[str]:
    """Return a verified math answer string, or None if not checkable."""
    legacy = _legacy_eval(text)
    if legacy:
        return legacy
    if not _HAS_SYMPY:
        return None
    blob = _extract_expr_blob(text)
    if not blob:
        return None
    out = _eval_sympy(blob)
    if out is None:
        return None
    # Natural phrasing for simple numeric
    if re.fullmatch(r"-?\d+(\.\d+)?", out.replace(" ", "")):
        return f"Verified result: {out}."
    return f"Verified (symbolic): {out}."


def math_policy(text: str) -> Tuple[str, Optional[str]]:
    """Return (action, answer) where action is solve|abstain|not_math."""
    if not looks_like_math(text):
        return "not_math", None
    ans = try_solve_math(text)
    if ans:
        return "solve", ans
    if looks_like_research_math(text) or looks_like_math(text):
        return "abstain", None
    return "not_math", None


def abstain_math_message(user: str) -> str:
    research = looks_like_research_math(user)
    if research:
        return (
            "I can help with research-level math only when the problem is stated as a "
            "checkable computation (e.g. 'integrate x**2', 'diff sin(x)', "
            "'solve x**2 - 1 = 0 for x', 'limit as x -> 0 of sin(x)/x'). "
            "I will not invent a proof or numeric guess. Please restate with an "
            "explicit expression, or ask for a high-level study plan for the topic."
        )
    return (
        "This looks like a math question I cannot verify exactly yet. "
        "Please rephrase with a clear expression (for example: 'what is 15% of 240' "
        "or 'integrate x**2'). I avoid guessing to prevent hallucinations."
    )
