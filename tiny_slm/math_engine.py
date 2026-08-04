"""Verified math engine for TinySLM (research-capable, hallucination-safe).

A 4M-param net cannot do PhD math by generation. This module:
  1) Detects math / research-math intent
  2) Solves with SymPy when the ask is machine-checkable
  3) Returns None (caller must abstain or tool-route) when not verifiable

No invented proofs or numeric guesses.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

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
    r"taylor|maclaurin|series expansion|"
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
    # compound first (before bare percent steals the match)
    pct_add = re.search(
        r"what\s+is\s+(\d+)\s*(?:%|percent)\s*of\s+(\d+)\s*,?\s*then\s+add\s+(\d+)",
        t,
    )
    if pct_add:
        a, b, c = int(pct_add.group(1)), int(pct_add.group(2)), int(pct_add.group(3))
        part = (a * b) // 100
        return f"{a}% of {b} is {part}; plus {c} equals {part + c}."
    # sum 1..n
    sumn = re.search(
        r"(?:sum|add)\s+(?:from\s+)?(?:k\s*=\s*)?1\s*(?:to|through|\.\.)\s*(\d+)",
        t,
    )
    if not sumn:
        sumn = re.search(r"sum\s+(?:of\s+)?(?:the\s+)?(?:numbers?\s+)?1\s*(?:to|through)\s*(\d+)", t)
    if sumn:
        n = int(sumn.group(1))
        if 1 <= n <= 1_000_000:
            total = n * (n + 1) // 2
            return f"Sum from 1 to {n} equals {total}."
    pct = re.search(r"what\s+is\s+(\d+)\s*(?:%|percent|pct)\s*of\s+(\d+)", t)
    if not pct:
        pct = re.search(r"(\d+)\s*(?:%|percent|pct)\s*of\s+(\d+)", t)
    if pct:
        a, b = int(pct.group(1)), int(pct.group(2))
        return f"{a}% of {b} equals {(a * b) // 100}."
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
        "summation": sp.summation,
        "series": sp.series,
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
    m = re.search(r"determinant of\s*(\[\[.+?\]\])", low)
    if m:
        return f"det(Matrix({m.group(1)}))"

    # eigenvalues of [[a,b],[c,d]]
    m = re.search(r"eigenvalues?\s+of\s*(\[\[.+?\]\])", low)
    if m:
        return f"Matrix({m.group(1)}).eigenvals()"

    # taylor / maclaurin: "taylor series of sin(x) around 0 order 5"
    m = re.search(
        r"(?:taylor|maclaurin)\s+(?:series\s+)?(?:expand\s+|of\s+)(.+?)"
        r"(?:\s+(?:around|at|about)\s+([-\d.]+|0))?"
        r"(?:\s+(?:to\s+)?(?:order|n\s*=)\s*(\d+))",
        low,
    )
    if m:
        expr = m.group(1).strip(" .?")
        point = m.group(2) or "0"
        order = m.group(3) or "5"
        # Infer variable from expr
        var = "x"
        vm = re.search(r"\b([a-z])\b", expr)
        if vm:
            var = vm.group(1)
        n = int(order)
        if 1 <= n <= 12:
            return f"series({expr}, {var}, {point}, {n})"

    # summation: sum k=1 to n of k**2
    m = re.search(
        r"sum(?:mation)?\s+([a-z])\s*=\s*(\d+)\s+to\s+(\d+)\s+of\s+(.+)$",
        low,
    )
    if m:
        var, a, b, expr = m.group(1), m.group(2), m.group(3), m.group(4).strip(" .?")
        return f"summation({expr}, ({var}, {a}, {b}))"

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


def _try_multihop(text: str) -> Optional[str]:
    """Verified multi-hop: split on 'then' and chain numeric CAS steps."""
    raw = (text or "").strip()
    if not re.search(r"\bthen\b", raw, re.I):
        return None
    # Dedicated percent-then-add (legacy) — only when pattern matches whole ask
    pct_add = _legacy_eval(raw)
    if pct_add and "plus" in pct_add and "%" in pct_add:
        return pct_add
    parts = re.split(r"\s*,?\s*\bthen\b\s+", raw, flags=re.I)
    if len(parts) < 2 or len(parts) > 4:
        return None
    notes: List[str] = []
    carry: Optional[str] = None
    for i, part in enumerate(parts):
        piece = part.strip(" .?")
        if i == 0:
            step = try_solve_math_once(piece)
            if not step:
                return None
            nums = re.findall(r"-?\d+(?:\.\d+)?", step)
            # Prefer last meaningful number (result)
            carry = nums[-1] if nums else None
            if carry is None:
                return None
            notes.append(step.rstrip("."))
            continue
        if carry is None:
            return None
        m = re.match(
            r"(add|plus|subtract|minus|multiply by|times|divide by)\s+(-?\d+(?:\.\d+)?)$",
            piece,
            re.I,
        )
        if not m:
            return None
        op, val = m.group(1).lower(), m.group(2)
        blob = {
            "add": f"({carry})+({val})",
            "plus": f"({carry})+({val})",
            "subtract": f"({carry})-({val})",
            "minus": f"({carry})-({val})",
            "multiply by": f"({carry})*({val})",
            "times": f"({carry})*({val})",
            "divide by": f"({carry})/({val})",
        }[op]
        out = _eval_sympy(blob) if _HAS_SYMPY else None
        if out is None:
            try:
                a_f, b_f = float(carry), float(val)
                if op in ("add", "plus"):
                    out = str(a_f + b_f)
                elif op in ("subtract", "minus"):
                    out = str(a_f - b_f)
                elif op in ("multiply by", "times"):
                    out = str(a_f * b_f)
                elif op == "divide by":
                    if b_f == 0:
                        return None
                    out = str(a_f / b_f)
            except Exception:
                return None
        if out is None:
            return None
        # Prefer clean ints
        try:
            if float(out) == int(float(out)):
                out = str(int(float(out)))
        except Exception:
            pass
        notes.append(f"{op} {val} -> {out}")
        carry = out
    if carry is None:
        return None
    return "Verified multi-hop: " + "; ".join(notes) + f" => {carry}."


def try_solve_math_once(text: str) -> Optional[str]:
    """Single-hop solve (no multi-hop recursion)."""
    legacy = _legacy_eval(text)
    if legacy:
        return legacy
    if not _HAS_SYMPY:
        return None
    blob = _extract_expr_blob(text)
    if not blob:
        return None
    # Method calls like Matrix(...).eigenvals() need direct eval
    if ".eigenvals()" in blob and _HAS_SYMPY:
        try:
            m = re.search(r"Matrix\((\[\[.+?\]\])\)\.eigenvals\(\)", blob)
            if m:
                mat = sp.Matrix(sp.sympify(m.group(1)))
                return f"Verified (symbolic): {mat.eigenvals()}."
        except Exception:
            return None
    # series(expr, var, point, n) — parse_expr can mishandle Order terms
    if blob.startswith("series(") and _HAS_SYMPY:
        try:
            m = re.match(
                r"series\((.+),\s*([a-z]),\s*([^,]+),\s*(\d+)\)$",
                blob,
            )
            if m:
                expr_s, var_s, point_s, n_s = m.groups()
                local_dict = _sympy_locals()
                expr = parse_expr(
                    expr_s,
                    local_dict=local_dict,
                    transformations=_TRANSFORMS,
                    evaluate=True,
                )
                var = sp.Symbol(var_s)
                point = parse_expr(point_s, local_dict=local_dict, evaluate=True)
                n = int(n_s)
                if 1 <= n <= 12:
                    ser = sp.series(expr, var, point, n)
                    return f"Verified (symbolic): {ser}."
        except Exception:
            return None
    out = _eval_sympy(blob)
    if out is None:
        return None
    if re.fullmatch(r"-?\d+(\.\d+)?", out.replace(" ", "")):
        return f"Verified result: {out}."
    return f"Verified (symbolic): {out}."


def try_solve_math(text: str) -> Optional[str]:
    """Return a verified math answer string, or None if not checkable."""
    if re.search(r"\bthen\b", text or "", re.I):
        hop = _try_multihop(text)
        if hop:
            return hop
    return try_solve_math_once(text)


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
    ulow = (user or "").lower()
    if re.search(r"\b(pde|partial differential|navier|schrodinger)\b", ulow):
        return (
            "That looks like an open or research-grade PDE/analysis ask. "
            "I only ship machine-checkable CAS results (integrals, Taylor series, "
            "eigenvalues, etc.) and will not invent a proof or closed form."
        )
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
