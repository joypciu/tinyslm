"""Syntax verification for coding answers (fail closed, no hallucination)."""

from __future__ import annotations

import ast
import re
from typing import List, Optional, Tuple


_FENCE = re.compile(r"```(?:python|py)?\s*([\s\S]*?)```", re.I)


def extract_python_blocks(text: str) -> List[str]:
    t = text or ""
    blocks = [m.group(1).strip() for m in _FENCE.finditer(t) if m.group(1).strip()]
    if blocks:
        return blocks
    # Heuristic: contiguous region starting at first def/class/import
    lines = t.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(r"\s*(def |class |import |from )", line):
            start = i
            break
    if start is None:
        return []
    chunk = "\n".join(lines[start:]).strip()
    # Cut trailing prose after blank line + sentence
    parts = re.split(r"\n\n(?=[A-Z])", chunk, maxsplit=1)
    return [parts[0].strip()] if parts[0].strip() else []


def verify_python_syntax(text: str) -> Tuple[bool, str]:
    """Return (ok, note). ok=True if at least one Python block parses."""
    blocks = extract_python_blocks(text)
    if not blocks:
        # Allow tiny snippets like "nums.sort()" without def
        snippet = (text or "").strip()
        if snippet and len(snippet) < 200 and not looks_like_prose(snippet):
            try:
                ast.parse(snippet)
                return True, "snippet-ok"
            except SyntaxError as exc:
                return False, f"syntax-error: {exc.msg}"
        return False, "no-python-block"
    errors = []
    for i, block in enumerate(blocks):
        try:
            ast.parse(block)
            return True, f"block-{i}-ok"
        except SyntaxError as exc:
            errors.append(exc.msg or "syntax-error")
    return False, "syntax-error: " + (errors[0] if errors else "unknown")


def looks_like_prose(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if re.search(r"\b(def |class |import |return |for |while |if )", t):
        return False
    # Sentences with spaces and punctuation, no code ops
    return bool(re.search(r"[.!?]", t)) and not re.search(r"[=\[\]{}():]", t)


def parse_io_examples(user: str) -> List[Tuple[str, str, str]]:
    """Parse example I/O from the user ask → (fn_hint, call_args, expected).

    Supports: f(1, 2) -> 3 | example: add(2, 3) -> 5 | add(2, 3) => 5
    """
    u = user or ""
    out: List[Tuple[str, str, str]] = []
    # Expected: short literal only so "f(1)->2 and f(3)->4" yields two examples
    lit = (
        r"(?:[-+]?\d+(?:\.\d+)?|None|True|False|"
        r"'[^']*'|\"[^\"]*\"|\[[^\]]*\]|\{[^}]*\})"
    )
    for m in re.finditer(
        rf"(?:example\s*:?\s*)?([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*(?:->|=>|=)\s*({lit})",
        u,
        re.I,
    ):
        fn, args, exp = m.group(1), m.group(2).strip(), m.group(3).strip()
        if fn.lower() in ("e", "g", "eg", "example"):
            continue
        out.append((fn, args, exp))
    return out[:6]


def _infer_spec_asserts(
    user: str, fn_names: List[str], class_names: List[str]
) -> List[str]:
    """Derive tiny behavioral asserts from the user ask (Spec-Assert Coding)."""
    u = (user or "").lower()
    names = set(fn_names)
    classes = set(class_names)
    asserts: List[str] = []

    # Example-driven TDD: user-provided I/O beats hardwired cards
    for fn, args, exp in parse_io_examples(user):
        if fn_names and fn not in names and fn.lower() not in {n.lower() for n in names}:
            # Map to the only function if names don't match exactly
            if len(fn_names) == 1:
                fn = fn_names[0]
            else:
                continue
        elif fn not in names and len(fn_names) == 1:
            fn = fn_names[0]
        # Normalize expected None / True / False / quotes
        exp_n = exp.strip()
        if re.fullmatch(r"none", exp_n, re.I):
            exp_n = "None"
        elif re.fullmatch(r"true|false", exp_n, re.I):
            exp_n = exp_n.capitalize()
        asserts.append(f"assert {fn}({args}) == {exp_n}")

    if "add" in names and any(w in u for w in ("add", "adds", "sum two")):
        if not any("add(" in a for a in asserts):
            asserts.append("assert add(2, 3) == 5")
    if "safe_div" in names or ("div" in u and "zero" in u):
        if "safe_div" in names and not any("safe_div" in a for a in asserts):
            asserts.extend(["assert safe_div(10, 2) == 5", "assert safe_div(10, 0) is None"])
    if "word_count" in names or "word count" in u:
        if "word_count" in names and not any("word_count" in a for a in asserts):
            asserts.append("assert word_count('a a b') == {'a': 2, 'b': 1}")
    if "fib" in names or "fibonacci" in u:
        if "fib" in names and not any("fib(" in a for a in asserts):
            asserts.extend(["assert fib(0) == 0", "assert fib(1) == 1", "assert fib(6) == 8"])
    # Class Spec-Assert: BankAccount deposit/withdraw
    if "BankAccount" in classes or "bankaccount" in u.replace(" ", ""):
        if "BankAccount" in classes:
            asserts.extend(
                [
                    "a = BankAccount(10)",
                    "a.deposit(5)",
                    "assert a.balance == 15",
                    "a.withdraw(3)",
                    "assert a.balance == 12",
                ]
            )
    # Dedupe while preserving order
    seen = set()
    uniq: List[str] = []
    for a in asserts:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq


def run_spec_asserts(code: str, user: str = "") -> Tuple[bool, str]:
    """Syntax + optional Spec-Assert micro-tests for known pure functions/classes."""
    ok, note = verify_python_syntax(code)
    if not ok:
        return False, note
    blocks = extract_python_blocks(code)
    src = blocks[0] if blocks else (code or "").strip()
    if not src:
        return False, "empty"
    if re.search(
        r"\b(import|open|exec|eval|compile|input|__|os|sys|subprocess|socket)\b",
        src,
        re.I,
    ):
        return True, "syntax-only-unsafe-skip"
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return False, f"syntax-error: {exc.msg}"
    fn_names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    class_names = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    asserts = _infer_spec_asserts(user, fn_names, class_names)
    if not asserts:
        return True, note
    ns: dict = {
        "__name__": "__spec_assert__",
        "__builtins__": {
            "range": range, "len": len, "min": min, "max": max, "sum": sum,
            "abs": abs, "enumerate": enumerate, "list": list, "dict": dict,
            "str": str, "int": int, "float": float, "bool": bool,
            "None": None, "True": True, "False": False,
            "ValueError": ValueError, "Exception": Exception,
            "isinstance": isinstance, "type": type, "object": object,
            "__build_class__": __build_class__,
            "__name__": "__spec_assert__",
        },
    }
    try:
        exec(compile(tree, "<spec>", "exec"), ns, ns)
        for line in asserts:
            exec(line, ns, ns)
        return True, f"spec-assert:{len(asserts)}"
    except AssertionError:
        return False, "spec-assert-fail"
    except Exception as exc:
        return False, f"spec-exec:{type(exc).__name__}"


def require_verified_code(user: str, answer: str) -> Optional[str]:
    """If the user asked for code, return answer only when syntax verifies; else None."""
    ulow = (user or "").lower()
    wants_code = any(
        w in ulow
        for w in (
            "python",
            "function",
            "class ",
            "write a",
            "implement",
            "script",
            "code",
            "tkinter",
            "def ",
        )
    )
    if not wants_code:
        return answer
    ok, _ = run_spec_asserts(answer, user)
    return answer if ok else None


def abstain_code_message(user: str) -> str:
    return (
        "I could not produce a syntax-verified Python solution for that request. "
        "Please narrow the ask (one function or one small script), or provide an "
        "example input/output. I will not ship unverified code that may be wrong."
    )
