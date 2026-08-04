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
    ok, _ = verify_python_syntax(answer)
    return answer if ok else None


def abstain_code_message(user: str) -> str:
    return (
        "I could not produce a syntax-verified Python solution for that request. "
        "Please narrow the ask (one function or one small script), or provide an "
        "example input/output. I will not ship unverified code that may be wrong."
    )
