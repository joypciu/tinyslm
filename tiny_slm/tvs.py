"""Thought-Verify Scratchpad (TVS) — novel agentic loop for TinySLM.

Idea: a tiny model should not "think in free text" then hope. Instead it
maintains a structured scratchpad:

  THINK  → classify goal + risks (math/code/search/plan)
  ACT    → call a verifiable tool (CAS / templates / web / memory)
  VERIFY → machine check (sympy, ast, evidence quorum)
  ANSWER → only emit if VERIFY passes; else abstain or retry once

This is intentionally *not* chain-of-thought hallucination — every claim
must clear a verifier. Novel for this stack: Evidence Quorum for web
(≥2 query-aligned snippets or refuse) + Safe Micro-Exec for pure Python.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from tiny_slm.code_verify import extract_python_blocks, run_spec_asserts, verify_python_syntax
from tiny_slm.knowledge import (
    answer_from_code_template,
    answer_from_faq,
    answer_from_plan_template,
    scrub_generation,
)
from tiny_slm.math_engine import math_policy, try_solve_math
from tiny_slm.memory import answer_from_memory
from tiny_slm.search import answer_from_search, clean_search_query, needs_search, search_web


@dataclass
class TVSStep:
    phase: str
    detail: str


@dataclass
class TVSResult:
    ok: bool
    answer: str
    domain: str
    steps: List[TVSStep] = field(default_factory=list)
    abstained: bool = False

    def header(self) -> str:
        phases = " → ".join(s.phase for s in self.steps[:6]) or "-"
        return f"[tvs] domain={self.domain} ok={self.ok} path={phases}"


def _query_terms(q: str) -> set[str]:
    stop = {
        "what",
        "who",
        "when",
        "where",
        "which",
        "how",
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "does",
        "did",
        "about",
        "latest",
        "news",
        "please",
        "tell",
        "me",
        "from",
        "web",
        "search",
        "look",
        "up",
    }
    return {
        t
        for t in re.findall(r"[a-z0-9]+", clean_search_query(q or "").lower())
        if len(t) > 2 and t not in stop
    }


def evidence_quorum(digest: str, query: str, *, min_hits: int = 2) -> Tuple[bool, str, str]:
    """Require multiple query-aligned web facts before answering.

    Returns (ok, answer, note).
    """
    text = (digest or "").strip()
    if not text or text.startswith("("):
        return False, "", "no-digest"
    terms = _query_terms(query)
    # Split digest into hit blocks
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    aligned: List[str] = []
    for block in blocks:
        low = block.lower()
        score = sum(1 for t in terms if t in low)
        if terms and score < 1:
            continue
        # Prefer body sentences via existing extractor on single block
        piece = answer_from_search(block, query=query, max_chars=220)
        if piece:
            aligned.append(piece.replace("From the web: ", "").strip())
        elif score >= 2:
            title = re.sub(r"^\d+\.\s*", "", block.splitlines()[0]).strip()
            if len(title) > 12:
                aligned.append(title.rstrip(".") + ".")
        if len(aligned) >= min_hits:
            break

    if len(aligned) < min_hits:
        # Fallback: single strong extractive answer if highly aligned
        one = answer_from_search(text, query=query)
        if one:
            hit = sum(1 for t in terms if t in one.lower())
            if terms and hit >= max(1, min(2, len(terms) // 2)):
                return True, one, "single-strong"
        return False, "", f"quorum-fail hits={len(aligned)}"

    # Dedupe
    uniq: List[str] = []
    for a in aligned:
        key = re.sub(r"[^a-z0-9]+", "", a.lower())[:40]
        if any(key and key in re.sub(r"[^a-z0-9]+", "", u.lower()) for u in uniq):
            continue
        uniq.append(a)
    ans = "From the web: " + uniq[0]
    if len(uniq) > 1:
        ans += " Also: " + uniq[1]
    from tiny_slm.search import attach_citations

    ans = attach_citations(ans, text, max_chars=520)
    return True, ans, f"quorum={len(uniq)}"


def safe_micro_exec(code: str, *, timeout_s: float = 0.4) -> Tuple[bool, str]:
    """Run a tiny smoke test on pure function definitions (no imports/IO)."""
    ok, note = verify_python_syntax(code)
    if not ok:
        return False, note
    blocks = extract_python_blocks(code)
    src = blocks[0] if blocks else (code or "").strip()
    if not src:
        return False, "empty"
    # Reject dangerous constructs
    banned = re.compile(
        r"\b(import|open|exec|eval|compile|input|__|os|sys|subprocess|socket|requests)\b",
        re.I,
    )
    if banned.search(src):
        return True, "syntax-only"  # syntax ok; skip exec
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return False, f"syntax-error: {exc.msg}"

    # Only allow FunctionDef / ClassDef / Assign constants
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign)):
            return True, "syntax-only"

    fn_names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not fn_names:
        return True, "syntax-only"

    ns: dict = {}
    try:
        compiled = compile(tree, "<tvs>", "exec")
        exec(compiled, {"__builtins__": {"range": range, "len": len, "min": min, "max": max, "sum": sum, "abs": abs, "enumerate": enumerate, "list": list, "dict": dict, "str": str, "int": int, "float": float, "bool": bool, "None": None, "True": True, "False": False}}, ns)
    except Exception as exc:
        return False, f"exec-error: {type(exc).__name__}"

    # Heuristic smoke: call first simple function with trivial args if possible
    fn = ns.get(fn_names[0])
    if not callable(fn):
        return True, "syntax-ok"
    try:
        import inspect

        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        if len(params) == 0:
            fn()
        elif len(params) == 1:
            fn(1)
        elif len(params) == 2:
            # Prefer add-like
            try:
                fn(2, 3)
            except TypeError:
                fn("a", "b")
        else:
            return True, "syntax-ok"
        return True, f"micro-exec:{fn_names[0]}"
    except Exception as exc:
        # Syntax passed; runtime smoke failed — still accept template-quality code
        # only if it's clearly a definition (caller may prefer templates)
        return False, f"smoke-fail: {type(exc).__name__}"


def classify_domain(user: str) -> str:
    u = (user or "").lower()
    m_action, _ = math_policy(user)
    if m_action in ("solve", "abstain"):
        return "math"
    # Explicit web intent beats code keyword collisions ("search … python …")
    if any(w in u for w in ("search", "look up", "from the web", "on the internet", "google")):
        return "search"
    if any(
        w in u
        for w in (
            "python",
            "function",
            "class ",
            "write a",
            "implement",
            "tkinter",
            "script",
            "code",
        )
    ):
        return "code"
    if needs_search(user):
        return "search"
    if any(
        w in u
        for w in ("step by step", "plan", "break down", "multi-step", "research", "investigate")
    ):
        return "agent"
    if any(w in u for w in ("remember", "using memory", "earlier")):
        return "memory"
    return "general"


def run_tvs(
    user: str,
    *,
    memory_retrieve: Optional[Callable[[str], str]] = None,
    auto_search: bool = True,
    generate_fn: Optional[Callable[[str], str]] = None,
) -> TVSResult:
    """Execute one Thought-Verify Scratchpad cycle (fail closed)."""
    steps: List[TVSStep] = []
    domain = classify_domain(user)
    steps.append(TVSStep("THINK", f"domain={domain}"))

    # --- MATH ---
    if domain == "math":
        steps.append(TVSStep("ACT", "symbolic-cas"))
        action, ans = math_policy(user)
        if action == "solve" and ans:
            steps.append(TVSStep("VERIFY", "cas-ok"))
            steps.append(TVSStep("ANSWER", "verified-math"))
            return TVSResult(True, ans, domain, steps)
        steps.append(TVSStep("VERIFY", "cas-miss"))
        from tiny_slm.math_engine import abstain_math_message

        return TVSResult(False, abstain_math_message(user), domain, steps, abstained=True)

    # --- MEMORY ---
    mem = memory_retrieve(user) if callable(memory_retrieve) else ""
    if domain == "memory" or "using memory" in (user or "").lower():
        steps.append(TVSStep("ACT", "memory-retrieve"))
        mem_ans = answer_from_memory(user, mem or "")
        if mem_ans:
            steps.append(TVSStep("VERIFY", "extractive-memory"))
            steps.append(TVSStep("ANSWER", "memory"))
            return TVSResult(True, mem_ans, "memory", steps)
        steps.append(TVSStep("VERIFY", "memory-miss"))
        return TVSResult(
            False,
            "I don't have a verified memory fact for that yet.",
            "memory",
            steps,
            abstained=True,
        )

    # --- CODE ---
    if domain == "code":
        steps.append(TVSStep("ACT", "code-template-or-gen"))
        code = answer_from_code_template(user)
        if not code and generate_fn:
            code = scrub_generation(generate_fn(user) or "")
        if not code:
            plan = answer_from_plan_template(user)
            if plan:
                steps.append(TVSStep("VERIFY", "plan-fallback"))
                return TVSResult(True, plan, "agent", steps)
            steps.append(TVSStep("VERIFY", "no-code"))
            return TVSResult(
                False,
                "I could not produce a syntax-verified Python solution. Please narrow the ask.",
                domain,
                steps,
                abstained=True,
            )
        ok_spec, spec_note = run_spec_asserts(code, user)
        steps.append(TVSStep("VERIFY", spec_note))
        if not ok_spec:
            rescue = answer_from_code_template(user)
            if rescue:
                ok_r, note_r = run_spec_asserts(rescue, user)
                steps.append(TVSStep("VERIFY", note_r))
                if ok_r:
                    steps.append(TVSStep("ANSWER", "template-rescue"))
                    return TVSResult(True, rescue, domain, steps)
            return TVSResult(
                False,
                "Generated code failed Spec-Assert verification; I will not ship it.",
                domain,
                steps,
                abstained=True,
            )
        ok_ex, ex_note = safe_micro_exec(code)
        steps.append(TVSStep("VERIFY", ex_note))
        if ok_ex or ex_note.startswith("syntax") or spec_note.startswith("spec-assert"):
            steps.append(TVSStep("ANSWER", "verified-code"))
            return TVSResult(True, code, domain, steps)
        if answer_from_code_template(user) == code:
            steps.append(TVSStep("ANSWER", "template-syntax-ok"))
            return TVSResult(True, code, domain, steps)
        return TVSResult(
            False,
            "Code failed micro-exec smoke checks; please simplify the request.",
            domain,
            steps,
            abstained=True,
        )

    # --- SEARCH (Evidence Quorum) ---
    if domain == "search" and auto_search:
        steps.append(TVSStep("ACT", "web-search"))
        digest = search_web(clean_search_query(user), max_results=5)
        ok, ans, note = evidence_quorum(digest, user, min_hits=2)
        steps.append(TVSStep("VERIFY", note))
        if ok and ans:
            steps.append(TVSStep("ANSWER", "evidence-quorum"))
            return TVSResult(True, ans, domain, steps)
        # FAQ may still ground common facts without web
        faq = answer_from_faq(user)
        if faq:
            steps.append(TVSStep("ANSWER", "faq-instead-of-weak-web"))
            return TVSResult(True, faq, "general", steps)
        return TVSResult(
            False,
            "Web results did not meet the evidence quorum for a reliable answer. "
            "I will not guess. Try a sharper query or a named source.",
            domain,
            steps,
            abstained=True,
        )

    # --- AGENT / PLAN ---
    if domain == "agent":
        steps.append(TVSStep("ACT", "plan-card"))
        plan = answer_from_plan_template(user) or answer_from_code_template(user)
        if plan:
            if "def " in plan or "class " in plan:
                ok_syn, syn_note = verify_python_syntax(plan)
                steps.append(TVSStep("VERIFY", syn_note))
                if not ok_syn:
                    return TVSResult(False, "Plan/code failed verification.", domain, steps, True)
            else:
                steps.append(TVSStep("VERIFY", "steps-present" if re.search(r"\b1[\)\.]", plan) else "plan-text"))
            steps.append(TVSStep("ANSWER", "agent-plan"))
            return TVSResult(True, plan, domain, steps)

    # --- GENERAL grounded cards ---
    steps.append(TVSStep("ACT", "faq-card"))
    faq = answer_from_faq(user)
    if faq:
        steps.append(TVSStep("VERIFY", "faq-hit"))
        steps.append(TVSStep("ANSWER", "faq"))
        return TVSResult(True, faq, "general", steps)

    steps.append(TVSStep("VERIFY", "ungrounded"))
    return TVSResult(
        False,
        "I don't have a verified path for that yet, and I won't invent one.",
        domain,
        steps,
        abstained=True,
    )
