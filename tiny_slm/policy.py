"""Production answer policy — calibrated route + anti-hallucination.

The small neural decoder is never treated as a knowledge oracle.
Every user ask maps to one primary action:

  grounded  — verified tool/card/memory/math/code (safe to answer)
  search    — need live/web evidence before answering
  agent     — multi-step plan / tools / swarm
  abstain   — refuse to invent; ask for a checkable form

This is the "intelligence" layer for a tiny model in production:
know when you know, know when to look up, know when to refuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from tiny_slm.agent import looks_agentic
from tiny_slm.compiler import CognitiveIR, compile_query
from tiny_slm.knowledge import answer_from_code_template, answer_from_faq, answer_from_plan_template
from tiny_slm.long_task import looks_long_task
from tiny_slm.math_engine import (
    abstain_math_message,
    looks_like_math,
    looks_like_research_math,
    math_policy,
)
from tiny_slm.memory import looks_like_recall
from tiny_slm.search import needs_search
from tiny_slm.swarm import looks_complex_query, should_spawn_swarm


ABSTAIN_GENERIC = (
    "I don't have a verified answer for that, and I won't guess. "
    "I can: (1) solve checkable math, (2) write/verify small Python, "
    "(3) plan multi-step tasks, (4) search the web for open facts, "
    "or (5) recall facts you stored in memory. Please rephrase toward one of those."
)

ABSTAIN_OPEN_FACT = (
    "That looks like an open factual question. I should check the web rather than "
    "invent an answer. Enable search / ask me to look it up, or rephrase as a "
    "topic I can ground (math, code, plan, or a stored memory fact)."
)


@dataclass
class RouteDecision:
    action: str  # grounded | search | agent | abstain
    ir: CognitiveIR
    reason: str
    message: Optional[str] = None  # set for immediate abstain replies
    allow_neural: bool = False

    def to_tag(self) -> str:
        return f"ACTION={self.action} reason={self.reason} {self.ir.to_tag()}"


def decide_route(
    user: str,
    *,
    auto_search: bool = True,
    force_search: bool = False,
    force_agent: bool = False,
    has_memory_hit: bool = False,
) -> RouteDecision:
    """Calibrated production router (CPU-cheap, fail-closed)."""
    q = (user or "").strip()
    ir = compile_query(q, auto_search=auto_search or force_search)

    if not q:
        return RouteDecision("abstain", ir, "empty", ABSTAIN_GENERIC, allow_neural=False)

    # --- Math: solve or abstain (never neural math) ---
    m_action, m_ans = math_policy(q)
    if m_action == "solve":
        ir.mode = "math"
        ir.need = ["math"]
        ir.verify = ["symbolic"]
        ir.confidence = 0.98
        ir.rationale = "verified-math"
        return RouteDecision("grounded", ir, "math-verified", allow_neural=False)
    if m_action == "abstain" or (looks_like_math(q) and m_ans is None):
        ir.mode = "abstain"
        ir.need = []
        ir.verify = ["symbolic"]
        ir.confidence = 0.9
        ir.rationale = "math-unverified"
        return RouteDecision(
            "abstain",
            ir,
            "math-unverified",
            abstain_math_message(q),
            allow_neural=False,
        )

    # --- Memory recall ---
    if looks_like_recall(q):
        if has_memory_hit:
            ir.mode = "recall"
            ir.confidence = 0.92
            return RouteDecision("grounded", ir, "memory-hit", allow_neural=False)
        ir.mode = "abstain"
        ir.confidence = 0.85
        return RouteDecision(
            "abstain",
            ir,
            "memory-miss",
            "I don't have a verified memory fact for that yet. "
            "Store it first (or ingest a document), then ask me to recall it.",
            allow_neural=False,
        )

    # --- Code cards (match even when IR says chat — e.g. "How do I reverse…") ---
    if answer_from_code_template(q):
        ir.mode = "code"
        ir.need = ["code"]
        ir.verify = ["syntax", "spec"]
        ir.confidence = max(ir.confidence, 0.92)
        ir.rationale = "code-card"
        return RouteDecision("grounded", ir, "code-card", allow_neural=False)

    # --- Coding / long tasks → agentic grounded (templates + verify) ---
    if ir.mode in ("code", "long_task") or looks_long_task(q):
        if answer_from_plan_template(q):
            return RouteDecision("grounded", ir, "code-or-plan-card", allow_neural=False)
        return RouteDecision("agent", ir, "code-agentic", allow_neural=False)

    # --- Forced agent / complex research ---
    if force_agent or ir.mode in ("swarm", "sara", "plan") or looks_complex_query(q):
        if should_spawn_swarm(q, has_card=bool(answer_from_faq(q))) and (auto_search or force_search):
            ir.mode = "swarm"
            return RouteDecision("agent", ir, "swarm-research", allow_neural=False)
        if looks_agentic(q) or force_agent or ir.mode in ("plan", "sara"):
            return RouteDecision("agent", ir, "agentic-plan", allow_neural=False)

    # --- FAQ grounded ---
    if answer_from_faq(q):
        ir.mode = "faq"
        ir.confidence = max(ir.confidence, 0.88)
        return RouteDecision("grounded", ir, "faq-card", allow_neural=False)

    if answer_from_plan_template(q):
        ir.mode = "plan"
        return RouteDecision("grounded", ir, "plan-card", allow_neural=False)

    # --- Search for open / live knowledge ---
    if force_search or ((auto_search or force_search) and (needs_search(q) or ir.mode == "search")):
        ir.mode = "search"
        ir.need = ["search"]
        ir.verify = ["extract"]
        ir.confidence = max(ir.confidence, 0.75)
        return RouteDecision("search", ir, "needs-web", allow_neural=False)

    # --- Compare without card: search or abstain, never invent ---
    if ir.mode == "compare":
        if auto_search:
            return RouteDecision("search", ir, "compare-needs-evidence", allow_neural=False)
        return RouteDecision(
            "abstain",
            ir,
            "compare-no-evidence",
            ABSTAIN_OPEN_FACT,
            allow_neural=False,
        )

    # --- Short chit-chat: allow light neural only for greetings ---
    low = q.lower().strip().rstrip("!")
    if low in (
        "hello",
        "hi",
        "hey",
        "thanks",
        "thank you",
        "good morning",
        "good evening",
        "how are you",
    ):
        ir.mode = "chat"
        ir.confidence = 0.7
        return RouteDecision("grounded", ir, "chitchat", allow_neural=True)

    # --- Research math phrasing already handled; other hard asks abstain ---
    if looks_like_research_math(q):
        return RouteDecision(
            "abstain",
            ir,
            "research-math",
            abstain_math_message(q),
            allow_neural=False,
        )

    # Open-ended factual / long speculative → abstain (production safe)
    if len(q) > 180 or ir.confidence < 0.55:
        return RouteDecision(
            "abstain",
            ir,
            "low-confidence",
            ABSTAIN_GENERIC if not needs_search(q) else ABSTAIN_OPEN_FACT,
            allow_neural=False,
        )

    # Default: do not free-decode open knowledge
    if auto_search and needs_search(q):
        return RouteDecision("search", ir, "default-search", allow_neural=False)

    return RouteDecision(
        "abstain",
        ir,
        "fail-closed",
        ABSTAIN_GENERIC,
        allow_neural=False,
    )
