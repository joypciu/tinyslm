"""Lightweight agentic controller for TinySLM (plan → tools → answer)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from tiny_slm.search import clean_search_query, search_web


@dataclass
class AgentState:
    goal: str
    plan: List[str] = field(default_factory=list)
    scratchpad: List[str] = field(default_factory=list)
    steps_done: int = 0


def looks_agentic(user: str) -> bool:
    u = user.lower()
    # Phrase triggers (substring OK)
    phrases = [
        "step by step",
        "find out",
        "long task",
        "multi-step",
        "after that",
        "break down",
        "summarize this",
        "using memory",
        "from the document",
        "from context",
    ]
    if any(p in u for p in phrases):
        return True
    # Word-boundary triggers — avoid "plan" matching "plants"
    words = [
        r"\bplan\b",
        r"\bplans\b",
        r"\bplanning\b",
        r"\bsteps\b",
        r"\bresearch\b",
        r"\binvestigate\b",
        r"\btodo\b",
        r"\bagent\b",
        r"\bcompare\b",
        r"\bversus\b",
        r"\bthen\b",
    ]
    return any(re.search(w, u) for w in words)


def build_plan(goal: str) -> List[str]:
    g = goal.lower()
    steps = []
    if any(w in g for w in ("search", "research", "find", "news", "latest", "look up")):
        steps.append("search")
    if any(w in g for w in ("memory", "document", "context", "earlier", "remember", "notes")):
        steps.append("memory")
    if any(w in g for w in ("compare", "versus", "vs")):
        steps.append("compare")
    if not steps:
        steps = ["memory", "reason"]
    else:
        steps.append("reason")
    # de-dupe preserve order
    out = []
    for s in steps:
        if s not in out:
            out.append(s)
    return out[:4]


def run_agent_tools(
    goal: str,
    memory_retrieve,
    auto_search: bool = True,
) -> Tuple[str, AgentState]:
    """Execute a tiny tool loop; returns context block for the LM."""
    state = AgentState(goal=goal, plan=build_plan(goal))
    blocks: List[str] = []
    blocks.append("Plan: " + " - ".join(state.plan))

    for step in state.plan:
        if step == "search" and auto_search:
            q = clean_search_query(goal)
            digest = search_web(q, max_results=3)
            note = f"[tool:search] {digest[:700]}"
            state.scratchpad.append(note)
            blocks.append(note)
            state.steps_done += 1
        elif step == "memory":
            mem = memory_retrieve(goal) if callable(memory_retrieve) else ""
            if mem:
                note = f"[tool:memory] {mem[:700]}"
                state.scratchpad.append(note)
                blocks.append(note)
            state.steps_done += 1
        elif step == "compare":
            # pull two keyword sides if present
            parts = re.split(r"\bvs\.?\b|\bversus\b|\bcompare\b", goal, flags=re.I)
            hint = " | ".join(p.strip() for p in parts if p.strip())[:200]
            note = f"[tool:compare] Focus sides: {hint or goal[:200]}"
            state.scratchpad.append(note)
            blocks.append(note)
            state.steps_done += 1
        elif step == "reason":
            state.scratchpad.append("[tool:reason] synthesize findings into a clear answer")
            state.steps_done += 1

    ctx = "\n".join(blocks)[:1200]
    return ctx, state
