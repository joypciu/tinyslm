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
    triggers = [
        "plan",
        "step by step",
        "steps",
        "research",
        "find out",
        "investigate",
        "long task",
        "multi-step",
        "then ",
        "after that",
        "break down",
        "todo",
        "agent",
        "compare",
        "summarize this",
        "using memory",
        "from the document",
        "from context",
    ]
    return any(t in u for t in triggers) or len(user) > 220


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
