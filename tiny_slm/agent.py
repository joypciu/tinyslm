"""Lightweight agentic controller for TinySLM (plan → contracted tools → answer)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from tiny_slm.contracts import (
    ContractResult,
    best_contract_answer,
    contract_compare,
    contract_memory,
    contract_search,
    contract_swarm,
    synthesize_from_contracts,
)
from tiny_slm.memory import repair_memory_query
from tiny_slm.search import clean_search_query, needs_search, repair_search_query, search_web
from tiny_slm.swarm import looks_complex_query, run_swarm


@dataclass
class AgentState:
    goal: str
    plan: List[str] = field(default_factory=list)
    scratchpad: List[str] = field(default_factory=list)
    steps_done: int = 0
    contracts: List[ContractResult] = field(default_factory=list)
    verified_answer: Optional[str] = None
    replanned: bool = False


def looks_agentic(user: str) -> bool:
    u = user.lower()
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
        "deep dive",
        "research",
        "investigate",
    ]
    if any(p in u for p in phrases):
        return True
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


def build_plan(goal: str, need: List[str] | None = None) -> List[str]:
    """Build tool plan; prefer CognitiveIR need list when provided."""
    g = goal.lower()
    steps: List[str] = []
    if need:
        for n in need:
            if n in ("swarm", "search", "memory", "compare", "reason", "code", "plan"):
                if n in ("code", "plan"):
                    steps.append("reason")
                else:
                    steps.append(n)
    if looks_complex_query(goal):
        steps.append("swarm")
    elif needs_search(goal) or any(
        w in g for w in ("search", "research", "find", "news", "latest", "look up", "collect")
    ):
        steps.append("search")
    if any(w in g for w in ("memory", "document", "context", "earlier", "remember", "notes")):
        steps.append("memory")
    if any(w in g for w in ("compare", "versus", "vs")):
        steps.append("compare")
    if any(w in g for w in ("python", "function", "implement", "tkinter", "script", "code")):
        steps.append("memory")
        steps.append("reason")
    if not steps:
        steps = ["memory", "reason"]
    else:
        steps.append("reason")
    out = []
    for s in steps:
        if s not in out:
            out.append(s)
    return out[:5]


def _failed_steps(contracts: Sequence[ContractResult]) -> List[str]:
    failed: List[str] = []
    for c in contracts:
        if not c.ok and c.step not in failed:
            failed.append(c.step)
    return failed


def _replan_steps(goal: str, need: Optional[List[str]], contracts: List[ContractResult]) -> List[str]:
    """One-shot replan: retry failed search/memory/swarm; keep compare/reason."""
    failed = set(_failed_steps(contracts))
    base = build_plan(goal, need=need)
    retry: List[str] = []
    for s in ("search", "memory", "swarm"):
        if s in failed or (s in base and not any(c.ok and c.step == s for c in contracts)):
            if s == "search" or s == "swarm":
                retry.append("search")  # prefer lighter search on replan
            else:
                retry.append(s)
    if any(w in (goal or "").lower() for w in ("compare", "versus", " vs ")):
        retry.append("compare")
    retry.append("reason")
    out: List[str] = []
    for s in retry:
        if s not in out:
            out.append(s)
    return out[:4] or ["search", "reason"]


def run_agent_tools(
    goal: str,
    memory_retrieve,
    auto_search: bool = True,
    need: Optional[List[str]] = None,
) -> Tuple[str, AgentState]:
    """Execute contracted tool loop; only verified artifacts feed the answer."""
    state = AgentState(goal=goal, plan=build_plan(goal, need=need))
    blocks: List[str] = []
    blocks.append("Plan: " + " - ".join(state.plan))

    def _run_steps(steps: List[str], *, pass_label: str) -> None:
        for step in steps:
            if step == "swarm" and auto_search:
                try:
                    swarm = run_swarm(
                        goal, max_workers=4, max_subgoals=5, max_pages_per_agent=2
                    )
                    cr = contract_swarm(swarm.answer or "", swarm.digest or "")
                    state.contracts.append(cr)
                    blocks.append(cr.line())
                    if cr.ok:
                        state.scratchpad.append(f"[tool:swarm_answer] {cr.artifact[:700]}")
                        blocks.append(f"[tool:swarm_answer] {cr.artifact[:700]}")
                    else:
                        state.scratchpad.append(f"[tool:swarm] {cr.proof}")
                except Exception as exc:
                    cr = ContractResult("swarm", False, "", f"failed:{type(exc).__name__}")
                    state.contracts.append(cr)
                    blocks.append(cr.line())
                state.steps_done += 1
            elif step == "search" and auto_search:
                q = clean_search_query(goal)
                digest = search_web(q, max_results=5)
                cr = contract_search(goal, digest)
                if not cr.ok:
                    q2 = repair_search_query(goal, failed_query=q)
                    if q2.lower() != q.lower():
                        digest2 = search_web(q2, max_results=5)
                        cr2 = contract_search(goal, digest2)
                        if cr2.ok:
                            cr2.proof = f"repaired:{q2[:60]}|{cr2.proof}"
                            cr = cr2
                        else:
                            blocks.append(f"[contract:search:repair-fail] q={q2[:80]}")
                state.contracts.append(cr)
                blocks.append(cr.line())
                if cr.ok:
                    state.scratchpad.append(f"[tool:extract] {cr.artifact}")
                    blocks.append(f"[tool:extract] {cr.artifact}")
                else:
                    state.scratchpad.append(f"[tool:search] {cr.proof}")
                state.steps_done += 1
            elif step == "memory":
                mem = memory_retrieve(goal) if callable(memory_retrieve) else ""
                cr = contract_memory(goal, mem or "")
                if (not cr.ok) or cr.proof == "context-only":
                    q2 = repair_memory_query(goal)
                    if q2.lower() != (goal or "").strip().lower():
                        mem2 = memory_retrieve(q2) if callable(memory_retrieve) else ""
                        blob = "\n".join(x for x in (mem2, mem) if x).strip()
                        cr2 = contract_memory(q2, blob)
                        if cr2.ok and cr2.proof == "extractive-hit":
                            cr2.proof = f"repaired:{q2[:50]}|{cr2.proof}"
                            cr = cr2
                        elif not cr.ok and cr2.ok:
                            cr2.proof = f"repaired:{q2[:50]}|{cr2.proof}"
                            cr = cr2
                state.contracts.append(cr)
                blocks.append(cr.line())
                if cr.ok:
                    state.scratchpad.append(f"[tool:memory] {cr.artifact[:700]}")
                    blocks.append(f"[tool:memory] {cr.artifact[:700]}")
                state.steps_done += 1
            elif step == "compare":
                cr = contract_compare(goal, state.contracts)
                state.contracts.append(cr)
                blocks.append(cr.line())
                if cr.ok:
                    state.scratchpad.append(f"[tool:compare] {cr.artifact}")
                    blocks.append(f"[tool:compare] {cr.artifact}")
                state.steps_done += 1
            elif step == "reason":
                synth = synthesize_from_contracts(state.contracts, goal)
                if synth:
                    cr = ContractResult("reason", True, synth, f"synthesize-{pass_label}")
                    state.verified_answer = synth
                else:
                    cr = ContractResult("reason", False, "", f"synthesize-empty-{pass_label}")
                state.contracts.append(cr)
                blocks.append(cr.line())
                state.scratchpad.append(
                    "[tool:reason] " + (synth[:200] if synth else "no PASS artifacts")
                )
                state.steps_done += 1

    _run_steps(state.plan, pass_label="pass1")

    # Prefer synthesis; fall back to best single artifact
    if not state.verified_answer:
        state.verified_answer = best_contract_answer(state.contracts)

    # One-shot replan when still empty and search/memory/swarm failed
    if not state.verified_answer and auto_search:
        failed = _failed_steps(state.contracts)
        if any(s in failed for s in ("search", "memory", "swarm")) or not any(
            c.ok for c in state.contracts if c.step in ("search", "swarm", "memory")
        ):
            state.replanned = True
            retry = _replan_steps(goal, need, state.contracts)
            blocks.append("Replan: " + " - ".join(retry))
            # Drop prior failed search contracts from blocking synthesis? keep for audit
            _run_steps(retry, pass_label="replan")
            if not state.verified_answer:
                state.verified_answer = (
                    synthesize_from_contracts(state.contracts, goal)
                    or best_contract_answer(state.contracts)
                )

    if state.verified_answer:
        blocks.append("[contract:final:PASS] verified tool answer ready")
    else:
        n_fail = sum(1 for c in state.contracts if not c.ok)
        blocks.append(f"[contract:final:FAIL] no verified artifact ({n_fail} failed steps)")

    ctx = "\n".join(blocks)[:1800]
    return ctx, state
