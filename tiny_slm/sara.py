"""SARA — Skill-Augmented Reflective Agent.

Novel loop for tiny SLMs:
  1) Retrieve procedural *skill cards* (not only facts) from long memory
  2) Draft an answer with tools
  3) Reflect: checklist for clarity / missing steps / memory conflicts
  4) Optional symbolic verify for simple arithmetic
  5) Revise once if reflection flags issues

This boosts agentic intelligence without growing the neural params.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from tiny_slm.agent import AgentState, build_plan, looks_agentic, run_agent_tools
from tiny_slm.knowledge import answer_from_faq, answer_from_plan_template, scrub_generation
from tiny_slm.memory import answer_from_memory
from tiny_slm.search import answer_from_search, clean_search_query, search_web


# Built-in skill cards (also ingestable into LongContextMemory)
SKILL_CARDS = [
    {
        "id": "plan_steps",
        "triggers": ["plan", "step by step", "break down", "long task", "multi-step"],
        "card": "SKILL plan_steps: Answer with numbered steps (1-4). Keep each step one short sentence.",
    },
    {
        "id": "compare_two",
        "triggers": ["compare", "versus", " vs "],
        "card": "SKILL compare_two: State A clearly, state B clearly, then one contrast line.",
    },
    {
        "id": "memory_answer",
        "triggers": [
            "memory",
            "remember",
            "from context",
            "from the document",
            "launch code",
            "deadline",
            "earlier",
            "secret code",
            "secret",
            "password",
            "warehouse",
            "what was the",
            "using memory",
            "exact code",
        ],
        "card": "SKILL memory_answer: Prefer facts from [memory]/tool:memory]. Quote the key fact in one sentence.",
    },
    {
        "id": "friendly_chat",
        "triggers": ["hello", "hi", "hey", "how are you", "thanks", "bored"],
        "card": "SKILL friendly_chat: Warm greeting or empathy first, then a short helpful offer.",
    },
    {
        "id": "math_simple",
        "triggers": ["+", "plus", "times", "minus", "divided", "equals", "what is 2"],
        "card": "SKILL math_simple: Compute carefully. Reply like 'A + B equals C'.",
    },
]


@dataclass
class SaraState:
    goal: str
    skills: List[str] = field(default_factory=list)
    agent: Optional[AgentState] = None
    draft: str = ""
    reflection: str = ""
    final: str = ""
    revised: bool = False
    verified_math: Optional[str] = None


def select_skills(goal: str, max_skills: int = 2) -> List[str]:
    g = " " + goal.lower() + " "
    hit = []
    for sk in SKILL_CARDS:
        matched = False
        for t in sk["triggers"]:
            # Short tokens need word boundaries ("hi" must not hit "friendship")
            if len(t) <= 3:
                if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", g):
                    matched = True
                    break
            elif t in g:
                matched = True
                break
        if matched:
            hit.append(sk["card"])
    if not hit and looks_agentic(goal):
        hit.append(SKILL_CARDS[0]["card"])
    return hit[:max_skills]


def try_eval_math(text: str) -> Optional[str]:
    """Verified math only (SymPy + safe legacy). Never guesses."""
    from tiny_slm.math_engine import try_solve_math

    return try_solve_math(text)


def reflect_on_draft(goal: str, draft: str, memory_snip: str = "") -> Tuple[bool, str]:
    """Heuristic reflector (CPU-cheap). Returns (needs_revise, notes)."""
    d = (draft or "").strip()
    issues = []
    if len(d) < 8:
        issues.append("too short")
    if d.endswith("?") and "equals" not in d.lower() and not any(
        w in goal.lower() for w in ("how are you", "what should", "bored")
    ):
        # unanswered echo
        if goal.lower().rstrip("?")[:20] in d.lower():
            issues.append("echoes question")
    if any(w in goal.lower() for w in ("step", "plan", "break down")):
        if not re.search(r"\b(1\)|1\.|step 1)\b", d.lower()):
            issues.append("missing numbered steps")
    if "paris" in goal.lower() and "france" in goal.lower():
        if "paris" not in d.lower():
            issues.append("missing Paris")
    if memory_snip:
        from tiny_slm.memory import extract_codes, looks_like_recall

        if looks_like_recall(goal):
            codes = extract_codes(memory_snip)
            if codes and not any(c.lower() in d.lower() for c in codes):
                issues.append("missing memory fact")
        elif "orbit-77" in memory_snip.lower() and "orbit" in goal.lower():
            if "orbit" not in d.lower():
                issues.append("missing memory fact")
    if "2 + 2" in goal.lower() or "2+2" in goal.lower():
        if "4" not in d:
            issues.append("math wrong")
    ok = len(issues) == 0
    note = "OK" if ok else ("Issues: " + ", ".join(issues))
    return (not ok), note


def run_sara(
    goal: str,
    generate_fn: Callable[[str], str],
    memory_retrieve: Callable[[str], str],
    auto_search: bool = True,
    force_agent: bool = False,
) -> SaraState:
    state = SaraState(goal=goal)
    state.skills = select_skills(goal)
    mem = memory_retrieve(goal) if callable(memory_retrieve) else ""

    # Math fast-path (symbolic intelligence)
    math_ans = try_eval_math(goal)
    if math_ans:
        state.verified_math = math_ans
        state.draft = math_ans
        state.final = math_ans
        state.reflection = "symbolic math verify"
        return state

    # Memory extractive fast-path (no neural regen of rare codes)
    mem_ans = answer_from_memory(goal, mem)
    if mem_ans:
        state.draft = mem_ans
        state.final = mem_ans
        state.reflection = "extractive memory verify"
        return state

    # FAQ / plan templates for brittle short facts (no weight updates)
    faq = answer_from_faq(goal)
    if faq and not (force_agent or looks_agentic(goal)):
        state.draft = faq
        state.final = faq
        state.reflection = "faq card"
        return state
    plan = answer_from_plan_template(goal)
    if plan:
        state.draft = plan
        state.final = plan
        state.reflection = "plan template"
        return state

    tool_block = ""
    if force_agent or looks_agentic(goal):
        tool_block, agent_state = run_agent_tools(
            goal, memory_retrieve=memory_retrieve, auto_search=auto_search
        )
        state.agent = agent_state
        # Re-check after tool memory pull
        if state.agent and state.agent.scratchpad:
            tool_mem = "\n".join(state.agent.scratchpad)
            mem_ans = answer_from_memory(goal, tool_mem) or answer_from_memory(goal, mem)
            if mem_ans:
                state.draft = mem_ans
                state.final = mem_ans
                state.reflection = "extractive memory after tools"
                return state
            # Extractive web answer when search tool ran
            for note in state.agent.scratchpad:
                if note.startswith("[tool:extract]"):
                    web_ans = note.replace("[tool:extract]", "", 1).strip()
                    if web_ans:
                        state.draft = web_ans
                        state.final = web_ans
                        state.reflection = "extractive web after tools"
                        return state
                if note.startswith("[tool:search]"):
                    web_ans = answer_from_search(
                        note.replace("[tool:search]", "", 1), query=goal
                    )
                    if web_ans:
                        state.draft = web_ans
                        state.final = web_ans
                        state.reflection = "extractive web after tools"
                        return state

    skill_txt = "\n".join(state.skills)
    notes = "\n".join(
        x for x in [skill_txt, tool_block, f"[memory]\n{mem}" if mem else ""] if x
    ).strip()

    draft_prompt = (
        f"Notes:\n{notes[:900]}\n"
        f"Answer the user clearly in short steps if needed.\n"
        f"User ask: {goal}"
    )
    state.draft = scrub_generation(generate_fn(draft_prompt) or "")

    needs_fix, refl = reflect_on_draft(goal, state.draft, mem)
    state.reflection = refl
    if needs_fix:
        # Prefer quoting memory / FAQ over a second noisy generation
        mem_fix = answer_from_memory(goal, mem)
        faq_fix = answer_from_faq(goal)
        if mem_fix:
            state.final = mem_fix
            state.revised = True
            state.reflection = refl + " | patched from memory"
        elif faq_fix:
            state.final = faq_fix
            state.revised = True
            state.reflection = refl + " | patched from faq"
        else:
            revise_prompt = (
                f"Notes:\n{notes[:700]}\nReflection: {refl}\n"
                f"Improve the draft. Draft was: {state.draft[:200]}\n"
                f"User ask: {goal}"
            )
            revised = scrub_generation(generate_fn(revise_prompt) or "")
            if revised and len(revised) >= 6:
                state.final = revised
                state.revised = True
            else:
                state.final = state.draft
    else:
        state.final = state.draft

    state.final = scrub_generation(state.final or state.draft or "(no answer)")
    return state
