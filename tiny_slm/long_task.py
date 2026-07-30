"""Long-task controller: plan → substeps → assemble (reliable multi-step work).

Keeps the tiny neural window focused on one sub-goal at a time while
LongContextMemory holds the full scratchpad — so long jobs stay reliable
without requiring a huge Transformer context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


@dataclass
class LongTaskState:
    goal: str
    steps: List[str] = field(default_factory=list)
    results: List[str] = field(default_factory=list)
    final: str = ""


_LONG_CUES = (
    "complete python",
    "write a complete",
    "write a class",
    "write a script",
    "write a recursive",
    "multi-step",
    "long task",
    "end to end",
    "end-to-end",
    "full program",
    "word_count",
    "bankaccount",
    "bank account",
    "fibonacci",
    "break this into code",
    "ship a tiny",
    "build a todo",
    "renames files",
)


def looks_long_task(user: str) -> bool:
    u = (user or "").lower()
    if len(user or "") >= 180:
        return True
    if any(c in u for c in _LONG_CUES):
        return True
    # Multiple distinct coding asks in one message
    hits = sum(
        1
        for k in ("def ", "class ", "try/", "csv", "file", "function", "script")
        if k in u
    )
    return hits >= 2


def build_long_steps(goal: str) -> List[str]:
    """Heuristic sub-plan for long coding / agentic goals."""
    g = goal.lower()
    steps: List[str] = []
    if any(k in g for k in ("word_count", "word count", "mapping each word")):
        steps = [
            "Define function signature and empty counts dict.",
            "Split text into words and update counts.",
            "Return the dict and show a tiny example.",
        ]
    elif any(k in g for k in ("bankaccount", "bank account", "deposit", "withdraw")):
        steps = [
            "Create BankAccount with __init__(balance).",
            "Implement deposit(amount).",
            "Implement withdraw(amount) with overdraft guard.",
        ]
    elif "fib" in g:
        steps = [
            "Write base cases for fib(0) and fib(1).",
            "Add the recursive case fib(n-1)+fib(n-2).",
        ]
    elif "input.txt" in g or "output.txt" in g:
        steps = [
            "Read and strip lines from input.txt.",
            "Skip blanks and collect unique lines.",
            "Write sorted unique lines to output.txt.",
        ]
    elif "csv" in g and ("average" in g or "top" in g):
        steps = [
            "Load CSV rows with names and scores.",
            "Compute average score.",
            "Sort and print top 3 names.",
        ]
    elif "todo" in g:
        steps = [
            "Define in-memory task list.",
            "Add commands: add, complete, list.",
            "Persist tasks to a file and test.",
        ]
    elif "rename" in g and "file" in g:
        steps = [
            "List files in the folder.",
            "Parse a date from each name.",
            "Dry-run new names then rename.",
        ]
    elif "break down" in g or "step by step" in g or "plan" in g:
        steps = [
            "Clarify the goal in one line.",
            "List 3-5 concrete steps.",
            "Name the first action to take now.",
        ]
    else:
        steps = [
            "Restate the goal briefly.",
            "Solve the hardest part first.",
            "Add a minimal example or check.",
            "Summarize the final answer.",
        ]
    return steps[:6]


def run_long_task(
    goal: str,
    *,
    solve_step: Callable[[str, str], str],
    memory_add: Optional[Callable[[str], None]] = None,
    assemble: bool = True,
) -> Tuple[str, LongTaskState]:
    """Execute substeps; `solve_step(goal, step)` returns one step result."""
    state = LongTaskState(goal=goal, steps=build_long_steps(goal))
    for i, step in enumerate(state.steps, 1):
        piece = (solve_step(goal, step) or "").strip()
        if not piece:
            piece = f"(step {i} pending)"
        state.results.append(piece)
        if memory_add is not None:
            memory_add(f"LONG_TASK step {i}/{len(state.steps)}: {step}\n{piece[:400]}")
    if assemble:
        # Prefer the richest code-looking chunk; else join all
        codeish = [r for r in state.results if any(t in r for t in ("def ", "class ", "try:", "import "))]
        if codeish:
            state.final = max(codeish, key=len)
            # If multiple code pieces, join uniquely
            if len(codeish) > 1:
                merged = []
                seen = set()
                for r in codeish:
                    key = re.sub(r"\s+", " ", r)[:80]
                    if key not in seen:
                        seen.add(key)
                        merged.append(r)
                state.final = "\n\n".join(merged)
        else:
            numbered = [f"{i}) {r}" for i, r in enumerate(state.results, 1)]
            state.final = "\n".join(numbered)
    else:
        state.final = "\n".join(state.results)
    return state.final, state
