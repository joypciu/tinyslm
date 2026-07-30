"""Synthesize parallel reader notes into a grounded multi-facet answer."""

from __future__ import annotations

import re
from typing import List

from tiny_slm.swarm.reader_agent import ReaderNote


def _goal_title(goal: str) -> str:
    g = re.sub(r"\s+", " ", (goal or "").strip())
    # Prefer first sentence / line before numbered list
    g = re.split(r"\n\s*1\s*[\)\.\-:]", g)[0].strip()
    if len(g) > 140:
        g = g[:137].rstrip() + "..."
    return g or "research goal"


def synthesize(goal: str, notes: List[ReaderNote]) -> str:
    lines: List[str] = []
    lines.append(f"Research summary for: {_goal_title(goal)}")
    all_sources: List[str] = []
    for i, note in enumerate(notes, 1):
        # Keep subgoal readable (strip "Research and answer: " prefix in display)
        label = re.sub(r"^Research and answer:\s*", "", note.subgoal, flags=re.I)
        lines.append(f"\n{i}) {label}")
        if note.bullets:
            for b in note.bullets[:5]:
                lines.append(f"   - {b}")
        else:
            lines.append("   - (no strong passages found for this facet)")
        for s in note.sources:
            if s not in all_sources:
                all_sources.append(s)
    if all_sources:
        lines.append("\nSources:")
        for s in all_sources[:12]:
            lines.append(f" - {s}")
    return "\n".join(lines).strip()
