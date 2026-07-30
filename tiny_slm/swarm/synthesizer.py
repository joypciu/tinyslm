"""Synthesize parallel reader notes into a grounded multi-facet answer."""

from __future__ import annotations

from typing import List

from tiny_slm.swarm.reader_agent import ReaderNote


def synthesize(goal: str, notes: List[ReaderNote]) -> str:
    lines: List[str] = []
    lines.append(f"Research summary for: {goal.strip()[:160]}")
    all_sources: List[str] = []
    for i, note in enumerate(notes, 1):
        lines.append(f"\n{i}) {note.subgoal}")
        if note.bullets:
            for b in note.bullets[:4]:
                lines.append(f"   - {b}")
        else:
            lines.append("   - (no strong passages found for this facet)")
        for s in note.sources:
            if s not in all_sources:
                all_sources.append(s)
    if all_sources:
        lines.append("\nSources:")
        for s in all_sources[:10]:
            lines.append(f" - {s}")
    return "\n".join(lines).strip()
