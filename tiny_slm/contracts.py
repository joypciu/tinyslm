"""Agent Step Contracts — every tool step must produce a verified artifact.

Novel for this stack: tools are not advisory notes. Each step returns a
ContractResult(ok, artifact, proof). Failed contracts are marked and excluded
from synthesis so the agent cannot launder empty search into a fluent lie.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from tiny_slm.tvs import evidence_quorum


@dataclass
class ContractResult:
    step: str
    ok: bool
    artifact: str
    proof: str

    def line(self) -> str:
        flag = "PASS" if self.ok else "FAIL"
        return f"[contract:{self.step}:{flag}] {self.proof}"


def contract_search(goal: str, digest: str) -> ContractResult:
    ok, ans, note = evidence_quorum(digest, goal, min_hits=2)
    if ok and ans:
        return ContractResult("search", True, ans, f"evidence-quorum {note}")
    # Single strong extract still allowed but marked weaker
    from tiny_slm.search import answer_from_search

    one = answer_from_search(digest, query=goal)
    if one and len(one) > 40:
        return ContractResult("search", True, one, "single-extract")
    return ContractResult(
        "search",
        False,
        "",
        f"no-verified-web ({note or 'empty'})",
    )


def contract_memory(goal: str, mem: str) -> ContractResult:
    from tiny_slm.memory import answer_from_memory

    if not (mem or "").strip():
        return ContractResult("memory", False, "", "empty-store")
    direct = answer_from_memory(goal, mem)
    if direct:
        return ContractResult("memory", True, direct, "extractive-hit")
    # Retrieved context still useful as support, but not a final answer
    if len(mem.strip()) >= 40:
        return ContractResult("memory", True, mem[:700], "context-only")
    return ContractResult("memory", False, "", "no-usable-memory")


def contract_swarm(answer: str, digest: str) -> ContractResult:
    a = (answer or "").strip()
    if len(a) >= 80 and "no strong passages" not in a.lower():
        # Prefer answers that cite sources
        has_src = "sources:" in a.lower() or "http" in (digest or "").lower()
        return ContractResult(
            "swarm",
            True,
            a[:900],
            "cited" if has_src else "summary-ok",
        )
    return ContractResult("swarm", False, "", "weak-or-empty-swarm")


def best_contract_answer(results: List[ContractResult]) -> Optional[str]:
    """Prefer extractive search/memory/swarm artifacts over empty reason steps."""
    priority = {"search": 3, "swarm": 3, "memory": 2, "compare": 1}
    ok = [r for r in results if r.ok and r.artifact.strip()]
    if not ok:
        return None
    ok.sort(key=lambda r: (-priority.get(r.step, 0), -len(r.artifact)))
    top = ok[0]
    if top.step == "memory" and top.proof == "context-only":
        # Don't ship raw memory dump as the user-facing answer
        for r in ok:
            if r.step in ("search", "swarm") and r.proof != "context-only":
                return r.artifact
        return None
    return top.artifact
