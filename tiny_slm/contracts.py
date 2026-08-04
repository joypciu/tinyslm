"""Agent Step Contracts — every tool step must produce a verified artifact.

Novel for this stack: tools are not advisory notes. Each step returns a
ContractResult(ok, artifact, proof). Failed contracts are marked and excluded
from synthesis so the agent cannot launder empty search into a fluent lie.
"""

from __future__ import annotations

import re
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
    from tiny_slm.search import citation_hosts

    a = (answer or "").strip()
    if len(a) < 80 or "no strong passages" in a.lower():
        return ContractResult("swarm", False, "", "weak-or-empty-swarm")
    hosts_d = citation_hosts(digest or "")
    hosts_a = citation_hosts(a)
    has_src = "sources:" in a.lower() or bool(hosts_a)
    # Host-Diversity: multi-publisher digests must not collapse to one host
    if len(hosts_d) >= 2 and len(hosts_a) < 2 and "sources:" not in a.lower():
        return ContractResult("swarm", False, "", "host-diversity-fail")
    if len(hosts_d) >= 2 and len(hosts_a) >= 2:
        return ContractResult("swarm", True, a[:900], f"cited-hosts={len(hosts_a)}")
    return ContractResult(
        "swarm",
        True,
        a[:900],
        "cited" if has_src else "summary-ok",
    )


def contract_compare(goal: str, evidence: List[ContractResult]) -> ContractResult:
    """Structured two-sided compare from PASS evidence (fail-closed if none)."""
    parts = re.split(r"\bvs\.?\b|\bversus\b|\bcompare\b", goal or "", flags=re.I)
    sides = [p.strip(" ?.") for p in parts if p.strip(" ?.")]
    if len(sides) < 2:
        # "Compare A and B"
        m = re.search(
            r"compare\s+(.+?)\s+and\s+(.+?)(?:\s*[.?]|$)",
            goal or "",
            re.I,
        )
        if m:
            sides = [m.group(1).strip(), m.group(2).strip()]
    if len(sides) < 2:
        return ContractResult("compare", False, "", "sides-unparsed")

    a_side, b_side = sides[0][:80], sides[1][:80]
    ok_ev = [
        r
        for r in evidence
        if r.ok and r.artifact.strip() and r.step in ("search", "swarm", "memory")
        and r.proof != "context-only"
    ]
    if not ok_ev:
        # Still emit a structured outline so reason can ask for evidence
        outline = f"A: {a_side}. B: {b_side}. Contrast: need verified evidence for both sides."
        return ContractResult("compare", True, outline, "sides-parsed-no-evidence")

    bits = []
    for r in ok_ev[:2]:
        snippet = re.sub(r"\s+", " ", r.artifact.strip())[:220]
        bits.append(snippet)
    body = (
        f"A ({a_side}): {bits[0]}"
        + (f" B ({b_side}): {bits[1]}" if len(bits) > 1 else f" B ({b_side}): (limited evidence)")
    )
    return ContractResult("compare", True, body[:900], f"evidence-sides={len(bits)}")


def synthesize_from_contracts(results: List[ContractResult], goal: str = "") -> Optional[str]:
    """Merge PASS artifacts into one grounded answer; None if nothing verifiable."""
    ok = [
        r
        for r in results
        if r.ok
        and r.artifact.strip()
        and not (r.step == "memory" and r.proof == "context-only")
        and r.step != "reason"
    ]
    if not ok:
        return None

    # Prefer a single strong extractive hit
    for r in ok:
        if r.step in ("search", "swarm") and (
            r.proof.startswith("evidence-quorum")
            or r.proof.startswith("cited")
            or "extract" in r.proof
        ):
            return r.artifact.strip()[:900]

    # Compare-shaped answer
    cmp = next((r for r in ok if r.step == "compare"), None)
    mem = next((r for r in ok if r.step == "memory"), None)
    web = next((r for r in ok if r.step in ("search", "swarm")), None)

    if cmp and (web or mem):
        parts = [cmp.artifact.strip()]
        if web:
            parts.append(web.artifact.strip()[:320])
        return " ".join(parts)[:900]

    if cmp and "need verified evidence" not in cmp.artifact.lower():
        return cmp.artifact.strip()[:900]

    if mem:
        return mem.artifact.strip()[:900]

    if web:
        return web.artifact.strip()[:900]

    # Last resort: best single artifact
    return best_contract_answer(results)


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
    if top.step == "compare" and "need verified evidence" in top.artifact.lower():
        for r in ok:
            if r.step in ("search", "swarm", "memory") and r.proof != "context-only":
                return r.artifact
        return None
    return top.artifact
