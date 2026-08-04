"""Cognitive Compiler — tiny IR for routing TinySLM's tool brain.

The 4M-param decoder is weak at long fluent answers but useful as a
*compiler*: turn a user ask into a short intermediate representation (IR),
then let deterministic tools / cards / memory / swarm execute it.

IR example:
  MODE=recall NEED=[memory] VERIFY=[extract] FACETS=[]

No weight updates — pure inference-time structure. Successful runs can
later be logged as distill traces (see tiny_slm.traces).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from tiny_slm.agent import looks_agentic
from tiny_slm.long_task import looks_long_task
from tiny_slm.memory import looks_like_recall
from tiny_slm.sara import try_eval_math
from tiny_slm.search import needs_search
from tiny_slm.swarm import looks_complex_query, should_spawn_swarm


_CODE_CUES = re.compile(
    r"\b("
    r"python|function|def |class |code|script|tkinter|pygame|pyqt|"
    r"list comprehension|sort|append|dictionary|loop|variable"
    r")\b",
    re.I,
)
_PLAN_CUES = re.compile(
    r"\b("
    r"step by step|break down|multi-step|roadmap|todo\b|plan\b|planning"
    r")\b",
    re.I,
)
_FAQ_CUES = re.compile(
    r"\b("
    r"what is|what's|whats|who is|define|definition|capital of|"
    r"how does .+ work|explain (what|how)"
    r")\b",
    re.I,
)


@dataclass
class CognitiveIR:
    """Compact cognitive intermediate representation."""

    mode: str
    need: List[str] = field(default_factory=list)
    verify: List[str] = field(default_factory=list)
    facets: List[str] = field(default_factory=list)
    confidence: float = 0.5
    rationale: str = ""

    def to_tag(self) -> str:
        need = ",".join(self.need) if self.need else "-"
        verify = ",".join(self.verify) if self.verify else "-"
        facets = ",".join(self.facets[:4]) if self.facets else "-"
        return (
            f"MODE={self.mode} NEED=[{need}] VERIFY=[{verify}] "
            f"FACETS=[{facets}] conf={self.confidence:.2f}"
        )

    def wants(self, tool: str) -> bool:
        return tool in self.need or self.mode == tool


def _dedupe(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    for x in items:
        if x and x not in out:
            out.append(x)
    return out


def compile_query(user: str, *, auto_search: bool = True) -> CognitiveIR:
    """Compile a user utterance into a CognitiveIR (heuristic, CPU-cheap)."""
    q = (user or "").strip()
    if not q:
        return CognitiveIR(mode="chat", confidence=0.2, rationale="empty")

    # Highest-confidence grounded modes first (safe for tiny nets)
    if try_eval_math(q) is not None:
        return CognitiveIR(
            mode="math",
            need=["math"],
            verify=["symbolic"],
            confidence=0.95,
            rationale="symbolic-math",
        )

    if looks_like_recall(q):
        return CognitiveIR(
            mode="recall",
            need=["memory"],
            verify=["extract"],
            confidence=0.9,
            rationale="memory-recall",
        )

    if _CODE_CUES.search(q) and (
        looks_long_task(q)
        or any(w in q.lower() for w in ("write", "implement", "build", "create", "function"))
    ):
        need = ["code"]
        if looks_long_task(q):
            need.append("long_task")
        return CognitiveIR(
            mode="code",
            need=need,
            verify=["syntax"],
            confidence=0.85,
            rationale="coding-ask",
        )

    if _PLAN_CUES.search(q) or (looks_agentic(q) and "compare" not in q.lower()):
        # Prefer plan cards for step-by-step; agent loop if research-y
        if auto_search and looks_complex_query(q):
            return CognitiveIR(
                mode="swarm",
                need=["swarm", "search", "sara"],
                verify=["cite"],
                confidence=0.8,
                rationale="complex-plan-research",
            )
        return CognitiveIR(
            mode="plan",
            need=["plan", "sara"],
            verify=["steps"],
            confidence=0.8,
            rationale="planning",
        )

    if auto_search and should_spawn_swarm(q, has_card=False):
        facets = _guess_facets(q)
        return CognitiveIR(
            mode="swarm",
            need=["swarm", "search"],
            verify=["cite"],
            facets=facets,
            confidence=0.78,
            rationale="multi-facet-research",
        )

    if looks_long_task(q):
        return CognitiveIR(
            mode="long_task",
            need=["long_task", "code", "plan"],
            verify=["steps"],
            confidence=0.75,
            rationale="long-task",
        )

    if "compare" in q.lower() or re.search(r"\bvs\b|versus", q.lower()):
        need = ["faq", "plan", "sara"]
        if auto_search and needs_search(q):
            need.append("search")
        return CognitiveIR(
            mode="compare",
            need=need,
            verify=["contrast"],
            confidence=0.72,
            rationale="compare",
        )

    if _FAQ_CUES.search(q) and len(q) < 160:
        need = ["faq"]
        if auto_search and needs_search(q):
            need.append("search")
        return CognitiveIR(
            mode="faq",
            need=need,
            verify=["card"],
            confidence=0.7,
            rationale="short-definition",
        )

    if auto_search and needs_search(q):
        return CognitiveIR(
            mode="search",
            need=["search"],
            verify=["extract"],
            confidence=0.68,
            rationale="web-knowledge",
        )

    if looks_agentic(q):
        return CognitiveIR(
            mode="sara",
            need=["sara", "memory"],
            verify=["reflect"],
            confidence=0.65,
            rationale="agentic",
        )

    return CognitiveIR(
        mode="chat",
        need=["neural"],
        verify=[],
        confidence=0.4,
        rationale="open-chat",
    )


def _guess_facets(user: str) -> List[str]:
    """Cheap facet hints for swarm IR (not a full decomposer)."""
    low = (user or "").lower()
    candidates = [
        ("auth", ("auth", "login", "oauth", "jwt", "password")),
        ("architecture", ("architecture", "design", "stack", "backend", "frontend")),
        ("risks", ("risk", "security", "threat", "privacy")),
        ("roadmap", ("roadmap", "milestone", "timeline", "phase")),
        ("cost", ("cost", "pricing", "budget", "latency")),
    ]
    hit = [name for name, keys in candidates if any(k in low for k in keys)]
    return hit[:4]


def pipeline_stages(ir: CognitiveIR) -> List[str]:
    """Ordered grounded/tool stages the chat runtime should try."""
    mode_pipes = {
        "math": ["math"],
        "recall": ["memory", "sara"],
        "code": ["code", "long_task", "sara", "neural"],
        "plan": ["plan", "sara", "neural"],
        "swarm": ["swarm", "search", "sara"],
        "long_task": ["code", "plan", "long_task", "sara"],
        "compare": ["faq", "plan", "sara", "search", "neural"],
        "faq": ["faq", "search", "neural"],
        "search": ["search", "faq", "neural"],
        "sara": ["memory", "sara", "neural"],
        "chat": ["faq", "neural"],
    }
    base = list(mode_pipes.get(ir.mode, ["faq", "neural"]))
    # Honor explicit needs while preserving order
    for n in ir.need:
        if n not in base:
            base.insert(0, n)
    return _dedupe(base)


def should_run_stage(ir: CognitiveIR, stage: str) -> bool:
    """Whether a grounded stage is worth trying for this IR."""
    stages = pipeline_stages(ir)
    if stage in stages:
        return True
    # Always allow extractive memory on recall-ish asks even if omitted
    if stage == "memory" and ir.mode in ("recall", "sara"):
        return True
    return False


def parse_ir_tag(tag: str) -> Optional[CognitiveIR]:
    """Parse a to_tag() string back into CognitiveIR (for traces/tests)."""
    if not tag or "MODE=" not in tag:
        return None
    m = re.search(r"MODE=(\w+)", tag)
    need_m = re.search(r"NEED=\[([^\]]*)\]", tag)
    ver_m = re.search(r"VERIFY=\[([^\]]*)\]", tag)
    fac_m = re.search(r"FACETS=\[([^\]]*)\]", tag)
    conf_m = re.search(r"conf=([0-9.]+)", tag)
    if not m:
        return None

    def _split(blob: Optional[re.Match]) -> List[str]:
        if not blob:
            return []
        raw = blob.group(1).strip()
        if not raw or raw == "-":
            return []
        return [p.strip() for p in raw.split(",") if p.strip()]

    return CognitiveIR(
        mode=m.group(1),
        need=_split(need_m),
        verify=_split(ver_m),
        facets=_split(fac_m),
        confidence=float(conf_m.group(1)) if conf_m else 0.5,
    )
