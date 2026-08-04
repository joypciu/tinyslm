"""Success-trace log for Cognitive Compiler runs → LoRA distill fuel.

When a grounded path (memory/math/faq/plan/code/web/swarm/sara) answers
well, we append a compact JSONL record. `prepare_distill_traces.py` turns
these into chat-format training text with heavy rehearsal so LoRA can
internalize frequent patterns without full FT.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACE_PATH = ROOT / "checkpoints" / "success_traces.jsonl"

_LOW_QUALITY = re.compile(
    r"(empty reply|not sure i followed|try a shorter|gibberish|�)",
    re.I,
)


@dataclass
class SuccessTrace:
    user: str
    answer: str
    mode: str
    source: str
    ir_tag: str = ""
    verify: List[str] = field(default_factory=list)
    ts: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not d.get("ts"):
            d["ts"] = time.time()
        return d


def looks_distillable(answer: str, *, min_len: int = 8, max_len: int = 900) -> bool:
    a = (answer or "").strip()
    if len(a) < min_len or len(a) > max_len:
        return False
    if _LOW_QUALITY.search(a):
        return False
    # Reject mostly-special-token or ultra-repetitive dumps
    if a.count("<") >= 3 and "def " not in a:
        return False
    return True


def strip_display_headers(display: str) -> str:
    """Pull the model body out of a chat display string."""
    text = display or ""
    if "[model]" in text:
        text = text.split("[model]", 1)[-1]
    text = re.sub(r"^\[ir\].*$", "", text, flags=re.M)
    return text.strip()


class TraceStore:
    """Append-only JSONL store (CPU-light, no deps beyond stdlib)."""

    def __init__(self, path: Path | str = DEFAULT_TRACE_PATH, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled

    def record(
        self,
        user: str,
        answer: str,
        *,
        mode: str,
        source: str,
        ir_tag: str = "",
        verify: Optional[List[str]] = None,
    ) -> bool:
        if not self.enabled:
            return False
        body = strip_display_headers(answer) if "[model]" in (answer or "") else (answer or "").strip()
        if not looks_distillable(body):
            return False
        if not (user or "").strip():
            return False
        trace = SuccessTrace(
            user=user.strip()[:500],
            answer=body[:900],
            mode=(mode or "chat")[:40],
            source=(source or "unknown")[:40],
            ir_tag=(ir_tag or "")[:240],
            verify=list(verify or [])[:6],
            ts=time.time(),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
        return True

    def load(self, limit: int = 0) -> List[SuccessTrace]:
        if not self.path.exists():
            return []
        out: List[SuccessTrace] = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out.append(
                    SuccessTrace(
                        user=str(row.get("user") or ""),
                        answer=str(row.get("answer") or ""),
                        mode=str(row.get("mode") or "chat"),
                        source=str(row.get("source") or "unknown"),
                        ir_tag=str(row.get("ir_tag") or ""),
                        verify=list(row.get("verify") or []),
                        ts=float(row.get("ts") or 0.0),
                    )
                )
                if limit and len(out) >= limit:
                    break
        return out

    def stats(self) -> dict:
        rows = self.load()
        by_mode: Dict[str, int] = {}
        for r in rows:
            by_mode[r.mode] = by_mode.get(r.mode, 0) + 1
        return {"path": str(self.path), "count": len(rows), "by_mode": by_mode}


def traces_to_chat_corpus(
    traces: Iterable[SuccessTrace],
    *,
    repeat: int = 3,
    max_items: int = 400,
) -> str:
    """Format traces as TinySLM chat training text (with light upsampling)."""
    items = [t for t in traces if looks_distillable(t.answer) and t.user.strip()]
    items = items[-max_items:]
    blocks: List[str] = []
    for t in items:
        u = t.user.replace("\n", " ").strip()[:400]
        a = t.answer.replace("\r\n", "\n").strip()[:700]
        block = f"<bos><user>{u}<eos>\n<assistant>{a}<eos>\n"
        blocks.extend([block] * max(1, repeat))
    return "".join(blocks)
