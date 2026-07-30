"""Collect compact HuggingFace corpora for TinySLM continual fine-tunes.

Outputs chat-style lines into data/hf_intelligence.txt mixed with rehearsal
so resume FT does not wipe existing chat/coding skills.

Designed for ~4M-param models: short turns, capped row counts, ASCII-friendly.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = DATA / "hf_intelligence.txt"

# Prefer small / streaming-friendly sources. Caps keep CPU FT tractable.
SOURCES = [
    # Short instruct — good for tiny models
    {
        "id": "databricks/databricks-dolly-15k",
        "split": "train",
        "max_rows": 1200,
        "kind": "dolly",
    },
    # Coding instruct
    {
        "id": "sahil2801/CodeAlpaca-20k",
        "split": "train",
        "max_rows": 800,
        "kind": "alpaca",
    },
]

REHEARSAL = r"""
<bos><user>Hello!<eos>
<assistant>Hi! I'm TinySLM. How are you doing today?<eos>
<bos><user>What is 2 + 2?<eos>
<assistant>2 + 2 equals 4.<eos>
<bos><user>What is the capital of France?<eos>
<assistant>The capital of France is Paris.<eos>
<bos><user>What is RAM?<eos>
<assistant>RAM is short-term computer memory the CPU uses to hold running programs and data.<eos>
<bos><user>What is Python?<eos>
<assistant>Python is a popular programming language used for websites, data work, automation, and learning to code.<eos>
<bos><user>Write a Python function that adds two numbers.<eos>
<assistant>def add(a, b):
    return a + b

# example: add(2, 3) -> 5<eos>
<bos><user>Plan a short study session step by step.<eos>
<assistant>1) Pick one topic. 2) Study for 20 focused minutes. 3) Write 3 notes. 4) Take a short break.<eos>
<bos><user>Notes:
Plan: memory - reason
Answer the user clearly in short steps if needed.
User ask: Break down this long task: learn Python basics.<eos>
<assistant>Step 1: install Python. Step 2: learn variables and print. Step 3: practice if/else. Step 4: write a tiny script.<eos>
"""


def _clean(text: str, limit: int = 480) -> str:
    text = (text or "").strip()
    text = text.replace("<bos>", "").replace("<eos>", "").replace("<user>", "").replace("<assistant>", "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text


def _pair(user: str, assistant: str) -> str | None:
    u, a = _clean(user, 360), _clean(assistant, 520)
    if len(u) < 8 or len(a) < 8:
        return None
    # Skip huge code dumps / multilingual noise for tiny BPE
    if sum(1 for ch in a if ord(ch) > 127) > max(40, len(a) // 4):
        return None
    return f"<bos><user>{u}<eos>\n<assistant>{a}<eos>"


def _from_dolly(row: dict) -> str | None:
    inst = row.get("instruction") or ""
    ctx = row.get("context") or ""
    resp = row.get("response") or ""
    user = inst if not ctx else f"{inst}\n\nContext: {ctx}"
    return _pair(user, resp)


def _from_alpaca(row: dict) -> str | None:
    inst = row.get("instruction") or row.get("prompt") or ""
    inp = row.get("input") or ""
    out = row.get("output") or row.get("completion") or ""
    user = inst if not inp else f"{inst}\n{inp}"
    return _pair(user, out)


def load_source(src: dict, seed: int) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("Install datasets: pip install datasets") from e

    print(f"Downloading {src['id']} ...")
    try:
        ds = load_dataset(src["id"], split=src["split"], streaming=True)
    except Exception as exc:
        print(f"  skip {src['id']}: {exc}")
        return []

    rng = random.Random(seed)
    rows: list[str] = []
    seen = 0
    for row in ds:
        seen += 1
        # Reservoir-ish: keep first max*3 then subsample for diversity
        if src["kind"] == "dolly":
            item = _from_dolly(row)
        else:
            item = _from_alpaca(row)
        if item:
            rows.append(item)
        if len(rows) >= src["max_rows"] * 2:
            break
        if seen > src["max_rows"] * 8:
            break
    if len(rows) > src["max_rows"]:
        rows = rng.sample(rows, src["max_rows"])
    print(f"  kept {len(rows)} from {src['id']} (scanned ~{seen})")
    return rows


def load_local_rehearsal() -> list[str]:
    chunks = [REHEARSAL.strip()]
    for name in (
        "seed.txt",
        "chat_smart.txt",
        "basic_fixes.txt",
        "coding_agentic.txt",
        "sara_skills.txt",
    ):
        path = DATA / name
        if path.exists():
            text = path.read_text(encoding="utf-8")
            # Take a compact head to avoid exploding the mix
            parts = [p.strip() for p in text.split("\n\n") if "<user>" in p][:40]
            chunks.extend(parts)
    out: list[str] = []
    for c in chunks:
        if "<user>" in c and "<assistant>" in c:
            out.append(c if c.startswith("<bos>") else c)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    collected: list[str] = []
    for src in SOURCES:
        collected.extend(load_source(src, seed=args.seed))

    rehearsal = load_local_rehearsal()
    # Upsample rehearsal so continual FT keeps identity / FAQ / coding
    mix = collected + rehearsal * 3
    random.Random(args.seed).shuffle(mix)
    body = "\n\n".join(mix) + "\n"
    args.out.write_text(body, encoding="utf-8")
    print(f"Wrote {args.out} ({len(mix)} dialogs, {args.out.stat().st_size:,} bytes)")
    print(f"  hf_new≈{len(collected)}  rehearsal_slots≈{len(rehearsal)*3}")


if __name__ == "__main__":
    main()
