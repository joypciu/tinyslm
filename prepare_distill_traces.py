"""Build a LoRA-ready distill curriculum from success traces + rehearsal.

Safe continual-learning path (no weight updates here):
  1) Chat logs grounded wins to checkpoints/success_traces.jsonl
  2) This script writes data/distill_traces.txt (+ optional seed rehearsal)
  3) Train with:
       python train.py --resume checkpoints/tinyslm.pt --reuse-tokenizer \\
         --adapter-rank 8 --freeze-base --steps 400 --lr 1e-4 --backup

Prefer this over narrow full FT (catastrophic forgetting risk).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tiny_slm.traces import DEFAULT_TRACE_PATH, TraceStore, traces_to_chat_corpus

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "distill_traces.txt"

# Compact rehearsal so LoRA does not overwrite basics when traces are sparse
REHEARSAL = r"""
<bos><user>Hello!<eos>
<assistant>Hi! I'm TinySLM. How can I help?<eos>
<bos><user>What is 2 + 2?<eos>
<assistant>2 + 2 equals 4.<eos>
<bos><user>What is RAM?<eos>
<assistant>RAM is short-term computer memory the CPU uses while programs run.<eos>
<bos><user>What is the capital of Canada?<eos>
<assistant>Ottawa is the capital of Canada.<eos>
<bos><user>Write a Python function that adds two numbers.<eos>
<assistant>def add(a, b):
    return a + b<eos>
<bos><user>Plan a short study session step by step.<eos>
<assistant>1) Pick one topic. 2) Study for 20 focused minutes. 3) Write 3 notes. 4) Take a short break.<eos>
<bos><user>Using memory, what is the launch code?<eos>
<assistant>From memory: ORBIT-77.<eos>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--repeat", type=int, default=4, help="Upsample each trace")
    parser.add_argument("--max-items", type=int, default=400)
    parser.add_argument("--rehearsal-repeat", type=int, default=8)
    parser.add_argument("--no-rehearsal", action="store_true")
    args = parser.parse_args()

    store = TraceStore(args.traces)
    traces = store.load()
    corpus = traces_to_chat_corpus(
        traces, repeat=max(1, args.repeat), max_items=max(1, args.max_items)
    )
    parts = []
    if not args.no_rehearsal:
        parts.append(REHEARSAL.strip() + "\n")
        # Extra rehearsal copies for tiny-data stability
        for _ in range(max(0, args.rehearsal_repeat - 1)):
            parts.append(REHEARSAL.strip() + "\n")
    parts.append(corpus)
    text = "".join(parts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    st = store.stats()
    print(
        f"Wrote {args.out} ({len(text):,} chars) from {st['count']} traces "
        f"by_mode={st['by_mode']}"
    )
    if st["count"] == 0:
        print("Note: no success traces yet — rehearsal-only curriculum written.")
        print("Chat with grounded answers first, or keep rehearsal and LoRA lightly.")


if __name__ == "__main__":
    main()
