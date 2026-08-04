"""Safe short LoRA distill tick on verified success traces.

Runs: seed → prepare_distill_traces → freeze-base LoRA (low LR, few steps).
Never full-FT the CSPE ~87M checkpoint. Prefer this for continual learning.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--adapter-rank", type=int, default=8)
    p.add_argument("--skip-seed", action="store_true")
    args = p.parse_args()

    if not args.skip_seed:
        subprocess.check_call([sys.executable, str(ROOT / "seed_verified_traces.py")], cwd=ROOT)
    subprocess.check_call([sys.executable, str(ROOT / "prepare_distill_traces.py")], cwd=ROOT)

    tick_dir = ROOT / "data" / "lora_tick"
    tick_dir.mkdir(parents=True, exist_ok=True)
    src = ROOT / "data" / "distill_traces.txt"
    if not src.exists():
        raise SystemExit("missing data/distill_traces.txt")
    shutil.copy2(src, tick_dir / "distill_traces.txt")

    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "--resume",
        str(ROOT / "checkpoints" / "tinyslm.pt"),
        "--reuse-tokenizer",
        "--adapter-rank",
        str(args.adapter_rank),
        "--freeze-base",
        "--backup",
        "--data-dir",
        str(tick_dir),
        "--chat-repeat",
        "2",
        "--batch-size",
        "2",
        "--grad-accum",
        "4",
        "--warmup",
        "10",
        "--steps",
        str(args.steps),
        "--lr",
        str(args.lr),
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)
    print("LoRA distill tick done. Verify with: python smoke_tick.py && python eval_advanced.py")


if __name__ == "__main__":
    main()
