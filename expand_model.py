"""Cascade Soft-Preserve Expand: 4.6M → ~60–100M without capability wipe.

Usage:
  python expand_model.py --target-params 85000000 --distill-steps 300

Steps:
  1) Backup current checkpoint (teacher)
  2) CSPE widen + deepen (soft-preserve)
  3) Shadow-teacher KL + CE rehearsal on chat curricula
  4) Freeze original layers; train new layers + LoRA
  5) Save checkpoints/tinyslm.pt
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train import TextDataset, cosine_lr, load_corpus
from tiny_slm.expand import cascade_expand, soft_preserve_error
from tiny_slm.model import TinySLM
from tiny_slm.tokenizer import TinyTokenizer

ROOT = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description="CSPE expand TinySLM toward 60–100M")
    p.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints" / "tinyslm.pt")
    p.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "tinyslm.pt")
    p.add_argument("--teacher-out", type=Path, default=None, help="Where to store teacher copy")
    p.add_argument("--target-params", type=int, default=85_000_000)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--tok", type=Path, default=ROOT / "checkpoints" / "tokenizer.json")
    p.add_argument("--distill-steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--distill-alpha", type=float, default=0.45, help="KL weight vs CE")
    p.add_argument("--distill-temp", type=float, default=2.0)
    p.add_argument("--chat-repeat", type=int, default=30)
    p.add_argument("--adapter-rank", type=int, default=8)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-distill", action="store_true")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    threads = os.cpu_count() or 4
    torch.set_num_threads(threads)
    device = torch.device(args.device)

    if not args.ckpt.exists():
        raise SystemExit(f"Missing {args.ckpt}")

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    teacher_path = args.teacher_out or (ROOT / "checkpoints" / f"tinyslm_teacher_pre_cspe_{tag}.pt")
    shutil.copy2(args.ckpt, teacher_path)
    print(f"Teacher backup: {teacher_path}")

    print("Loading student seed (current model)...")
    student, step0 = TinySLM.load_checkpoint(str(args.ckpt), map_location="cpu")
    print(f"  before: {student.n_parameters():,} params  arch={student.config.arch}")

    print("Loading frozen teacher for shadow distill...")
    teacher, _ = TinySLM.load_checkpoint(str(teacher_path), map_location=str(device))
    teacher.to(device)
    teacher.eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)

    print(f"CSPE expand toward ~{args.target_params/1e6:.0f}M...")
    student, report = cascade_expand(student, target_params=args.target_params)
    print(
        f"  plan={report['plan']}\n"
        f"  merged_lora={report['merged_lora']}\n"
        f"  params {report['old_params']:,} -> {report['new_params']:,}\n"
        f"  layers={report['n_layer']} embd={report['n_embd']} ff={report['intermediate_size']}"
    )

    # Attach LoRA on expanded model; freeze original-depth blocks' base weights
    n_lora = student.attach_adapters(rank=args.adapter_rank, alpha=16.0)
    stats = student.freeze_for_continual(train_new_layers=True)
    # Also unfreeze ln_f / wte lightly? Keep frozen for max preserve — only new layers + LoRA
    print(f"  LoRA wrapped={n_lora} trainable={stats['trainable']:,}")

    student.to(device)
    try:
        err = soft_preserve_error(teacher.cpu(), student.cpu())
        student.to(device)
        teacher.to(device)
        print(f"  soft-preserve logit MAE (informational): {err:.4f}")
    except Exception as exc:
        print(f"  soft-preserve check skipped: {type(exc).__name__}")

    # Save expanded weights even before distill (recoverable)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    student.save_checkpoint(args.out, step=step0)
    student.config.save(args.out.parent / "config.json")
    print(f"Saved expanded checkpoint (pre-distill): {args.out}")

    if args.skip_distill or args.distill_steps <= 0:
        print("Skip distill. Done.")
        return

    print("Loading rehearsal corpus for shadow-teacher distill...")
    text = load_corpus(args.data_dir, chat_repeat=args.chat_repeat)
    tok = TinyTokenizer.load(args.tok)
    ids: list[int] = []
    chunk = 100_000
    for i in range(0, len(text), chunk):
        ids.extend(tok.encode(text[i : i + chunk]))
    print(f"Tokens: {len(ids):,}")

    block = student.config.block_size
    ds = TextDataset(ids, block)
    if len(ds) < 1:
        raise SystemExit("Corpus too small")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    opt = torch.optim.AdamW(
        (p for p in student.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=0.05,
        betas=(0.9, 0.95),
    )

    student.train()
    teacher.eval()
    data_iter = iter(loader)
    t0 = time.time()
    running = 0.0
    opt.zero_grad(set_to_none=True)
    Ttemp = max(args.distill_temp, 1e-6)
    alpha = min(max(args.distill_alpha, 0.0), 0.95)

    for step in range(1, args.distill_steps + 1):
        lr = cosine_lr(step, args.distill_steps, args.lr, args.warmup)
        for pg in opt.param_groups:
            pg["lr"] = lr
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        logits_s, ce, _ = student(x, y)
        with torch.no_grad():
            logits_t, _, _ = teacher(x)
        # Shadow teacher KL (same vocab — architecture-agnostic)
        log_p = F.log_softmax(logits_s / Ttemp, dim=-1)
        q = F.softmax(logits_t / Ttemp, dim=-1)
        kl = F.kl_div(log_p, q, reduction="batchmean") * (Ttemp ** 2)
        loss = (1.0 - alpha) * ce + alpha * kl
        (loss / args.grad_accum).backward()

        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        running += float(loss.item())
        if step % 25 == 0 or step == 1 or step == args.distill_steps:
            avg = running / (25 if step >= 25 else step)
            running = 0.0
            elapsed = time.time() - t0
            print(
                f"step {step:4d}/{args.distill_steps}  loss={loss.item():.4f}  "
                f"ce={ce.item():.4f}  kl={kl.item():.4f}  avg={avg:.4f}  "
                f"lr={lr:.2e}  {elapsed:.1f}s"
            )
        if step % 100 == 0 or step == args.distill_steps:
            student.save_checkpoint(args.out, optimizer=opt, step=step0 + step)
            student.config.save(args.out.parent / "config.json")
            print(f"Saved {args.out} ({student.n_parameters():,} params)")

    print(
        f"Done CSPE. params={student.n_parameters():,}  "
        f"teacher={teacher_path}  student={args.out}"
    )
    print("Verify with: python smoke_tick.py && python eval_advanced.py")


if __name__ == "__main__":
    main()
