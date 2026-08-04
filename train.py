"""Train TinySLM v2 — sample-efficient CPU training."""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from tiny_slm.config import TinySLMConfig
from tiny_slm.model import TinySLM
from tiny_slm.tokenizer import TinyTokenizer

ROOT = Path(__file__).resolve().parent


class TextDataset(Dataset):
    def __init__(self, ids: list[int], block_size: int):
        self.ids = ids
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.ids) - self.block_size - 1)

    def __getitem__(self, i: int):
        chunk = self.ids[i : i + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_corpus(data_dir: Path, chat_repeat: int = 40) -> str:
    """Upsample chat-style files so the model learns dialogue without huge corpora."""
    chat_parts: list[str] = []
    other_parts: list[str] = []
    for path in sorted(data_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        print(f"  + {path.name}: {path.stat().st_size:,} bytes")
        name = path.name.lower()
        # HF bulk corpora: keep once (already include rehearsal). Don't x60-upsample.
        if name.startswith("hf_") or "intelligence" in name:
            other_parts.append(text)
        elif any(
            k in name
            for k in (
                "seed",
                "chat",
                "instruct",
                "agent",
                "interactive",
                "basic",
                "sara",
                "coding",
                "rehearsal",
                "intelligence",
                "production",
                "router",
                "distill",
            )
        ):
            chat_parts.append(text)
        else:
            other_parts.append(text)
    if not chat_parts and not other_parts:
        raise FileNotFoundError(f"No .txt files in {data_dir}. Run: python prepare_data.py")

    chunks: list[str] = []
    if chat_parts:
        chat = "\n\n".join(chat_parts)
        chunks.extend([chat] * max(1, chat_repeat))
        print(f"  chat upsampled x{max(1, chat_repeat)}")
    if other_parts:
        chunks.append("\n\n".join(other_parts))
    return "\n\n".join(chunks)


def cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.1 + 0.9 * base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinySLM v2 from scratch")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-kv-head", type=int, default=1)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--chat-repeat", type=int, default=50)
    parser.add_argument("--threads", type=int, default=0, help="0 = auto (all CPU cores)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume", type=Path, default=None, help="Resume/finetune from checkpoint")
    parser.add_argument(
        "--extend-block-size",
        type=int,
        default=0,
        help="On resume only: safely widen RoPE block_size (e.g. 512). Never shrinks.",
    )
    parser.add_argument(
        "--grow-layers",
        type=int,
        default=0,
        help="On resume: append N near-identity blocks (expand capacity, keep old weights).",
    )
    parser.add_argument(
        "--adapter-rank",
        type=int,
        default=0,
        help="Attach LoRA of this rank (0=off). Prefer with --freeze-base for safe continual FT.",
    )
    parser.add_argument(
        "--adapter-alpha",
        type=float,
        default=16.0,
        help="LoRA alpha scaling (scale=alpha/rank).",
    )
    parser.add_argument(
        "--freeze-base",
        action="store_true",
        help="Freeze base weights; train LoRA and/or newly grown layers only.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Copy current tinyslm.pt to tinyslm_pre_expand.pt before overwriting.",
    )
    parser.add_argument("--reuse-tokenizer", action="store_true", help="Reuse checkpoints/tokenizer.json")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    threads = args.threads or os.cpu_count() or 4
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(max(1, min(4, threads)))
    print(f"CPU threads: {threads}")

    print("Loading corpus...")
    text = load_corpus(args.data_dir, chat_repeat=args.chat_repeat)

    tok_path = args.out_dir / "tokenizer.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_tokenizer and tok_path.exists():
        print(f"Reusing tokenizer {tok_path}")
        tokenizer = TinyTokenizer.load(tok_path)
    else:
        print(f"Training BPE tokenizer (vocab={args.vocab_size})...")
        tok_text = text if len(text) < 120_000 else text[:120_000]
        tokenizer = TinyTokenizer()
        tokenizer.train([tok_text], vocab_size=args.vocab_size, max_chars=60_000)
        tokenizer.save(tok_path)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    print("Encoding corpus...")
    chunk_size = 100_000
    ids: list[int] = []
    for start in range(0, len(text), chunk_size):
        ids.extend(tokenizer.encode(text[start : start + chunk_size]))
        if start == 0 or (start // chunk_size) % 5 == 0:
            print(
                f"  encoded {min(start + chunk_size, len(text)):,}/{len(text):,} chars "
                f"-> {len(ids):,} tokens"
            )
    print(f"Tokens: {len(ids):,}")

    device = torch.device(args.device)
    start_step = 0
    if args.resume and Path(args.resume).exists():
        print(f"Resuming from {args.resume}")
        if args.backup:
            import shutil
            from datetime import datetime

            src = Path(args.resume)
            tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            bak = args.out_dir / f"tinyslm_pre_expand_{tag}.pt"
            shutil.copy2(src, bak)
            print(f"Backup: {bak}")
        model, start_step = TinySLM.load_checkpoint(str(args.resume), map_location=str(device))
        config = model.config
        # Safe context widen (RoPE-only; weights unchanged). Never from-scratch.
        if args.extend_block_size and args.extend_block_size > config.block_size:
            old_b = config.block_size
            model.extend_block_size(args.extend_block_size)
            config = model.config
            print(f"Extended block_size {old_b} -> {config.block_size} (weights preserved)")
        elif args.block_size != config.block_size:
            print(f"Note: keeping checkpoint block_size={config.block_size} (CLI had {args.block_size})")
        if args.grow_layers and args.grow_layers > 0:
            old_l = config.n_layer
            new_l = model.grow_layers(args.grow_layers)
            config = model.config
            print(f"Grew layers {old_l} -> {new_l} (old blocks frozen-capable)")
        if args.adapter_rank and args.adapter_rank > 0:
            n_wrap = model.attach_adapters(rank=args.adapter_rank, alpha=args.adapter_alpha)
            config = model.config
            print(f"Attached LoRA rank={args.adapter_rank} on {n_wrap} linears")
        if args.freeze_base:
            stats = model.freeze_for_continual(train_new_layers=True)
            print(
                f"Continual freeze: trainable={stats['trainable']:,} "
                f"(lora={stats['lora_params']:,}, grown={stats['grown_layer_params']:,}, "
                f"frozen={stats['frozen_params']:,})"
            )
        model.to(device)
    else:
        config = TinySLMConfig(
            vocab_size=tokenizer.vocab_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_kv_head=args.n_kv_head,
            n_embd=args.n_embd,
            block_size=args.block_size,
            dropout=args.dropout,
            bos_token_id=tokenizer.bos_id,
            eos_token_id=tokenizer.eos_id,
            pad_token_id=tokenizer.pad_id,
            unk_token_id=tokenizer.unk_id,
        )
        model = TinySLM(config).to(device)
    config.save(args.out_dir / "config.json")
    print(
        f"Parameters: {model.n_parameters():,} "
        f"(~{model.n_parameters() * 4 / 1e6:.1f} MB float32) "
        f"arch={config.arch} MQA kv_heads={config.n_kv_head}"
    )

    ds = TextDataset(ids, config.block_size)
    if len(ds) < 1:
        raise RuntimeError("Corpus too small for block_size. Add more text or lower --block-size.")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
    )
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=0.05,
        betas=(0.9, 0.95),
    )

    model.train()
    data_iter = iter(loader)
    t0 = time.time()
    running = 0.0
    opt.zero_grad(set_to_none=True)

    for step in range(1, args.steps + 1):
        lr = cosine_lr(step, args.steps, args.lr, args.warmup)
        for pg in opt.param_groups:
            pg["lr"] = lr

        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, y)
        (loss / args.grad_accum).backward()

        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        running += loss.item()
        if step % 50 == 0 or step == 1:
            avg = running / (50 if step >= 50 else step)
            running = 0.0
            elapsed = time.time() - t0
            tok_s = step * args.batch_size * config.block_size / max(elapsed, 1e-6)
            print(
                f"step {step:5d}/{args.steps}  loss={loss.item():.4f}  "
                f"avg={avg:.4f}  lr={lr:.2e}  {tok_s:.0f} tok/s  {elapsed:.1f}s"
            )

        if step % 200 == 0 or step == args.steps:
            ckpt = args.out_dir / "tinyslm.pt"
            model.save_checkpoint(ckpt, optimizer=opt, step=start_step + step)
            print(f"Saved {ckpt}")

    print("Done. Chat with: python app.py")


if __name__ == "__main__":
    main()
