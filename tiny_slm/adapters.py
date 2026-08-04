"""Expandable adapters for TinySLM — train new capacity without wiping base weights.

Design goals:
  - Freeze the pretrained decoder; train small LoRA deltas (or new grown layers).
  - Zero-init so behavior matches the base model at attach time.
  - LongContextMemory is untouched (external BM25 store).
  - Checkpoints remain resume-safe via strict=False + config.adapter_rank.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Low-rank adapter around a frozen Linear. Starts as exact identity (B=0)."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        if rank < 1:
            raise ValueError("adapter rank must be >= 1")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        in_f = base.in_features
        out_f = base.out_features
        # Freeze base weights (keeps prior knowledge)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.zeros(self.rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B stays zero → adapter contributes nothing until trained

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # x @ A^T @ B^T
        delta = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + delta * self.scale

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias


def _wrap_linear(module: nn.Module, attr: str, rank: int, alpha: float) -> bool:
    lin = getattr(module, attr, None)
    if not isinstance(lin, nn.Linear):
        return False
    if isinstance(lin, LoRALinear):
        return False
    setattr(module, attr, LoRALinear(lin, rank=rank, alpha=alpha))
    return True


def count_lora_linears(model: nn.Module) -> int:
    """How many LoRALinear wrappers are already attached."""
    return sum(1 for mod in model.modules() if isinstance(mod, LoRALinear))


def inject_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0) -> int:
    """Attach LoRA to attention + MLP projections. Returns number of *new* wraps."""
    from tiny_slm.model import CausalMQA, SwiGLUMLP  # local import avoids cycles

    n = 0
    for mod in model.modules():
        if isinstance(mod, CausalMQA):
            for name in ("wq", "wk", "wv", "wo"):
                if _wrap_linear(mod, name, rank, alpha):
                    n += 1
        elif isinstance(mod, SwiGLUMLP):
            for name in ("w1", "w2", "w3"):
                if _wrap_linear(mod, name, rank, alpha):
                    n += 1
    return n


def freeze_base_parameters(model: nn.Module) -> tuple[int, int]:
    """Freeze everything except LoRA + newly grown block params (requires_grad already True).

    Heuristic: keep requires_grad on params whose name contains lora_ OR that are
    already trainable and belong to blocks beyond the original frozen set.
    Caller typically freezes all, then unfreezes adapters / new layers.
    """
    frozen = 0
    kept = 0
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad_(True)
            kept += p.numel()
        else:
            p.requires_grad_(False)
            frozen += p.numel()
    return frozen, kept


def unfreeze_modules(modules: Iterable[nn.Module]) -> int:
    n = 0
    for mod in modules:
        for p in mod.parameters():
            p.requires_grad_(True)
            n += p.numel()
    return n


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adapter_parameter_names(model: nn.Module) -> List[str]:
    return [n for n, _ in model.named_parameters() if "lora_" in n]
