"""TinySLM v2 — CPU-efficient decoder (from-scratch weights).

Architecture choices (inspired by modern SLMs, not their weights):
  - RMSNorm          → cheaper than LayerNorm on CPU
  - RoPE             → no position embedding table; better with little data
  - Multi-Query Attn → 1 KV head → tiny KV cache, fast token-by-token decode
  - SwiGLU MLP       → better sample efficiency than plain GELU FFN
  - Weight tying     → fewer params
  - KV cache generate→ O(1) attention cost per new token (critical on CPU)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TinySLMConfig

KVCache = Tuple[torch.Tensor, torch.Tensor]  # (k, v) each (B, n_kv_head, T, head_dim)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep in float32 for stability on CPU even if x is half later
        orig = x.dtype
        x32 = x.float()
        var = x32.pow(2).mean(dim=-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return (self.weight * x32).to(orig)


def build_rope_cache(
    head_dim: int,
    max_seq: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(max_seq, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freq)  # (T, half)
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D) with even D; cos/sin: (T, D/2)."""
    B, H, T, D = x.shape
    x = x.view(B, H, T, D // 2, 2)
    x1, x2 = x[..., 0], x[..., 1]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # (1,1,T,D/2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack((out1, out2), dim=-1).flatten(-2)


class CausalMQA(nn.Module):
    """Multi-Query Attention (shared KV) for fast CPU inference."""

    def __init__(self, config: TinySLMConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.reps = self.n_head // self.n_kv_head

        self.wq = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.wk = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.wv = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.wo = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # RoPE on q/k. When decoding with cache, positions continue after past length.
        past_len = past_kv[0].size(2) if past_kv is not None else 0
        if past_len == 0:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        else:
            # Only the new tokens (usually T=1) need positions past_len..past_len+T
            cos_new = cos[past_len : past_len + T]
            sin_new = sin[past_len : past_len + T]
            q = apply_rope(q, cos_new, sin_new)
            k = apply_rope(k, cos_new, sin_new)
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        cache: Optional[KVCache] = (k, v) if use_cache else None

        # Expand KV to match Q heads
        if self.reps > 1:
            k = k.repeat_interleave(self.reps, dim=1)
            v = v.repeat_interleave(self.reps, dim=1)

        # Prefer fused SDPA when available (faster / less RAM)
        if hasattr(F, "scaled_dot_product_attention"):
            # Prefill: causal. Decode with cache: attend to all past keys.
            causal = past_len == 0 and T > 1
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=causal,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            if past_len == 0:
                causal_mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
                att = att.masked_fill(~causal_mask, float("-inf"))
            att = self.attn_drop(F.softmax(att, dim=-1))
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        return self.resid_drop(self.wo(y)), cache


class SwiGLUMLP(nn.Module):
    def __init__(self, config: TinySLMConfig):
        super().__init__()
        h = config.intermediate_size
        self.w1 = nn.Linear(config.n_embd, h, bias=config.bias)  # gate
        self.w3 = nn.Linear(config.n_embd, h, bias=config.bias)  # up
        self.w2 = nn.Linear(h, config.n_embd, bias=config.bias)  # down
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, config: TinySLMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = CausalMQA(config)
        self.mlp_norm = RMSNorm(config.n_embd)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        h, cache = self.attn(self.attn_norm(x), cos, sin, past_kv=past_kv, use_cache=use_cache)
        x = x + h
        x = x + self.mlp(self.mlp_norm(x))
        return x, cache


class TinySLM(nn.Module):
    def __init__(self, config: TinySLMConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        head_dim = config.n_embd // config.n_head
        # Buffers filled lazily on first forward for correct device/dtype
        self.register_buffer("_rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("_rope_sin", torch.empty(0), persistent=False)
        self._rope_head_dim = head_dim

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("wo.weight") or pn.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _ensure_rope(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        need = (
            self._rope_cos.numel() == 0
            or self._rope_cos.device != device
            or self._rope_cos.size(0) < self.config.block_size
        )
        if need:
            cos, sin = build_rope_cache(
                self._rope_head_dim,
                self.config.block_size,
                self.config.rope_theta,
                device,
                dtype,
            )
            self._rope_cos = cos
            self._rope_sin = sin
        return self._rope_cos, self._rope_sin

    def n_parameters(self) -> int:
        # Count unique tensors only (weight tying)
        return sum({id(p): p.numel() for p in self.parameters()}.values())

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_kvs: Optional[List[Optional[KVCache]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[KVCache]]]:
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.config.block_size}")

        x = self.drop(self.wte(idx))
        cos, sin = self._ensure_rope(x.device, x.dtype)

        new_kvs: Optional[List[KVCache]] = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past = past_kvs[i] if past_kvs is not None else None
            past_len = past[0].size(2) if past is not None else 0
            if past_len + T > self.config.block_size:
                raise ValueError("KV cache + new tokens exceed block_size")
            x, cache = block(x, cos, sin, past_kv=past, use_cache=use_cache)
            if use_cache and cache is not None:
                new_kvs.append(cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.config.pad_token_id,
            )
        return logits, loss, new_kvs

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        eos = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        # Prefill once, then decode token-by-token with KV cache
        if idx.size(1) > self.config.block_size:
            idx = idx[:, -self.config.block_size :]

        logits, _, past = self(idx, use_cache=True)
        for _ in range(max_new_tokens):
            logits_last = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
                logits_last = logits_last.clone()
                logits_last[logits_last < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits_last, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if (next_id == eos).all():
                break
            # Truncate cache if we would exceed context (rare for chat)
            if past is not None and past[0][0].size(2) >= self.config.block_size:
                past = None
                logits, _, past = self(idx[:, -self.config.block_size :], use_cache=True)
            else:
                logits, _, past = self(next_id, past_kvs=past, use_cache=True)
        return idx

    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None, step: int = 0) -> None:
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.state_dict(),
            "config": asdict_config(self.config),
            "step": step,
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path: str, map_location: str = "cpu") -> Tuple["TinySLM", int]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        config = TinySLMConfig.from_dict(payload["config"])
        model = cls(config)
        model.load_state_dict(payload["model"], strict=False)
        return model, int(payload.get("step", 0))


def asdict_config(config: TinySLMConfig) -> dict:
    from dataclasses import asdict

    return asdict(config)
