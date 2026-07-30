"""Efficient TinySLM config — CPU / low-RAM friendly defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
import json


@dataclass
class TinySLMConfig:
    # Compact Llama-style decoder optimized for CPU decode + sample efficiency
    vocab_size: int = 2048
    n_layer: int = 6
    n_head: int = 8
    n_kv_head: int = 1  # Multi-Query Attention → tiny KV cache, fast CPU generate
    n_embd: int = 256
    intermediate_size: int = 0  # 0 → auto (SwiGLU ~8/3 * n_embd, multiple of 64)
    block_size: int = 256
    dropout: float = 0.0  # 0 is better for tiny data; enable only if overfitting
    bias: bool = False
    rope_theta: float = 500000.0  # high-base RoPE for longer effective windows
    arch: str = "tinyslm_v2"
    # Neural window (RAM-bound). Long tasks use LongContextMemory up to 2M tokens.
    max_memory_tokens: int = 2_000_000
    # Soft factor for RoPE when decoding past the last training length (1.0 = off).
    rope_scale: float = 1.0

    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    unk_token_id: int = 3

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        if self.intermediate_size <= 0:
            # Llama SwiGLU sizing: ~8/3 d, rounded to 64
            hidden = int(8 * self.n_embd / 3)
            self.intermediate_size = max(64, ((hidden + 63) // 64) * 64)

    def n_params_estimate(self) -> int:
        d = self.n_embd
        h = self.n_head
        kv = self.n_kv_head
        hd = d // h
        emb = self.vocab_size * d
        # q + k + v + o
        attn = d * d + 2 * (kv * hd) * d + d * d
        # SwiGLU: gate + up + down
        mlp = 2 * d * self.intermediate_size + self.intermediate_size * d
        norms = 2 * d
        return emb + self.n_layer * (attn + mlp + norms) + d

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TinySLMConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    @classmethod
    def from_dict(cls, data: dict) -> "TinySLMConfig":
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})
