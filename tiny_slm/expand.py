"""Cascade Soft-Preserve Expand (CSPE) — scale TinySLM without wiping skill.

Novel (for this stack) expansion recipe:
  1) Merge LoRA into base (absorb current intelligence into dense weights)
  2) Soft-preserve *widen*: copy old subspace; zero new residual channels so
     f_new(x) ≈ f_old(x) at attach time (Net2Wider-style, MQA/SwiGLU aware)
  3) Soft-preserve *deepen*: append zero-init residual blocks (existing grow_layers)
  4) Shadow-teacher distill afterward (KL to pre-expand teacher + CE rehearsal)

Target band for production: ~60–100M params while keeping tools/policy intact.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from tiny_slm.adapters import LoRALinear
from tiny_slm.config import TinySLMConfig
from tiny_slm.model import Block, CausalMQA, SwiGLUMLP, TinySLM


def merge_lora_into_base_(model: TinySLM) -> int:
    """Fold LoRA deltas into base Linear weights and unwrap. Returns merges."""
    n = 0
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                with torch.no_grad():
                    child.base.weight.add_((child.lora_B @ child.lora_A) * child.scale)
                setattr(parent, name, child.base)
                n += 1
    model.config.adapter_rank = 0
    if model.config.arch.endswith("_lora"):
        model.config.arch = model.config.arch[: -len("_lora")]
    return n


def _copy_linear(
    src: nn.Linear,
    dst: nn.Linear,
    *,
    zero_new_outputs: bool = False,
) -> None:
    """Copy overlapping weights; optionally silence brand-new output rows."""
    with torch.no_grad():
        so, si = src.weight.shape
        do, di = dst.weight.shape
        o, i = min(so, do), min(si, di)
        dst.weight.zero_()
        dst.weight[:o, :i].copy_(src.weight[:o, :i])
        if zero_new_outputs and do > so:
            dst.weight[so:, :].zero_()
        if src.bias is not None and dst.bias is not None:
            dst.bias.zero_()
            dst.bias[: min(so, do)].copy_(src.bias[: min(so, do)])


def _widen_rmsnorm(src: nn.Module, dst: nn.Module, old_dim: int, new_dim: int) -> None:
    """Pad RMSNorm; rescale old gains to offset variance dilution from zero-pad."""
    with torch.no_grad():
        old = src.weight
        new = dst.weight
        new.zero_()
        # New channels start at 0 gain; old gains compensated for longer mean()
        scale = math.sqrt(float(old_dim) / float(new_dim)) if new_dim else 1.0
        new[: old.numel()].copy_(old * scale)


def widen_model(
    model: TinySLM,
    new_n_embd: int,
    new_intermediate: int = 0,
    new_n_head: Optional[int] = None,
    new_n_kv_head: Optional[int] = None,
) -> TinySLM:
    """Return a wider model with soft-preserved weights (same depth)."""
    old_cfg = model.config
    new_n_embd = int(new_n_embd)
    if new_n_embd < old_cfg.n_embd:
        raise ValueError("widen_model refuses to shrink n_embd")
    if new_n_embd == old_cfg.n_embd and (
        not new_intermediate or new_intermediate == old_cfg.intermediate_size
    ):
        return model

    n_head = int(new_n_head or old_cfg.n_head)
    n_kv = int(new_n_kv_head or old_cfg.n_kv_head)
    if new_n_embd % n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")
    if n_head % n_kv != 0:
        raise ValueError("n_head must be divisible by n_kv_head")

    if new_intermediate and new_intermediate > 0:
        ff = int(new_intermediate)
    else:
        # Keep Llama-ish ~8/3 ratio, multiple of 64
        hidden = int(8 * new_n_embd / 3)
        ff = max(old_cfg.intermediate_size, ((hidden + 63) // 64) * 64)

    new_cfg = TinySLMConfig(
        vocab_size=old_cfg.vocab_size,
        n_layer=old_cfg.n_layer,
        n_head=n_head,
        n_kv_head=n_kv,
        n_embd=new_n_embd,
        intermediate_size=ff,
        block_size=old_cfg.block_size,
        dropout=old_cfg.dropout,
        bias=old_cfg.bias,
        rope_theta=old_cfg.rope_theta,
        arch="tinyslm_v2_cspe",
        max_memory_tokens=old_cfg.max_memory_tokens,
        rope_scale=old_cfg.rope_scale,
        adapter_rank=0,
        adapter_alpha=old_cfg.adapter_alpha,
        base_n_layer=int(getattr(old_cfg, "base_n_layer", 0) or old_cfg.n_layer),
        bos_token_id=old_cfg.bos_token_id,
        eos_token_id=old_cfg.eos_token_id,
        pad_token_id=old_cfg.pad_token_id,
        unk_token_id=old_cfg.unk_token_id,
    )
    student = TinySLM(new_cfg)

    with torch.no_grad():
        # Embeddings / tied lm_head
        student.wte.weight.zero_()
        student.wte.weight[:, : old_cfg.n_embd].copy_(model.wte.weight)
        _widen_rmsnorm(model.ln_f, student.ln_f, old_cfg.n_embd, new_n_embd)

        for ob, nb in zip(model.blocks, student.blocks):
            _widen_rmsnorm(ob.attn_norm, nb.attn_norm, old_cfg.n_embd, new_n_embd)
            _widen_rmsnorm(ob.mlp_norm, nb.mlp_norm, old_cfg.n_embd, new_n_embd)
            # Attention: silence new residual channels via wo
            _copy_linear(ob.attn.wq, nb.attn.wq, zero_new_outputs=False)
            _copy_linear(ob.attn.wk, nb.attn.wk, zero_new_outputs=False)
            _copy_linear(ob.attn.wv, nb.attn.wv, zero_new_outputs=False)
            _copy_linear(ob.attn.wo, nb.attn.wo, zero_new_outputs=True)
            # MLP: new ff rows small; down-proj silences new embd outs + new ff cols start 0
            _copy_linear(ob.mlp.w1, nb.mlp.w1, zero_new_outputs=False)
            _copy_linear(ob.mlp.w3, nb.mlp.w3, zero_new_outputs=False)
            _copy_linear(ob.mlp.w2, nb.mlp.w2, zero_new_outputs=True)
            # Tiny noise on brand-new ff rows so they can learn (not stuck at 0 gate)
            if nb.mlp.w1.out_features > ob.mlp.w1.out_features:
                start = ob.mlp.w1.out_features
                nn.init.normal_(nb.mlp.w1.weight[start:], std=0.02 / math.sqrt(2 * new_cfg.n_layer))
                nn.init.normal_(nb.mlp.w3.weight[start:], std=0.02 / math.sqrt(2 * new_cfg.n_layer))

    student.config.arch = "tinyslm_v2_cspe"
    return student


def plan_cspe_target(target_params: int = 85_000_000) -> dict:
    """Pick a width/depth recipe in the 60–100M band."""
    target_params = int(target_params)
    # Prefer keeping n_head=8 (stable head_dim growth) for soft-preserve copies
    recipes = [
        dict(n_embd=768, n_layer=14, n_head=8, n_kv_head=1, intermediate_size=2048),
        dict(n_embd=768, n_layer=12, n_head=8, n_kv_head=1, intermediate_size=2048),
        dict(n_embd=640, n_layer=16, n_head=8, n_kv_head=1, intermediate_size=1728),
        dict(n_embd=768, n_layer=16, n_head=12, n_kv_head=2, intermediate_size=2048),
    ]
    best = None
    best_err = 1e18
    for r in recipes:
        cfg = TinySLMConfig(vocab_size=1024, block_size=512, **r)
        est = cfg.n_params_estimate()
        if est < 60_000_000 or est > 110_000_000:
            continue
        err = abs(est - target_params)
        if err < best_err:
            best_err = err
            best = {**r, "est_params": est}
    if best is None:
        best = dict(
            n_embd=768,
            n_layer=14,
            n_head=8,
            n_kv_head=1,
            intermediate_size=2048,
            est_params=TinySLMConfig(
                vocab_size=1024,
                n_embd=768,
                n_layer=14,
                n_head=8,
                n_kv_head=1,
                intermediate_size=2048,
                block_size=512,
            ).n_params_estimate(),
        )
    return best


def cascade_expand(
    model: TinySLM,
    *,
    target_params: int = 85_000_000,
    recipe: Optional[dict] = None,
) -> Tuple[TinySLM, dict]:
    """Full CSPE: merge LoRA → widen → deepen. Returns (new_model, report)."""
    report = {"merged_lora": 0, "old_params": model.n_parameters()}
    report["merged_lora"] = merge_lora_into_base_(model)

    plan = recipe or plan_cspe_target(target_params)
    report["plan"] = plan

    # Record original depth before widen (widen keeps depth)
    old_layers = int(model.config.n_layer)
    if not getattr(model.config, "base_n_layer", 0):
        model.config.base_n_layer = old_layers

    student = widen_model(
        model,
        new_n_embd=int(plan["n_embd"]),
        new_intermediate=int(plan["intermediate_size"]),
        new_n_head=int(plan["n_head"]),
        new_n_kv_head=int(plan["n_kv_head"]),
    )
    student.config.base_n_layer = old_layers

    extra = max(0, int(plan["n_layer"]) - int(student.config.n_layer))
    if extra:
        student.grow_layers(extra)
        # grow_layers overwrites arch; restore CSPE tag
        student.config.arch = "tinyslm_v2_cspe_expandable"

    report["new_params"] = student.n_parameters()
    report["n_layer"] = student.config.n_layer
    report["n_embd"] = student.config.n_embd
    report["intermediate_size"] = student.config.intermediate_size
    return student, report


@torch.no_grad()
def soft_preserve_error(teacher: TinySLM, student: TinySLM, vocab: int = 1024) -> float:
    """Mean abs logit delta on random tokens (should be tiny right after CSPE)."""
    teacher.eval()
    student.eval()
    x = torch.randint(0, min(vocab, teacher.config.vocab_size), (2, 16))
    # Student may have larger block but same vocab
    lt, _, _ = teacher(x)
    # Student forward uses its own embd — compare only if same depth path works
    ls, _, _ = student(x)
    return float((lt - ls).abs().mean().item())
