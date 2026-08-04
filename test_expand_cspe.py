"""Unit tests for Cascade Soft-Preserve Expand (no long train)."""

from __future__ import annotations

import torch

from tiny_slm.config import TinySLMConfig
from tiny_slm.expand import cascade_expand, merge_lora_into_base_, plan_cspe_target, widen_model
from tiny_slm.model import TinySLM


def test_plan_in_band() -> None:
    plan = plan_cspe_target(85_000_000)
    assert 60_000_000 <= plan["est_params"] <= 110_000_000, plan


def test_widen_soft_preserve() -> None:
    cfg = TinySLMConfig(
        vocab_size=128, n_layer=2, n_head=4, n_kv_head=1, n_embd=64, intermediate_size=128, block_size=32
    )
    m = TinySLM(cfg)
    m.attach_adapters(rank=4, alpha=8.0)
    # Train LoRA a tiny bit so merge is non-trivial
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.add_(0.01)
    x = torch.randint(0, 128, (2, 8))
    with torch.no_grad():
        before, _, _ = m(x)
    n_merge = merge_lora_into_base_(m)
    assert n_merge > 0
    wide = widen_model(m, new_n_embd=128, new_intermediate=256, new_n_head=4)
    assert wide.config.n_embd == 128
    assert wide.n_parameters() > m.n_parameters()
    with torch.no_grad():
        after, _, _ = wide(x)
    # Soft-preserve: logits should stay in the same ballpark (not random)
    mae = (before - after).abs().mean().item()
    assert mae < 0.5, f"widen drifted too far: {mae}"


def test_cascade_grows_to_midsize_shape() -> None:
    cfg = TinySLMConfig(
        vocab_size=128, n_layer=2, n_head=4, n_kv_head=1, n_embd=64, intermediate_size=128, block_size=32
    )
    m = TinySLM(cfg)
    # Tiny custom recipe (not 85M — unit-speed)
    student, report = cascade_expand(
        m,
        recipe=dict(n_embd=128, n_layer=4, n_head=4, n_kv_head=1, intermediate_size=256),
    )
    assert student.config.n_layer == 4
    assert student.config.n_embd == 128
    assert student.config.base_n_layer == 2
    assert report["new_params"] > report["old_params"]
    x = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        logits, loss, _ = student(x, x)
    assert logits.shape[-1] == 128
    assert loss is not None


def main() -> None:
    tests = [test_plan_in_band, test_widen_soft_preserve, test_cascade_grows_to_midsize_shape]
    fails = []
    for fn in tests:
        try:
            fn()
            print(f"  OK {fn.__name__}")
        except Exception as exc:
            fails.append(f"{fn.__name__}: {exc}")
            print(f"  FAIL {fn.__name__}: {exc}")
    if fails:
        raise SystemExit(1)
    print("test_expand_cspe OK")


if __name__ == "__main__":
    main()
