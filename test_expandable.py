"""Quick unit checks for expandable continual-learning path (no full train)."""

from __future__ import annotations

import copy

import torch

from tiny_slm.adapters import trainable_parameter_count
from tiny_slm.config import TinySLMConfig
from tiny_slm.model import TinySLM


def main() -> None:
    cfg = TinySLMConfig(vocab_size=128, n_layer=2, n_head=4, n_kv_head=1, n_embd=64, block_size=32)
    m = TinySLM(cfg)
    base_params = m.n_parameters()

    # Grow layers — near-identity at attach
    x = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        logits0, _, _ = m(x)
    m.grow_layers(1)
    assert m.config.n_layer == 3
    with torch.no_grad():
        logits1, _, _ = m(x)
    # Zero-init residual outs → logits should stay very close
    delta = (logits0 - logits1).abs().mean().item()
    assert delta < 1e-4, f"grow_layers drifted too much: {delta}"

    # LoRA attach — identity at start
    n = m.attach_adapters(rank=4, alpha=8.0)
    assert n > 0
    with torch.no_grad():
        logits2, _, _ = m(x)
    delta2 = (logits1 - logits2).abs().mean().item()
    assert delta2 < 1e-5, f"LoRA attach drifted: {delta2}"

    stats = m.freeze_for_continual(train_new_layers=True)
    assert stats["trainable"] > 0
    assert trainable_parameter_count(m) == stats["trainable"]
    assert trainable_parameter_count(m) < m.n_parameters()

    # Save / load roundtrip with LoRA keys
    path = "checkpoints/_expand_test.pt"
    m.save_checkpoint(path, step=1)
    m2, step = TinySLM.load_checkpoint(path)
    assert step == 1
    assert any("lora_" in k for k in m2.state_dict())
    with torch.no_grad():
        a, _, _ = m(x)
        b, _, _ = m2(x)
    assert (a - b).abs().mean().item() < 1e-5

    print("expand_ok", True)
    print("base_params", base_params)
    print("grown_layers", m.config.n_layer)
    print("lora_wrapped", n)
    print("trainable", stats["trainable"])
    print("OVERALL PASS")


if __name__ == "__main__":
    main()
