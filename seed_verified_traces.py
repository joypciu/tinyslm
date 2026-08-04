"""CLI: mint verified math/code success traces for LoRA distill."""

from __future__ import annotations

from tiny_slm.seed_traces import seed_verified_traces
from tiny_slm.traces import TraceStore


def main() -> None:
    n = seed_verified_traces()
    st = TraceStore().stats()
    print(f"Seeded {n} verified traces. Store: {st}")


if __name__ == "__main__":
    main()
