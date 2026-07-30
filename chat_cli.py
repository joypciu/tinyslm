"""CLI chat — no Gradio required."""

from __future__ import annotations

import argparse
from pathlib import Path

from tiny_slm.chat import TinyChat

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with TinySLM in the terminal")
    parser.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints" / "tinyslm.pt")
    parser.add_argument("--tok", type=Path, default=ROOT / "checkpoints" / "tokenizer.json")
    args = parser.parse_args()

    chat = TinyChat(ckpt_path=args.ckpt, tok_path=args.tok)
    print(
        f"TinySLM ready ({chat.model.n_parameters():,} params). "
        "Commands: quit | /clear | /memory | /search <q>\n"
    )

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"quit", "exit", "/q"}:
            break
        if user.lower() in {"clear", "/clear"}:
            chat.reset()
            chat.clear_memory()
            print("(history + memory cleared; skills re-seeded)")
            continue
        if user.lower() in {"/memory", "memory"}:
            st = chat.memory.stats()
            print(
                f"(memory {st['tokens']:,}/{st['max_tokens']:,} tok, "
                f"{st['chunks']} chunks, {st['fill_pct']}% full; "
                f"history turns={len(chat.history)})"
            )
            continue
        force = user.lower().startswith("/search ")
        if force:
            user = user[8:].strip()
        reply, _ = chat.generate_reply(user, force_search=force)
        print(f"TinySLM: {reply}\n")


if __name__ == "__main__":
    main()
