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
    mem_path = ROOT / "checkpoints" / "memory_store.json"
    print(
        f"TinySLM ready ({chat.model.n_parameters():,} params). "
        "Commands: quit | /clear | /memory | /save | /load | /ingest <path> | /search <q>\n"
        "Auto web search runs when a question looks like it needs live facts.\n"
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
        if user.lower() in {"/save", "save"}:
            chat.save_memory(mem_path)
            print(f"(memory saved to {mem_path})")
            continue
        if user.lower() in {"/load", "load"}:
            info = chat.load_memory(mem_path)
            print(f"(memory loaded: {info})")
            continue
        if user.lower().startswith("/ingest "):
            path = Path(user[8:].strip().strip('"'))
            if not path.exists():
                print(f"(missing file: {path})")
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            info = chat.ingest(text, source=f"file:{path.name}")
            print(f"(ingested {path.name}: {info})")
            continue
        force = user.lower().startswith("/search ")
        if force:
            user = user[8:].strip()
        reply, _ = chat.generate_reply(user, force_search=force)
        print(f"TinySLM: {reply}\n")


if __name__ == "__main__":
    main()
