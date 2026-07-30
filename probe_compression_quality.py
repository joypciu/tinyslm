"""Probe automatic context compression + general/code reply quality."""

from __future__ import annotations

from tiny_slm.chat import TinyChat


def body(ans: str) -> str:
    if "[model]" in ans:
        ans = ans.split("[model]")[-1]
    return ans.strip()


def main() -> None:
    c = TinyChat(auto_search=False)
    c.clear_memory()
    c.reset()
    block = c.model.config.block_size
    print("=== COMPRESSION MECHANICS ===")
    print(f"block_size={block} params={c.model.n_parameters():,}")

    print("turn | hist | win | prompt_ids | truncated | hist_compressed | mem_tok")
    for i in range(1, 12):
        user = (
            f"Turn {i}. "
            + ("Detailed notes about algorithms, databases, networks, and habits. " * 8)
        )
        if i == 3:
            user += " Remember project token ALPHA_COMPRESS_99 for later."
        c.generate_reply(user, temperature=0.25, max_new_tokens=48)
        win = c._history_window()
        prompt = c._build_prompt("ping")
        ids = c.tokenizer.encode(prompt)
        # Same left-truncation as generate path
        truncated = len(ids) >= block
        hist_compressed = len(c.history) > len(win)
        print(
            f"{i:4d} | {len(c.history):4d} | {len(win):3d} | {len(ids):10d} | "
            f"{str(truncated):9s} | {str(hist_compressed):15s} | {c.memory.stats()['tokens']}"
        )

    q = "Using memory, what was the secret code or token related to ALPHA?"
    ans, _ = c.generate_reply(q, temperature=0.1)
    ret = c.memory.retrieve(q, top_k=3)
    print("needle_mem", "ALPHA_COMPRESS_99" in ret)
    print("needle_reply", body(ans)[:140])

    print("\n=== GENERAL REPLIES ===")
    c2 = TinyChat(auto_search=False)
    for q in [
        "Hello!",
        "What is photosynthesis?",
        "Give me a short sleep tip.",
        "Explain gravity simply.",
        "Why do we sleep?",
        "What is the capital of Canada?",
    ]:
        c2.reset()
        c2.clear_memory()
        a, _ = c2.generate_reply(q, temperature=0.3, max_new_tokens=80)
        print("Q:", q)
        print("A:", body(a)[:220])
        print()

    print("=== CODE REPLIES ===")
    for q in [
        "What is Python?",
        "Break down this long task: learn Python basics.",
        "Write a Python function that adds two numbers.",
        "What does if/else do in Python?",
        "Explain what a for loop is in programming.",
        "How do I reverse a string in Python?",
        "What is a variable in programming?",
    ]:
        c2.reset()
        c2.clear_memory()
        a, _ = c2.generate_reply(
            q,
            temperature=0.25,
            max_new_tokens=100,
            force_agent=("Break" in q),
        )
        print("Q:", q)
        print("A:", body(a)[:260])
        print()


if __name__ == "__main__":
    main()
