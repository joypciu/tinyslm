"""Quick quality probe: grounded FAQ/code + auto web + compression recall."""

from __future__ import annotations

from tiny_slm.chat import TinyChat


def body(ans: str) -> str:
    return (ans.split("[model]")[-1] if "[model]" in ans else ans).strip()


def main() -> None:
    fails = []
    c = TinyChat(auto_search=False)

    offline = [
        ("Hello!", ["tinyslm", "hello", "hi"]),
        ("What is 2 + 2?", ["4"]),
        ("What is Python?", ["python"]),
        ("Write a Python function that adds two numbers.", ["def add"]),
        ("What is machine learning?", ["learning"]),
        ("What is a transformer model?", ["attention", "transformer"]),
    ]
    for q, needles in offline:
        c.reset()
        c.clear_memory()
        a, _ = c.generate_reply(q, temperature=0.2)
        b = body(a).lower()
        if not any(n.lower() in b for n in needles):
            fails.append(f"offline:{q}->{body(a)[:80]!r}")

    # Compression + memory
    c.reset()
    c.clear_memory()
    c.generate_reply(
        "Remember project token ZETA_PROBE_77 for later. " + ("pad notes. " * 20),
        temperature=0.2,
        max_new_tokens=20,
        use_sara=False,
    )
    for _ in range(4):
        c.generate_reply("Continue chatting briefly about focus.", temperature=0.2, max_new_tokens=20)
    a, _ = c.generate_reply(
        "Using memory, what was the secret code or token related to ZETA?",
        temperature=0.1,
    )
    if "ZETA_PROBE_77" not in body(a):
        fails.append(f"memory:{body(a)[:80]!r}")

    # Live web (best-effort)
    c2 = TinyChat(auto_search=True)
    c2.reset()
    c2.clear_memory()
    a, dig = c2.generate_reply("Who is Ada Lovelace?", temperature=0.2)
    if "[web]" not in a and not dig:
        fails.append("web:no digest")
    elif "ada" not in body(a).lower() and "lovelace" not in body(a).lower():
        fails.append(f"web:{body(a)[:100]!r}")

    print(f"probe_intelligence fails={len(fails)}")
    for f in fails:
        print(" ", f)
    if fails:
        raise SystemExit(1)
    print("probe_intelligence OK")


if __name__ == "__main__":
    main()
