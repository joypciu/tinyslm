"""Eval agentic + coding quality (templates + neural, search off)."""

from __future__ import annotations

from tiny_slm.chat import TinyChat


def body(ans: str) -> str:
    return (ans.split("[model]")[-1] if "[model]" in ans else ans).strip()


CODING = [
    ("Write a Python function that adds two numbers.", ["def add", "return"]),
    ("How do I reverse a string in Python?", ["[::-1]", "reversed"]),
    ("Explain what a for loop is in programming.", ["for", "in"]),
    ("What does if/else do in Python?", ["if", "else"]),
    ("What is a variable in programming?", ["variable", "value"]),
    ("How do I read a file in Python?", ["open(", "read"]),
    ("Sort a list in Python.", ["sort", "sorted"]),
    ("Write a list comprehension that squares numbers.", ["for", "in"]),
    ("What is a dict in Python?", ["{", "name"]),
    ("How do I append to a list in Python?", ["append"]),
]

AGENTIC = [
    ("Plan a short study session step by step.", ["1)", "2)", "study"]),
    ("Break down this long task: learn Python basics.", ["step", "python"]),
    ("Compare France and Japan capitals.", ["paris", "tokyo"]),
    ("Break down this long task: debug a small Python script.", ["error", "bug", "step"]),
    ("Plan a short coding practice session step by step.", ["1)", "function", "test"]),
]

GENERAL = [
    ("Hello!", ["tinyslm", "hi", "hello"]),
    ("What is 2 + 2?", ["4"]),
    ("What is RAM?", ["memory"]),
    ("What is Python?", ["python"]),
]


def score(
    chat: TinyChat,
    items,
    *,
    agent: bool = False,
    grounded: bool = True,
) -> tuple[int, int, list]:
    ok = 0
    details = []
    for q, needles in items:
        chat.reset()
        chat.clear_memory()
        ans, _ = chat.generate_reply(
            q,
            temperature=0.2,
            top_k=20,
            max_new_tokens=100,
            force_agent=agent,
            use_sara=grounded,
            use_grounded=grounded,
        )
        b = body(ans)
        hit = any(n.lower() in b.lower() for n in needles)
        ok += int(hit)
        details.append((hit, q, b[:140]))
    return ok, len(items), details


def main() -> None:
    chat = TinyChat(auto_search=False)
    print("--- full stack (grounded+SARA) ---")
    c_ok, c_n, c_d = score(chat, CODING)
    a_ok, a_n, a_d = score(chat, AGENTIC, agent=True)
    g_ok, g_n, g_d = score(chat, GENERAL)
    print(
        f"coding={c_ok}/{c_n} agentic={a_ok}/{a_n} general={g_ok}/{g_n} "
        f"step={chat.step} params={chat.model.n_parameters():,}"
    )
    print("--- neural-only (no FAQ/plan/code/math/memory extract) ---")
    nc_ok, nc_n, nc_d = score(chat, CODING, grounded=False)
    na_ok, na_n, na_d = score(chat, AGENTIC, agent=False, grounded=False)
    ng_ok, ng_n, ng_d = score(chat, GENERAL, grounded=False)
    print(f"neural_coding={nc_ok}/{nc_n} neural_agentic={na_ok}/{na_n} neural_general={ng_ok}/{ng_n}")
    for good, q, a in c_d + a_d + g_d:
        print(f"  [{'PASS' if good else 'FAIL'}] {q} -> {a}".encode("ascii", "replace").decode())
    print("neural samples:")
    for good, q, a in nc_d + na_d + ng_d:
        print(f"  [{'PASS' if good else 'FAIL'}] {q} -> {a}".encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
