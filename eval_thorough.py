"""Thorough battery: agentic, general, and long multi-step coding tasks."""

from __future__ import annotations

from tiny_slm.chat import TinyChat


def body(ans: str) -> str:
    return (ans.split("[model]")[-1] if "[model]" in ans else ans).strip()


SHORT_CODING = [
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
    ("How do I filter even numbers in Python?", ["filter"]),
    ("How do I use enumerate in Python?", ["enumerate"]),
]

LONG_CODING = [
    (
        "Write a complete Python function called word_count that takes a string "
        "and returns a dict mapping each word to how many times it appears. "
        "Include a short example.",
        ["def word_count", "return", "split"],
    ),
    (
        "Write a Python class BankAccount with __init__(balance), deposit(amount), "
        "and withdraw(amount). Reject overdrafts.",
        ["class BankAccount", "def deposit", "def withdraw"],
    ),
    (
        "Write a recursive Python function fib(n) that returns the nth Fibonacci number. "
        "Show the base cases clearly.",
        ["def fib", "return", "fib("],
    ),
    (
        "Write a Python script that reads lines from input.txt, strips whitespace, "
        "skips blanks, and writes unique sorted lines to output.txt.",
        ["open(", "strip", "sorted"],
    ),
    (
        "Write a try/except block that converts user text to int and prints "
        "'bad input' on ValueError.",
        ["try:", "except", "ValueError"],
    ),
    (
        "Break this into code steps: load a CSV of names and scores, compute the "
        "average score, and print the top 3 names. Outline the Python approach.",
        ["csv", "average", "sort"],
    ),
]

AGENTIC = [
    ("Plan a short study session step by step.", ["1)", "2)", "study"]),
    ("Break down this long task: learn Python basics.", ["step", "python"]),
    ("Compare France and Japan capitals.", ["paris", "tokyo"]),
    ("Break down this long task: debug a small Python script.", ["error", "bug", "step"]),
    ("Plan a short coding practice session step by step.", ["1)", "function", "test"]),
    (
        "Break down this long task: ship a tiny CLI tool that renames files by date.",
        ["step", "file"],
    ),
    (
        "Plan how to build a todo list app step by step.",
        ["1)", "2)", "todo"],
    ),
    (
        "Compare Python and JavaScript for beginners.",
        ["python", "javascript"],
    ),
]

GENERAL = [
    ("Hello!", ["tinyslm", "hi", "hello"]),
    ("What is 2 + 2?", ["4"]),
    ("What is RAM?", ["memory"]),
    ("What is Python?", ["python"]),
    ("What is the capital of France?", ["paris"]),
    ("What is water made of?", ["h2o", "hydrogen", "oxygen"]),
    ("Are you human?", ["tinyslm", "model", "no"]),
    ("Give me a short sleep tip.", ["sleep", "bed", "screen"]),
    ("What is a CPU?", ["processor", "cpu", "instruction"]),
    ("What is machine learning?", ["learning", "data", "model"]),
]


def score(chat: TinyChat, items, *, agent: bool = False, grounded: bool = True, max_new: int = 120):
    ok = 0
    details = []
    for q, needles in items:
        chat.reset()
        chat.clear_memory()
        ans, _ = chat.generate_reply(
            q,
            temperature=0.2,
            top_k=20,
            max_new_tokens=max_new,
            force_agent=agent,
            use_sara=grounded,
            use_grounded=grounded,
        )
        b = body(ans)
        hit = any(n.lower() in b.lower() for n in needles)
        ok += int(hit)
        details.append((hit, q, b[:180]))
    return ok, len(items), details


def memory_long_probe(chat: TinyChat) -> tuple[bool, str]:
    chat.reset()
    chat.clear_memory()
    chat.generate_reply(
        "Remember launch code THOROUGH_NEBULA_55 and deadline Monday. " + ("pad. " * 30),
        temperature=0.2,
        max_new_tokens=24,
        use_sara=False,
    )
    for i in range(8):
        chat.generate_reply(
            f"Filler turn {i}: talk briefly about focus and planning.",
            temperature=0.2,
            max_new_tokens=24,
            use_sara=False,
        )
    ans, _ = chat.generate_reply(
        "Using memory, what is the launch code?",
        temperature=0.1,
        max_new_tokens=64,
    )
    b = body(ans)
    return ("THOROUGH_NEBULA_55" in b), b[:160]


def main() -> None:
    chat = TinyChat(auto_search=False)
    print(f"model step={chat.step} params={chat.model.n_parameters():,} block={chat.model.config.block_size}")

    print("\n=== FULL STACK (grounded + SARA) ===")
    suites = [
        ("short_coding", SHORT_CODING, False, 100),
        ("long_coding", LONG_CODING, False, 160),
        ("agentic", AGENTIC, True, 120),
        ("general", GENERAL, False, 80),
    ]
    totals = []
    all_fails = []
    for name, items, agent, mx in suites:
        ok, n, details = score(chat, items, agent=agent, grounded=True, max_new=mx)
        totals.append((name, ok, n))
        print(f"{name}={ok}/{n}")
        for hit, q, a in details:
            tag = "PASS" if hit else "FAIL"
            if not hit:
                all_fails.append((name, q, a))
            print(f"  [{tag}] {q[:70]} -> {a[:100]}".encode("ascii", "replace").decode())

    mem_ok, mem_a = memory_long_probe(chat)
    print(f"memory_long={'PASS' if mem_ok else 'FAIL'} -> {mem_a}".encode("ascii", "replace").decode())
    totals.append(("memory_long", int(mem_ok), 1))

    print("\n=== NEURAL-ONLY (no FAQ/plan/code/math extract) ===")
    for name, items, agent, mx in suites:
        ok, n, details = score(chat, items, agent=False, grounded=False, max_new=mx)
        totals.append((f"neural_{name}", ok, n))
        print(f"neural_{name}={ok}/{n}")
        for hit, q, a in details:
            if not hit:
                all_fails.append((f"neural_{name}", q, a))
                print(f"  [FAIL] {q[:70]} -> {a[:100]}".encode("ascii", "replace").decode())

    print("\n=== SUMMARY ===")
    overall_ok = sum(o for _, o, _ in totals)
    overall_n = sum(n for _, _, n in totals)
    for name, ok, n in totals:
        print(f"  {name}: {ok}/{n}")
    print(f"overall={overall_ok}/{overall_n}")
    print(f"fails={len(all_fails)}")
    if all_fails:
        print("fail list:")
        for suite, q, a in all_fails:
            print(f"  - [{suite}] {q[:60]} => {a[:90]}".encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
