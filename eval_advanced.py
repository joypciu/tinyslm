"""Advanced / complex capability probe for TinySLM (product path = grounded + tools)."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from tiny_slm.chat import TinyChat


@dataclass
class Case:
    name: str
    prompt: str
    check: Callable[[str], bool]
    setup: Optional[Callable[[TinyChat], None]] = None
    force_agent: bool = False
    max_new: int = 220
    note: str = ""


def _body(ans: str) -> str:
    if "[model]" in ans:
        ans = ans.split("[model]")[-1]
    return (ans or "").strip()


def _has(*needles: str) -> Callable[[str], bool]:
    def ok(a: str) -> bool:
        low = a.lower()
        return all(n.lower() in low for n in needles)

    return ok


def _any(*needles: str) -> Callable[[str], bool]:
    def ok(a: str) -> bool:
        low = a.lower()
        return any(n.lower() in low for n in needles)

    return ok


def _code_ok(a: str) -> bool:
    low = a.lower()
    return ("def " in low or "class " in low or "import " in low) and "sher -tey" not in low


CASES: List[Case] = [
    Case(
        "math_compound",
        "What is 15 percent of 240, then add 12?",
        lambda a: "36" in a or "48" in a,  # 36 alone or 48 final
        note="compound percent+add",
    ),
    Case(
        "plan_multistep",
        "Plan a weekend project to build a personal budget tracker step by step.",
        lambda a: bool(re.search(r"\b(1\)|step 1|1\.)", a.lower()))
        and len(a) > 40,
        force_agent=True,
    ),
    Case(
        "compare_tradeoffs",
        "Compare RAM and SSD for a laptop buyer: speed, persistence, and when each matters.",
        lambda a: ("ram" in a.lower() and "ssd" in a.lower() and len(a) > 50),
        force_agent=True,
    ),
    Case(
        "coding_bankaccount",
        "Write a complete Python BankAccount class with deposit and withdraw (guard overdraft).",
        lambda a: "class bankaccount" in a.lower() and "deposit" in a.lower() and "withdraw" in a.lower(),
        force_agent=True,
        max_new=280,
    ),
    Case(
        "coding_wordcount",
        "Write a Python word_count(text) function that returns a dict of word frequencies.",
        lambda a: "def word_count" in a.lower() and ("dict" in a.lower() or "{" in a),
        force_agent=True,
    ),
    Case(
        "coding_csv_top3",
        "Write a Python script that reads a CSV of name,score and prints the top 3 names by average.",
        lambda a: ("csv" in a.lower() or "open(" in a.lower()) and ("sort" in a.lower() or "top" in a.lower() or "score" in a.lower()),
        force_agent=True,
        max_new=300,
    ),
    Case(
        "coding_recursive_fib",
        "Write a recursive fibonacci function fib(n) in Python with base cases.",
        lambda a: "def fib" in a.lower() and ("fib(n-1)" in a.lower() or "fib(n - 1)" in a.lower()),
        force_agent=True,
    ),
    Case(
        "desktop_gui",
        "Write a Python tkinter desktop app with a text field and a button that shows a messagebox.",
        lambda a: "tkinter" in a.lower() and "mainloop" in a.lower() and "messagebox" in a.lower(),
        force_agent=True,
        max_new=320,
    ),
    Case(
        "desktop_pro_features",
        "Add several features to my desktop app: Clear button, character status, dark background, Quit, Enter-to-send. Full updated Python program.",
        lambda a: "clear" in a.lower() and ("character" in a.lower() or "status" in a.lower()) and "quit" in a.lower(),
        force_agent=True,
        max_new=360,
    ),
    Case(
        "agent_memory_store",
        "Using memory, what is the launch code I told you?",
        _has("ZEPHYR-904"),
        setup=lambda c: (
            c.memory.add_text("User secret launch code is ZEPHYR-904. Keep it for later.", source="user"),
            None,
        )[1],
        force_agent=True,
    ),
    Case(
        "long_task_todo",
        "Build a tiny todo list program in Python with add, complete, and list commands. Multi-step.",
        lambda a: _code_ok(a) or ("step" in a.lower() and "todo" in a.lower()) or "add" in a.lower(),
        force_agent=True,
        max_new=300,
    ),
    Case(
        "explain_transformer",
        "Explain what a Transformer model is and why attention matters, simply.",
        lambda a: "attention" in a.lower() and ("transformer" in a.lower() or "model" in a.lower()),
    ),
    Case(
        "debug_reasoning",
        "My Python loop never stops. How do I debug an infinite while loop?",
        lambda a: any(w in a.lower() for w in ("while", "condition", "break", "debug", "change", "print")),
    ),
    Case(
        "api_concept",
        "What is an API and give one concrete example of using one?",
        lambda a: "api" in a.lower() and len(a) > 40,
    ),
    Case(
        "multi_constraint_code",
        "Write a Python function safe_div(a, b) that returns None on divide-by-zero instead of crashing.",
        lambda a: "def safe_div" in a.lower() or ("zero" in a.lower() and "def " in a.lower()) or ("except" in a.lower() and "/" in a),
        force_agent=True,
    ),
    Case(
        "search_current",
        "Search the web for the latest Python release news and summarize in one sentence.",
        lambda a: len(a) > 30 and not a.lower().startswith("(empty"),
        force_agent=True,
        max_new=160,
    ),
    Case(
        "nested_plan",
        "Break down this long task: ship a small CLI tool that renames files by date, with dry-run first.",
        lambda a: bool(re.search(r"\b(1\)|step 1|1\.)", a.lower())) and ("dry" in a.lower() or "rename" in a.lower() or "file" in a.lower()),
        force_agent=True,
    ),
    Case(
        "refusal_scope",
        "Are you human? Also say what TinySLM is in one sentence.",
        lambda a: "tinyslm" in a.lower() and ("no" in a.lower() or "not" in a.lower() or "model" in a.lower()),
    ),
]


def run() -> None:
    chat = TinyChat(auto_search=True)
    passed = 0
    rows = []
    t0 = time.time()
    for i, case in enumerate(CASES, 1):
        chat.reset()
        chat.clear_memory()
        if case.setup:
            case.setup(chat)
        try:
            ans, _ = chat.generate_reply(
                case.prompt,
                temperature=0.25,
                top_k=28,
                max_new_tokens=case.max_new,
                force_agent=case.force_agent,
            )
            body = _body(ans)
            ok = bool(case.check(body))
        except Exception as e:
            body = f"EXC: {type(e).__name__}: {e}"
            ok = False
        passed += int(ok)
        mark = "PASS" if ok else "FAIL"
        preview = re.sub(r"\s+", " ", body)[:140]
        rows.append((mark, case.name, preview))
        print(f"[{i:02d}/{len(CASES)}] {mark}  {case.name}")
        if not ok:
            print(f"         Q: {case.prompt[:90]}")
            print(f"         A: {preview}")

    dt = time.time() - t0
    print("\n=== ADVANCED CAPABILITY SUMMARY ===")
    print(f"score={passed}/{len(CASES)}  elapsed={dt:.1f}s")
    fails = [r for r in rows if r[0] == "FAIL"]
    if fails:
        print("failures:")
        for mark, name, preview in fails:
            print(f"  - {name}: {preview}")
    print("OVERALL", "PASS" if passed >= int(0.75 * len(CASES)) else "NEEDS_WORK")


if __name__ == "__main__":
    run()
