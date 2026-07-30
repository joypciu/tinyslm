"""Coding + agentic curriculum for safe resume fine-tunes (with chat rehearsal)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# Mix: coding snippets, agentic plans, memory/tool style, plus chat rehearsal
TEXT = r"""
<bos><user>Hello!<eos>
<assistant>Hi! I'm TinySLM. How are you doing today?<eos>
<bos><user>What is 2 + 2?<eos>
<assistant>2 + 2 equals 4.<eos>
<bos><user>What is the capital of France?<eos>
<assistant>The capital of France is Paris.<eos>
<bos><user>What is RAM?<eos>
<assistant>RAM is short-term computer memory the CPU uses to hold running programs and data.<eos>
<bos><user>What is Python?<eos>
<assistant>Python is a popular programming language used for websites, data work, automation, and learning to code.<eos>
<bos><user>Write a Python function that adds two numbers.<eos>
<assistant>def add(a, b):
    return a + b

# example: add(2, 3) -> 5<eos>
<bos><user>How do I reverse a string in Python?<eos>
<assistant>s = 'hello'
reversed_s = s[::-1]  # 'olleh'
# or: ''.join(reversed(s))<eos>
<bos><user>Explain what a for loop is in programming.<eos>
<assistant>A for loop repeats work for each item:
for x in [1, 2, 3]:
    print(x)<eos>
<bos><user>What does if/else do in Python?<eos>
<assistant>if/else picks a branch:
if n > 0:
    print('positive')
else:
    print('not positive')<eos>
<bos><user>What is a variable in programming?<eos>
<assistant>A variable is a named place that stores a value, e.g. score = 10 in Python.<eos>
<bos><user>How do I read a file in Python?<eos>
<assistant>with open('notes.txt', encoding='utf-8') as f:
    text = f.read()<eos>
<bos><user>How do I write a file in Python?<eos>
<assistant>with open('out.txt', 'w', encoding='utf-8') as f:
    f.write('hello')<eos>
<bos><user>Show a try/except example in Python.<eos>
<assistant>try:
    n = int(text)
except ValueError:
    print('not a number')<eos>
<bos><user>What is a class in Python?<eos>
<assistant>class Dog:
    def __init__(self, name):
        self.name = name

Dog('Rex').name<eos>
<bos><user>Use map in Python.<eos>
<assistant>nums = [1, 2, 3]
list(map(lambda x: x * 2, nums))  # [2, 4, 6]<eos>
<bos><user>Sort a list in Python.<eos>
<assistant>nums = [3, 1, 2]
nums.sort()
sorted(nums)<eos>
<bos><user>Notes:
Plan: memory - reason
Answer the user clearly in short steps if needed.
User ask: Plan a short study session step by step.<eos>
<assistant>1) Pick one topic. 2) Study for 20 focused minutes. 3) Write 3 notes. 4) Take a short break.<eos>
<bos><user>Notes:
Plan: memory - reason
Answer the user clearly in short steps if needed.
User ask: Break down this long task: learn Python basics.<eos>
<assistant>Step 1: install Python. Step 2: learn variables and print. Step 3: practice if/else. Step 4: write a tiny script.<eos>
<bos><user>Notes:
Plan: compare - reason
Focus sides: France and Japan capitals
Answer the user clearly in short steps if needed.
User ask: Compare France and Japan capitals.<eos>
<assistant>France's capital is Paris. Japan's capital is Tokyo.<eos>
<bos><user>Notes:
Plan: memory - reason
[tool:memory] launch code is ORBIT-77 and deadline is Friday
Answer the user clearly in short steps if needed.
User ask: Using memory, what is the launch code?<eos>
<assistant>The launch code is ORBIT-77.<eos>
<bos><user>Notes:
Plan: search - reason
[tool:search] Ada Lovelace was an English mathematician and writer.
[tool:extract] From the web: Ada Lovelace was an English mathematician.
Answer the user clearly in short steps if needed.
User ask: Who is Ada Lovelace?<eos>
<assistant>From the web: Ada Lovelace was an English mathematician and writer.<eos>
<bos><user>Notes:
Plan: memory - reason
Answer the user clearly in short steps if needed.
User ask: Break down this long task: debug a small Python script.<eos>
<assistant>Step 1: read the error message. Step 2: print key variables. Step 3: fix one bug. Step 4: rerun the script.<eos>
<bos><user>Notes:
Plan: memory - reason
Answer the user clearly in short steps if needed.
User ask: Plan a short coding practice session step by step.<eos>
<assistant>1) Pick one small function. 2) Write tests for two cases. 3) Implement it. 4) Refactor names.<eos>
<bos><user>What is an API?<eos>
<assistant>An API is a defined way for programs to talk to each other - request in, data or action out.<eos>
<bos><user>What is Git?<eos>
<assistant>Git is a version-control tool that tracks changes in code so you can branch, merge, and roll back.<eos>
<bos><user>What is a neural network?<eos>
<assistant>A neural network is a layered model of weighted connections that learns patterns from data.<eos>
<bos><user>I'm bored.<eos>
<assistant>Want a short joke, a tiny story, or a fun fact? Pick one and I'll go with it.<eos>
<bos><user>Give me a short sleep tip.<eos>
<assistant>Keep a regular bedtime, dim screens late, and keep the room cool and quiet.<eos>
<bos><user>Explain gravity simply.<eos>
<assistant>Gravity is the force that pulls masses together - it keeps us on Earth and planets in orbit around the Sun.<eos>
""".strip()

# Rehearsal: greetings + Python hard; keep math light (oversampling math collapsed "What is Python?").
_GREET = """
<bos><user>Hello!<eos>
<assistant>Hi! I'm TinySLM. How are you doing today?<eos>
<bos><user>Hi!<eos>
<assistant>Hi! I'm TinySLM. How are you doing today?<eos>
<bos><user>Hello<eos>
<assistant>Hi! I'm TinySLM. How are you doing today?<eos>
""".strip()

_PYTHON = """
<bos><user>What is Python?<eos>
<assistant>Python is a popular programming language used for websites, data work, automation, and learning to code.<eos>
<bos><user>What is Python used for?<eos>
<assistant>Python is a programming language used for websites, data work, automation, and learning to code.<eos>
<bos><user>Tell me about Python.<eos>
<assistant>Python is a popular programming language used for websites, data work, automation, and learning to code.<eos>
""".strip()

_MATH_LIGHT = """
<bos><user>What is 2 + 2?<eos>
<assistant>2 + 2 equals 4.<eos>
""".strip()

_CODE_LIGHT = """
<bos><user>Write a Python function that adds two numbers.<eos>
<assistant>def add(a, b):
    return a + b

# example: add(2, 3) -> 5<eos>
<bos><user>What is a variable in programming?<eos>
<assistant>A variable is a named place that stores a value, e.g. score = 10 in Python.<eos>
<bos><user>How do I read a file in Python?<eos>
<assistant>with open('notes.txt', encoding='utf-8') as f:
    text = f.read()<eos>
<bos><user>What is RAM?<eos>
<assistant>RAM is short-term computer memory the CPU uses to hold running programs and data.<eos>
""".strip()


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "coding_agentic.txt"
    body = (
        TEXT
        + "\n"
        + ("\n" + _GREET) * 12
        + ("\n" + _PYTHON) * 14
        + ("\n" + _MATH_LIGHT) * 2
        + ("\n" + _CODE_LIGHT) * 4
        + "\n"
    )
    path.write_text(body, encoding="utf-8")
    print(f"Wrote {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
