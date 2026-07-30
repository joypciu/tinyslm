"""Compact intelligence boost curriculum (chat rehearsal + coding concepts)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "intelligence_boost.txt"

TEXT = r"""
<bos><user>Hello!<eos>
<assistant>Hi! I'm TinySLM. How are you doing today?<eos>
<bos><user>What is 2 + 2?<eos>
<assistant>2 + 2 equals 4.<eos>
<bos><user>What is a CPU?<eos>
<assistant>A CPU (central processing unit) is the main chip that runs instructions in a computer.<eos>
<bos><user>What is RAM?<eos>
<assistant>RAM is short-term computer memory the CPU uses to hold running programs and data.<eos>
<bos><user>What is Python?<eos>
<assistant>Python is a popular programming language used for websites, data work, automation, and learning to code.<eos>
<bos><user>Explain what a for loop is in programming.<eos>
<assistant>A for loop repeats work for each item:
for x in [1, 2, 3]:
    print(x)<eos>
<bos><user>What is a for loop?<eos>
<assistant>A for loop repeats work for each item:
for x in [1, 2, 3]:
    print(x)<eos>
<bos><user>What is a variable in programming?<eos>
<assistant>A variable is a named place that stores a value, e.g. score = 10 in Python.<eos>
<bos><user>Sort a list in Python.<eos>
<assistant>nums = [3, 1, 2]
nums.sort()
sorted(nums)<eos>
<bos><user>What is machine learning?<eos>
<assistant>Machine learning lets programs improve from examples instead of only hard-coded rules.<eos>
<bos><user>What is AI?<eos>
<assistant>AI is software that learns patterns from data to help with language, vision, planning, and similar tasks.<eos>
<bos><user>How do I write a function in Python?<eos>
<assistant>def greet(name):
    return f'Hello, {name}'

greet('Ada')<eos>
<bos><user>What does return do in a function?<eos>
<assistant>return sends a value back to the caller and ends the function early if reached.<eos>
<bos><user>What is a boolean?<eos>
<assistant>A boolean is True or False - used for yes/no decisions in if statements and loops.<eos>
<bos><user>What is an API?<eos>
<assistant>An API is a defined way for programs to talk to each other, like requesting data from a web service.<eos>
<bos><user>What is debugging?<eos>
<assistant>Debugging means finding and fixing mistakes: read the error, reproduce it, change one thing, and retest.<eos>
<bos><user>Explain recursion simply.<eos>
<assistant>Recursion is a function calling itself with a smaller problem until a base case stops it.<eos>
<bos><user>What is Big O notation roughly?<eos>
<assistant>Big O describes how runtime or memory grows as input size grows - smaller growth is usually better.<eos>
<bos><user>Plan a short study session step by step.<eos>
<assistant>1) Pick one topic. 2) Study for 20 focused minutes. 3) Write 3 notes. 4) Take a short break.<eos>
<bos><user>Compare France and Japan capitals.<eos>
<assistant>France's capital is Paris. Japan's capital is Tokyo.<eos>
<bos><user>Break down this long task: learn Python basics.<eos>
<assistant>Step 1: install Python. Step 2: learn variables and print. Step 3: practice if/else. Step 4: write a tiny script.<eos>
<bos><user>What is the capital of France?<eos>
<assistant>The capital of France is Paris.<eos>
<bos><user>Are you human?<eos>
<assistant>No, I'm TinySLM, a small language model running on your computer.<eos>
<bos><user>Write a Python function that adds two numbers.<eos>
<assistant>def add(a, b):
    return a + b

# example: add(2, 3) -> 5<eos>
<bos><user>How do I reverse a string in Python?<eos>
<assistant>s = 'hello'
reversed_s = s[::-1]  # 'olleh'
# or: ''.join(reversed(s))<eos>
<bos><user>What does if/else do in Python?<eos>
<assistant>if/else picks a branch:
if n > 0:
    print('positive')
else:
    print('not positive')<eos>
<bos><user>Notes:
Plan: memory - reason
Answer the user clearly in short steps if needed.
User ask: Explain what a for loop is in programming.<eos>
<assistant>A for loop repeats work for each item:
for x in [1, 2, 3]:
    print(x)<eos>
""".strip()

# Upsample hard misses so LoRA sees them often
HARD = """
<bos><user>Explain what a for loop is in programming.<eos>
<assistant>A for loop repeats work for each item:
for x in [1, 2, 3]:
    print(x)<eos>
<bos><user>What is a CPU?<eos>
<assistant>A CPU (central processing unit) is the main chip that runs instructions in a computer.<eos>
"""

def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    body = TEXT + "\n\n" + "\n\n".join([HARD.strip()] * 12) + "\n"
    OUT.write_text(body, encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
