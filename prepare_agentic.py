"""Agentic + long-memory curriculum (aligned with inference prompt style)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

TEXT = r"""
<bos><user>What is water made of?<eos>
<assistant>Water is H2O - two hydrogen atoms and one oxygen atom.<eos>
<bos><user>What is 2 + 2?<eos>
<assistant>2 + 2 equals 4.<eos>
<bos><user>What is the capital of France?<eos>
<assistant>The capital of France is Paris.<eos>
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
<assistant>France capital is Paris. Japan capital is Tokyo.<eos>
<bos><user>Notes:
Plan: memory - reason
[tool:memory] launch code is ORBIT-77 and deadline is Friday
Answer the user clearly in short steps if needed.
User ask: Using memory, what is the launch code?<eos>
<assistant>The launch code is ORBIT-77.<eos>
<bos><user>Notes:
Plan: search - reason
[tool:search] sleep tips: regular bedtime, less screens, cool quiet room
Answer the user clearly in short steps if needed.
User ask: Research healthy sleep tips and summarize.<eos>
<assistant>Keep a regular bedtime, reduce screens at night, and keep the room cool and quiet.<eos>
<bos><user>Notes:
Plan: memory - reason
Answer the user clearly in short steps if needed.
User ask: I have a multi-step goal: finish homework then rest.<eos>
<assistant>First finish the hardest homework problem, then take a short rest with water and stretch.<eos>
<bos><user>[memory]
project deadline is Friday
Question: What is the project deadline from memory?<eos>
<assistant>The project deadline is Friday.<eos>
""".strip()


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "agentic.txt"
    path.write_text(TEXT + "\n", encoding="utf-8")
    # Keep fixes file in sync style
    (DATA / "agent_fixes.txt").write_text(TEXT + "\n", encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
