"""SARA skill / reflection fine-tune examples."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

TEXT = r"""
<bos><user>What is 2 + 2?<eos>
<assistant>2 + 2 equals 4.<eos>
<bos><user>Notes:
SKILL math_simple: Compute carefully. Reply like 'A + B equals C'.
Answer the user clearly in short steps if needed.
User ask: What is 2 + 2?<eos>
<assistant>2 + 2 equals 4.<eos>
<bos><user>Notes:
SKILL plan_steps: Answer with numbered steps (1-4). Keep each step one short sentence.
Plan: memory - reason
Reflection: Issues: missing numbered steps
Improve the draft. Draft was: Study hard every day.
User ask: Plan a short study session step by step.<eos>
<assistant>1) Pick one topic. 2) Study for 20 focused minutes. 3) Write 3 notes. 4) Take a short break.<eos>
<bos><user>Notes:
SKILL compare_two: State A clearly, state B clearly, then one contrast line.
Plan: compare - reason
Answer the user clearly in short steps if needed.
User ask: Compare France and Japan capitals.<eos>
<assistant>France capital is Paris. Japan capital is Tokyo. They are different countries.<eos>
<bos><user>Notes:
SKILL memory_answer: Prefer facts from [memory]/tool:memory]. Quote the key fact in one sentence.
[memory]
launch code is ORBIT-77
Answer the user clearly in short steps if needed.
User ask: Using memory, what is the launch code?<eos>
<assistant>The launch code is ORBIT-77.<eos>
""".strip()


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "sara_skills.txt"
    path.write_text(TEXT + "\n", encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
