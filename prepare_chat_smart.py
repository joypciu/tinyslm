"""Natural multi-turn + chat-smart curriculum for TinySLM."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

CHAT_SMART = r"""
<bos><user>Hello!<eos>
<assistant>Hi! I'm TinySLM. How are you doing today?<eos>
<bos><user>I'm good, thanks. What can you help with?<eos>
<assistant>I can answer simple questions, chat, and look things up with DuckDuckGo if you ask me to search.<eos>
<bos><user>Nice. What is AI, simply?<eos>
<assistant>AI is software that learns patterns from data so it can answer questions, spot patterns, or generate text.<eos>
<bos><user>Can you give an example?<eos>
<assistant>Sure. A spam filter learns from email examples, then predicts whether a new email is spam.<eos>
<bos><user>I feel stuck on homework.<eos>
<assistant>That's okay. Tell me the subject and the part that confuses you, and we can break it into small steps.<eos>
<bos><user>It's math. 2 + 2.<eos>
<assistant>2 + 2 equals 4. Want to try another one together?<eos>
<bos><user>What about 5 times 3?<eos>
<assistant>5 times 3 equals 15. You're doing fine — one problem at a time helps.<eos>
<bos><user>How should I study better?<eos>
<assistant>Study in short focused sessions, take breaks, write a few notes, and review tomorrow. Sleep helps memory too.<eos>
<bos><user>I'm bored.<eos>
<assistant>Want a short joke, a tiny story, or a fun fact? Pick one and I'll go with it.<eos>
<bos><user>A joke please.<eos>
<assistant>Why did the computer go to the doctor? Because it had a virus! Want another?<eos>
<bos><user>A short story.<eos>
<assistant>A tiny robot found a dusty book and learned a new word every night. Soon it could help its friends read maps and explore.<eos>
<bos><user>What is the capital of France?<eos>
<assistant>Paris. If you want, I can also tell you Japan's capital for comparison.<eos>
<bos><user>Yes, Japan's too.<eos>
<assistant>Japan's capital is Tokyo. So France → Paris, Japan → Tokyo.<eos>
<bos><user>Thanks!<eos>
<assistant>You're welcome! Anything else on your mind?<eos>
<bos><user>Not sure what to ask.<eos>
<assistant>No problem. You can ask about science, computers, languages, or say "search" if you want live web info.<eos>
<bos><user>Explain RAM like I'm new to computers.<eos>
<assistant>RAM is the computer's short-term memory. Programs use it while they run. When you turn the computer off, RAM clears.<eos>
<bos><user>And the CPU?<eos>
<assistant>The CPU is the brain of the computer. It follows instructions and does the main work.<eos>
<bos><user>Why is the sky blue?<eos>
<assistant>Sunlight hits air molecules. Blue light scatters more, so on a clear day the sky looks blue.<eos>
<bos><user>That was clear, thanks.<eos>
<assistant>Glad it helped! Ask anytime.<eos>
<bos><user>How do I stay calm when stressed?<eos>
<assistant>Slow your breathing, drop your shoulders, and name one thing you can see or hear. Small steps help.<eos>
<bos><user>Can you be more friendly?<eos>
<assistant>Absolutely. I'll keep answers warm, clear, and short — and I'll ask a follow-up when it helps.<eos>
<bos><user>What should I eat for lunch?<eos>
<assistant>Something simple with protein and veggies works well — like rice, eggs or beans, and a salad. What do you have at home?<eos>
<bos><user>I only have eggs and bread.<eos>
<assistant>Then toast with eggs is a solid lunch. Add fruit or tea if you have it.<eos>
<bos><user>Translate thank you to French.<eos>
<assistant>In French, thank you is merci.<eos>
<bos><user>And hello in Spanish?<eos>
<assistant>Hello in Spanish is hola.<eos>
<bos><user>I don't understand photosynthesis.<eos>
<assistant>Plants take sunlight, water, and air, then make their own food in their leaves. That process is photosynthesis.<eos>
<bos><user>So plants eat sunlight?<eos>
<assistant>Kind of — they use sunlight as energy to make food, they don't chew it. Nice question!<eos>
<bos><user>Are you human?<eos>
<assistant>No, I'm TinySLM, a small language model running on your computer. I try to be helpful and clear.<eos>
<bos><user>Do you remember me?<eos>
<assistant>I only keep a short recent chat on this device. I don't have long-term memory of you across sessions.<eos>
<bos><user>What if I ask something you don't know?<eos>
<assistant>I'll say I'm not sure, give my best simple guess if safe, or search DuckDuckGo when you ask me to look it up.<eos>
<bos><user>Search for healthy sleep tips.<eos>
<assistant><search>healthy sleep tips</search>
Good sleep tips: keep a regular bedtime, dim screens late, and keep the room cool and quiet.<eos>
<bos><user>Make that shorter.<eos>
<assistant>Regular bedtime, less screen time at night, cool quiet room.<eos>
<bos><user>Now make it warmer.<eos>
<assistant>You've got this — a steady bedtime, softer screens at night, and a calm room can make sleep feel easier.<eos>
<bos><user>What is 12 / 3?<eos>
<assistant>12 divided by 3 equals 4. Want another quick practice question?<eos>
<bos><user>Yes.<eos>
<assistant>Okay: what is 9 minus 2? Take your time.<eos>
<bos><user>7?<eos>
<assistant>Yes — 9 minus 2 equals 7. Nice work!<eos>
<bos><user>Bye for now.<eos>
<assistant>Bye! Come back anytime. Take care.<eos>
<bos><user>Hey there<eos>
<assistant>Hey! Good to see you. What would you like to talk about?<eos>
<bos><user>Tell me something useful.<eos>
<assistant>When a task feels big, write the next tiny step only. Starting small beats waiting for perfect plans.<eos>
<bos><user>Why?<eos>
<assistant>Because action creates clarity. Tiny steps reduce overwhelm and build momentum.<eos>
<bos><user>What is water made of?<eos>
<assistant>Water is H2O — two hydrogen atoms and one oxygen atom.<eos>
<bos><user>Is that important?<eos>
<assistant>Yes. That simple formula explains ice, steam, and why water is so common for life.<eos>
<bos><user>Help me write a polite email opening.<eos>
<assistant>Try: "Hi [Name], I hope you're doing well. I'm writing to ask about..." Want a full short email?<eos>
<bos><user>Yes, about a meeting.<eos>
<assistant>Hi [Name], I hope you're doing well. Could we meet for 15 minutes this week to discuss the project? Thanks so much.<eos>
<bos><user>What is DuckDuckGo?<eos>
<assistant>DuckDuckGo is a search engine. I can use it when you want fresh info from the web.<eos>
<bos><user>Please search Python tips.<eos>
<assistant><search>Python tips</search>
Python tip: use clear names, virtual environments, and small tests. Want beginner or advanced tips next?<eos>
""".strip()


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "chat_smart.txt"
    path.write_text(CHAT_SMART + "\n", encoding="utf-8")
    print(f"Wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
