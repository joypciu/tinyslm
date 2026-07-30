"""Tiny grounded FAQ / definition cards for TinySLM.

Used as an inference-time fast-path (like symbolic math) so common short
facts do not depend on the 4M-param generator. No weight updates — avoids
catastrophic forgetting from narrow fine-tunes.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# (match_fn patterns as lowercase substrings OR regex, answer)
# Prefer specific multi-word cues first.
_FAQ: List[Tuple[List[str], str]] = [
    (
        ["what is ram", "what's ram", "whats ram", "define ram", "ram?"],
        "RAM is short-term computer memory the CPU uses to hold running programs and data.",
    ),
    (
        ["what is cpu", "what's cpu", "whats cpu", "define cpu"],
        "A CPU (central processing unit) is the main chip that runs instructions in a computer.",
    ),
    (
        ["what is a gpu", "what is gpu", "what's gpu"],
        "A GPU accelerates graphics and many parallel math workloads, including some AI tasks.",
    ),
    (
        ["what is water made of", "water made of", "composition of water"],
        "Water is H2O - two hydrogen atoms and one oxygen atom.",
    ),
    (
        ["sleep tip", "short sleep tip", "tip for sleep", "better sleep"],
        "Keep a regular bedtime, dim screens late, and keep the room cool and quiet.",
    ),
    (
        ["i'm bored", "im bored", "i am bored"],
        "Want a short joke, a tiny story, or a fun fact? Pick one and I'll go with it.",
    ),
    # Multi-entity compares must be matched before single-country cards
    (
        [
            "compare france and japan",
            "france and japan capital",
            "japan and france capital",
        ],
        "France's capital is Paris. Japan's capital is Tokyo.",
    ),
    (
        ["capital of france", "france capital", "france's capital"],
        "The capital of France is Paris.",
    ),
    (
        ["capital of japan", "japan capital", "japan's capital"],
        "The capital of Japan is Tokyo.",
    ),
    (
        ["are you human", "are you a human", "are you a person"],
        "No, I'm TinySLM, a small language model running on your computer.",
    ),
    (
        ["who are you", "what are you"],
        "I'm TinySLM, a tiny from-scratch chat model with long memory and light tools.",
    ),
    (
        ["hello!", "hello", "hi!", "hi", "hey!", "hey", "good morning", "good evening"],
        "Hi! I'm TinySLM. How are you doing today?",
    ),
    (
        ["what is ssd", "what's ssd", "whats ssd", "define ssd"],
        "An SSD is solid-state storage - fast disk memory that keeps files when the computer is off.",
    ),
    (
        ["what is the internet", "what's the internet", "what is internet"],
        "The internet is a global network that lets computers share websites, messages, and data.",
    ),
    (
        ["polite email", "write a polite email", "email tip"],
        "Start with a clear greeting, state your request in one short paragraph, then close politely.",
    ),
    (
        ["focus tip", "stay focused", "how to focus"],
        "Pick one task, silence notifications for 20 minutes, then take a short break.",
    ),
    (
        ["thank you", "thanks", "thanks!", "thank you!"],
        "You're welcome - happy to help anytime.",
    ),
    (
        ["how are you", "how are you doing", "how's it going", "hows it going"],
        "I'm doing well - ready to help. What would you like to talk about?",
    ),
    (
        ["what is wifi", "what's wifi", "what is wi fi", "what's wi fi", "what is wi-fi"],
        "Wi-Fi is wireless networking that lets devices join a local network and reach the internet.",
    ),
    (
        ["difference between ram and ssd", "ram vs ssd", "ram versus ssd"],
        "RAM is fast temporary working memory; an SSD is slower long-term storage that keeps files when power is off.",
    ),
    (
        ["what is a browser", "what's a browser", "what is web browser"],
        "A web browser is an app for opening websites - for example Chrome, Firefox, Edge, or Safari.",
    ),
    (
        ["what is photosynthesis", "photosynthesis", "how leaves make food"],
        "Photosynthesis is how plants make food from sunlight, water, and carbon dioxide.",
    ),
    (
        ["capital of germany", "germany capital", "germany's capital"],
        "The capital of Germany is Berlin.",
    ),
    (
        ["capital of italy", "italy capital", "italy's capital"],
        "The capital of Italy is Rome.",
    ),
    (
        ["capital of bangladesh", "bangladesh capital", "bangladesh's capital"],
        "The capital of Bangladesh is Dhaka.",
    ),
    (
        ["tell me a joke", "short joke", "joke please"],
        "Why did the computer go to the doctor? Because it had a virus.",
    ),
    (
        ["what should i do", "what do you suggest", "any suggestion"],
        "Pick one small task you can finish in 10 minutes - starting tiny beats overthinking.",
    ),
    (
        [
            "friendly one-sentence reply",
            "give a friendly",
            "one short tip",
            "short tip for me",
            "remind me to stay brief",
            "keep answers warm and brief",
        ],
        "Stay kind, keep it short, and take the next small step when you're ready.",
    ),
    (
        ["friendship", "be a good friend", "kindness tip"],
        "Listen first, keep promises small and real, and say thank you when someone helps you.",
    ),
    (
        ["what is oxygen", "what's oxygen"],
        "Oxygen is the gas living things need to breathe; it makes up much of the air with nitrogen.",
    ),
    (
        ["what is a laptop", "what's a laptop"],
        "A laptop is a portable computer with a screen, keyboard, battery, CPU, and memory in one device.",
    ),
]


_ECHO_ONLY = re.compile(
    r"^(what is\s+)?([A-Za-z0-9\- ]{1,40})\??$",
    re.I,
)


def answer_from_faq(user: str) -> Optional[str]:
    """Return a short grounded answer when the question matches a FAQ card."""
    u = (user or "").strip().lower()
    if not u:
        return None
    # Normalize light punctuation for matching
    norm = re.sub(r"[^\w\s\?']+", " ", u)
    norm = re.sub(r"\s+", " ", norm).strip()
    # Both countries mentioned → prefer compare card even if cue order slips
    if "france" in norm and "japan" in norm and "capital" in norm:
        return (
            "France's capital is Paris. Japan's capital is Tokyo."
        )
    # Exact short greetings / thanks before looser substring cards
    bare = norm.rstrip("?").strip()
    if bare in ("hello", "hi", "hey", "good morning", "good evening"):
        return "Hi! I'm TinySLM. How are you doing today?"
    if bare in ("thanks", "thank you", "thx"):
        return "You're welcome - happy to help anytime."
    for cues, ans in _FAQ:
        # Skip ultra-short greeting cues for long non-greeting utterances
        for cue in cues:
            if cue in ("hi", "hey", "hello", "hi!", "hey!", "hello!") and len(norm) > 24:
                continue
            if cue in norm or cue.rstrip("?") == bare:
                return ans
            if cue.endswith("?") and bare == cue.rstrip("?"):
                return ans
    if bare in ("ram",):
        return _FAQ[0][1]
    if bare in ("cpu",):
        return _FAQ[1][1]
    return None


def scrub_generation(text: str) -> str:
    """Remove prompt leakage and collapsed echoes from model drafts."""
    t = (text or "").strip()
    if not t:
        return t
    # Cut SARA / prompt echoes
    for marker in (
        "\nUser ask:",
        "\nuser ask:",
        "\nNotes:",
        "\nReflection:",
        "\nDraft was:",
        "\nQuestion:",
        "\nSKILL ",
        "\n[tool:",
        "\n[memory]",
        "\n[agent]",
    ):
        if marker in t:
            t = t.split(marker, 1)[0].strip()
    # Also cut inline leakage without leading newline
    t = re.split(r"\bUser ask\s*:", t, maxsplit=1)[0].strip()
    lines = []
    for line in t.splitlines():
        low = line.strip().lower()
        if low.startswith(
            ("user ask:", "notes:", "reflection:", "draft was:", "skill ", "[tool:")
        ):
            break
        lines.append(line)
    t = "\n".join(lines).strip()
    # Drop a trailing broken fragment (often cut mid-word by max tokens)
    if t and not t[-1] in ".!?)" and " " in t:
        words = t.split()
        last = words[-1]
        if len(last) <= 2 or not re.search(r"[aeiouy]", last.lower()):
            t = " ".join(words[:-1]).rstrip(",;:") + ("." if words[:-1] else "")
    return t.strip()


_PLAN_TEMPLATES: List[Tuple[List[str], str]] = [
    (
        ["study session", "short study", "plan a short study"],
        "1) Pick one topic. 2) Study for 20 focused minutes. 3) Write 3 notes. 4) Take a short break.",
    ),
    (
        ["learn python", "python basics", "learn python basics"],
        "Step 1: install Python. Step 2: learn variables and print. Step 3: practice if/else. Step 4: write a tiny script.",
    ),
    (
        ["homework", "small homework", "plan homework"],
        "1) List the due work. 2) Do the hardest item first for 15 minutes. 3) Check answers. 4) Pack what you need for tomorrow.",
    ),
    (
        ["morning routine", "plan my morning", "morning plan"],
        "1) Wake and drink water. 2) Light stretch. 3) Review today's top 3 tasks. 4) Start the first task.",
    ),
    (
        ["exercise", "workout", "short workout", "plan a workout"],
        "1) Warm up 3 minutes. 2) Do 3 simple moves. 3) Rest briefly between sets. 4) Stretch and drink water.",
    ),
    (
        ["grocery", "shopping list", "plan shopping"],
        "1) Check what you already have. 2) List meals for 2-3 days. 3) Add staples. 4) Shop with the list only.",
    ),
    (
        ["summarize", "summary", "summarise"],
        "1) Name the topic in one line. 2) List 3 key points. 3) End with one takeaway sentence.",
    ),
]


def answer_from_plan_template(user: str) -> Optional[str]:
    """Deterministic short plans for common agentic asks (no training)."""
    norm = re.sub(r"[^\w\s\?']+", " ", (user or "").lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    if not any(w in norm for w in ("plan", "step", "break down", "steps")):
        # still allow explicit learn-python style tasks
        if "python" not in norm:
            return None
    for cues, ans in _PLAN_TEMPLATES:
        if any(c in norm for c in cues):
            return ans
    return None


def looks_like_echo(user: str, reply: str) -> bool:
    """True if the model mostly echoed the question."""
    u = re.sub(r"[^\w\s]", "", (user or "").lower()).strip()
    r = re.sub(r"[^\w\s]", "", (reply or "").lower()).strip()
    if not r:
        return True
    if r == u or r.rstrip("?") == u.rstrip("?"):
        return True
    # "What is RAM?" -> "RAM?" / "RAM"
    um = _ECHO_ONLY.match((user or "").strip())
    if um:
        topic = (um.group(2) or "").strip().lower()
        if topic and r in {topic, f"what is {topic}"}:
            return True
    return False
