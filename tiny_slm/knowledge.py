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
    (
        ["what is python", "what's python", "whats python"],
        "Python is a popular programming language used for websites, data work, automation, and learning to code.",
    ),
    (
        ["how to learn", "how do i learn", "best way to learn"],
        "Practice a little every day: one concept, one tiny exercise, then explain it in your own words.",
    ),
    (
        ["what is ai", "what's ai", "what is artificial intelligence"],
        "AI is software that learns patterns from data to help with language, vision, planning, and similar tasks.",
    ),
    (
        ["what is tinyslm", "what's tinyslm", "about tinyslm"],
        "TinySLM is a tiny from-scratch chat model with a short neural window plus up to 2M tokens of retrieved memory.",
    ),
    (
        ["explain gravity", "what is gravity", "what's gravity"],
        "Gravity is the force that pulls masses together - it keeps us on Earth and planets in orbit around the Sun.",
    ),
    (
        ["why do we sleep", "why sleep", "why do humans sleep"],
        "Sleep lets the body and brain rest, repair, and consolidate memories from the day.",
    ),
    (
        ["what is json", "what's json"],
        "JSON is a simple text format for data objects and lists, widely used in APIs and config files.",
    ),
    (
        ["what is an api", "what's an api", "what is api"],
        "An API is a defined way for programs to talk to each other - request in, data or action out.",
    ),
    (
        ["what is git", "what's git"],
        "Git is a version-control tool that tracks changes in code so you can branch, merge, and roll back.",
    ),
    (
        ["what is http", "what's http"],
        "HTTP is the protocol browsers and servers use to request and send web pages and API data.",
    ),
    (
        ["what is sql", "what's sql"],
        "SQL is a language for asking databases for data - for example SELECT name FROM users;",
    ),
    (
        ["what is html", "what's html"],
        "HTML is the markup language that structures web page content with tags like h1, p, and a.",
    ),
    (
        ["what is css", "what's css"],
        "CSS styles web pages - colors, layout, fonts - separate from the HTML structure.",
    ),
    (
        ["what is javascript", "what's javascript", "what is js"],
        "JavaScript is the programming language that makes web pages interactive in the browser.",
    ),
    (
        ["what is linux", "what's linux"],
        "Linux is an open-source operating system kernel used in servers, phones, and many computers.",
    ),
    (
        ["what is docker", "what's docker"],
        "Docker packages an app and its dependencies into a container so it runs the same on different machines.",
    ),
    (
        ["what is dns", "what's dns"],
        "DNS translates human website names like example.com into IP addresses computers can route to.",
    ),
    (
        ["what is a database", "what's a database"],
        "A database stores structured data so programs can insert, query, update, and delete it reliably.",
    ),
    (
        ["what is rest", "what's rest", "what is a rest api"],
        "REST is a common API style using HTTP methods (GET, POST, PUT, DELETE) on resource URLs.",
    ),
    (
        ["what is kubernetes", "what's kubernetes"],
        "Kubernetes schedules and manages containers across machines so apps stay available and scalable.",
    ),
    (
        ["what is recursion", "what's recursion"],
        "Recursion is when a function solves a problem by calling itself on a smaller piece until a base case.",
    ),
    (
        ["what is machine learning", "what's machine learning", "what is ml"],
        "Machine learning lets programs improve from examples instead of only hard-coded rules.",
    ),
    (
        ["what is a neural network", "what's a neural network"],
        "A neural network is a layered model of weighted connections that learns patterns from data.",
    ),
    (
        ["what is overfitting", "what's overfitting"],
        "Overfitting means a model memorizes training data too well and performs worse on new data.",
    ),
    (
        ["what is a tokenizer", "what's a tokenizer"],
        "A tokenizer splits text into tokens (pieces the model reads), often words or subwords.",
    ),
    (
        ["what is attention", "what's attention", "self attention"],
        "Attention lets a model weigh which parts of the input matter most when building each output.",
    ),
    (
        ["what is a transformer", "what's a transformer model"],
        "A Transformer is a neural architecture that uses attention layers to model sequences efficiently.",
    ),
]

_CODE_CARDS: List[Tuple[List[str], str]] = [
    (
        ["function that adds", "add two numbers", "adds two numbers"],
        "def add(a, b):\n    return a + b\n\n# example: add(2, 3) -> 5",
    ),
    (
        ["reverse a string", "reverse string", "string reverse"],
        "s = 'hello'\nreversed_s = s[::-1]  # 'olleh'\n# or: ''.join(reversed(s))",
    ),
    (
        ["for loop", "what is a for loop", "explain for loop"],
        "A for loop repeats work for each item:\nfor x in [1, 2, 3]:\n    print(x)",
    ),
    (
        ["if/else", "if else", "what does if/else"],
        "if/else picks a branch:\nif n > 0:\n    print('positive')\nelse:\n    print('not positive')",
    ),
    (
        ["what is a variable", "what's a variable", "variable in programming"],
        "A variable is a named place that stores a value, e.g. score = 10 in Python.",
    ),
    (
        ["hello world", "print hello"],
        "print('Hello, world!')",
    ),
    (
        ["list comprehension"],
        "squares = [n*n for n in range(5)]  # [0, 1, 4, 9, 16]",
    ),
    (
        ["dictionary", "what is a dict", "python dict", "dict in python"],
        "person = {'name': 'Ada', 'year': 1815}\nprint(person['name'])",
    ),
    (
        ["append to a list", "list append", "how do i append"],
        "nums = [1, 2]\nnums.append(3)  # [1, 2, 3]",
    ),
    (
        ["try except", "exception handling", "try/except"],
        "try:\n    n = int(text)\nexcept ValueError:\n    print('not a number')",
    ),
    (
        ["read a file", "open a file", "read file in python"],
        "with open('notes.txt', encoding='utf-8') as f:\n    text = f.read()",
    ),
    (
        ["class in python", "what is a class", "define a class"],
        "class Dog:\n    def __init__(self, name):\n        self.name = name\n\nDog('Rex').name",
    ),
    (
        ["write a file", "save a file", "write file in python"],
        "with open('out.txt', 'w', encoding='utf-8') as f:\n    f.write('hello')",
    ),
    (
        ["while loop", "what is a while loop"],
        "n = 3\nwhile n > 0:\n    print(n)\n    n -= 1",
    ),
    (
        ["map function", "use map in python"],
        "nums = [1, 2, 3]\nlist(map(lambda x: x * 2, nums))  # [2, 4, 6]",
    ),
    (
        ["filter function", "use filter in python"],
        "nums = [1, 2, 3, 4]\nlist(filter(lambda x: x % 2 == 0, nums))  # [2, 4]",
    ),
    (
        ["sort a list", "sort list python"],
        "nums = [3, 1, 2]\nnums.sort()          # in place\nsorted(nums)       # new list",
    ),
    (
        ["enumerate", "use enumerate"],
        "for i, name in enumerate(['a', 'b']):\n    print(i, name)  # 0 a / 1 b",
    ),
]


def answer_from_code_template(user: str) -> Optional[str]:
    """Short grounded code snippets for common programming asks."""
    norm = re.sub(r"[^\w\s\?/']+", " ", (user or "").lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    if not any(
        w in norm
        for w in (
            "python",
            "code",
            "function",
            "loop",
            "variable",
            "programming",
            "string",
            "if/else",
            "if else",
            "print",
            "list comprehension",
            "dict",
            "dictionary",
            "class",
            "file",
            "except",
            "exception",
            "while",
            "write",
            "save",
            "map",
            "filter",
            "lambda",
            "sort",
            "append",
            "enumerate",
        )
    ):
        return None
    for cues, ans in _CODE_CARDS:
        if any(c in norm for c in cues):
            return ans
    return None

# country key -> (display name, capital)
_CAPITALS = {
    "france": ("France", "Paris"),
    "japan": ("Japan", "Tokyo"),
    "germany": ("Germany", "Berlin"),
    "italy": ("Italy", "Rome"),
    "bangladesh": ("Bangladesh", "Dhaka"),
    "india": ("India", "New Delhi"),
    "usa": ("the United States", "Washington, D.C."),
    "us": ("the United States", "Washington, D.C."),
    "united states": ("the United States", "Washington, D.C."),
    "uk": ("the United Kingdom", "London"),
    "united kingdom": ("the United Kingdom", "London"),
    "canada": ("Canada", "Ottawa"),
    "australia": ("Australia", "Canberra"),
    "spain": ("Spain", "Madrid"),
    "brazil": ("Brazil", "Brasília"),
    "china": ("China", "Beijing"),
    "pakistan": ("Pakistan", "Islamabad"),
}


def _capital_answer(norm: str) -> Optional[str]:
    if "france" in norm and "japan" in norm and "capital" in norm:
        return "France's capital is Paris. Japan's capital is Tokyo."
    m = re.search(r"capital of (?:the )?([a-z ]+?)(?:\?|$)", norm)
    if m:
        key = m.group(1).strip()
        # longest key first
        for ck in sorted(_CAPITALS, key=len, reverse=True):
            if key == ck or key.startswith(ck + " ") or key.endswith(" " + ck):
                name, cap = _CAPITALS[ck]
                return f"The capital of {name} is {cap}."
            if key == ck:
                name, cap = _CAPITALS[ck]
                return f"The capital of {name} is {cap}."
        if key in _CAPITALS:
            name, cap = _CAPITALS[key]
            return f"The capital of {name} is {cap}."
    for ck, (name, cap) in _CAPITALS.items():
        if f"{ck} capital" in norm or f"{ck}'s capital" in norm:
            return f"The capital of {name} is {cap}."
    return None


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
    cap = _capital_answer(norm)
    if cap:
        return cap
    # Exact short greetings / thanks before looser substring cards
    bare = norm.rstrip("?").strip()
    if bare in ("hello", "hi", "hey", "good morning", "good evening"):
        return "Hi! I'm TinySLM. How are you doing today?"
    if bare in ("thanks", "thank you", "thx"):
        return "You're welcome - happy to help anytime."
    for cues, ans in _FAQ:
        for cue in cues:
            c = cue.rstrip("?").strip()
            if len(c) <= 3:
                if not re.search(rf"(?<!\w){re.escape(c)}(?!\w)", norm):
                    continue
                if len(norm) > 24 and c in ("hi", "hey"):
                    continue
                return ans
            if c in norm or c == bare:
                return ans
            if cue.endswith("?") and bare == c:
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
        ["debug", "debug a", "fix a bug", "python script"],
        "Step 1: read the error message. Step 2: print key variables. Step 3: fix one bug. Step 4: rerun the script.",
    ),
    (
        ["coding practice", "practice session", "coding session"],
        "1) Pick one small function. 2) Write tests for two cases. 3) Implement it. 4) Refactor names.",
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
    (
        ["next small step", "plan my next", "my next small step", "next step"],
        "1) Name the outcome. 2) Do the smallest useful action for 10 minutes. 3) Check it off. 4) Pick the following step.",
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


def repair_truncated_greeting(user: str, reply: str) -> str:
    """Tiny nets often emit only the second half of the greeting — or drift off-topic."""
    bare = re.sub(r"[^\w\s]", "", (user or "").lower()).strip()
    if bare not in ("hello", "hi", "hey", "good morning", "good evening"):
        return reply
    t = (reply or "").strip()
    if not t:
        return "Hi! I'm TinySLM. How are you doing today?"
    low = t.lower()
    if "tinyslm" in low or low.startswith(("hi!", "hi ", "hello", "hey")):
        return t
    if any(p in low for p in ("how are you", "how can i help", "doing today")):
        return "Hi! I'm TinySLM. " + t
    # Collapsed into unrelated prompts ("What is 2 + 2?")
    return "Hi! I'm TinySLM. How are you doing today?"


def looks_off_topic_math(user: str, reply: str) -> bool:
    """True when a non-math ask (e.g. What is Python?) collapses to arithmetic."""
    u = (user or "").lower()
    if re.search(r"\d\s*[\+\-\*\/]\s*\d", u) or "plus" in u or "minus" in u:
        return False
    if not any(k in u for k in ("python", "ram", "cpu", "water", "france", "japan", "sort")):
        return False
    r = (reply or "").lower()
    return bool(re.search(r"\d\s*\+\s*\d", r) and "equal" in r)


def looks_wrong_sort_answer(user: str, reply: str) -> bool:
    """Sort questions that drift into generic Python-definition chatter."""
    u = (user or "").lower()
    if "sort" not in u:
        return False
    r = (reply or "").lower()
    if "sort" in r or "sorted" in r:
        return False
    return any(
        p in r
        for p in (
            "programming language",
            "websites, data",
            "popular programming",
            "ask me to plan",
            "say search",
        )
    )


def looks_wrong_coding_answer(user: str, reply: str) -> bool:
    """True when a coding ask is missing its key tokens (drift/gibberish)."""
    u = (user or "").lower()
    r = (reply or "").lower()
    rules = (
        (("reverse",), ("[::-1]", "reversed")),
        (("append",), ("append",)),
        (("dict", "dictionary"), ("{",)),
        (("list comprehension", "comprehension that square"), ("for", "in")),
        (("read a file", "read file"), ("open(", "read")),
    )
    for cues, need in rules:
        if any(c in u for c in cues) and not any(n in r for n in need):
            return True
    return False


def looks_low_quality(reply: str) -> bool:
    """Heuristic for collapsed / gibberish tiny-model drafts."""
    t = (reply or "").strip()
    if len(t) < 4:
        return True
    low = t.lower()
    if low.startswith("plan:") and len(t) < 48:
        return True
    if "user ask:" in low or "notes:" in low:
        return True
    letters = re.findall(r"[A-Za-z]", t)
    # Short math answers ("2 + 2 equals 4.") are digit-heavy; don't treat that as gibberish.
    has_digit = bool(re.search(r"\d", t))
    has_word = bool(re.search(r"[A-Za-z]{3,}", t))
    if has_digit and not has_word:
        return True
    if letters and len(letters) / max(1, len(t)) < 0.45:
        if not (has_digit and has_word and len(t) < 48):
            return True
    words = re.findall(r"[A-Za-z']+", low)
    if words and len(set(words)) <= 2 and len(words) >= 5:
        return True
    # Broken mid-word endings without sentence close
    if len(t) > 40 and t[-1] not in ".!?)" and re.search(r"\b\w{1,2}$", t):
        if not re.search(r"[.!?]", t):
            return True
    return False


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
