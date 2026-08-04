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
        ["what is cpu", "what's cpu", "whats cpu", "define cpu", "what is a cpu", "what's a cpu"],
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
        [
            "difference between ram and ssd",
            "ram vs ssd",
            "ram versus ssd",
            "compare ram and ssd",
            "compare ram",
            "ram and ssd",
            "ssd for a laptop",
        ],
        (
            "RAM is fast temporary working memory (speed for open apps; clears when power is off). "
            "An SSD is persistent storage that keeps files when the laptop is off. "
            "Buy enough RAM for smooth multitasking; buy SSD capacity for apps and files."
        ),
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
    (
        ["what is an api", "what's an api", "what is api"],
        "An API is a defined way for programs to talk to each other, like requesting data from a web service.",
    ),
    (
        ["what is debugging", "what's debugging", "how to debug"],
        "Debugging means finding and fixing mistakes: read the error, reproduce it, change one thing, and retest.",
    ),
    (
        ["what is a boolean", "what's a boolean", "boolean in programming"],
        "A boolean is True or False - used for yes/no decisions in if statements and loops.",
    ),
]

_CODE_CARDS: List[Tuple[List[str], str]] = [
    (
        ["function that adds", "add two numbers", "adds two numbers"],
        "def add(a, b):\n    return a + b\n\n# example: add(2, 3) -> 5",
    ),
    (
        ["safe_div", "safe div", "divide-by-zero", "divide by zero", "returns none on divide"],
        "def safe_div(a, b):\n    if b == 0:\n        return None\n    return a / b\n\n# example: safe_div(10, 0) -> None",
    ),
    (
        ["reverse a string", "reverse string", "string reverse"],
        "s = 'hello'\nreversed_s = s[::-1]  # 'olleh'\n# or: ''.join(reversed(s))",
    ),
    (
        ["for loop", "what is a for loop", "explain for loop", "explain what a for loop"],
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
        ["filter a list", "use filter in python", "filter function", "filter even", "filter"],
        "nums = [1, 2, 3, 4]\nlist(filter(lambda n: n % 2 == 0, nums))  # [2, 4]",
    ),
    (
        ["enumerate", "use enumerate"],
        "for i, x in enumerate(['a', 'b']):\n    print(i, x)",
    ),
    (
        ["word_count", "word count", "mapping each word"],
        "def word_count(text):\n    counts = {}\n    for w in text.split():\n        counts[w] = counts.get(w, 0) + 1\n    return counts\n\n# example: word_count('a a b') -> {'a': 2, 'b': 1}",
    ),
    (
        ["bankaccount", "bank account", "deposit(amount)", "withdraw(amount)"],
        "class BankAccount:\n    def __init__(self, balance=0):\n        self.balance = balance\n    def deposit(self, amount):\n        self.balance += amount\n    def withdraw(self, amount):\n        if amount > self.balance:\n            raise ValueError('overdraft')\n        self.balance -= amount",
    ),
    (
        ["fibonacci", "fib(n)", "recursive python function fib"],
        "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)\n\n# fib(0)=0, fib(1)=1, fib(6)=8",
    ),
    (
        ["unique sorted lines", "input.txt", "output.txt", "skips blanks"],
        "lines = []\nwith open('input.txt', encoding='utf-8') as f:\n    for line in f:\n        s = line.strip()\n        if s:\n            lines.append(s)\nwith open('output.txt', 'w', encoding='utf-8') as f:\n    for s in sorted(set(lines)):\n        f.write(s + '\\n')",
    ),
    (
        ["csv of names", "average score", "top 3 names"],
        "import csv\nrows = list(csv.DictReader(open('scores.csv', encoding='utf-8')))\navg = sum(float(r['score']) for r in rows) / len(rows)\ntop = sorted(rows, key=lambda r: float(r['score']), reverse=True)[:3]\nprint('average', avg)\nprint([r['name'] for r in top])",
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
    (
        ["pygame", "blue screen"],
        "import pygame\n"
        "pygame.init()\n"
        "screen = pygame.display.set_mode((400, 300))\n"
        "pygame.display.set_caption('TinySLM Pygame')\n"
        "running = True\n"
        "while running:\n"
        "    for event in pygame.event.get():\n"
        "        if event.type == pygame.QUIT:\n"
        "            running = False\n"
        "    screen.fill((30, 90, 200))  # blue\n"
        "    pygame.display.flip()\n"
        "pygame.quit()\n"
        "# Run: python app.py",
    ),
    (
        ["notepad", "text widget", "save button writing", "desktop notepad"],
        "import tkinter as tk\n"
        "from tkinter import filedialog\n\n"
        "root = tk.Tk()\n"
        "root.title('TinySLM Notepad')\n"
        "text = tk.Text(root, width=50, height=15)\n"
        "text.pack(padx=8, pady=8)\n\n"
        "def save_notes():\n"
        "    with open('notes.txt', 'w', encoding='utf-8') as f:\n"
        "        f.write(text.get('1.0', 'end-1c'))\n\n"
        "tk.Button(root, text='Save', command=save_notes).pack(pady=6)\n"
        "root.mainloop()\n"
        "# Run: python notepad.py",
    ),
    (
        ["canvas", "color picker", "fills it red", "fill it red"],
        "import tkinter as tk\n\n"
        "root = tk.Tk()\n"
        "root.title('TinySLM Canvas')\n"
        "canvas = tk.Canvas(root, width=240, height=140, bg='white')\n"
        "canvas.pack(padx=8, pady=8)\n\n"
        "def paint_red():\n"
        "    canvas.delete('all')\n"
        "    canvas.create_rectangle(0, 0, 240, 140, fill='red', outline='')\n\n"
        "tk.Button(root, text='Fill red', command=paint_red).pack(pady=6)\n"
        "root.mainloop()\n"
        "# Run: python canvas_demo.py",
    ),
    (
        ["pyqt5", "pyqt", "desktop calculator", "qt widgets"],
        "import sys\n"
        "from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QPushButton, QVBoxLayout, QLabel\n\n"
        "app = QApplication(sys.argv)\n"
        "win = QWidget()\n"
        "win.setWindowTitle('TinySLM Calculator')\n"
        "a = QLineEdit(); b = QLineEdit(); out = QLabel('Result:')\n"
        "btn = QPushButton('Add')\n"
        "def add():\n"
        "    try:\n"
        "        out.setText('Result: ' + str(float(a.text()) + float(b.text())))\n"
        "    except ValueError:\n"
        "        out.setText('Result: bad input')\n"
        "btn.clicked.connect(add)\n"
        "lay = QVBoxLayout(); lay.addWidget(a); lay.addWidget(b); lay.addWidget(btn); lay.addWidget(out)\n"
        "win.setLayout(lay); win.show(); sys.exit(app.exec_())\n"
        "# Run: python app.py  (pip install PyQt5)",
    ),
    (
        [
            "several features",
            "clear button",
            "status label",
            "dark background",
            "quit button",
            "pressing enter",
            "full updated",
            "add several features",
            "updated python program",
            "character",
            "empties the text",
        ],
        "import tkinter as tk\n"
        "from tkinter import messagebox\n\n"
        "root = tk.Tk()\n"
        "root.title('TinySLM Desktop Pro')\n"
        "root.geometry('420x220')\n"
        "root.configure(bg='#1e1e1e')\n\n"
        "fg = '#f0f0f0'\n"
        "tk.Label(root, text='Type something:', bg='#1e1e1e', fg=fg).pack(pady=6)\n"
        "entry = tk.Entry(root, width=42, bg='#2d2d2d', fg=fg, insertbackground=fg)\n"
        "entry.pack(pady=4)\n"
        "status = tk.Label(root, text='Characters: 0', bg='#1e1e1e', fg='#9cdcfe')\n"
        "status.pack(pady=4)\n\n"
        "def update_status(event=None):\n"
        "    status.config(text=f'Characters: {len(entry.get())}')\n\n"
        "def show_message(event=None):\n"
        "    msg = entry.get().strip() or 'Hello from TinySLM!'\n"
        "    messagebox.showinfo('Message', msg)\n\n"
        "def clear_text():\n"
        "    entry.delete(0, 'end')\n"
        "    update_status()\n\n"
        "def quit_app():\n"
        "    root.destroy()\n\n"
        "entry.bind('<KeyRelease>', update_status)\n"
        "entry.bind('<Return>', show_message)\n\n"
        "row = tk.Frame(root, bg='#1e1e1e')\n"
        "row.pack(pady=10)\n"
        "tk.Button(row, text='Show message', command=show_message).pack(side='left', padx=4)\n"
        "tk.Button(row, text='Clear', command=clear_text).pack(side='left', padx=4)\n"
        "tk.Button(row, text='Quit', command=quit_app).pack(side='left', padx=4)\n"
        "root.mainloop()\n"
        "# Run: python app.py",
    ),
    (
        [
            "modernise",
            "modernize",
            "modern ui",
            "modernise the ui",
            "modernize the ui",
            "cleaner modern layout",
            "accent color",
            "better typography",
            "rounded-looking",
            "subtle border",
            "modern layout",
        ],
        "import tkinter as tk\n"
        "from tkinter import messagebox\n\n"
        "BG = '#12141a'\n"
        "PANEL = '#1c2030'\n"
        "FG = '#e8ecf4'\n"
        "MUTED = '#9aa3b5'\n"
        "ACCENT = '#3d8bfd'\n"
        "ACCENT_FG = '#ffffff'\n"
        "ENTRY_BG = '#0f1219'\n"
        "BORDER = '#2a3144'\n\n"
        "root = tk.Tk()\n"
        "root.title('TinySLM Desktop Modern')\n"
        "root.geometry('520x340')\n"
        "root.minsize(460, 300)\n"
        "root.configure(bg=BG)\n\n"
        "shell = tk.Frame(root, bg=BG, padx=28, pady=22)\n"
        "shell.pack(fill='both', expand=True)\n\n"
        "tk.Label(\n"
        "    shell, text='TinySLM',\n"
        "    bg=BG, fg=FG, font=('Segoe UI', 20, 'bold'),\n"
        ").pack(anchor='w')\n"
        "tk.Label(\n"
        "    shell, text='Modern desktop messenger',\n"
        "    bg=BG, fg=MUTED, font=('Segoe UI', 10),\n"
        ").pack(anchor='w', pady=(2, 16))\n\n"
        "card = tk.Frame(\n"
        "    shell, bg=PANEL, highlightbackground=BORDER,\n"
        "    highlightthickness=1, padx=16, pady=14,\n"
        ")\n"
        "card.pack(fill='x')\n\n"
        "tk.Label(\n"
        "    card, text='Message', bg=PANEL, fg=MUTED,\n"
        "    font=('Segoe UI', 9),\n"
        ").pack(anchor='w')\n"
        "entry = tk.Entry(\n"
        "    card, font=('Segoe UI', 12), bg=ENTRY_BG, fg=FG,\n"
        "    insertbackground=FG, relief='flat',\n"
        "    highlightthickness=1, highlightbackground=BORDER,\n"
        "    highlightcolor=ACCENT,\n"
        ")\n"
        "entry.pack(fill='x', pady=(6, 10), ipady=8)\n\n"
        "status = tk.Label(\n"
        "    card, text='Characters: 0', bg=PANEL, fg=MUTED,\n"
        "    font=('Segoe UI', 9),\n"
        ")\n"
        "status.pack(anchor='w')\n\n"
        "def update_status(event=None):\n"
        "    status.config(text=f'Characters: {len(entry.get())}')\n\n"
        "def show_message(event=None):\n"
        "    msg = entry.get().strip() or 'Hello from TinySLM!'\n"
        "    messagebox.showinfo('Message', msg)\n\n"
        "def clear_text():\n"
        "    entry.delete(0, 'end')\n"
        "    update_status()\n\n"
        "def quit_app():\n"
        "    root.destroy()\n\n"
        "entry.bind('<KeyRelease>', update_status)\n"
        "entry.bind('<Return>', show_message)\n\n"
        "row = tk.Frame(shell, bg=BG)\n"
        "row.pack(anchor='w', pady=(18, 0))\n"
        "tk.Button(\n"
        "    row, text='Send', command=show_message,\n"
        "    bg=ACCENT, fg=ACCENT_FG, activebackground='#5a9fff',\n"
        "    activeforeground=ACCENT_FG, relief='flat', bd=0,\n"
        "    font=('Segoe UI', 10, 'bold'), padx=18, pady=8,\n"
        ").pack(side='left', padx=(0, 8))\n"
        "tk.Button(\n"
        "    row, text='Clear', command=clear_text,\n"
        "    bg=PANEL, fg=FG, activebackground=BORDER,\n"
        "    activeforeground=FG, relief='flat', bd=0,\n"
        "    font=('Segoe UI', 10), padx=14, pady=8,\n"
        "    highlightthickness=1, highlightbackground=BORDER,\n"
        ").pack(side='left', padx=(0, 8))\n"
        "tk.Button(\n"
        "    row, text='Quit', command=quit_app,\n"
        "    bg=PANEL, fg=MUTED, activebackground=BORDER,\n"
        "    activeforeground=FG, relief='flat', bd=0,\n"
        "    font=('Segoe UI', 10), padx=14, pady=8,\n"
        "    highlightthickness=1, highlightbackground=BORDER,\n"
        ").pack(side='left')\n"
        "root.mainloop()\n"
        "# Run: python app.py",
    ),
    (
        [
            "tkinter",
            "gui app",
            "desktop app",
            "desktop software",
            "python software for desktop",
            "window and a button",
            "text field",
            "messagebox",
            "desktop",
        ],
        "import tkinter as tk\nfrom tkinter import messagebox\n\n"
        "root = tk.Tk()\n"
        "root.title('TinySLM Desktop Demo')\n"
        "root.geometry('360x160')\n\n"
        "tk.Label(root, text='Type something:').pack(pady=6)\n"
        "entry = tk.Entry(root, width=40)\n"
        "entry.pack(pady=4)\n\n"
        "def on_click():\n"
        "    msg = entry.get().strip() or 'Hello from TinySLM!'\n"
        "    messagebox.showinfo('Message', msg)\n\n"
        "tk.Button(root, text='Show message', command=on_click).pack(pady=10)\n"
        "root.mainloop()\n"
        "# Run: python app.py",
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
            "safe_div",
            "safe div",
            "divide",
            "zero",
            "while",
            "write",
            "save",
            "map",
            "filter",
            "lambda",
            "sort",
            "append",
            "enumerate",
            "word_count",
            "word count",
            "fibonacci",
            "fib",
            "bankaccount",
            "bank account",
            "deposit",
            "withdraw",
            "csv",
            "input.txt",
            "output.txt",
            "recursive",
            "desktop",
            "tkinter",
            "gui",
            "software",
            "window",
            "button",
            "messagebox",
            "pygame",
            "pyqt",
            "pyqt5",
            "calculator",
            "clear",
            "status",
            "quit",
            "dark",
            "features",
            "modern",
            "modernise",
            "modernize",
            "ui",
            "layout",
            "typography",
            "accent",
        )
    ):
        return None
    # Prefer the card with the strongest cue match (longer / more specific cues win).
    feature_ask = any(
        f in norm
        for f in (
            "several features",
            "clear button",
            "status label",
            "dark background",
            "quit button",
            "pressing enter",
            "full updated",
            "add several",
            "updated python",
            "characters were typed",
        )
    )
    modern_ask = any(
        f in norm
        for f in (
            "modernise",
            "modernize",
            "modern ui",
            "modern layout",
            "cleaner modern",
            "accent color",
            "better typography",
            "rounded-looking",
            "subtle border",
        )
    )
    best: Optional[tuple[int, str]] = None
    for cues, ans in _CODE_CARDS:
        hits = [c for c in cues if c in norm]
        if not hits:
            continue
        # Don't let the basic desktop demo steal feature-upgrade asks.
        if feature_ask and "Desktop Demo" in ans and "Desktop Pro" not in ans and "Desktop Modern" not in ans:
            continue
        if modern_ask and "Desktop Demo" in ans:
            continue
        if modern_ask and "Desktop Pro" in ans and "Desktop Modern" not in ans:
            continue
        score = max(len(c) for c in hits) * 10 + len(hits)
        if feature_ask and "Desktop Pro" in ans and not modern_ask:
            score += 500
        if modern_ask and "Desktop Modern" in ans:
            score += 800
        if best is None or score > best[0]:
            best = (score, ans)
    return best[1] if best else None

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
                # Avoid "ram?" matching any long sentence that merely mentions RAM
                if len(norm) > 16 and bare not in (
                    c,
                    f"what is {c}",
                    f"what's {c}",
                    f"whats {c}",
                    f"define {c}",
                ):
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
        [
            "weekend project",
            "budget tracker",
            "personal budget",
            "plan a weekend",
            "weekend project to build",
        ],
        "1) Define must-have features (add expense, list, total). "
        "2) Sketch a tiny data model. "
        "3) Build a CLI or simple GUI for those features. "
        "4) Save data to a file and do a quick test pass.",
    ),
    (
        ["learn python", "python basics", "learn python basics"],
        "Step 1: install Python. Step 2: learn variables and print. Step 3: practice if/else. Step 4: write a tiny script.",
    ),
    (
        ["debug", "debug a", "fix a bug", "debug a small python"],
        "Step 1: read the error message. Step 2: print key variables. Step 3: fix one bug. Step 4: rerun the script.",
    ),
    (
        ["coding practice", "practice session", "coding session"],
        "1) Pick one small function. 2) Write tests for two cases. 3) Implement it. 4) Refactor names.",
    ),
    (
        ["todo list app", "build a todo", "todo list"],
        "1) Define a task list. 2) Add add/complete/list commands. 3) Save tasks to a file. 4) Test the happy path.",
    ),
    (
        ["renames files", "rename files by date", "cli tool that renames"],
        "Step 1: list files in the folder. Step 2: parse each file date. Step 3: build the new name. Step 4: rename safely with a dry-run first.",
    ),
    (
        ["compare python and javascript", "python and javascript", "python vs javascript"],
        "Python is often simpler for beginners and data scripts. JavaScript runs in browsers and powers interactive web pages. Start with the one matching your first project.",
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
    if not any(w in norm for w in ("plan", "step", "break down", "steps", "compare")):
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


def repair_short_definition(user: str, reply: str) -> str:
    """Replace drifted answers for a few high-value definition asks."""
    u = re.sub(r"\s+", " ", (user or "").lower()).strip()
    t = (reply or "").strip()
    low = t.lower()
    if u.startswith("what is python") or u in ("what is python?", "what's python?"):
        if "python" in low and "language" in low:
            return t
        return (
            "Python is a popular programming language used for websites, "
            "data work, automation, and learning to code."
        )
    if u.startswith("what is ram") or u in ("what is ram?", "what's ram?"):
        if "memory" in low:
            return t
        return (
            "RAM is short-term computer memory the CPU uses to hold "
            "running programs and data."
        )
    if u.startswith("what is a cpu") or u.startswith("what is cpu") or u in ("what is a cpu?", "what is cpu?"):
        if "processor" in low or ("cpu" in low and "instruction" in low):
            return t
        return "A CPU is the main processor chip that runs a computer's instructions."
    if "machine learning" in u:
        if "learning" in low and ("data" in low or "example" in low or "model" in low):
            return t
        return "Machine learning lets programs improve from examples instead of only hard-coded rules."
    if "water made" in u or u.startswith("what is water"):
        if "h2o" in low or "hydrogen" in low:
            return t
        return "Water is H2O - two hydrogen atoms and one oxygen atom."
    return t


def repair_plan_answer(user: str, reply: str) -> str:
    """Snap weak agentic drafts to plan templates when available."""
    plan = answer_from_plan_template(user)
    if not plan:
        return reply
    low = (reply or "").lower()
    # Keep a decent plan/compare draft
    if any(x in low for x in ("step 1", "1)", "2)", "python", "javascript", "paris")) and len(low) > 40:
        if not looks_low_quality(reply):
            return reply
    return plan


def repair_coding_answer(user: str, reply: str) -> str:
    """If a coding ask clearly missed its tokens, snap to the grounded card."""
    if not (
        looks_wrong_coding_answer(user, reply)
        or looks_wrong_sort_answer(user, reply)
    ):
        return reply
    return answer_from_code_template(user) or reply


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
        (("filter",), ("filter",)),
        (("enumerate",), ("enumerate",)),
        (("word_count", "word count", "mapping each word"), ("def word_count", "split")),
        (("bankaccount", "bank account", "deposit(amount)", "withdraw(amount)"), ("class bankaccount", "def deposit")),
        (("fibonacci", "fib(n)", "function fib"), ("def fib", "fib(")),
        (("input.txt", "output.txt", "skips blanks"), ("open(", "sorted")),
        (("csv of names", "average score", "top 3 names"), ("csv", "average")),
        (("try/except", "valueerror", "converts user text to int"), ("try:", "except")),
        (("if/else", "if else", "what does if"), ("if", "else")),
        (("adds two numbers", "function that adds", "def add"), ("def add", "return")),
        (
            ("desktop", "tkinter", "gui app", "desktop app", "desktop software"),
            ("tkinter", "mainloop", "button"),
        ),
        (("pygame",), ("pygame", "display")),
        (("pyqt5", "pyqt", "desktop calculator"), ("pyqt5", "qapplication", "qpushbutton")),
        (
            ("several features", "clear button", "status label", "quit button", "dark background"),
            ("clear", "quit", "characters", "bind"),
        ),
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
    # Code snippets are punctuation-heavy; don't treat as gibberish.
    if any(
        m in t
        for m in ("def ", "lambda", "[::-1]", "append(", "filter(", "open(", "for n in", "for x in")
    ) or ("=" in t and ("[" in t or "(" in t)):
        return False
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
