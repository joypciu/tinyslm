import tkinter as tk
from tkinter import messagebox

BG = '#12141a'
PANEL = '#1c2030'
FG = '#e8ecf4'
MUTED = '#9aa3b5'
ACCENT = '#3d8bfd'
ACCENT_FG = '#ffffff'
ENTRY_BG = '#0f1219'
BORDER = '#2a3144'

root = tk.Tk()
root.title('TinySLM Desktop Modern')
root.geometry('520x340')
root.minsize(460, 300)
root.configure(bg=BG)

shell = tk.Frame(root, bg=BG, padx=28, pady=22)
shell.pack(fill='both', expand=True)

tk.Label(
    shell, text='TinySLM',
    bg=BG, fg=FG, font=('Segoe UI', 20, 'bold'),
).pack(anchor='w')
tk.Label(
    shell, text='Modern desktop messenger',
    bg=BG, fg=MUTED, font=('Segoe UI', 10),
).pack(anchor='w', pady=(2, 16))

card = tk.Frame(
    shell, bg=PANEL, highlightbackground=BORDER,
    highlightthickness=1, padx=16, pady=14,
)
card.pack(fill='x')

tk.Label(
    card, text='Message', bg=PANEL, fg=MUTED,
    font=('Segoe UI', 9),
).pack(anchor='w')
entry = tk.Entry(
    card, font=('Segoe UI', 12), bg=ENTRY_BG, fg=FG,
    insertbackground=FG, relief='flat',
    highlightthickness=1, highlightbackground=BORDER,
    highlightcolor=ACCENT,
)
entry.pack(fill='x', pady=(6, 10), ipady=8)

status = tk.Label(
    card, text='Characters: 0', bg=PANEL, fg=MUTED,
    font=('Segoe UI', 9),
)
status.pack(anchor='w')

def update_status(event=None):
    status.config(text=f'Characters: {len(entry.get())}')

def show_message(event=None):
    msg = entry.get().strip() or 'Hello from TinySLM!'
    messagebox.showinfo('Message', msg)

def clear_text():
    entry.delete(0, 'end')
    update_status()

def quit_app():
    root.destroy()

entry.bind('<KeyRelease>', update_status)
entry.bind('<Return>', show_message)

row = tk.Frame(shell, bg=BG)
row.pack(anchor='w', pady=(18, 0))
tk.Button(
    row, text='Send', command=show_message,
    bg=ACCENT, fg=ACCENT_FG, activebackground='#5a9fff',
    activeforeground=ACCENT_FG, relief='flat', bd=0,
    font=('Segoe UI', 10, 'bold'), padx=18, pady=8,
).pack(side='left', padx=(0, 8))
tk.Button(
    row, text='Clear', command=clear_text,
    bg=PANEL, fg=FG, activebackground=BORDER,
    activeforeground=FG, relief='flat', bd=0,
    font=('Segoe UI', 10), padx=14, pady=8,
    highlightthickness=1, highlightbackground=BORDER,
).pack(side='left', padx=(0, 8))
tk.Button(
    row, text='Quit', command=quit_app,
    bg=PANEL, fg=MUTED, activebackground=BORDER,
    activeforeground=FG, relief='flat', bd=0,
    font=('Segoe UI', 10), padx=14, pady=8,
    highlightthickness=1, highlightbackground=BORDER,
).pack(side='left')
root.mainloop()
