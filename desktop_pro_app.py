import tkinter as tk
from tkinter import messagebox

root = tk.Tk()
root.title('TinySLM Desktop Pro')
root.geometry('420x220')
root.configure(bg='#1e1e1e')

fg = '#f0f0f0'
tk.Label(root, text='Type something:', bg='#1e1e1e', fg=fg).pack(pady=6)
entry = tk.Entry(root, width=42, bg='#2d2d2d', fg=fg, insertbackground=fg)
entry.pack(pady=4)
status = tk.Label(root, text='Characters: 0', bg='#1e1e1e', fg='#9cdcfe')
status.pack(pady=4)

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

row = tk.Frame(root, bg='#1e1e1e')
row.pack(pady=10)
tk.Button(row, text='Show message', command=show_message).pack(side='left', padx=4)
tk.Button(row, text='Clear', command=clear_text).pack(side='left', padx=4)
tk.Button(row, text='Quit', command=quit_app).pack(side='left', padx=4)
root.mainloop()
