"""Gradio chat UI for TinySLM + DuckDuckGo."""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from tiny_slm.chat import TinyChat

ROOT = Path(__file__).resolve().parent


def build_demo(chat: TinyChat) -> gr.Blocks:
    with gr.Blocks(title="TinySLM") as demo:
        mem_path = ROOT / "checkpoints" / "memory_store.json"
        status = gr.Markdown("")
        gr.Markdown(
            """
            # TinySLM
            From-scratch SLM with **2M memory**, SARA tools, FAQ/math/code fast-paths, and **auto DuckDuckGo**
            when a question needs live facts. Tip: say *search* / *news* to force lookup; use Save/Load for memory.
            """
        )
        chatbot = gr.Chatbot(height=420, label="Chat")
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask something… (tip: say 'search' or 'news' to force web lookup)",
                scale=4,
                show_label=False,
            )
            send = gr.Button("Send", variant="primary", scale=1)
        with gr.Row():
            force_search = gr.Checkbox(label="Force DuckDuckGo search", value=False)
            temperature = gr.Slider(0.1, 1.5, value=0.4, step=0.1, label="Temperature")
            clear = gr.Button("Clear")
            save_mem = gr.Button("Save memory")
            load_mem = gr.Button("Load memory")
        ingest_file = gr.File(
            label="Ingest text/markdown into 2M memory",
            file_types=[".txt", ".md", ".csv"],
            type="filepath",
        )

        def respond(message, history, do_search, temp):
            history = list(history or [])
            message = (message or "").strip()
            if not message:
                return history, "", ""
            reply, _ = chat.generate_reply(
                message,
                temperature=float(temp),
                force_search=bool(do_search),
            )
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply},
            ]
            st = chat.memory.stats()
            note = (
                f"memory {st['tokens']:,}/{st['max_tokens']:,} tok · "
                f"{st['chunks']} chunks · history {len(chat.history)} turns"
            )
            return history, "", note

        def on_clear():
            chat.reset()
            chat.clear_memory()
            return [], "memory cleared (skills re-seeded)"

        def on_save():
            chat.save_memory(mem_path)
            return f"saved → {mem_path}"

        def on_load():
            info = chat.load_memory(mem_path)
            return f"loaded {info}"

        def on_ingest(path):
            if not path:
                return "no file"
            p = Path(path)
            text = p.read_text(encoding="utf-8", errors="ignore")
            info = chat.ingest(text, source=f"file:{p.name}")
            return f"ingested {p.name}: {info}"

        send.click(
            respond,
            [msg, chatbot, force_search, temperature],
            [chatbot, msg, status],
        )
        msg.submit(
            respond,
            [msg, chatbot, force_search, temperature],
            [chatbot, msg, status],
        )
        clear.click(on_clear, outputs=[chatbot, status])
        save_mem.click(on_save, outputs=[status])
        load_mem.click(on_load, outputs=[status])
        ingest_file.change(on_ingest, inputs=[ingest_file], outputs=[status])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints" / "tinyslm.pt")
    parser.add_argument("--tok", type=Path, default=ROOT / "checkpoints" / "tokenizer.json")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    chat = TinyChat(ckpt_path=args.ckpt, tok_path=args.tok)
    print(
        f"Loaded TinySLM ({chat.model.n_parameters():,} params, train step {chat.step}) on {chat.device}"
    )
    demo = build_demo(chat)
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
