"""Gradio chat UI for TinySLM + DuckDuckGo."""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from tiny_slm.chat import TinyChat

ROOT = Path(__file__).resolve().parent


def build_demo(chat: TinyChat) -> gr.Blocks:
    with gr.Blocks(title="TinySLM") as demo:
        gr.Markdown(
            """
            # TinySLM
            A small language model **trained from scratch** on your machine.
            Uses DuckDuckGo when a question looks like it needs live web info.
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
            temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature")
            clear = gr.Button("Clear")

        def respond(message, history, do_search, temp):
            history = list(history or [])
            message = (message or "").strip()
            if not message:
                return history, ""
            reply, _ = chat.generate_reply(
                message,
                temperature=float(temp),
                force_search=bool(do_search),
            )
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply},
            ]
            return history, ""

        def on_clear():
            chat.reset()
            return []

        send.click(respond, [msg, chatbot, force_search, temperature], [chatbot, msg])
        msg.submit(respond, [msg, chatbot, force_search, temperature], [chatbot, msg])
        clear.click(on_clear, outputs=[chatbot])

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
