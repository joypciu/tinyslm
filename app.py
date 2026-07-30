"""Gradio chat UI for TinySLM — modern single-composition workspace."""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from tiny_slm.chat import TinyChat

ROOT = Path(__file__).resolve().parent

# Cool mist + forest ink — not purple, not cream/terracotta, not pure dark.
_THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#eef7f5",
        c100="#d5ebe6",
        c200="#aad7cd",
        c300="#78bdb0",
        c400="#4a9e8f",
        c500="#2f7f72",
        c600="#24665c",
        c700="#1e514a",
        c800="#1a423c",
        c900="#143530",
        c950="#0c221f",
    ),
    secondary_hue=gr.themes.Color(
        c50="#f3f5f4",
        c100="#e2e7e5",
        c200="#c5cecb",
        c300="#9aaba5",
        c400="#6f877f",
        c500="#526c65",
        c600="#415651",
        c700="#364642",
        c800="#2e3a37",
        c900="#283230",
        c950="#151c1a",
    ),
    neutral_hue=gr.themes.Color(
        c50="#f6f7f6",
        c100="#e9ecea",
        c200="#d3d9d6",
        c300="#afbbb5",
        c400="#84948d",
        c500="#667770",
        c600="#515f59",
        c700="#434e49",
        c800="#39413e",
        c900="#323835",
        c950="#1b201e",
    ),
    font=[gr.themes.GoogleFont("Figtree"), "ui-sans-serif", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#e8efeC",
    body_background_fill_dark="#e8efeC",
    body_text_color="#143530",
    body_text_color_dark="#143530",
    block_background_fill="rgba(255,255,255,0.55)",
    block_background_fill_dark="rgba(255,255,255,0.55)",
    block_border_width="0px",
    block_label_text_color="#24665c",
    block_title_text_color="#143530",
    button_primary_background_fill="#24665c",
    button_primary_background_fill_hover="#1e514a",
    button_primary_text_color="#f4faf8",
    button_secondary_background_fill="rgba(255,255,255,0.7)",
    button_secondary_background_fill_hover="rgba(255,255,255,0.95)",
    button_secondary_text_color="#1e514a",
    input_background_fill="rgba(255,255,255,0.82)",
    input_border_color="#c5cecb",
    input_border_width="1px",
    shadow_drop="none",
    shadow_drop_lg="none",
)

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&family=Syne:wght@600;700;800&display=swap');

:root {
  --ink: #143530;
  --ink-soft: #3d5651;
  --accent: #24665c;
  --accent-soft: #4a9e8f;
  --mist: #e8efec;
  --panel: rgba(255, 255, 255, 0.62);
  --line: rgba(20, 53, 48, 0.10);
  --radius: 14px;
}

html, body, .gradio-container {
  font-family: 'Figtree', ui-sans-serif, sans-serif !important;
  color: var(--ink) !important;
}

.gradio-container {
  max-width: 920px !important;
  margin: 0 auto !important;
  padding: 1.25rem 1rem 2.5rem !important;
}

/* Atmosphere: cool mist field + soft teal wash (not flat) */
.gradio-container,
.main,
footer {
  background: transparent !important;
}

body {
  min-height: 100vh;
  background:
    radial-gradient(1200px 700px at 12% -10%, rgba(74, 158, 143, 0.28), transparent 55%),
    radial-gradient(900px 600px at 100% 8%, rgba(36, 102, 92, 0.14), transparent 50%),
    linear-gradient(165deg, #dfeae6 0%, #eef3f0 42%, #e4ebe7 100%) !important;
  background-attachment: fixed !important;
}

/* Subtle paper grain */
body::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  opacity: 0.35;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.45'/%3E%3C/svg%3E");
  mix-blend-mode: soft-light;
}

.gradio-container > * {
  position: relative;
  z-index: 1;
}

/* Brand composition */
.tslm-hero {
  animation: tslmRise 0.7s cubic-bezier(0.22, 1, 0.36, 1) both;
  margin: 0.25rem 0 1.1rem;
  padding: 0.15rem 0 1rem;
  border-bottom: 1px solid var(--line);
}

.tslm-brand {
  font-family: 'Syne', ui-sans-serif, sans-serif !important;
  font-weight: 800 !important;
  font-size: clamp(2.35rem, 5.5vw, 3.35rem) !important;
  letter-spacing: -0.04em !important;
  line-height: 1.02 !important;
  color: var(--ink) !important;
  margin: 0 !important;
}

.tslm-tag {
  margin: 0.55rem 0 0 !important;
  max-width: 36rem;
  font-size: 1.02rem !important;
  line-height: 1.45 !important;
  color: var(--ink-soft) !important;
  font-weight: 500 !important;
}

.tslm-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem 1.1rem;
  margin-top: 0.85rem;
  font-size: 0.82rem;
  letter-spacing: 0.01em;
  color: var(--accent);
  font-weight: 600;
}

.tslm-meta span {
  opacity: 0.92;
}

.tslm-meta span::before {
  content: "";
  display: inline-block;
  width: 0.4rem;
  height: 0.4rem;
  margin-right: 0.45rem;
  border-radius: 1px;
  background: var(--accent-soft);
  vertical-align: 0.1rem;
}

/* Chat plane */
#tslm-chat {
  animation: tslmRise 0.85s cubic-bezier(0.22, 1, 0.36, 1) 0.08s both;
  border: 1px solid var(--line) !important;
  border-radius: var(--radius) !important;
  background: var(--panel) !important;
  backdrop-filter: blur(10px);
  overflow: hidden;
}

#tslm-chat .bubble-wrap,
#tslm-chat .message-wrap {
  font-size: 0.98rem !important;
  line-height: 1.55 !important;
}

#tslm-composer {
  animation: tslmRise 0.9s cubic-bezier(0.22, 1, 0.36, 1) 0.14s both;
  margin-top: 0.85rem !important;
}

#tslm-msg textarea {
  min-height: 3.1rem !important;
  font-size: 1.02rem !important;
  border-radius: 12px !important;
  border-color: var(--line) !important;
  background: rgba(255,255,255,0.88) !important;
  box-shadow: none !important;
  transition: border-color 0.2s ease, background 0.2s ease;
}

#tslm-msg textarea:focus {
  border-color: var(--accent-soft) !important;
  background: #fff !important;
  outline: none !important;
}

#tslm-send {
  min-height: 3.1rem !important;
  border-radius: 12px !important;
  font-weight: 700 !important;
  letter-spacing: 0.01em;
  transition: transform 0.18s ease, background 0.18s ease;
}

#tslm-send:hover {
  transform: translateY(-1px);
}

#tslm-tools {
  animation: tslmRise 0.95s cubic-bezier(0.22, 1, 0.36, 1) 0.18s both;
  margin-top: 0.55rem !important;
  gap: 0.55rem !important;
}

#tslm-tools button {
  border-radius: 10px !important;
  border: 1px solid var(--line) !important;
  font-weight: 600 !important;
  box-shadow: none !important;
}

#tslm-status {
  margin-top: 0.65rem !important;
  padding: 0.55rem 0.75rem !important;
  border-radius: 10px !important;
  background: rgba(36, 102, 92, 0.06) !important;
  border: 1px solid var(--line) !important;
  color: var(--ink-soft) !important;
  font-size: 0.88rem !important;
  font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
}

#tslm-status p {
  margin: 0 !important;
}

#tslm-advanced {
  margin-top: 0.85rem !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--radius) !important;
  background: rgba(255,255,255,0.42) !important;
  backdrop-filter: blur(8px);
}

#tslm-advanced summary,
#tslm-advanced .label-wrap {
  font-weight: 700 !important;
  color: var(--ink) !important;
}

#tslm-examples {
  margin-top: 0.75rem !important;
}

#tslm-examples button {
  border-radius: 10px !important;
  border: 1px solid var(--line) !important;
  background: rgba(255,255,255,0.55) !important;
  color: var(--ink-soft) !important;
  font-size: 0.86rem !important;
  font-weight: 500 !important;
  box-shadow: none !important;
  transition: background 0.18s ease, color 0.18s ease, border-color 0.18s ease;
}

#tslm-examples button:hover {
  background: rgba(255,255,255,0.9) !important;
  color: var(--accent) !important;
  border-color: var(--accent-soft) !important;
}

footer {
  display: none !important;
}

@keyframes tslmRise {
  from {
    opacity: 0;
    transform: translateY(12px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (max-width: 640px) {
  .gradio-container {
    padding: 0.85rem 0.7rem 1.75rem !important;
  }
  .tslm-brand {
    font-size: 2.1rem !important;
  }
  #tslm-tools {
    flex-direction: column !important;
  }
}
"""


def build_demo(chat: TinyChat) -> gr.Blocks:
    mem_path = ROOT / "checkpoints" / "memory_store.json"
    params = f"{chat.model.n_parameters():,}"
    device = str(chat.device)
    st0 = chat.memory.stats()
    boot_status = (
        f"ready · {params} params · {device} · "
        f"memory {st0['tokens']:,}/{st0['max_tokens']:,} tok · {st0['chunks']} chunks"
    )

    with gr.Blocks(title="TinySLM", fill_height=True) as demo:
        gr.HTML(
            f"""
            <div class="tslm-hero">
              <h1 class="tslm-brand">TinySLM</h1>
              <p class="tslm-tag">
                From-scratch SLM with 2M memory, skill cards, live search, and a parallel research swarm.
              </p>
              <div class="tslm-meta">
                <span>{params} params</span>
                <span>2M-token memory</span>
                <span>swarm + search</span>
              </div>
            </div>
            """
        )

        chatbot = gr.Chatbot(
            height=460,
            show_label=False,
            elem_id="tslm-chat",
            placeholder="Ask anything — short FAQ, math, plans, or a multi-part research brief.",
        )

        with gr.Row(elem_id="tslm-composer"):
            msg = gr.Textbox(
                placeholder="Message TinySLM… (say search / news to force lookup)",
                scale=5,
                show_label=False,
                elem_id="tslm-msg",
                autofocus=True,
            )
            send = gr.Button("Send", variant="primary", scale=1, elem_id="tslm-send")

        with gr.Row(elem_id="tslm-tools"):
            force_search = gr.Checkbox(label="Force search", value=False, scale=1)
            temperature = gr.Slider(
                0.1, 1.5, value=0.4, step=0.1, label="Temperature", scale=2
            )
            clear = gr.Button("Clear", scale=1)
            save_mem = gr.Button("Save memory", scale=1)
            load_mem = gr.Button("Load memory", scale=1)

        status = gr.Markdown(boot_status, elem_id="tslm-status")

        with gr.Accordion("Memory ingest", open=False, elem_id="tslm-advanced"):
            ingest_file = gr.File(
                label="Drop .txt / .md / .csv into the 2M memory store",
                file_types=[".txt", ".md", ".csv"],
                type="filepath",
            )

        examples = gr.Examples(
            examples=[
                ["What is RAM?"],
                ["Plan a productive weekend in three steps."],
                ["Write a Python function that safely divides two numbers."],
                [
                    "Design a multi-tenant SaaS notes API. Cover: "
                    "1) monolith vs modular monolith vs microservices "
                    "2) JWT vs session auth and tenancy "
                    "3) Python stack 4) two-week roadmap 5) security risks."
                ],
            ],
            inputs=[msg],
            elem_id="tslm-examples",
            label="Try",
        )

        def respond(message, history, do_search, temp):
            history = list(history or [])
            message = (message or "").strip()
            if not message:
                return history, "", boot_status
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
                f"{st['chunks']} chunks · history {len(chat.history)} turns · {device}"
            )
            return history, "", note

        def on_clear():
            chat.reset()
            chat.clear_memory()
            st = chat.memory.stats()
            return (
                [],
                f"cleared · skills re-seeded · memory {st['tokens']:,}/{st['max_tokens']:,} tok",
            )

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

        # Keep examples referenced so Gradio doesn't warn about unused nodes.
        _ = examples

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
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
        theme=_THEME,
        css=_CSS,
    )


if __name__ == "__main__":
    main()
