# TinySLM

From-scratch small language model with **2M-token long memory**, DuckDuckGo search, and a lightweight **agentic** tool loop.

Repo: [joypciu/tinyslm](https://github.com/joypciu/tinyslm)

## Why this design

A ~4M-param model on 2–3 GB RAM cannot fully attend over 2,000,000 tokens (KV cache alone would be huge). TinySLM instead uses:

1. **Neural window** (RoPE + Multi-Query Attention + KV cache) for fast local decode  
2. **LongContextMemory** rated for **2,000,000 tokens** with BM25-lite retrieval  
3. Retrieved chunks + optional web search injected into the neural window  

## Features

- Custom Transformer SLM trained from scratch (no pretrained HF weights)
- Chat UI (Gradio) + CLI with memory save/load
- **LongContextMemory** (~2M tokens) with BM25 retrieval + extractive recall
- **SARA** agent loop (skills → tools → reflect) without growing params
- FAQ / math / plan / code fast-paths (inference-time; avoid brittle generator misses)
- Auto DuckDuckGo when needed + extractive multi-snippet answers (news-aware)
- CPU-friendly (~4.4M params, `block_size=256`)
- Regression gate: `python smoke_tick.py` / `python eval_ready.py`

## Quick start

```bash
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:7860
```

Or terminal chat:

```bash
python chat_cli.py
```

Rebuild data + train (optional):

```bash
python prepare_data.py --chat-only
python prepare_chat_smart.py
python prepare_agentic.py
python train.py --steps 1500 --block-size 256
# Prefer inference-side fixes over narrow retrain (catastrophic forgetting risk).
# If you must fine-tune, resume + rehearse old chat data at low LR:
python train.py --resume checkpoints/tinyslm.pt --reuse-tokenizer --steps 400 --lr 1e-4
python smoke_tick.py
python eval_ready.py
python test_long_chat.py --turns 100
```

## Layout

```text
tiny_slm/     model, memory, agent, SARA, knowledge, chat, search
checkpoints/  tinyslm.pt + tokenizer (+ optional memory_store.json)
prepare_*.py  curricula
train.py      train / resume
app.py        Gradio UI
chat_cli.py   terminal chat
smoke_tick.py fast regression smoke
```
