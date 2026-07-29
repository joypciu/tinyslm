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
- Chat UI (Gradio) + CLI
- DuckDuckGo search
- Agentic plan → memory/search → synthesize
- CPU-friendly (~17–25 MB float32 weights)

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
# fine-tune without wiping weights:
python train.py --resume checkpoints/tinyslm.pt --reuse-tokenizer --steps 400 --lr 1e-4
python eval_ready.py
```

## Layout

```text
tiny_slm/     model, memory, agent, chat, search
checkpoints/  tinyslm.pt + tokenizer
prepare_*.py  curricula
train.py      train / resume
app.py        Gradio UI
```
