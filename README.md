# TinySLM

From-scratch small language model (~4.4M params) with **2M-token long memory**, DuckDuckGo search, inference-time skill cards, and a **parallel web research swarm**.

Repo: [joypciu/tinyslm](https://github.com/joypciu/tinyslm)

## Why this design

A ~4M-param model on 2–3 GB RAM cannot fully attend over 2,000,000 tokens (KV cache alone would be huge). TinySLM instead uses:

1. **Neural window** (RoPE + Multi-Query Attention + KV cache, `block_size=512`) for fast local decode
2. **LongContextMemory** rated for **2,000,000 tokens** with BM25-lite retrieval
3. Retrieved chunks + optional web search injected into the neural window
4. **Inference-time cards** (FAQ / math / plan / code) for reliable answers the tiny generator would miss
5. **Swarm** for complex multi-facet asks: decompose → parallel search/crawl → FastEmbed vector RAG → synthesize

Product quality is mostly **tooling + retrieval**, not raw neural generation.

## Features

- Custom Transformer SLM trained from scratch (no pretrained HF base weights)
- Expandable continual learning: **LoRA adapters**, optional layer growth, freeze-base FT
- Chat UI (Gradio) + CLI with memory save/load
- **LongContextMemory** (~2M tokens) with BM25 retrieval + extractive recall
- **SARA** agent loop (skills → tools → reflect) without growing params
- FAQ / math / plan / code fast-paths
- Auto DuckDuckGo when needed + extractive multi-snippet answers
- **Multi-agent swarm**: numbered facets → parallel workers → crawl + vector RAG + citations
- CPU-friendly (~4.4M params + LoRA, `block_size=512`)

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

## Checks

```bash
python smoke_tick.py
python eval_ready.py
python eval_advanced.py
python test_swarm_unit.py
python test_swarm_e2e.py          # needs network
python test_long_chat.py --turns 100
```

## Train / continual FT (optional)

Prefer inference-side fixes over narrow retrain (catastrophic forgetting risk).

```bash
python prepare_data.py --chat-only
python prepare_chat_smart.py
python prepare_agentic.py
# Optional HF curricula (Dolly + CodeAlpaca + rehearsal):
# python prepare_hf_intelligence.py

python train.py --steps 1500 --block-size 512
# Resume + rehearse at low LR:
python train.py --resume checkpoints/tinyslm.pt --reuse-tokenizer --steps 400 --lr 1e-4

# Safer continual FT (LoRA + freeze base):
python train.py --resume checkpoints/tinyslm.pt --reuse-tokenizer \
  --adapter-rank 8 --freeze-base --steps 400 --lr 1e-4 --backup
```

## Layout

```text
tiny_slm/           model, memory, agent, SARA, knowledge, chat, search
tiny_slm/swarm/     decompose → search/crawl/reader → FastEmbed store → synthesize
tiny_slm/adapters.py  LoRA / expandable continual FT helpers
checkpoints/        tinyslm.pt + tokenizer (+ optional memory_store.json)
prepare_*.py        curricula
train.py            train / resume / LoRA / grow-layers
app.py              Gradio UI
chat_cli.py         terminal chat
smoke_tick.py       fast regression smoke
eval_*.py           readiness / advanced / thorough suites
test_swarm_*.py     swarm unit + e2e
```

## Swarm (complex asks)

For design/research briefs with numbered facets (architecture, auth, stack, roadmap, risks, …), chat routes to the swarm after short cards:

1. Decompose into focused sub-goals (checklist items win over naive “A vs B”)
2. Parallel search + bounded page crawl (chrome / video hosts filtered)
3. FastEmbed cosine index over chunks
4. Per-facet reader notes → cited research summary

Requires `numpy` + `fastembed` (already in `requirements.txt`).
