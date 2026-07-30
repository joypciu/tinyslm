# 100-turn long-chat stress test

- Turns: **40** in one continuous session
- Total time: **0.28s** (avg **0.007s**/turn, p95 **0.012s**)
- Neural block_size: **512** | prompt history window: **last 3 turns** (adaptive 2–3)
- Final memory: **4,748 tokens**, **73 chunks** (0.24% of 2M cap)
- Needle recall — memory retrieve: **2/2**, model reply: **2/2**
- Managed long chat: **True**

## How context is compressed

1. Up to the last 3 short chat turns enter the Transformer prompt (falls back to 2 when recent turns are long); block_size=512 truncates token ids from the left if needed.
2. Every turn is appended to LongContextMemory (up to 2,000,000 tokens). Older dialogue is not kept in the neural KV cache; it is retrieved by BM25 when the user asks about past facts, then answered extractively when possible.
3. If memory exceeds max_tokens, oldest chunks are evicted and the index rebuilt.

## Needle probes

| Planted turn | Code | Memory HIT | Model HIT |
|---:|---|:---:|:---:|
| 1 | `BLUE_LANTERN_CODE` | yes | yes |
| 25 | `ORANGE_RIVER_42` | yes | yes |

Full JSON: `long_chat_100_report.json`
