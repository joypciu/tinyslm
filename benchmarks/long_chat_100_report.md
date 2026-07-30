# 100-turn long-chat stress test

- Turns: **100** in one continuous session
- Total time: **88.55s** (avg **0.885s**/turn, p95 **2.381s**)
- Neural block_size: **256** | prompt history window: **last 2 turns**
- Final memory: **10,716 tokens**, **138 chunks** (0.54% of 2M cap)
- Needle recall — memory retrieve: **4/4**, model reply: **4/4**
- Managed long chat: **True**

## How context is compressed

1. Only the last 2 chat turns enter the Transformer prompt; block_size=256 truncates token ids from the left if needed.
2. Every turn is appended to LongContextMemory (up to 2,000,000 tokens). Older dialogue is not kept in the neural KV cache; it is retrieved by BM25 when the user asks about past facts.
3. If memory exceeds max_tokens, oldest chunks are evicted and the index rebuilt.

## Needle probes

| Planted turn | Code | Memory HIT | Model HIT |
|---:|---|:---:|:---:|
| 1 | `BLUE_LANTERN_CODE` | yes | yes |
| 25 | `ORANGE_RIVER_42` | yes | yes |
| 50 | `SILVER_OWL_99` | yes | yes |
| 75 | `GREEN_COMET_7` | yes | yes |

Full JSON: `long_chat_100_report.json`
